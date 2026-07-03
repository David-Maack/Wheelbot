"""Universe refresh — overlay, guardrails, persistence, apply semantics."""

from __future__ import annotations

from datetime import UTC, date, datetime
from typing import Any

import pandas as pd
import pytest

from core.models import (
    Position,
    PositionState,
    Quote,
    UniverseEntry,
    WatchlistEntry,
    WatchlistRun,
    WatchlistRunStatus,
)
from core.watchlists import effective_universe, overlay_universe
from intelligence.universe_refresh import enforce_guardrails, run_universe_refresh

UNIVERSE_YAML = """\
tickers:
  - symbol: AAA
    tier: 1
    strategies: [strat_a]
  - symbol: BBB
    tier: 1
    strategies: [strat_a]
  - symbol: CCC
    tier: 2
    strategies: [strat_b]
  - symbol: DDD
    tier: 1
    strategies: [strat_a, strat_b]
  - symbol: PARKED3
    tier: 3
    strategies: []
banned:
  - GME
"""

CONFIG: dict[str, Any] = {
    "account": {"id": "primary"},
    "strategies": [
        {"id": "strat_a", "display_name": "A", "type": "wheel", "enabled": True,
         "max_concurrent": 2, "params": {"dte_min": 30}},
        {"id": "strat_b", "display_name": "B", "type": "vertical_spread", "enabled": True,
         "max_concurrent": 2, "params": {"direction": "bull_put"}},
    ],
    "universe_refresh": {
        "enabled": True,
        "auto_apply": False,
        "max_adds_per_strategy": 2,
        "max_drops_per_strategy": 2,
        "min_symbols_per_strategy": 1,
        "exclude_strategies": [],
        "min_price": 5.0,
        "max_price": 1200.0,
        "min_median_volume": 5_000_000,
        "earnings_min_days": 7,
        "candidate_pool": ["NEWT"],
        "pinned": {},
    },
}


def _universe() -> dict[str, Any]:
    return {
        "tickers": [
            UniverseEntry(symbol="AAA", tier=1, strategies=["strat_a"]),
            UniverseEntry(symbol="BBB", tier=1, strategies=["strat_a"]),
            UniverseEntry(symbol="CCC", tier=2, strategies=["strat_b"]),
            UniverseEntry(symbol="DDD", tier=1, strategies=["strat_a", "strat_b"]),
            UniverseEntry(symbol="PARKED3", tier=3, strategies=[]),
        ],
        "banned": ["GME"],
        "banned_rules": [],
    }


@pytest.fixture
def config_dir(tmp_path, monkeypatch):
    (tmp_path / "universe.yaml").write_text(UNIVERSE_YAML, encoding="utf-8")
    (tmp_path / "config.yaml").write_text("{}", encoding="utf-8")
    monkeypatch.setenv("WHEELBOT_CONFIG_DIR", str(tmp_path))
    return tmp_path


# ---------------------------------------------------------------- overlay ----

def test_overlay_swaps_membership_for_refreshed_strategies():
    out = overlay_universe(_universe(), {"strat_a": ["AAA", "CCC"]})
    by = {t.symbol: t for t in out["tickers"]}
    assert "strat_a" in by["AAA"].strategies
    assert "strat_a" not in by["BBB"].strategies         # dropped
    assert "strat_a" in by["CCC"].strategies             # added
    assert by["CCC"].strategies.count("strat_b") == 1    # untouched tag preserved
    # strat_b was not refreshed — its yaml membership is intact.
    assert "strat_b" in by["DDD"].strategies
    assert "strat_a" not in by["DDD"].strategies         # DDD dropped from strat_a


def test_overlay_synthesizes_tier2_for_unknown_symbols():
    out = overlay_universe(_universe(), {"strat_a": ["AAA", "ZZZT"]})
    by = {t.symbol: t for t in out["tickers"]}
    assert by["ZZZT"].tier == 2
    assert by["ZZZT"].strategies == ["strat_a"]


def test_overlay_refuses_banned_and_tier3():
    out = overlay_universe(_universe(), {"strat_a": ["AAA", "GME", "PARKED3"]})
    by = {t.symbol: t for t in out["tickers"]}
    assert "GME" not in by
    assert by["PARKED3"].strategies == []
    assert [t for t in out["tickers"] if "strat_a" in t.strategies] == [by["AAA"]]


@pytest.mark.asyncio
async def test_effective_universe_disabled_returns_yaml(db_repos, config_dir):
    config = {"universe_refresh": {"enabled": False}}
    out = await effective_universe(db_repos, config)
    by = {t.symbol: t for t in out["tickers"]}
    assert "strat_a" in by["BBB"].strategies


@pytest.mark.asyncio
async def test_effective_universe_overlays_applied_run(db_repos, config_dir):
    run_id = await db_repos.watchlists.insert_run(WatchlistRun(
        run_date=date(2026, 7, 4), status=WatchlistRunStatus.PROPOSED,
        created_at=datetime.now(UTC),
    ))
    for sym, action in (("AAA", "keep"), ("BBB", "drop"), ("CCC", "add")):
        await db_repos.watchlists.insert_entry(WatchlistEntry(
            run_id=run_id, strategy_id="strat_a", symbol=sym, action=action,
        ))
    # Proposed runs are NOT consumed.
    out = await effective_universe(db_repos, CONFIG)
    assert "strat_a" in {t.symbol: t for t in out["tickers"]}["BBB"].strategies

    await db_repos.watchlists.apply_run(run_id, applied_by="test")
    out = await effective_universe(db_repos, CONFIG)
    by = {t.symbol: t for t in out["tickers"]}
    assert "strat_a" not in by["BBB"].strategies
    assert "strat_a" in by["CCC"].strategies
    assert "strat_b" in by["DDD"].strategies  # non-refreshed strategy untouched


@pytest.mark.asyncio
async def test_apply_run_supersedes_previous(db_repos):
    first = await db_repos.watchlists.insert_run(WatchlistRun(
        run_date=date(2026, 7, 4), created_at=datetime.now(UTC)))
    second = await db_repos.watchlists.insert_run(WatchlistRun(
        run_date=date(2026, 7, 11), created_at=datetime.now(UTC)))
    await db_repos.watchlists.apply_run(first, applied_by="test")
    await db_repos.watchlists.apply_run(second, applied_by="test")
    assert str((await db_repos.watchlists.get_run(first)).status) == "superseded"
    assert str((await db_repos.watchlists.get_run(second)).status) == "applied"
    applied = await db_repos.watchlists.latest_run(status="applied")
    assert applied.id == second


# ------------------------------------------------------------- guardrails ----

def _guard(parsed, **over):
    kwargs = dict(
        current={"strat_a": ["AAA", "BBB", "DDD"]},
        protected={"strat_a": set()},
        eligible_adds={"CCC", "NEWT", "EEE"},
        known_strategies={"strat_a"},
        max_adds=2,
        max_drops=2,
        min_symbols=1,
    )
    kwargs.update(over)
    return enforce_guardrails(parsed, **kwargs)


def _actions(rows):
    return {r["symbol"]: r["action"] for r in rows}


def test_guardrails_protected_symbols_cannot_be_dropped():
    parsed = {"watchlists": [{"strategy_id": "strat_a", "symbols": [
        {"symbol": "AAA", "action": "drop", "score": 10},
        {"symbol": "BBB", "action": "keep", "score": 70},
        {"symbol": "DDD", "action": "keep", "score": 70},
    ]}]}
    result, notes = _guard(parsed, protected={"strat_a": {"AAA"}})
    assert _actions(result["strat_a"])["AAA"] == "keep"
    assert any("vetoed" in n for n in notes)


def test_guardrails_add_requires_quant_eligibility():
    parsed = {"watchlists": [{"strategy_id": "strat_a", "symbols": [
        {"symbol": "ILLIQ", "action": "add", "score": 90},
        {"symbol": "NEWT", "action": "add", "score": 80},
    ]}]}
    result, notes = _guard(parsed)
    actions = _actions(result["strat_a"])
    assert "ILLIQ" not in actions
    assert actions["NEWT"] == "add"
    assert any("failed quant gate" in n for n in notes)


def test_guardrails_churn_caps_clamp_adds_and_drops():
    parsed = {"watchlists": [{"strategy_id": "strat_a", "symbols": [
        {"symbol": "CCC", "action": "add", "score": 90},
        {"symbol": "NEWT", "action": "add", "score": 85},
        {"symbol": "EEE", "action": "add", "score": 80},   # 3rd add → clamped
        {"symbol": "AAA", "action": "drop", "score": 10},
        {"symbol": "BBB", "action": "drop", "score": 20},
        {"symbol": "DDD", "action": "drop", "score": 30},  # 3rd drop → kept
    ]}]}
    result, _ = _guard(parsed, min_symbols=0)
    actions = _actions(result["strat_a"])
    assert "EEE" not in actions                    # lowest-score add clamped
    assert actions["CCC"] == "add" and actions["NEWT"] == "add"
    assert actions["DDD"] == "keep"                # least-confident drop clamped
    assert actions["AAA"] == "drop" and actions["BBB"] == "drop"


def test_guardrails_min_size_reverts_drops():
    parsed = {"watchlists": [{"strategy_id": "strat_a", "symbols": [
        {"symbol": "AAA", "action": "drop", "score": 10},
        {"symbol": "BBB", "action": "drop", "score": 20},
        {"symbol": "DDD", "action": "keep", "score": 60},
    ]}]}
    result, notes = _guard(parsed, min_symbols=2)
    actions = _actions(result["strat_a"])
    # 3 members - 2 drops = 1 < 2 → the least-confident drop (BBB) reverts.
    assert actions["AAA"] == "drop"
    assert actions["BBB"] == "keep"
    assert any("min size" in n for n in notes)


def test_guardrails_omitted_members_are_kept():
    parsed = {"watchlists": [{"strategy_id": "strat_a", "symbols": [
        {"symbol": "AAA", "action": "keep", "score": 70},
        # BBB and DDD omitted by the model
    ]}]}
    result, _ = _guard(parsed)
    actions = _actions(result["strat_a"])
    assert actions == {"AAA": "keep", "BBB": "keep", "DDD": "keep"}


def test_guardrails_unknown_strategy_ignored():
    parsed = {"watchlists": [{"strategy_id": "made_up", "symbols": [
        {"symbol": "AAA", "action": "add", "score": 90},
    ]}]}
    result, notes = _guard(parsed)
    assert "made_up" not in result
    assert any("unknown" in n for n in notes)


# ------------------------------------------------------------ end-to-end ----

class _StubBroker:
    name = "stub"

    async def get_quote(self, symbol: str) -> Quote:
        return Quote(symbol=symbol, bid=19.9, ask=20.1)


class _StubIvr:
    async def stats(self, symbol):
        from data.ivr import IvStats

        return IvStats(current=0.30, low=0.20, high=0.40, n_points=30, rank=45.0, percentile=55.0)


class _StubAnthropic:
    def __init__(self, response: dict[str, Any]):
        self._response = response
        self.calls: list[dict[str, Any]] = []

    async def call(self, **kwargs):
        self.calls.append(kwargs)
        return {
            "parsed": self._response,
            "raw_text": "",
            "tokens_in": 1000,
            "tokens_out": 500,
            "cost_usd": 0.10,
            # None — the real AnthropicClient inserts the llm_decisions row this
            # id points at; the stub doesn't, and the FK is enforced in tests.
            "decision_id": None,
        }


class _StubEarnings:
    next_date = None
    source = "stub"


@pytest.fixture
def patched_data(monkeypatch):
    vol = pd.DataFrame({"Volume": [6_000_000] * 60, "Close": [20.0] * 60})
    monkeypatch.setattr("intelligence.universe_refresh.safe_history", lambda *a, **k: vol)
    monkeypatch.setattr(
        "intelligence.universe_refresh.next_earnings", lambda *a, **k: _StubEarnings()
    )


@pytest.mark.asyncio
async def test_run_refresh_skipped_when_disabled(db_repos):
    result = await run_universe_refresh(
        broker=_StubBroker(), repos=db_repos, ivr=_StubIvr(),
        anthropic=_StubAnthropic({}), config={"universe_refresh": {"enabled": False}},
    )
    assert result == {"skipped": "disabled"}


@pytest.mark.asyncio
async def test_run_refresh_writes_proposed_run(db_repos, config_dir, patched_data):
    stub = _StubAnthropic({
        "decision": "refresh_complete",
        "summary": "swap BBB for NEWT on strat_a",
        "watchlists": [{
            "strategy_id": "strat_a",
            "symbols": [
                {"symbol": "AAA", "action": "keep", "score": 70, "rationale": "solid"},
                {"symbol": "BBB", "action": "drop", "score": 20, "rationale": "IV dead"},
                {"symbol": "DDD", "action": "keep", "score": 60, "rationale": "fine"},
                {"symbol": "NEWT", "action": "add", "score": 85, "rationale": "rich IV"},
            ],
        }],
    })
    result = await run_universe_refresh(
        broker=_StubBroker(), repos=db_repos, ivr=_StubIvr(),
        anthropic=stub, config=CONFIG, run_date=date(2026, 7, 4),
    )
    assert result["status"] == "proposed"
    assert result["adds"] == 1 and result["drops"] == 1
    # Proposed ≠ applied: membership is unchanged until approval.
    assert await db_repos.watchlists.applied_membership() == {}
    entries = await db_repos.watchlists.entries_for_run(result["run_id"])
    by_sym = {(e.strategy_id, e.symbol): e.action for e in entries}
    assert by_sym[("strat_a", "NEWT")] == "add"
    assert by_sym[("strat_a", "BBB")] == "drop"
    # The candidate payload excluded banned + tier-3 names.
    sent = stub.calls[0]["user_payload"]
    sent_symbols = {c["symbol"] for c in sent["candidates"]}
    assert "GME" not in sent_symbols and "PARKED3" not in sent_symbols
    assert "NEWT" in sent_symbols


@pytest.mark.asyncio
async def test_run_refresh_auto_apply(db_repos, config_dir, patched_data):
    config = {**CONFIG, "universe_refresh": {**CONFIG["universe_refresh"], "auto_apply": True}}
    stub = _StubAnthropic({
        "decision": "refresh_complete",
        "summary": "add NEWT",
        "watchlists": [{
            "strategy_id": "strat_a",
            "symbols": [
                {"symbol": "AAA", "action": "keep", "score": 70},
                {"symbol": "BBB", "action": "keep", "score": 60},
                {"symbol": "DDD", "action": "keep", "score": 60},
                {"symbol": "NEWT", "action": "add", "score": 85},
            ],
        }],
    })
    result = await run_universe_refresh(
        broker=_StubBroker(), repos=db_repos, ivr=_StubIvr(),
        anthropic=stub, config=config, run_date=date(2026, 7, 4),
    )
    assert result["status"] == "applied"
    membership = await db_repos.watchlists.applied_membership()
    assert set(membership["strat_a"]) == {"AAA", "BBB", "DDD", "NEWT"}


@pytest.mark.asyncio
async def test_run_refresh_open_position_cannot_be_dropped(db_repos, config_dir, patched_data):
    await db_repos.positions.insert(Position(
        account_id="primary", symbol="BBB", strategy_id="strat_a",
        state=PositionState.CSP_OPEN, state_changed_at=datetime.now(UTC).replace(tzinfo=None),
    ))
    stub = _StubAnthropic({
        "decision": "refresh_complete",
        "summary": "drop BBB",
        "watchlists": [{
            "strategy_id": "strat_a",
            "symbols": [
                {"symbol": "AAA", "action": "keep", "score": 70},
                {"symbol": "BBB", "action": "drop", "score": 10},
                {"symbol": "DDD", "action": "keep", "score": 60},
            ],
        }],
    })
    result = await run_universe_refresh(
        broker=_StubBroker(), repos=db_repos, ivr=_StubIvr(),
        anthropic=stub, config=CONFIG, run_date=date(2026, 7, 4),
    )
    entries = await db_repos.watchlists.entries_for_run(result["run_id"])
    bbb = next(e for e in entries if e.symbol == "BBB")
    assert bbb.action == "keep"
    assert any("vetoed" in n for n in result["guardrail_notes"])


@pytest.mark.asyncio
async def test_run_refresh_parse_failure_marks_run_failed(db_repos, config_dir, patched_data):
    result = await run_universe_refresh(
        broker=_StubBroker(), repos=db_repos, ivr=_StubIvr(),
        anthropic=_StubAnthropic({"text": "not json"}),
        config=CONFIG, run_date=date(2026, 7, 4),
    )
    assert result["status"] == "failed"
    run = await db_repos.watchlists.get_run(result["run_id"])
    assert str(run.status) == "failed"
    # Fail-open: nothing applied, membership untouched.
    assert await db_repos.watchlists.applied_membership() == {}


# ------------------------------------------------------------- discovery ----

def _discovery_config(**disc_over):
    disc = {"enabled": True, "top_n": 10, "max_new_candidates": 2, "chain_spread_max_pct": 15.0}
    disc.update(disc_over)
    return {**CONFIG, "universe_refresh": {**CONFIG["universe_refresh"], "discovery": disc}}


def _keep_all_response():
    return {
        "decision": "refresh_complete",
        "summary": "no changes",
        "watchlists": [{
            "strategy_id": "strat_a",
            "symbols": [
                {"symbol": "AAA", "action": "keep", "score": 70},
                {"symbol": "BBB", "action": "keep", "score": 60},
                {"symbol": "DDD", "action": "keep", "score": 60},
            ],
        }],
    }


class _TradableChainBroker(_StubBroker):
    """Every symbol has a tight-quoted option chain."""

    async def get_option_chain(self, underlying, expiration=None, option_type=None):
        from datetime import date as _date

        from core.models import OptionContract, OptionType

        return [OptionContract(
            underlying=underlying, occ_symbol=f"{underlying}260717P00010000",
            strike=10.0, expiration=_date(2026, 7, 17), option_type=OptionType.PUT,
            bid=1.00, ask=1.05,
        )]


class _NoChainBroker(_StubBroker):
    async def get_option_chain(self, underlying, expiration=None, option_type=None):
        return []


@pytest.mark.asyncio
async def test_discovery_feeds_capped_new_names_to_payload(
    db_repos, config_dir, patched_data, monkeypatch
):
    async def fake_discover(config, **kw):
        return ["NEW1", "NEW2", "NEW3", "GME", "PARKED3", "AAA"]

    monkeypatch.setattr("intelligence.universe_refresh.discover_candidates", fake_discover)
    stub = _StubAnthropic(_keep_all_response())
    result = await run_universe_refresh(
        broker=_TradableChainBroker(), repos=db_repos, ivr=_StubIvr(),
        anthropic=stub, config=_discovery_config(), run_date=date(2026, 7, 4),
    )
    assert result["status"] in ("no_changes", "proposed")
    sent = stub.calls[0]["user_payload"]
    by_symbol = {c["symbol"]: c for c in sent["candidates"]}
    # Banned (GME), tier-3 (PARKED3) never entered; AAA was already in the pool
    # so it is NOT flagged as discovered.
    assert "GME" not in by_symbol and "PARKED3" not in by_symbol
    assert by_symbol["AAA"]["newly_discovered"] is False
    # max_new_candidates=2 → exactly two discovered names survive, flagged.
    discovered = [s for s, c in by_symbol.items() if c["newly_discovered"]]
    assert len(discovered) == 2
    assert set(discovered) <= {"NEW1", "NEW2", "NEW3"}
    assert all(by_symbol[s]["add_eligible"] for s in discovered)


@pytest.mark.asyncio
async def test_discovery_requires_tradable_chain(db_repos, config_dir, patched_data, monkeypatch):
    async def fake_discover(config, **kw):
        return ["NEW1"]

    monkeypatch.setattr("intelligence.universe_refresh.discover_candidates", fake_discover)
    stub = _StubAnthropic(_keep_all_response())
    await run_universe_refresh(
        broker=_NoChainBroker(), repos=db_repos, ivr=_StubIvr(),
        anthropic=stub, config=_discovery_config(), run_date=date(2026, 7, 4),
    )
    sent = stub.calls[0]["user_payload"]
    # No options market → dropped from the payload entirely.
    assert "NEW1" not in {c["symbol"] for c in sent["candidates"]}


@pytest.mark.asyncio
async def test_discovery_failure_falls_back_to_curated_pool(
    db_repos, config_dir, patched_data, monkeypatch
):
    async def boom(config, **kw):
        return []  # discover_candidates itself fails open to []

    monkeypatch.setattr("intelligence.universe_refresh.discover_candidates", boom)
    stub = _StubAnthropic(_keep_all_response())
    result = await run_universe_refresh(
        broker=_TradableChainBroker(), repos=db_repos, ivr=_StubIvr(),
        anthropic=stub, config=_discovery_config(), run_date=date(2026, 7, 4),
    )
    assert result["status"] in ("no_changes", "proposed")
    sent = stub.calls[0]["user_payload"]
    # The curated pool still went through.
    assert {"AAA", "BBB", "CCC", "DDD"} <= {c["symbol"] for c in sent["candidates"]}


@pytest.mark.asyncio
async def test_discovery_disabled_by_default(db_repos, config_dir, patched_data, monkeypatch):
    called = False

    async def fake_discover(config, **kw):
        nonlocal called
        called = True
        return ["NEW1"]

    monkeypatch.setattr("intelligence.universe_refresh.discover_candidates", fake_discover)
    stub = _StubAnthropic(_keep_all_response())
    # CONFIG has no discovery block → tier 0 must not run at all.
    await run_universe_refresh(
        broker=_TradableChainBroker(), repos=db_repos, ivr=_StubIvr(),
        anthropic=stub, config=CONFIG, run_date=date(2026, 7, 4),
    )
    assert called is False
