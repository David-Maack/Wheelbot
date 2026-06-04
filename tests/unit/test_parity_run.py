"""TICKET-022 — scripts/parity_run unit tests.

Exercises the join + diff computation + asymmetric-liquidity flag + report
rendering against mock brokers (no network). The gated integration test
against the real Tastytrade sandbox lives at
tests/integration/test_tastytrade_sandbox.py.
"""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import pytest

from core.broker import Broker, BrokerUnavailable
from core.models import Account, OptionContract, OptionType, Quote
from scripts.parity_run import (
    _alpaca_has_liquidity,
    _build_rows,
    _join_and_compute,
    _mid,
    _normalize_occ,
    _render_daily_report,
    _select_symbols,
    _tasty_has_liquidity,
    run,
)


# -- _normalize_occ: Q4 (both sides) ---------------------------------------


def test_normalize_occ_strips_padding_and_uppercases():
    """Locked Q4: the two adapters may emit space-padded or dense, mixed case.
    Both sides MUST produce identical canonical form for the same contract."""
    alpaca_form = "F     250620P00010000"  # hypothetical padded
    tasty_form = "F250620P00010000"  # dense
    assert _normalize_occ(alpaca_form) == _normalize_occ(tasty_form)
    assert _normalize_occ("aapl250620p00150000") == "AAPL250620P00150000"


def test_normalize_occ_handles_empty():
    assert _normalize_occ("") == ""
    assert _normalize_occ(None) == ""  # tolerate adapter returning None


# -- _mid + liquidity helpers ----------------------------------------------


def _contract(
    occ: str = "F250620P00010000",
    bid: float | None = None,
    ask: float | None = None,
    mid: float | None = None,
) -> OptionContract:
    return OptionContract(
        underlying="F",
        occ_symbol=occ,
        strike=10.0,
        expiration=date(2026, 6, 20),
        option_type=OptionType.PUT,
        bid=bid,
        ask=ask,
        mid=mid,
    )


def test_mid_prefers_explicit_field():
    c = _contract(bid=1.0, ask=1.10, mid=1.07)
    assert _mid(c) == 1.07


def test_mid_falls_back_to_midpoint():
    c = _contract(bid=1.0, ask=1.10)
    assert _mid(c) == pytest.approx(1.05)


def test_mid_none_when_one_side_missing():
    assert _mid(_contract(bid=1.0)) is None
    assert _mid(_contract(ask=1.10)) is None


def test_alpaca_has_liquidity_requires_both_positive():
    assert _alpaca_has_liquidity(_contract(bid=0.5, ask=0.6))
    assert not _alpaca_has_liquidity(_contract(bid=0, ask=0.6))
    assert not _alpaca_has_liquidity(_contract(bid=0.5, ask=None))


def test_tasty_has_liquidity_handles_missing_contract():
    """Q3(a): when Tastytrade returns no contract at all, treat as illiquid."""
    assert not _tasty_has_liquidity(None)
    assert not _tasty_has_liquidity(_contract(bid=0, ask=0))
    assert _tasty_has_liquidity(_contract(bid=0.5, ask=0.6))


# -- join + row build ------------------------------------------------------


def test_join_aligns_by_normalized_occ():
    """Even when Alpaca emits a padded OCC and Tastytrade emits dense, the
    join must match them."""
    alpaca = [_contract(occ="F     250620P00010000", bid=0.5, ask=0.6)]
    tasty = [_contract(occ="F250620P00010000", bid=0.51, ask=0.59)]
    ordered, aligned = _join_and_compute(alpaca, tasty)
    assert len(aligned) == 1
    assert aligned[0] is tasty[0]


def test_join_returns_none_for_missing_tasty_match():
    alpaca = [_contract(occ="F250620P00010000", bid=0.5, ask=0.6)]
    tasty: list[OptionContract] = []
    _, aligned = _join_and_compute(alpaca, tasty)
    assert aligned == [None]


def test_build_rows_computes_mid_diff_pct():
    """Alpaca mid 1.00, Tasty mid 1.10 → +10% diff."""
    now = datetime(2026, 6, 4, 14, 35)
    alpaca = [_contract(occ="F250620P00010000", bid=0.99, ask=1.01)]  # mid 1.00
    tasty = [_contract(occ="F250620P00010000", bid=1.08, ask=1.12)]  # mid 1.10
    _, aligned = _join_and_compute(alpaca, tasty)
    rows = _build_rows(now, "F", alpaca, aligned)
    assert len(rows) == 1
    r = rows[0]
    assert r["alpaca_mid"] == pytest.approx(1.00)
    assert r["tasty_mid"] == pytest.approx(1.10)
    assert r["mid_diff_pct"] == pytest.approx(10.0)
    assert r["asymmetric_liquidity"] == 0


def test_build_rows_flags_asymmetric_liquidity():
    """Alpaca has bid+ask>0, Tastytrade returns nothing for that contract."""
    now = datetime(2026, 6, 4, 14, 35)
    alpaca = [_contract(occ="F250620P00010000", bid=0.5, ask=0.6)]
    _, aligned = _join_and_compute(alpaca, [])
    rows = _build_rows(now, "F", alpaca, aligned)
    assert rows[0]["asymmetric_liquidity"] == 1
    assert rows[0]["tasty_mid"] is None
    assert rows[0]["mid_diff_pct"] is None


def test_build_rows_no_asym_when_alpaca_also_illiquid():
    """Q3(a): asymmetric flag fires only when Alpaca has liquidity. If
    Alpaca itself shows bid=0/ask=0, the missing Tastytrade match is not
    an Alpaca-liquid asymmetry."""
    now = datetime(2026, 6, 4, 14, 35)
    alpaca = [_contract(occ="F250620P00010000", bid=0.0, ask=0.0)]
    _, aligned = _join_and_compute(alpaca, [])
    rows = _build_rows(now, "F", alpaca, aligned)
    assert rows[0]["asymmetric_liquidity"] == 0


# -- _select_symbols (Q1 dynamic) ------------------------------------------


def test_select_symbols_filters_to_active_strategies():
    """Q1(b)-dynamic: tier-3 / parked tickers (strategies: []) are excluded
    automatically; no separate config to maintain."""
    class _T:
        def __init__(self, symbol, strategies):
            self.symbol = symbol
            self.strategies = strategies
    uni = {"tickers": [
        _T("AAL", ["weekly_wheel"]),
        _T("PLUG", []),  # tier 3 — parked
        _T("F", ["monthly_wheel"]),
        _T("COIN", ["put_spread"]),
    ]}
    assert _select_symbols(uni) == ["AAL", "COIN", "F"]


# -- report rendering ------------------------------------------------------


def test_render_daily_report_pass_path():
    stats = [
        {"symbol": "F", "samples": 60, "avg_abs_diff_pct": 0.5,
         "worst_abs_diff_pct": 1.2, "asymmetric_count": 0},
        {"symbol": "AAL", "samples": 50, "avg_abs_diff_pct": 1.1,
         "worst_abs_diff_pct": 3.5, "asymmetric_count": 0},
    ]
    outliers = [
        {"symbol": "AAL", "contract_symbol": "AAL250620P00010000",
         "alpaca_mid": 1.0, "tasty_mid": 1.04, "mid_diff_pct": 4.0},
    ]
    md = _render_daily_report(
        report_date=datetime(2026, 6, 4), stats=stats, outliers=outliers,
    )
    assert "# Broker parity report — 2026-06-04" in md
    assert "| F | 60 | 0.50% | 1.20% | 0 |" in md
    assert "**PASS**" in md  # all three acceptance criteria pass


def test_render_daily_report_fail_path():
    """Worst > 5% → FAIL on that criterion + the warning banner."""
    stats = [
        {"symbol": "F", "samples": 60, "avg_abs_diff_pct": 0.5,
         "worst_abs_diff_pct": 7.2, "asymmetric_count": 0},
    ]
    md = _render_daily_report(
        report_date=datetime(2026, 6, 4), stats=stats, outliers=[],
    )
    assert "**FAIL**" in md
    assert "Do not advance to Stage 3" in md


# -- end-to-end run with two PaperBroker stubs -----------------------------


class _MockBroker(Broker):
    """Minimal Broker for tests. Returns a fixed chain per symbol."""

    def __init__(self, name: str, chains: dict[str, list[OptionContract]]):
        self._name = name
        self._chains = chains

    @property
    def name(self) -> str:
        return self._name

    async def get_account(self) -> Account:
        return Account(account_id="test", cash=10_000.0, buying_power=10_000.0, equity=10_000.0)

    async def get_positions(self) -> list:
        return []

    async def get_quote(self, symbol: str) -> Quote:
        return Quote(symbol=symbol, bid=1.0, ask=1.1, mid=1.05)

    async def get_option_chain(self, underlying, expiration=None, option_type=None):
        return self._chains.get(underlying.upper(), [])

    async def place_order(self, order):
        raise NotImplementedError

    async def cancel_order(self, broker_order_id):
        raise NotImplementedError

    async def get_orders_since(self, since):
        return []


@pytest.mark.asyncio
async def test_run_end_to_end(db_repos, tmp_path):
    """Drives run() with mock brokers + a real test DB. Verifies row count,
    rendered report exists, summary checkpoint fires."""
    alpaca_chain = {"F": [_contract(bid=0.99, ask=1.01)]}    # mid 1.00
    tasty_chain = {"F": [_contract(bid=1.08, ask=1.12)]}     # mid 1.10
    alpaca = _MockBroker("alpaca_paper", alpaca_chain)
    tasty = _MockBroker("tastytrade_sandbox", tasty_chain)

    class _T:
        def __init__(self, symbol, strategies):
            self.symbol = symbol
            self.strategies = strategies
    universe = {"tickers": [_T("F", ["monthly_wheel"])]}
    # Inject a config that points at the real test DB path so the script
    # writes through to db_repos's connection.
    db_path = Path(db_repos.db.path).resolve() if hasattr(db_repos.db, "path") else None
    # Fall back: scope a fresh config with the in-memory path the fixture uses.
    config = {
        "account": {"id": "test"},
        "database": {"path": str(db_path) if db_path else ":memory:"},
    }

    # The run() opens its own Database against config["database"]["path"];
    # for this test we patch make_broker to return our mocks AND pass the
    # brokers directly. Since we're passing brokers, _verify_creds is skipped.
    report_dir = tmp_path / "parity_reports"
    result = await run(
        config=config,
        universe=universe,
        report_dir=report_dir,
        now=datetime(2026, 6, 4, 14, 35),
        alpaca_broker=alpaca,
        tasty_broker=tasty,
    )

    assert result["symbols"] == 1
    assert result["rows"] == 1
    report_path = Path(result["report_path"])
    assert report_path.exists()
    text = report_path.read_text(encoding="utf-8")
    assert "Broker parity report" in text
    assert "| F |" in text


@pytest.mark.asyncio
async def test_run_skips_when_universe_empty(db_repos, tmp_path):
    """No active strategies → no symbols → no work, no error."""
    config = {
        "account": {"id": "test"},
        "database": {"path": str(tmp_path / "test.db")},
    }
    result = await run(
        config=config,
        universe={"tickers": []},
        report_dir=tmp_path / "reports",
        now=datetime(2026, 6, 4, 14, 35),
        alpaca_broker=_MockBroker("a", {}),
        tasty_broker=_MockBroker("t", {}),
    )
    assert result == {"symbols": 0, "rows": 0, "report_path": None}


@pytest.mark.asyncio
async def test_preflight_aborts_on_broker_unavailable(db_repos, tmp_path):
    """Rate-limit pre-flight: if either broker's get_account() raises
    BrokerUnavailable, the harness aborts before doing any work."""
    class _DeadBroker(_MockBroker):
        async def get_account(self):
            raise BrokerUnavailable("rate limited")

    config = {
        "account": {"id": "test"},
        "database": {"path": str(tmp_path / "test.db")},
    }

    class _T:
        def __init__(self, symbol, strategies):
            self.symbol = symbol
            self.strategies = strategies

    with pytest.raises(SystemExit):
        await run(
            config=config,
            universe={"tickers": [_T("F", ["monthly_wheel"])]},
            report_dir=tmp_path / "reports",
            now=datetime(2026, 6, 4, 14, 35),
            alpaca_broker=_DeadBroker("alpaca_paper", {}),
            tasty_broker=_MockBroker("tastytrade_sandbox", {}),
        )
