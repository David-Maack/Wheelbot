"""execution/router news_check integration."""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from typing import Any

import pytest

from core.models import (
    OptionContract,
    OptionType,
    OrderType,
    PositionState,
    UniverseEntry,
)
from execution.router import OrderRouter
from platforms.paper_broker import PaperBroker
from strategies.wheel import Proposal


class _NewsStub:
    """A callable that returns a fixed news_check result.

    Accepts the optional `profile` parameter the TICKET-014 router signature
    now passes for MULTI_LEG_OPEN news_checks. Single-leg call sites still
    call with just `symbol`."""

    def __init__(self, decision: str, rationale: str = ""):
        self.decision = decision
        self.rationale = rationale
        self.calls = 0
        self.profiles_seen: list[str | None] = []

    async def __call__(self, symbol: str, profile: str | None = None):
        self.calls += 1
        self.profiles_seen.append(profile)
        return self  # NewsCheckResult-shaped (decision/rationale)


def _config() -> dict:
    return {
        "account": {"id": "test", "broker": "paper"},
        "wheel": {
            "buying_power_floor_pct": 20,
            "max_position_pct_of_account": 30,
            "max_concurrent_positions": 4,
            "open_interest_min": 100,
            "volume_min": 50,
            "bid_ask_spread_max_pct": 10.0,
        },
        "regime": {"enabled": False},
        "execution": {"dry_run": False, "retry_max_attempts": 1, "retry_initial_backoff_seconds": 0, "retry_max_backoff_seconds": 0},
    }


def _universe() -> dict:
    return {
        "tickers": [UniverseEntry(symbol="F", name="Ford", tier=1, overrides={})],
        "banned": [],
        "banned_rules": [],
    }


def _contract() -> OptionContract:
    today = date(2025, 6, 1)
    return OptionContract(
        underlying="F",
        occ_symbol="F250706P00009500",
        strike=9.5,
        expiration=today + timedelta(days=35),
        option_type=OptionType.PUT,
        bid=0.39, ask=0.41, delta=-0.25,
        open_interest=1000, volume=200,
    )


def _proposal(qty: int = 2) -> Proposal:
    return Proposal(
        symbol="F",
        contract=_contract(),
        order_type=OrderType.SELL_TO_OPEN,
        quantity=qty,
        rationale="csp test",
    )


async def _noop_sleep(_):
    return None


@pytest.fixture(autouse=True)
def _stub_earnings(monkeypatch):
    monkeypatch.setattr("risk.limits.in_blackout", lambda *a, **k: None)


@pytest.mark.asyncio
async def test_proceed_places_unchanged(db_repos):
    broker = PaperBroker(cash=20_000)
    news = _NewsStub("proceed")
    router = OrderRouter(broker, db_repos, _config(), _universe(), news_checker=news)
    result = await router.place(_proposal(qty=2), sleep=_noop_sleep, today=date(2025, 6, 1))
    assert result.placed is not None
    assert result.placed.quantity == 2
    assert result.news_decision == "proceed"
    assert news.calls == 1


@pytest.mark.asyncio
async def test_block_does_not_submit(db_repos):
    broker = PaperBroker(cash=20_000)
    news = _NewsStub("block", "FDA warning")
    router = OrderRouter(broker, db_repos, _config(), _universe(), news_checker=news)
    result = await router.place(_proposal(qty=2), sleep=_noop_sleep, today=date(2025, 6, 1))
    assert result.placed is None
    assert result.news_decision == "block"
    pos = await db_repos.positions.get_by_symbol("test", "F")
    assert pos is None
    rows = await db_repos.orders.list_recent("test")
    assert rows == []


@pytest.mark.asyncio
async def test_caution_halves_qty_when_possible(db_repos):
    broker = PaperBroker(cash=20_000)
    news = _NewsStub("caution", "shaky guidance")
    router = OrderRouter(broker, db_repos, _config(), _universe(), news_checker=news)
    result = await router.place(_proposal(qty=2), sleep=_noop_sleep, today=date(2025, 6, 1))
    assert result.placed is not None
    assert result.placed.quantity == 1
    assert result.news_decision == "caution"
    assert result.quantity_adjusted == 1


@pytest.mark.asyncio
async def test_caution_halving_preserves_strategy_id(db_repos):
    """Regression: rebuilding the Proposal on a caution-halve previously dropped
    strategy_id, silently re-tagging a halved weekly_wheel CSP as monthly_wheel
    (wrong idempotency key + position lookup). It must be preserved."""
    broker = PaperBroker(cash=40_000)
    news = _NewsStub("caution", "shaky guidance")
    router = OrderRouter(broker, db_repos, _config(), _universe(), news_checker=news)
    weekly = Proposal(
        symbol="F",
        contract=_contract(),
        order_type=OrderType.SELL_TO_OPEN,
        quantity=4,
        rationale="csp test",
        strategy_id="weekly_wheel",
    )
    result = await router.place(weekly, sleep=_noop_sleep, today=date(2025, 6, 1))
    assert result.placed is not None
    assert result.placed.quantity == 2            # halved
    assert result.placed.strategy_id == "weekly_wheel"  # NOT defaulted to monthly_wheel
    pos = await db_repos.positions.get_by_symbol("test", "F", strategy_id="weekly_wheel")
    assert pos is not None


@pytest.mark.asyncio
async def test_advisory_mode_caution_proceeds_full_size(db_repos):
    """Advisory mode (paper testing): a 'caution' is logged but the order
    proceeds at FULL size — only a hard 'block' cancels."""
    broker = PaperBroker(cash=20_000)
    news = _NewsStub("caution", "elevated IV")
    cfg = _config()
    cfg["intelligence"] = {"news_check_advisory": True}
    router = OrderRouter(broker, db_repos, cfg, _universe(), news_checker=news)
    result = await router.place(_proposal(qty=2), sleep=_noop_sleep, today=date(2025, 6, 1))
    assert result.placed is not None
    assert result.placed.quantity == 2          # NOT halved
    assert result.news_decision == "caution"


@pytest.mark.asyncio
async def test_advisory_mode_still_blocks_on_block(db_repos):
    """Advisory only softens 'caution' — a hard 'block' still cancels."""
    broker = PaperBroker(cash=20_000)
    news = _NewsStub("block", "fraud probe")
    cfg = _config()
    cfg["intelligence"] = {"news_check_advisory": True}
    router = OrderRouter(broker, db_repos, cfg, _universe(), news_checker=news)
    result = await router.place(_proposal(qty=2), sleep=_noop_sleep, today=date(2025, 6, 1))
    assert result.placed is None
    assert result.news_decision == "block"


@pytest.mark.asyncio
async def test_caution_with_qty_one_blocks(db_repos):
    """Spec stretch: caution + qty=1 → block (can't halve a single contract)."""
    broker = PaperBroker(cash=20_000)
    news = _NewsStub("caution", "shaky guidance")
    router = OrderRouter(broker, db_repos, _config(), _universe(), news_checker=news)
    result = await router.place(_proposal(qty=1), sleep=_noop_sleep, today=date(2025, 6, 1))
    assert result.placed is None
    assert result.news_decision == "caution"
    pos = await db_repos.positions.get_by_symbol("test", "F")
    assert pos is None


@pytest.mark.asyncio
async def test_cc_entry_skips_news_check(db_repos):
    """Spec §9.2 says news check is for new CSPs only — CCs should bypass it."""
    broker = PaperBroker(cash=20_000)
    today = date(2025, 6, 1)
    cc_contract = OptionContract(
        underlying="F",
        occ_symbol="F250706C00010500",
        strike=10.5,
        expiration=today + timedelta(days=35),
        option_type=OptionType.CALL,  # ← key part
        bid=0.39, ask=0.41, delta=0.25,
        open_interest=1000, volume=200,
    )
    # Pre-seed shares so the CC has cost basis.
    from core.models import Position, PositionState
    from datetime import UTC, datetime
    await db_repos.positions.insert(
        Position(
            account_id="test", symbol="F",
            state=PositionState.SHARES_HELD,
            shares=100, cost_basis=10.0,
            state_changed_at=datetime.now(UTC).replace(tzinfo=None),
        )
    )
    cc_proposal = Proposal(
        symbol="F",
        contract=cc_contract,
        order_type=OrderType.SELL_TO_OPEN,
        quantity=1,
        rationale="cc test",
    )
    news = _NewsStub("block", "would block if asked")
    router = OrderRouter(broker, db_repos, _config(), _universe(), news_checker=news)
    result = await router.place(cc_proposal, sleep=_noop_sleep, today=today)
    assert news.calls == 0  # the news check should NOT have been invoked
    assert result.placed is not None
    assert result.news_decision is None


@pytest.mark.asyncio
async def test_no_news_checker_means_no_news_decision(db_repos):
    """Backward compatible — router with no news_checker skips the gate."""
    broker = PaperBroker(cash=20_000)
    router = OrderRouter(broker, db_repos, _config(), _universe(), news_checker=None)
    result = await router.place(_proposal(qty=1), sleep=_noop_sleep, today=date(2025, 6, 1))
    assert result.placed is not None
    assert result.news_decision is None


# -- TICKET-014 Phase 2: news_check on MULTI_LEG_OPEN ----------------------


def _mleg_legs() -> list:
    """4-leg iron condor on F — distinct enough to test the news_check
    plumbing without depending on real chain selection."""
    from core.models import OrderLeg
    today = date(2025, 6, 1)
    exp = today + timedelta(days=35)
    return [
        OrderLeg(
            contract_symbol="F250706P00009000",
            underlying="F", option_type=OptionType.PUT, strike=9.0,
            expiration=exp, action=OrderType.BUY_TO_OPEN,
        ),
        OrderLeg(
            contract_symbol="F250706P00009500",
            underlying="F", option_type=OptionType.PUT, strike=9.5,
            expiration=exp, action=OrderType.SELL_TO_OPEN,
        ),
        OrderLeg(
            contract_symbol="F250706C00010500",
            underlying="F", option_type=OptionType.CALL, strike=10.5,
            expiration=exp, action=OrderType.SELL_TO_OPEN,
        ),
        OrderLeg(
            contract_symbol="F250706C00011000",
            underlying="F", option_type=OptionType.CALL, strike=11.0,
            expiration=exp, action=OrderType.BUY_TO_OPEN,
        ),
    ]


def _condor_proposal(qty: int = 2, profile: str | None = "neutral_range"):
    from strategies.spreads import DIRECTION_IRON_CONDOR, MultiLegProposal
    return MultiLegProposal(
        symbol="F",
        legs=_mleg_legs(),
        net_credit_per_spread=0.50,
        max_loss_per_spread=450.0,
        width_dollars=0.5,
        quantity=qty,
        rationale="condor test",
        strategy_id="iron_condor",
        direction=DIRECTION_IRON_CONDOR,
        news_check_profile=profile,
    )


@pytest.mark.asyncio
async def test_multi_leg_no_news_check_when_profile_unset(db_repos):
    """Existing put_spread / bear_call_spread proposals leave
    news_check_profile=None — router must NOT fire news_check (preserves
    pre-TICKET-014 behavior for those strategies)."""
    from strategies.spreads import DIRECTION_BULL_PUT, MultiLegProposal
    broker = PaperBroker(cash=20_000)
    news = _NewsStub("block", "would block if asked")
    router = OrderRouter(broker, db_repos, _config(), _universe(), news_checker=news)
    vanilla = MultiLegProposal(
        symbol="F",
        legs=_mleg_legs()[:2],  # 2-leg bull put shape
        net_credit_per_spread=0.30,
        max_loss_per_spread=70.0,
        width_dollars=0.5,
        quantity=1,
        rationale="vanilla put_spread",
        strategy_id="put_spread",
        direction=DIRECTION_BULL_PUT,
        # news_check_profile defaults to None
    )
    result = await router.place_multi_leg(
        vanilla, sleep=_noop_sleep, today=date(2025, 6, 1),
    )
    assert news.calls == 0
    assert result.placed is not None


@pytest.mark.asyncio
async def test_multi_leg_news_check_fires_with_profile_when_set(db_repos):
    """iron_condor proposal sets news_check_profile=neutral_range → router
    fires news_check with that profile."""
    broker = PaperBroker(cash=20_000)
    news = _NewsStub("proceed")
    router = OrderRouter(broker, db_repos, _config(), _universe(), news_checker=news)
    result = await router.place_multi_leg(
        _condor_proposal(qty=2), sleep=_noop_sleep, today=date(2025, 6, 1),
    )
    assert news.calls == 1
    assert news.profiles_seen == ["neutral_range"]
    assert result.placed is not None
    assert result.news_decision == "proceed"


@pytest.mark.asyncio
async def test_multi_leg_news_block_skips_open(db_repos):
    """block decision → no order placed, RouteResult carries the decision."""
    broker = PaperBroker(cash=20_000)
    news = _NewsStub("block", "FOMC tomorrow")
    router = OrderRouter(broker, db_repos, _config(), _universe(), news_checker=news)
    result = await router.place_multi_leg(
        _condor_proposal(qty=2), sleep=_noop_sleep, today=date(2025, 6, 1),
    )
    assert news.calls == 1
    assert result.placed is None
    assert result.news_decision == "block"
    assert "FOMC tomorrow" in (result.news_rationale or "")


@pytest.mark.asyncio
async def test_multi_leg_news_caution_qty1_skips(db_repos):
    """caution + qty=1 + not-advisory → skip (can't half a structured package)."""
    broker = PaperBroker(cash=20_000)
    news = _NewsStub("caution", "earnings inside window")
    cfg = _config()
    cfg["intelligence"] = {"news_check_advisory": False}
    router = OrderRouter(broker, db_repos, cfg, _universe(), news_checker=news)
    result = await router.place_multi_leg(
        _condor_proposal(qty=1), sleep=_noop_sleep, today=date(2025, 6, 1),
    )
    assert news.calls == 1
    assert result.placed is None
    assert result.news_decision == "caution"
    assert "qty=1 cannot be halved" in (result.news_rationale or "")


@pytest.mark.asyncio
async def test_multi_leg_news_caution_qty2_halves(db_repos):
    """caution + qty=2 + not-advisory → place at qty=1."""
    broker = PaperBroker(cash=20_000)
    news = _NewsStub("caution", "non-trivial signal")
    cfg = _config()
    cfg["intelligence"] = {"news_check_advisory": False}
    router = OrderRouter(broker, db_repos, cfg, _universe(), news_checker=news)
    result = await router.place_multi_leg(
        _condor_proposal(qty=2), sleep=_noop_sleep, today=date(2025, 6, 1),
    )
    assert news.calls == 1
    assert result.placed is not None
    assert result.placed.quantity == 1  # halved from 2


@pytest.mark.asyncio
async def test_multi_leg_news_caution_advisory_proceeds_full_size(db_repos):
    """caution + advisory mode → place at full size, log the advisory note."""
    broker = PaperBroker(cash=20_000)
    news = _NewsStub("caution", "minor signal")
    cfg = _config()
    cfg["intelligence"] = {"news_check_advisory": True}
    router = OrderRouter(broker, db_repos, cfg, _universe(), news_checker=news)
    result = await router.place_multi_leg(
        _condor_proposal(qty=2), sleep=_noop_sleep, today=date(2025, 6, 1),
    )
    assert news.calls == 1
    assert result.placed is not None
    assert result.placed.quantity == 2  # full size preserved
