"""strategies/wheel orchestrator dispatches per state."""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta

import pytest

from core.models import OptionContract, OptionType, Position, PositionState, Quote, UniverseEntry
from core.strategies import StrategyDefinition
from platforms.paper_broker import PaperBroker
from strategies.wheel import propose_all, propose_for_symbol


class _FakePositionsRepo:
    def __init__(self, positions: dict[tuple[str, str], Position] | None = None):
        self._by_key = positions or {}

    async def get_by_symbol(
        self, account_id: str, symbol: str, strategy_id: str | None = None
    ) -> Position | None:
        return self._by_key.get((account_id, symbol))


class _FakeRepos:
    def __init__(self, positions: dict[tuple[str, str], Position] | None = None):
        self.positions = _FakePositionsRepo(positions)


class _NullIvr:
    async def iv_rank(self, symbol: str) -> float | None:
        return None

    async def iv_percentile(self, symbol: str) -> float | None:
        return None


def _config() -> dict:
    return {
        "account": {"id": "test", "broker": "paper"},
        "wheel": {
            "csp_delta_min": 0.20,
            "csp_delta_max": 0.30,
            "cc_delta_min": 0.20,
            "cc_delta_max": 0.30,
            "dte_min": 30,
            "dte_max": 45,
            "open_interest_min": 100,
            "volume_min": 50,
            "bid_ask_spread_max_pct": 10.0,
        },
    }


def _universe(tier: int = 1) -> dict:
    entry = UniverseEntry(symbol="F", name="Ford", tier=tier, overrides={})
    return {"tickers": [entry], "banned": [], "banned_rules": []}


def _put(strike: float = 9.5) -> OptionContract:
    today = date(2025, 6, 1)
    return OptionContract(
        underlying="F",
        occ_symbol="F250706P00009500",
        strike=strike,
        expiration=today + timedelta(days=35),
        option_type=OptionType.PUT,
        bid=0.39,
        ask=0.41,
        delta=-0.25,
        open_interest=1000,
        volume=200,
    )


def _call(strike: float = 10.5) -> OptionContract:
    today = date(2025, 6, 1)
    return OptionContract(
        underlying="F",
        occ_symbol="F250706C00010500",
        strike=strike,
        expiration=today + timedelta(days=35),
        option_type=OptionType.CALL,
        bid=0.29,
        ask=0.31,
        delta=0.25,
        open_interest=1000,
        volume=200,
    )


@pytest.mark.asyncio
async def test_idle_state_proposes_csp():
    broker = PaperBroker()
    broker.seed_quote(Quote(symbol="F", bid=10.0, ask=10.04))
    broker.seed_chain("F", [_put()])
    repos = _FakeRepos()  # no positions → defaults to IDLE
    proposal = await propose_for_symbol(
        broker, repos, "F", _config(), _universe(), _NullIvr(), today=date(2025, 6, 1)
    )
    assert proposal is not None
    assert proposal.contract.option_type == OptionType.PUT
    assert proposal.quantity == 1
    assert proposal.requires_screen is False  # tier 1
    assert proposal.requires_human is False


@pytest.mark.asyncio
async def test_shares_held_proposes_cc_with_quantity_from_share_count():
    broker = PaperBroker()
    broker.seed_quote(Quote(symbol="F", bid=10.0, ask=10.04))
    broker.seed_chain("F", [_call()])

    pos = Position(
        account_id="test",
        symbol="F",
        state=PositionState.SHARES_HELD,
        shares=300,
        cost_basis=10.0,
        state_changed_at=datetime.now(UTC).replace(tzinfo=None),
    )
    repos = _FakeRepos({("test", "F"): pos})
    proposal = await propose_for_symbol(
        broker, repos, "F", _config(), _universe(), _NullIvr(), today=date(2025, 6, 1)
    )
    assert proposal is not None
    assert proposal.contract.option_type == OptionType.CALL
    assert proposal.quantity == 3  # 300 / 100


@pytest.mark.asyncio
async def test_csp_open_state_yields_no_proposal():
    broker = PaperBroker()
    broker.seed_quote(Quote(symbol="F", bid=10.0, ask=10.04))
    broker.seed_chain("F", [_put()])
    pos = Position(
        account_id="test",
        symbol="F",
        state=PositionState.CSP_OPEN,
        shares=0,
        state_changed_at=datetime.now(UTC).replace(tzinfo=None),
    )
    repos = _FakeRepos({("test", "F"): pos})
    proposal = await propose_for_symbol(
        broker, repos, "F", _config(), _universe(), _NullIvr(), today=date(2025, 6, 1)
    )
    assert proposal is None


@pytest.mark.asyncio
async def test_shares_held_without_cost_basis_skips():
    broker = PaperBroker()
    broker.seed_quote(Quote(symbol="F", bid=10.0, ask=10.04))
    broker.seed_chain("F", [_call()])
    pos = Position(
        account_id="test",
        symbol="F",
        state=PositionState.SHARES_HELD,
        shares=100,
        cost_basis=None,  # unknown — refuse to propose
        state_changed_at=datetime.now(UTC).replace(tzinfo=None),
    )
    repos = _FakeRepos({("test", "F"): pos})
    proposal = await propose_for_symbol(
        broker, repos, "F", _config(), _universe(), _NullIvr(), today=date(2025, 6, 1)
    )
    assert proposal is None


@pytest.mark.asyncio
async def test_tier_2_marks_proposal_requires_screen():
    broker = PaperBroker()
    broker.seed_quote(Quote(symbol="F", bid=10.0, ask=10.04))
    broker.seed_chain("F", [_put()])
    proposal = await propose_for_symbol(
        broker, _FakeRepos(), "F", _config(), _universe(tier=2), _NullIvr(), today=date(2025, 6, 1)
    )
    assert proposal is not None
    assert proposal.requires_screen is True
    assert proposal.requires_human is False


@pytest.mark.asyncio
async def test_tier_3_marks_proposal_requires_human():
    broker = PaperBroker()
    broker.seed_quote(Quote(symbol="F", bid=10.0, ask=10.04))
    broker.seed_chain("F", [_put()])
    proposal = await propose_for_symbol(
        broker, _FakeRepos(), "F", _config(), _universe(tier=3), _NullIvr(), today=date(2025, 6, 1)
    )
    assert proposal is not None
    assert proposal.requires_human is True


# -- Orphan position management (Sprint 14) ---------------------------------


def _strategy_def() -> StrategyDefinition:
    return StrategyDefinition(
        id="weekly_wheel",
        display_name="Weekly Wheel",
        type="wheel",
        enabled=True,
        max_concurrent=4,
        params={"csp_delta_min": 0.20, "csp_delta_max": 0.30,
                "cc_delta_min": 0.20, "cc_delta_max": 0.30,
                "dte_min": 7, "dte_max": 14,
                "open_interest_min": 100, "volume_min": 50,
                "bid_ask_spread_max_pct": 10.0},
    )


@pytest.mark.asyncio
async def test_propose_all_manages_orphan_shares_held_outside_universe(db_repos):
    """Regression for the 2026-05-22 COIN situation: position assigned under
    weekly_wheel but the symbol is no longer in weekly_wheel's universe.
    The orchestrator must still propose a covered call so the wheel cycle
    can complete (called-away or CC expires → IDLE)."""
    broker = PaperBroker()
    broker.seed_quote(Quote(symbol="ORPHAN", bid=180.0, ask=180.5))
    broker.seed_chain("ORPHAN", [
        OptionContract(
            underlying="ORPHAN",
            occ_symbol="ORPHAN250706C00185000",
            strike=185.0,
            expiration=date(2025, 6, 1) + timedelta(days=10),
            option_type=OptionType.CALL,
            bid=2.50, ask=2.60, delta=0.22,
            open_interest=500, volume=100,
        ),
    ])
    # Position exists under weekly_wheel but is not in the universe.
    now = datetime.now(UTC).replace(tzinfo=None)
    await db_repos.positions.insert(
        Position(
            account_id="test",
            symbol="ORPHAN",
            strategy_id="weekly_wheel",
            state=PositionState.SHARES_HELD,
            shares=100,
            cost_basis=180.0,
            state_changed_at=now,
        )
    )
    # Universe contains a different ticker — ORPHAN is NOT in it.
    other_universe = {
        "tickers": [UniverseEntry(symbol="F", tier=1, strategies=["weekly_wheel"])],
        "banned": [], "banned_rules": [],
    }
    broker.seed_chain("F", [])  # no chain → no proposal from F

    proposals = await propose_all(
        broker, db_repos, _config(), other_universe, _NullIvr(),
        today=date(2025, 6, 1), strategy=_strategy_def(),
    )
    # Should have at least one CC proposal on the orphan ORPHAN position.
    orphan_proposals = [p for p in proposals if p.symbol == "ORPHAN"]
    assert len(orphan_proposals) == 1
    assert orphan_proposals[0].contract.option_type == OptionType.CALL


@pytest.mark.asyncio
async def test_propose_all_does_not_open_new_csp_on_orphan_idle(db_repos):
    """Orphan position in IDLE state must NOT trigger a new CSP entry — the
    operator removed the symbol from the universe for a reason."""
    broker = PaperBroker()
    broker.seed_quote(Quote(symbol="ORPHAN", bid=10.0, ask=10.04))
    broker.seed_chain("ORPHAN", [
        OptionContract(
            underlying="ORPHAN",
            occ_symbol="ORPHAN250706P00009500",
            strike=9.5,
            expiration=date(2025, 6, 1) + timedelta(days=10),
            option_type=OptionType.PUT,
            bid=0.30, ask=0.32, delta=-0.25,
            open_interest=500, volume=100,
        ),
    ])
    now = datetime.now(UTC).replace(tzinfo=None)
    await db_repos.positions.insert(
        Position(
            account_id="test",
            symbol="ORPHAN",
            strategy_id="weekly_wheel",
            state=PositionState.IDLE,
            shares=0,
            state_changed_at=now,
        )
    )
    other_universe = {
        "tickers": [UniverseEntry(symbol="F", tier=1, strategies=["weekly_wheel"])],
        "banned": [], "banned_rules": [],
    }
    broker.seed_chain("F", [])

    proposals = await propose_all(
        broker, db_repos, _config(), other_universe, _NullIvr(),
        today=date(2025, 6, 1), strategy=_strategy_def(),
    )
    # No new entry on the orphan IDLE position.
    assert all(p.symbol != "ORPHAN" for p in proposals)
