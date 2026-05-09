"""strategies/wheel orchestrator dispatches per state."""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta

import pytest

from core.models import OptionContract, OptionType, Position, PositionState, Quote, UniverseEntry
from platforms.paper_broker import PaperBroker
from strategies.wheel import propose_for_symbol


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
