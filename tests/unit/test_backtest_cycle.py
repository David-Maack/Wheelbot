"""scripts/backtest_cycle — decision-point replay using stored chain snapshots."""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta

import pytest

from core.models import (
    ChainSnapshot,
    OptionContract,
    OptionType,
    Order,
    OrderStatus,
    OrderType,
    WheelCycle,
)
from scripts.backtest_cycle import backtest


def _utc():
    return datetime.now(UTC).replace(tzinfo=None)


def _put_contract(occ: str, strike: float, mid: float, days_out: int = 35) -> OptionContract:
    today = date.today()
    return OptionContract(
        underlying="F",
        occ_symbol=occ,
        strike=strike,
        expiration=today + timedelta(days=days_out),
        option_type=OptionType.PUT,
        bid=mid - 0.02, ask=mid + 0.02,
        delta=-0.25,
        open_interest=1000, volume=200,
    )


@pytest.mark.asyncio
async def test_returns_none_for_unknown_cycle(db_repos):
    report = await backtest(db_repos, cycle_id=99999)
    assert report is None


@pytest.mark.asyncio
async def test_agreement_when_original_is_top_yield(db_repos):
    cid = await db_repos.cycles.insert(
        WheelCycle(account_id="primary", symbol="F", started_at=_utc())
    )
    # CSP entry: high-yield original
    await db_repos.orders.insert(
        Order(
            account_id="primary", symbol="F",
            cycle_id=cid,
            order_type=OrderType.SELL_TO_OPEN,
            contract_symbol="HIGH_YIELD",
            strike=9.5, expiration=date.today() + timedelta(days=35),
            option_type=OptionType.PUT,
            quantity=1, limit_price=0.50, fill_price=0.50,
            status=OrderStatus.FILLED,
            placed_at=_utc(),
            client_order_id="wb-1",
        )
    )
    # Snapshot: same chain that was evaluated. HIGH_YIELD has the best mid/strike.
    chain_contracts = [
        _put_contract("HIGH_YIELD", strike=9.5, mid=0.50),
        _put_contract("LOW_YIELD", strike=9.0, mid=0.20),
    ]
    await db_repos.chain_snapshots.insert(
        ChainSnapshot(
            captured_at=_utc(),
            symbol="F",
            side="put",
            underlying_price=10.0,
            contracts=[c.model_dump(mode="json") for c in chain_contracts],
            cycle_id=cid,
        )
    )

    report = await backtest(db_repos, cycle_id=cid)
    assert report is not None
    assert len(report.decisions) == 1
    d = report.decisions[0]
    assert d.original_occ == "HIGH_YIELD"
    assert d.current_pick_occ == "HIGH_YIELD"
    assert d.diverged is False


@pytest.mark.asyncio
async def test_divergence_when_original_was_not_best_yield(db_repos):
    cid = await db_repos.cycles.insert(
        WheelCycle(account_id="primary", symbol="F", started_at=_utc())
    )
    # CSP entry on the WORSE-yield contract.
    await db_repos.orders.insert(
        Order(
            account_id="primary", symbol="F",
            cycle_id=cid,
            order_type=OrderType.SELL_TO_OPEN,
            contract_symbol="LOW_YIELD",
            strike=9.0, expiration=date.today() + timedelta(days=35),
            option_type=OptionType.PUT,
            quantity=1, limit_price=0.20, fill_price=0.20,
            status=OrderStatus.FILLED,
            placed_at=_utc(),
            client_order_id="wb-2",
        )
    )
    chain_contracts = [
        _put_contract("HIGH_YIELD", strike=9.5, mid=0.50),
        _put_contract("LOW_YIELD", strike=9.0, mid=0.20),
    ]
    await db_repos.chain_snapshots.insert(
        ChainSnapshot(
            captured_at=_utc(),
            symbol="F",
            side="put",
            underlying_price=10.0,
            contracts=[c.model_dump(mode="json") for c in chain_contracts],
            cycle_id=cid,
        )
    )

    report = await backtest(db_repos, cycle_id=cid)
    assert report is not None
    d = report.decisions[0]
    assert d.diverged is True
    assert d.current_pick_occ == "HIGH_YIELD"


@pytest.mark.asyncio
async def test_no_chain_data_skips_decision(db_repos):
    cid = await db_repos.cycles.insert(
        WheelCycle(account_id="primary", symbol="F", started_at=_utc())
    )
    await db_repos.orders.insert(
        Order(
            account_id="primary", symbol="F",
            cycle_id=cid,
            order_type=OrderType.SELL_TO_OPEN,
            contract_symbol="MYSTERY",
            strike=9.0, expiration=date.today() + timedelta(days=35),
            option_type=OptionType.PUT,
            quantity=1, limit_price=0.20, fill_price=0.20,
            status=OrderStatus.FILLED,
            placed_at=_utc(),
            client_order_id="wb-3",
        )
    )
    report = await backtest(db_repos, cycle_id=cid)
    assert report is not None
    assert len(report.decisions) == 1
    assert report.decisions[0].diverged is False
    assert "no chain stored" in report.decisions[0].reason
