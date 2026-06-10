"""TICKET-016 — Calendar Spread.

Sell a front-month call, buy a back-month call at the SAME strike. Net
DEBIT. Profits from the front decaying faster than the back (theta
differential) and from IV rising (long vega). Best when IV is LOW and the
underlying stays NEAR the strike through front-month expiration. Max profit
at the strike at front expiry. Defined risk = the net debit.

This is the counter-cyclical piece — it wants LOW IV (the INVERTED IVR gate:
enter only when IVR ≤ ivr_max), the opposite of the premium-selling
strategies. When premium strategies sit out below their IVR floor, calendars
can step in.

Structurally a 2-leg MLEG (same strike, two expirations) — PF-1 confirmed
both brokers accept mixed-expiration in one order. Reuses MultiLegProposal /
place_multi_leg / SPREAD_* states (distinguished by strategy_id=="calendar").
It is the first net-DEBIT MLEG open (verticals/condors open for credit);
net_credit_per_spread is negative and the router's limit-price math handles
it like a spread close. width_dollars=0 → the reconciler's _open_cycle_for_spread
records capital_at_risk = the debit (verified by test).

v1: call only, ATM strike, single front/back. Closes-and-reenters (no
front roll). Force-closed at 2 DTE so the partial-expiry path is unreachable
(a reconciler guard flags MANUAL_INTERVENTION if it ever fires).

news_check: neutral_pin profile fires on the MLEG open via the router.
Out of scope: double/diagonal calendars, put calendars, front-leg rolling.
"""

from __future__ import annotations

from datetime import date
from typing import Any

from core.broker import Broker
from core.checkpoint import log_checkpoint
from core.models import OptionContract, OptionType, OrderLeg, OrderType, PositionState
from core.strategies import StrategyDefinition
from data.chain import ChainFilters, fetch_filtered_chain
from data.ivr import IVRProvider
from db.repo import Repos
from strategies.spreads import (
    DIRECTION_CALENDAR,
    MultiLegProposal,
    _make_chain_recorder,
    _tier_flags,
)

NEWS_CHECK_PROFILE = "neutral_pin"


def _mid(c: OptionContract) -> float | None:
    if c.bid is None or c.ask is None:
        return None
    m = (c.bid + c.ask) / 2
    return m if m > 0 else None


async def select_calendar(
    broker: Broker,
    symbol: str,
    params: dict[str, Any],
    *,
    today: date | None = None,
    record_chain: Any | None = None,
    ivr: IVRProvider | None = None,
) -> tuple[OptionContract, OptionContract, float] | None:
    """Return (short_front, long_back, net_debit_per_share) for an ATM calendar,
    or None. INVERTED IVR gate: skip when IVR is too HIGH (calendars want low
    IV). Same strike, front + back expirations."""
    today = today or date.today()

    # INVERTED IVR gate — the opposite of premium strategies. Enter only when
    # implied vol is LOW (a calendar is long vega; it profits if IV rises).
    if ivr is not None:
        rank = await ivr.iv_rank(symbol)
        ivr_max = float(params.get("ivr_max", 100))
        if rank is not None and rank > ivr_max:
            log_checkpoint(
                "calendar_skip_ivr_too_high", status="ok",
                symbol=symbol, ivr=rank, ivr_max=ivr_max,
            )
            return None

    # ATM band for both expirations (~0.50 delta).
    band = ChainFilters(
        dte_min=int(params.get("short_dte_min", 7)),
        dte_max=int(params.get("short_dte_max", 21)),
        delta_min=0.40, delta_max=0.60,
        open_interest_min=int(params.get("open_interest_min", 0)),
        volume_min=int(params.get("volume_min", 0)),
        bid_ask_spread_max_pct=float(params.get("bid_ask_spread_max_pct", 100.0)),
    )
    back_band = ChainFilters(
        dte_min=int(params.get("long_dte_min", 45)),
        dte_max=int(params.get("long_dte_max", 75)),
        delta_min=0.40, delta_max=0.60,
        open_interest_min=int(params.get("open_interest_min", 0)),
        volume_min=int(params.get("volume_min", 0)),
        bid_ask_spread_max_pct=float(params.get("bid_ask_spread_max_pct", 100.0)),
    )
    front = await fetch_filtered_chain(broker, symbol, "call", band, today=today)
    back = await fetch_filtered_chain(broker, symbol, "call", back_band, today=today)
    if record_chain is not None:
        await record_chain(symbol, "call", front)
    if not front or not back:
        log_checkpoint(
            "calendar_no_candidates", status="ok", symbol=symbol,
            n_front=len(front), n_back=len(back),
        )
        return None

    back_by_strike = {c.strike: c for c in back}
    # ATM = front delta closest to 0.50, among strikes present in BOTH expiries.
    paired = [
        (abs(abs(f.delta) - 0.50), f, back_by_strike[f.strike])
        for f in front
        if f.delta is not None and f.strike in back_by_strike
    ]
    if not paired:
        log_checkpoint("calendar_no_matching_strike", status="ok", symbol=symbol)
        return None
    paired.sort(key=lambda t: t[0])
    _, short_front, long_back = paired[0]

    sf_mid = _mid(short_front)
    lb_mid = _mid(long_back)
    if sf_mid is None or lb_mid is None:
        log_checkpoint("calendar_missing_mids", status="ok", symbol=symbol)
        return None
    net_debit = lb_mid - sf_mid  # buy back, sell front → positive debit
    if net_debit <= 0:
        # Back cheaper than front (inverted term structure) — not a calendar.
        log_checkpoint("calendar_non_positive_debit", status="ok",
                       symbol=symbol, net_debit=net_debit)
        return None
    cap = float(params.get("max_debit_per_spread_usd", 0) or 0)
    if cap > 0 and net_debit * 100 > cap:
        log_checkpoint(
            "calendar_skip_over_cap", status="ok", symbol=symbol,
            strike=short_front.strike, net_debit=net_debit, cap=cap,
        )
        return None
    log_checkpoint(
        "calendar_selected", status="ok", symbol=symbol,
        strike=short_front.strike, net_debit=net_debit,
        front_dte=(short_front.expiration - today).days,
        back_dte=(long_back.expiration - today).days,
    )
    return short_front, long_back, net_debit


def _build_legs(short_front: OptionContract, long_back: OptionContract) -> list[OrderLeg]:
    """Sell the front, buy the back — same strike, different expirations."""
    def _leg(c: OptionContract, action: OrderType) -> OrderLeg:
        return OrderLeg(
            contract_symbol=c.occ_symbol, underlying=c.underlying,
            option_type=c.option_type, strike=c.strike, expiration=c.expiration,
            action=action, ratio_qty=1,
        )
    return [
        _leg(short_front, OrderType.SELL_TO_OPEN),
        _leg(long_back, OrderType.BUY_TO_OPEN),
    ]


async def propose_for_symbol(
    broker: Broker,
    repos: Repos,
    symbol: str,
    config: dict[str, Any],
    universe: dict[str, Any],
    *,
    today: date | None = None,
    strategy: StrategyDefinition | None = None,
    ivr: IVRProvider | None = None,
) -> MultiLegProposal | None:
    today = today or date.today()
    if strategy is None:
        log_checkpoint("calendar_skip_no_strategy", status="fail", symbol=symbol)
        return None
    account_id = config.get("account", {}).get("id", "primary")
    position = await repos.positions.get_by_symbol(account_id, symbol, strategy_id=strategy.id)
    state = position.state if position else PositionState.IDLE
    if state not in (PositionState.IDLE, PositionState.SPREAD_CLOSED):
        log_checkpoint("calendar_skip_state", status="ok",
                       symbol=symbol, strategy=strategy.id, state=str(state))
        return None

    record = _make_chain_recorder(
        repos,
        cycle_id=position.current_cycle_id if position else None,
        strategy_id=strategy.id,
    )
    picked = await select_calendar(
        broker, symbol, strategy.params, today=today, record_chain=record, ivr=ivr,
    )
    if picked is None:
        return None
    short_front, long_back, net_debit = picked
    legs = _build_legs(short_front, long_back)
    needs_screen, needs_human = _tier_flags(symbol, universe)
    rationale = (
        f"calendar[{symbol}] strike={short_front.strike} "
        f"front_dte={(short_front.expiration - today).days} "
        f"back_dte={(long_back.expiration - today).days} debit={net_debit:.2f}"
    )
    return MultiLegProposal(
        symbol=symbol,
        legs=legs,
        # NET DEBIT → negative net_credit (router pays up to this, like a close).
        net_credit_per_spread=-net_debit,
        max_loss_per_spread=net_debit * 100,
        width_dollars=0.0,  # same strike → reconciler records capital = debit
        quantity=1,
        rationale=rationale,
        strategy_id=strategy.id,
        requires_screen=needs_screen,
        requires_human=needs_human,
        direction=DIRECTION_CALENDAR,
        news_check_profile=NEWS_CHECK_PROFILE,
    )


async def propose_all(
    broker: Broker,
    repos: Repos,
    config: dict[str, Any],
    universe: dict[str, Any],
    *,
    today: date | None = None,
    strategy: StrategyDefinition | None = None,
    ivr: IVRProvider | None = None,
) -> list[MultiLegProposal]:
    out: list[MultiLegProposal] = []
    for entry in universe["tickers"]:
        p = await propose_for_symbol(
            broker, repos, entry.symbol, config, universe,
            today=today, strategy=strategy, ivr=ivr,
        )
        if p is not None:
            out.append(p)
    log_checkpoint(
        "calendar_propose_all", status="ok",
        strategy=strategy.id if strategy else "calendar", n_proposals=len(out),
    )
    return out
