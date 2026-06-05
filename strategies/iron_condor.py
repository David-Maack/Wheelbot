"""TICKET-014 — Iron Condor strategy.

Sell an OTM put spread + sell an OTM call spread on the same underlying at
the same expiration. Four legs:

    long put  (lowest strike)  — BUY_TO_OPEN
    short put                  — SELL_TO_OPEN
    short call                 — SELL_TO_OPEN
    long call (highest strike) — BUY_TO_OPEN

Defined-risk neutral-range trade. Net credit received. Max profit = net
credit (underlying expires between short strikes). Max loss = wing_width −
net_credit (underlying blows through either wing). Profits in range-bound
markets; short gamma on both sides → managed tighter than a single spread
(default 25% profit close vs 50% for verticals).

Universe: indices only for v1 (SPY, QQQ, IWM — per the runbook Stage 2
acceptance criteria, these are the most liquid neutral-range candidates).
The universe lives in `config/universe.yaml` per the existing convention;
adding `iron_condor` to each ticker's `strategies:` list opts it in.

Regime gate: iron_condor AND-combines `csps_allowed` + `bear_calls_allowed`
in risk/limits.py::_rule_regime_multi_leg. Current 4-bucket regime: NEUTRAL
passes; BULL_TREND fails (call wing), BEAR_TREND fails (put wing), HIGH_VOL
fails (both). When TICKET-013's 6-bucket regime + explicit
iron_condor_allowed column lands, the gate switches to read that directly.

news_check: fires the `neutral_range` profile via the router's MULTI_LEG_OPEN
news_check block. Each proposal carries `news_check_profile="neutral_range"`.
For broad-market indices, the prompt focuses on macro catalysts (FOMC,
CPI/NFP, geopolitical) since single-name news is less relevant.

Out of scope (defer): iron butterflies, asymmetric/broken-wing condors,
jade lizards. Symmetric condor only. Single-name underlyings deferred to
post-paper-validation.
"""

from __future__ import annotations

from dataclasses import dataclass
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
    DIRECTION_IRON_CONDOR,
    MultiLegProposal,
    _make_chain_recorder,
    _tier_flags,
)


# news_check profile fired by the router on MULTI_LEG_OPEN for this strategy.
NEWS_CHECK_PROFILE = "neutral_range"


@dataclass(frozen=True, slots=True)
class IronCondorCandidate:
    """Four legs of a symmetric iron condor at one expiration."""

    long_put: OptionContract
    short_put: OptionContract
    short_call: OptionContract
    long_call: OptionContract
    net_credit_per_spread: float    # dollars per share, positive = credit received
    max_loss_per_spread: float      # dollars total per package
    wing_width_dollars: float       # symmetric — put wing == call wing


def _mid(c: OptionContract) -> float | None:
    if c.bid is None or c.ask is None:
        return None
    mid = (c.bid + c.ask) / 2
    return mid if mid > 0 else None


def _find_protective_long(
    chain: list[OptionContract],
    short: OptionContract,
    *,
    wing_width: float,
    side: OptionType,
) -> OptionContract | None:
    """Find the long leg that caps risk on the named side.

    For PUT: target strike = short.strike − wing_width; prefer exact match,
    fall back to nearest at-or-below the target.
    For CALL: target strike = short.strike + wing_width; prefer exact match,
    fall back to nearest at-or-above the target.

    Returns None when nothing in the chain can wing this short.
    """
    same_expiry = [
        c for c in chain
        if c.expiration == short.expiration
        and c.option_type == side
        and c.bid is not None
        and c.ask is not None
    ]
    if not same_expiry:
        return None
    if side == OptionType.PUT:
        # Put wing protects against down-move: long strike < short strike.
        target = short.strike - wing_width
        candidates = [c for c in same_expiry if c.strike < short.strike]
        candidates.sort(key=lambda c: c.strike, reverse=True)
        at_or_below = [c for c in candidates if c.strike <= target]
        return at_or_below[0] if at_or_below else (candidates[-1] if candidates else None)
    # CALL wing protects against up-move: long strike > short strike.
    target = short.strike + wing_width
    candidates = [c for c in same_expiry if c.strike > short.strike]
    candidates.sort(key=lambda c: c.strike)  # ascending — lowest strike first
    at_or_above = [c for c in candidates if c.strike >= target]
    return at_or_above[0] if at_or_above else (candidates[-1] if candidates else None)


def _pick_short_at_delta(
    chain: list[OptionContract], *, target_abs_delta: float,
) -> OptionContract | None:
    """Closest contract to |delta| = target. Returns None if chain empty
    or no contract has a delta."""
    scored = [c for c in chain if c.delta is not None]
    if not scored:
        return None
    return min(scored, key=lambda c: abs(abs(c.delta) - target_abs_delta))


async def select_iron_condor(
    broker: Broker,
    symbol: str,
    params: dict[str, Any],
    *,
    today: date | None = None,
    record_chain: Any | None = None,
    ivr: IVRProvider | None = None,
) -> IronCondorCandidate | None:
    """Return a symmetric iron condor candidate, or None if nothing qualifies.

    Selection:
      1. (Optional) IVR ≥ ivr_min — skip when premium is too thin.
      2. Find short put closest to |delta| = short_put_delta_target.
      3. Find short call closest to |delta| = short_call_delta_target on
         the SAME expiration as the short put. If expirations diverge,
         skip — condor v1 is single-expiration.
      4. Find long put at short_put.strike − wing_width (or nearest below).
      5. Find long call at short_call.strike + wing_width (or nearest above).
      6. net_credit = (sp − lp) + (sc − lc). Reject if
         net_credit / wing_width < min_credit_pct / 100.
      7. max_loss = (wing_width − net_credit) × 100.
    """
    today = today or date.today()

    if ivr is not None:
        rank = await ivr.iv_rank(symbol)
        ivr_min = float(params.get("ivr_min", 0))
        if rank is not None and rank < ivr_min:
            log_checkpoint(
                "condor_skip_ivr",
                status="ok", symbol=symbol, ivr=rank, ivr_min=ivr_min,
            )
            return None

    dte_min = int(params.get("dte_min", 30))
    dte_max = int(params.get("dte_max", 45))
    short_put_target = float(params.get("short_put_delta_target", 0.16))
    short_call_target = float(params.get("short_call_delta_target", 0.16))
    wing_width = float(params.get("wing_width", 5.0))
    min_credit_pct = float(params.get("min_credit_pct", 30.0))

    # Pull short-leg candidates with the delta band around the targets.
    # Use a permissive band (±0.10 around target) so we have multiple
    # expirations to pick from.
    put_filters = ChainFilters(
        dte_min=dte_min, dte_max=dte_max,
        delta_min=max(0.01, short_put_target - 0.10),
        delta_max=short_put_target + 0.10,
        open_interest_min=int(params.get("open_interest_min", 0)),
        volume_min=int(params.get("volume_min", 0)),
        bid_ask_spread_max_pct=float(params.get("bid_ask_spread_max_pct", 100.0)),
    )
    call_filters = ChainFilters(
        dte_min=dte_min, dte_max=dte_max,
        delta_min=max(0.01, short_call_target - 0.10),
        delta_max=short_call_target + 0.10,
        open_interest_min=int(params.get("open_interest_min", 0)),
        volume_min=int(params.get("volume_min", 0)),
        bid_ask_spread_max_pct=float(params.get("bid_ask_spread_max_pct", 100.0)),
    )

    put_candidates = await fetch_filtered_chain(broker, symbol, "put", put_filters, today=today)
    call_candidates = await fetch_filtered_chain(broker, symbol, "call", call_filters, today=today)
    if record_chain is not None:
        await record_chain(symbol, "put", put_candidates)
        await record_chain(symbol, "call", call_candidates)
    if not put_candidates or not call_candidates:
        log_checkpoint(
            "condor_no_short_candidates",
            status="ok", symbol=symbol,
            n_puts=len(put_candidates), n_calls=len(call_candidates),
        )
        return None

    short_put = _pick_short_at_delta(put_candidates, target_abs_delta=short_put_target)
    if short_put is None:
        log_checkpoint("condor_no_short_put", status="ok", symbol=symbol)
        return None
    # Force same-expiration: filter calls to the short put's expiry.
    same_exp_calls = [c for c in call_candidates if c.expiration == short_put.expiration]
    if not same_exp_calls:
        log_checkpoint(
            "condor_no_matching_expiration",
            status="ok", symbol=symbol,
            short_put_exp=short_put.expiration.isoformat(),
        )
        return None
    short_call = _pick_short_at_delta(same_exp_calls, target_abs_delta=short_call_target)
    if short_call is None:
        log_checkpoint("condor_no_short_call", status="ok", symbol=symbol)
        return None

    # Pull the wider chain (no delta filter) to find long wings.
    wide_filters = ChainFilters(
        dte_min=dte_min, dte_max=dte_max,
        delta_min=0.0, delta_max=1.0,
        open_interest_min=0, volume_min=0,
        bid_ask_spread_max_pct=100.0,
    )
    wide_puts = await fetch_filtered_chain(broker, symbol, "put", wide_filters, today=today)
    wide_calls = await fetch_filtered_chain(broker, symbol, "call", wide_filters, today=today)
    long_put = _find_protective_long(
        wide_puts, short_put, wing_width=wing_width, side=OptionType.PUT,
    )
    long_call = _find_protective_long(
        wide_calls, short_call, wing_width=wing_width, side=OptionType.CALL,
    )
    if long_put is None or long_call is None:
        log_checkpoint(
            "condor_no_viable_structure",
            status="ok", symbol=symbol,
            have_long_put=long_put is not None,
            have_long_call=long_call is not None,
        )
        return None

    sp_mid = _mid(short_put)
    lp_mid = _mid(long_put)
    sc_mid = _mid(short_call)
    lc_mid = _mid(long_call)
    if any(m is None for m in (sp_mid, lp_mid, sc_mid, lc_mid)):
        log_checkpoint("condor_missing_mids", status="ok", symbol=symbol)
        return None
    net_credit = (sp_mid - lp_mid) + (sc_mid - lc_mid)
    if net_credit <= 0:
        log_checkpoint(
            "condor_non_positive_credit",
            status="ok", symbol=symbol, net_credit=net_credit,
        )
        return None
    actual_put_wing = short_put.strike - long_put.strike
    actual_call_wing = long_call.strike - short_call.strike
    # Symmetric condor: take the LARGER wing as the package-level wing
    # width for max_loss accounting (the worst-case exposure is the wider
    # wing). v1 enforces symmetric structure via the strike-selection
    # logic, so this is normally a no-op.
    package_width = max(actual_put_wing, actual_call_wing)
    if package_width <= 0:
        log_checkpoint("condor_zero_width", status="ok", symbol=symbol)
        return None
    credit_pct = (net_credit / package_width) * 100.0
    if credit_pct < min_credit_pct:
        log_checkpoint(
            "condor_skip_low_credit",
            status="ok", symbol=symbol,
            short_put_strike=short_put.strike, short_call_strike=short_call.strike,
            credit=net_credit, width=package_width,
            credit_pct=credit_pct, min_credit_pct=min_credit_pct,
        )
        return None
    max_loss = (package_width - net_credit) * 100.0
    log_checkpoint(
        "condor_selected",
        status="ok", symbol=symbol,
        short_put=short_put.occ_symbol, long_put=long_put.occ_symbol,
        short_call=short_call.occ_symbol, long_call=long_call.occ_symbol,
        credit=net_credit, width=package_width, max_loss=max_loss,
        short_put_delta=short_put.delta, short_call_delta=short_call.delta,
    )
    return IronCondorCandidate(
        long_put=long_put,
        short_put=short_put,
        short_call=short_call,
        long_call=long_call,
        net_credit_per_spread=net_credit,
        max_loss_per_spread=max_loss,
        wing_width_dollars=package_width,
    )


def _build_legs(c: IronCondorCandidate) -> list[OrderLeg]:
    """4 legs in canonical iron-condor order: long put, short put, short
    call, long call. Wire-order doesn't matter to the brokers (legs are
    keyed by OCC symbol) but operator displays read this consistently."""
    def _leg(contract: OptionContract, action: OrderType) -> OrderLeg:
        return OrderLeg(
            contract_symbol=contract.occ_symbol,
            underlying=contract.underlying,
            option_type=contract.option_type,
            strike=contract.strike,
            expiration=contract.expiration,
            action=action,
            ratio_qty=1,
        )
    return [
        _leg(c.long_put, OrderType.BUY_TO_OPEN),
        _leg(c.short_put, OrderType.SELL_TO_OPEN),
        _leg(c.short_call, OrderType.SELL_TO_OPEN),
        _leg(c.long_call, OrderType.BUY_TO_OPEN),
    ]


def _sizing_quantity(
    strategy: StrategyDefinition,
    candidate: IronCondorCandidate,
    *,
    size_multiplier: float = 1.0,
) -> int:
    """Same shape as strategies/spreads.py::_sizing_quantity but reads
    `max_capital_per_condor_usd` (the condor-specific cap)."""
    cap = float(strategy.params.get("max_capital_per_condor_usd", 0) or 0)
    if cap <= 0 or candidate.max_loss_per_spread <= 0:
        return 1
    effective_cap = cap * float(size_multiplier)
    return int(effective_cap // candidate.max_loss_per_spread)


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
    size_multiplier: float = 1.0,
) -> MultiLegProposal | None:
    today = today or date.today()
    if strategy is None:
        log_checkpoint("condor_skip_no_strategy", status="fail", symbol=symbol)
        return None
    account_id = config.get("account", {}).get("id", "primary")
    strategy_id = strategy.id

    position = await repos.positions.get_by_symbol(
        account_id, symbol, strategy_id=strategy_id,
    )
    state = position.state if position else PositionState.IDLE
    # Only propose when flat. Open condors close via separate proposals.
    if state not in (PositionState.IDLE, PositionState.SPREAD_CLOSED):
        log_checkpoint(
            "condor_skip_state",
            status="ok", symbol=symbol, strategy=strategy_id, state=str(state),
        )
        return None

    record = _make_chain_recorder(
        repos,
        cycle_id=position.current_cycle_id if position else None,
        strategy_id=strategy_id,
    )
    candidate = await select_iron_condor(
        broker, symbol, strategy.params,
        today=today, record_chain=record, ivr=ivr,
    )
    if candidate is None:
        return None

    quantity = _sizing_quantity(strategy, candidate, size_multiplier=size_multiplier)
    if quantity <= 0:
        log_checkpoint(
            "condor_skip_over_cap",
            status="skip", symbol=symbol, strategy=strategy_id,
            max_loss=candidate.max_loss_per_spread,
            cap=float(strategy.params.get("max_capital_per_condor_usd", 0) or 0),
        )
        return None

    legs = _build_legs(candidate)
    needs_screen, needs_human = _tier_flags(symbol, universe)
    rationale = (
        f"iron_condor[{strategy_id}] "
        f"LP/SP={candidate.long_put.strike}/{candidate.short_put.strike} "
        f"SC/LC={candidate.short_call.strike}/{candidate.long_call.strike} "
        f"width={candidate.wing_width_dollars:.2f} "
        f"credit={candidate.net_credit_per_spread:.2f} "
        f"max_loss={candidate.max_loss_per_spread:.2f} qty={quantity}"
    )
    return MultiLegProposal(
        symbol=symbol,
        legs=legs,
        net_credit_per_spread=candidate.net_credit_per_spread,
        max_loss_per_spread=candidate.max_loss_per_spread,
        width_dollars=candidate.wing_width_dollars,
        quantity=quantity,
        rationale=rationale,
        strategy_id=strategy_id,
        requires_screen=needs_screen,
        requires_human=needs_human,
        direction=DIRECTION_IRON_CONDOR,
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
    size_multiplier: float = 1.0,
) -> list[MultiLegProposal]:
    out: list[MultiLegProposal] = []
    for entry in universe["tickers"]:
        proposal = await propose_for_symbol(
            broker, repos, entry.symbol, config, universe,
            today=today, strategy=strategy, ivr=ivr,
            size_multiplier=size_multiplier,
        )
        if proposal is not None:
            out.append(proposal)
    log_checkpoint(
        "condor_propose_all",
        status="ok",
        strategy=strategy.id if strategy else "iron_condor",
        n_proposals=len(out),
    )
    return out
