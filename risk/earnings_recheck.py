"""Mid-cycle earnings recheck — TICKET-006.

The entry-time earnings blackout (risk/limits.py rule 4) only fires at order
placement. Companies often confirm earnings dates 2-3 weeks in advance, which
means a CSP opened cleanly can later have earnings INSIDE its remaining DTE
with no second look. This module is that second look.

Runs from scripts/run_bot.py::_post_tick AFTER reconcile and the kill-switch
check, BEFORE the strategy proposers. Self-rate-limited via a tick counter
(default ~1 hour at 5-min cadence) — we don't need to re-fetch earnings every
minute.

Two actions:
  flag_manual  — sets the position to MANUAL_INTERVENTION and notifies Discord.
                 ALWAYS runs, even when the kill switch is tripped. A tripped
                 kill switch must not hide the signal the operator needs to
                 see to clear it.
  close        — builds a BUY_TO_CLOSE Proposal carrying
                 trigger_reason="earnings_recheck_close" and routes it.
                 SKIPPED when the kill switch is tripped (logs
                 earnings_recheck_close_skipped_kill_switch for future audit;
                 deciding whether defensive closes should bypass the kill
                 switch entirely is a separate ticket).

SPREAD_OPEN + action='close' always falls back to flag_manual until the
generic "build a close proposal for one open spread" helper lands with
TICKET-014 (iron condor).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timezone
from typing import Any, Callable

from core.broker import Broker
from core.checkpoint import log_checkpoint
from core.models import (
    OptionContract,
    OrderStatus,
    OrderType,
    Position,
    PositionState,
    StateLog,
    StateLogTrigger,
)
from core.notify import notify
from data.earnings import EarningsLookup, next_earnings as default_next_earnings
from db.repo import Repos
from execution.router import OrderRouter
from strategies.wheel import Proposal


# Action values stored on the result. Distinct strings (not the config action
# values) so a tripped close path is observable in the returned list and not
# silently dropped.
ACTION_FLAG_MANUAL = "flag_manual"
ACTION_CLOSE_PROPOSED = "close_proposed"
ACTION_CLOSE_SKIPPED_KILL_SWITCH = "close_skipped_kill_switch"
ACTION_CLOSE_SPREAD_UNSUPPORTED = "close_spread_unsupported_fallback_flag_manual"
ACTION_PROVIDER_UNAVAILABLE = "provider_unavailable"
ACTION_OUTSIDE_WINDOW = "outside_window"  # not returned, but used internally


@dataclass(frozen=True, slots=True)
class EarningsRecheckResult:
    """One per affected position. Lets tests assert without parsing logs."""

    position_id: int
    symbol: str
    strategy_id: str | None
    action_taken: str             # one of the ACTION_* constants above
    earnings_date: date | None    # None when provider unavailable
    short_expiration: date | None  # None when we couldn't resolve the leg
    rationale: str


# Optional dependency-injectable hook so tests don't have to monkeypatch the
# data.earnings module. Production callers use the default which threads
# through to data.earnings.next_earnings (with its 6h in-process TTL cache).
NextEarningsFn = Callable[[str], EarningsLookup]


async def _latest_short_for_cycle(repos: Repos, cycle_id: int | None):
    """Most-recent FILLED SELL_TO_OPEN for the cycle — same shape strategies/
    wheel_close.py uses to recover the live short leg. Returns None for cycles
    without one (shouldn't happen for CSP_OPEN/CC_OPEN positions, but defensive)."""
    if cycle_id is None:
        return None
    c = await repos.db.connect()
    async with c.execute(
        "SELECT * FROM orders WHERE cycle_id = ? AND order_type = ? "
        "AND status = ? ORDER BY filled_at DESC LIMIT 1",
        (cycle_id, OrderType.SELL_TO_OPEN.value, OrderStatus.FILLED.value),
    ) as cur:
        row = await cur.fetchone()
    if row is None:
        return None
    from core.models import Order
    from db.repo import JSON_FIELDS_BY_TABLE, _row_to_dict
    return Order(**_row_to_dict(row, JSON_FIELDS_BY_TABLE["orders"]))


async def _latest_spread_short_expiration(repos: Repos, cycle_id: int | None) -> date | None:
    """For a SPREAD_OPEN position, recover the short-leg expiration from the
    cycle's latest filled MULTI_LEG_OPEN raw_request. Both legs of our credit
    spreads share the same expiration, so we can take either — we still pick
    the short leg explicitly so the semantics are unambiguous in logs."""
    if cycle_id is None:
        return None
    c = await repos.db.connect()
    async with c.execute(
        "SELECT raw_request FROM orders WHERE cycle_id = ? AND order_type = ? "
        "AND status = ? ORDER BY filled_at DESC LIMIT 1",
        (cycle_id, OrderType.MULTI_LEG_OPEN.value, OrderStatus.FILLED.value),
    ) as cur:
        row = await cur.fetchone()
    if row is None or row["raw_request"] is None:
        return None
    import json
    try:
        legs = json.loads(row["raw_request"]).get("legs") or []
    except (ValueError, TypeError):
        return None
    for leg in legs:
        action = str(leg.get("action", "")).upper()
        if action.startswith("SELL_TO_OPEN"):
            exp = leg.get("expiration")
            if exp:
                try:
                    return datetime.fromisoformat(str(exp)).date()
                except ValueError:
                    return None
    return None


def is_in_earnings_window(
    earnings_date: date, short_expiration: date, *, days_before: int, days_after: int
) -> bool:
    """True when an earnings event poses risk to a position with the given
    short-leg expiration: earnings happens up to `days_before` days BEFORE the
    expiration (we'd hold through the catalyst), or up to `days_after` days
    AFTER (gamma is still bleeding through the close).

    NOT the same predicate as risk/limits.py::in_blackout — that one is a
    window around the EARNINGS date used at entry-time selection. This is a
    window around the EXPIRATION used at mid-cycle recheck, where the question
    is 'is the open position now exposed to earnings'. Same days_before /
    days_after config knobs read from the same block so the dashboard badge
    and the recheck mutation cannot disagree.
    """
    diff = (short_expiration - earnings_date).days
    return -days_after <= diff <= days_before


def _build_close_proposal(
    position: Position, short_order, today: date
) -> Proposal | None:
    """Single-leg BUY_TO_CLOSE Proposal carrying the structured trigger_reason."""
    if (
        short_order.contract_symbol is None
        or short_order.strike is None
        or short_order.expiration is None
        or short_order.option_type is None
    ):
        return None
    contract = OptionContract(
        underlying=position.symbol,
        occ_symbol=short_order.contract_symbol,
        strike=float(short_order.strike),
        expiration=short_order.expiration,
        option_type=short_order.option_type,
    )
    rationale = (
        f"earnings_recheck[{position.strategy_id}] mid-cycle catalyst "
        f"({short_order.expiration} short DTE)"
    )
    return Proposal(
        symbol=position.symbol,
        contract=contract,
        order_type=OrderType.BUY_TO_CLOSE,
        quantity=int(short_order.quantity or 1),
        rationale=rationale,
        strategy_id=position.strategy_id or "monthly_wheel",
        trigger_reason="earnings_recheck_close",
    )


async def _flag_manual(
    repos: Repos,
    position: Position,
    earnings_date: date,
    short_expiration: date,
    config_action: str,
) -> None:
    """Mutate position state → MANUAL_INTERVENTION + write state_log row +
    Discord. Idempotent: skip the notify if the position is already flagged."""
    if position.state == PositionState.MANUAL_INTERVENTION:
        return
    reason = (
        f"earnings_appeared_mid_cycle earnings={earnings_date} "
        f"short_exp={short_expiration} configured_action={config_action}"
    )
    if position.id is not None:
        await repos.positions.update_state(
            position.id, PositionState.MANUAL_INTERVENTION, reason,
        )
        await repos.state_log.insert(
            StateLog(
                position_id=position.id,
                from_state=position.state,
                to_state=PositionState.MANUAL_INTERVENTION,
                reason=reason,
                triggered_by=StateLogTrigger.STRATEGY,
                created_at=datetime.now(timezone.utc).replace(tzinfo=None),
            )
        )
    await notify(
        "earnings.recheck_action",
        f"{position.symbol} flagged — earnings inside remaining DTE",
        symbol=position.symbol,
        strategy=position.strategy_id,
        position_id=position.id,
        action="flag_manual",
        earnings_date=str(earnings_date),
        short_expiration=str(short_expiration),
        days_to_earnings=(earnings_date - datetime.now(timezone.utc).date()).days,
    )


async def check_open_positions_for_new_earnings(
    *,
    repos: Repos,
    router: OrderRouter | None,
    config: dict[str, Any],
    today: date | None = None,
    kill_switch_tripped: bool = False,
    recheck_state: dict[str, int],
    next_earnings_fn: NextEarningsFn | None = None,
) -> list[EarningsRecheckResult]:
    """Main entry point. See module docstring.

    Self-rate-limited: increments recheck_state["ticks_since_check"] on each
    call, only does the per-position work when it crosses check_interval_ticks.
    """
    cfg = (config.get("risk", {}) or {}).get("earnings_recheck", {}) or {}
    if not bool(cfg.get("enabled", False)):
        return []

    interval = int(cfg.get("check_interval_ticks", 12))
    recheck_state["ticks_since_check"] = recheck_state.get("ticks_since_check", 0) + 1
    if recheck_state["ticks_since_check"] < interval:
        return []
    recheck_state["ticks_since_check"] = 0  # reset for the next window

    today = today or datetime.now(timezone.utc).date()
    config_action = str(cfg.get("action", "flag_manual")).lower()
    days_before = int(cfg.get("days_before", 5))
    days_after = int(cfg.get("days_after", 2))
    lookup_fn = next_earnings_fn or default_next_earnings

    account_id = (config.get("account") or {}).get("id", "primary")
    active = await repos.positions.list_active(account_id)

    results: list[EarningsRecheckResult] = []
    eligible_states = (
        PositionState.CSP_OPEN,
        PositionState.CC_OPEN,
        PositionState.SPREAD_OPEN,
    )
    for pos in active:
        if pos.state not in eligible_states:
            continue
        if pos.id is None or pos.current_cycle_id is None:
            continue

        # Resolve the short-leg expiration to know when "remaining DTE" ends.
        if pos.state in (PositionState.CSP_OPEN, PositionState.CC_OPEN):
            short = await _latest_short_for_cycle(repos, pos.current_cycle_id)
            short_expiration = short.expiration if short is not None else None
        else:  # SPREAD_OPEN
            short = None
            short_expiration = await _latest_spread_short_expiration(
                repos, pos.current_cycle_id
            )
        if short_expiration is None:
            continue  # can't reason about "inside remaining DTE" without the leg

        # Provider lookup. None → log distinctly (not silent) and skip.
        lookup = lookup_fn(pos.symbol)
        if lookup.next_date is None:
            log_checkpoint(
                "earnings_recheck_provider_unavailable",
                status="skip",
                symbol=pos.symbol,
                strategy=pos.strategy_id,
                position_id=pos.id,
            )
            results.append(EarningsRecheckResult(
                position_id=pos.id,
                symbol=pos.symbol,
                strategy_id=pos.strategy_id,
                action_taken=ACTION_PROVIDER_UNAVAILABLE,
                earnings_date=None,
                short_expiration=short_expiration,
                rationale="next_earnings returned None",
            ))
            continue

        earnings_date = lookup.next_date
        # Only act if earnings is in the future AND inside the configured window
        # around the short-leg expiration. (Past earnings, or distant earnings,
        # are irrelevant — they were either handled at entry or aren't a threat.)
        if earnings_date < today:
            continue
        if not is_in_earnings_window(
            earnings_date, short_expiration,
            days_before=days_before, days_after=days_after,
        ):
            continue

        days_to_earnings = (earnings_date - today).days

        # Action dispatch. Order matters: spread-close fallback BEFORE
        # kill-switch check because the spread fallback degrades to flag_manual
        # which we still want to run even with kill switch tripped.
        if config_action == "close" and pos.state == PositionState.SPREAD_OPEN:
            log_checkpoint(
                "earnings_recheck_close_spread_unsupported",
                status="ok",
                symbol=pos.symbol, strategy=pos.strategy_id,
                position_id=pos.id, earnings_date=str(earnings_date),
                note="spread close-action unsupported until TICKET-014; falling back to flag_manual",
            )
            await _flag_manual(repos, pos, earnings_date, short_expiration, config_action)
            results.append(EarningsRecheckResult(
                position_id=pos.id, symbol=pos.symbol, strategy_id=pos.strategy_id,
                action_taken=ACTION_CLOSE_SPREAD_UNSUPPORTED,
                earnings_date=earnings_date, short_expiration=short_expiration,
                rationale=f"spread close-action fell back to flag_manual; days_to_earnings={days_to_earnings}",
            ))
            continue

        if config_action == "close":
            if kill_switch_tripped:
                log_checkpoint(
                    "earnings_recheck_close_skipped_kill_switch",
                    status="skip",
                    symbol=pos.symbol, strategy=pos.strategy_id,
                    position_id=pos.id,
                    earnings_date=str(earnings_date),
                    short_expiration=str(short_expiration),
                    note="defensive close blocked by kill switch — see TICKET-006 docs",
                )
                results.append(EarningsRecheckResult(
                    position_id=pos.id, symbol=pos.symbol, strategy_id=pos.strategy_id,
                    action_taken=ACTION_CLOSE_SKIPPED_KILL_SWITCH,
                    earnings_date=earnings_date, short_expiration=short_expiration,
                    rationale=f"kill switch tripped; days_to_earnings={days_to_earnings}",
                ))
                continue
            if short is None or router is None:
                continue
            proposal = _build_close_proposal(pos, short, today)
            if proposal is None:
                continue
            log_checkpoint(
                "earnings_recheck_close_proposed",
                status="ok",
                symbol=pos.symbol, strategy=pos.strategy_id,
                position_id=pos.id, earnings_date=str(earnings_date),
                short_expiration=str(short_expiration),
                days_to_earnings=days_to_earnings,
            )
            await router.place(proposal, today=today)
            await notify(
                "earnings.recheck_action",
                f"{pos.symbol} earnings-recheck close placed",
                symbol=pos.symbol, strategy=pos.strategy_id,
                position_id=pos.id, action="close",
                earnings_date=str(earnings_date),
                short_expiration=str(short_expiration),
                days_to_earnings=days_to_earnings,
            )
            results.append(EarningsRecheckResult(
                position_id=pos.id, symbol=pos.symbol, strategy_id=pos.strategy_id,
                action_taken=ACTION_CLOSE_PROPOSED,
                earnings_date=earnings_date, short_expiration=short_expiration,
                rationale=f"close proposal routed; days_to_earnings={days_to_earnings}",
            ))
            continue

        # action == 'flag_manual' (or any unknown value — fall through to the
        # safe default since flag_manual is purely a state annotation).
        await _flag_manual(repos, pos, earnings_date, short_expiration, config_action)
        results.append(EarningsRecheckResult(
            position_id=pos.id, symbol=pos.symbol, strategy_id=pos.strategy_id,
            action_taken=ACTION_FLAG_MANUAL,
            earnings_date=earnings_date, short_expiration=short_expiration,
            rationale=f"flagged; days_to_earnings={days_to_earnings}",
        ))

    return results
