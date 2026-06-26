"""Order router.

Single funnel for placing orders. Responsibilities:

1. Run RiskGate (§8 rules 1-7). On failure: do not submit, do not write to DB.
2. Generate a deterministic client_order_id so retries are idempotent.
3. Submit to broker with exponential backoff on BrokerUnavailable.
4. Persist Order with status=PENDING. Insert/upgrade Position to *_PENDING.
5. Honor execution.dry_run — when true, no broker call AND no DB write.

What this module DOES NOT do:
- Transition positions to *_OPEN. That is exclusively the reconciler's job.
- Decide what to trade. The wheel orchestrator builds the Proposal upstream.
- Roll/exit logic.
"""

from __future__ import annotations

import asyncio
import hashlib
import uuid
from dataclasses import dataclass, replace
from datetime import UTC, date, datetime
from typing import Any

from core.broker import Broker, BrokerUnavailable, OrderRejected
from core.checkpoint import checkpoint, log_checkpoint
from core.models import Order, OrderStatus, OrderType, Position, PositionState
from db.repo import Repos
from execution.market_hours import within_entry_window
from risk.limits import RiskCheckFailed, RiskGate
from strategies.spreads import MultiLegProposal
from strategies.wheel import Proposal


# TICKET-015: BUY_TO_OPEN joins the open set — PMCC's long-LEAP entry is the
# first routed BUY_TO_OPEN (synthetic stock legs are inserted, never routed).
# This makes PMCC long opens respect the entry-window cutoff like any entry.
_OPEN_ORDER_TYPES = (
    OrderType.SELL_TO_OPEN,
    OrderType.BUY_TO_OPEN,
    OrderType.MULTI_LEG_OPEN,
)


@dataclass(frozen=True, slots=True)
class RouterConfig:
    dry_run: bool = False
    retry_max_attempts: int = 5
    retry_initial_backoff_seconds: float = 1.0
    retry_max_backoff_seconds: float = 60.0
    # Cancel-and-replace any PENDING/PARTIAL order whose deterministic
    # client_order_id matches the new proposal AND has been pending longer
    # than this threshold. Avoids the broker-side "client_order_id must be
    # unique" rejection cycle when market moves past the stale limit.
    stale_pending_minutes: float = 15.0
    # Slippage concession on multi-leg credit-spread OPENS. We price at
    # net_credit − open_slippage so the order is marketable enough to fill;
    # mid-to-mid credit often won't cross.
    open_slippage: float = 0.05
    # Slippage concession on multi-leg CLOSES. A spread buyback is a debit
    # (net_credit < 0); pricing exactly at mid often won't cross, so we concede
    # a little — willing to pay net_credit − close_slippage (more negative) to
    # actually exit. Same subtraction direction as opens. Set to 0 to disable.
    close_slippage: float = 0.05
    # Don't OPEN new positions in the final N minutes before the 16:00 ET
    # close. Opens placed near/after close can't fill or be managed. Closes
    # and rolls are exempt.
    entry_cutoff_minutes_before_close: int = 15
    # news_check enforcement. When True (advisory mode, used during paper
    # testing), a "caution" verdict is logged but the order proceeds at FULL
    # size; only an explicit "block" cancels. When False (full enforcement),
    # "caution" halves the size (and a 1-lot caution becomes a block). The
    # underlying news_check still runs + records either way.
    news_check_advisory: bool = False


@dataclass(frozen=True, slots=True)
class RouteResult:
    proposal: Proposal | MultiLegProposal
    placed: Order | None  # None when dry-run, risk-failed, or news-blocked
    risk_failed: bool = False
    risk_failure_rule: str | None = None
    risk_failure_detail: str | None = None
    dry_run: bool = False
    news_decision: str | None = None  # "proceed" | "caution" | "block" | None
    news_rationale: str | None = None
    quantity_adjusted: int | None = None  # final qty if news_check forced a halve
    skipped_duplicate_pending: bool = False  # active PENDING order, too young to replace
    skipped_outside_entry_window: bool = False  # open attempted too close to / after close


def _utcnow() -> datetime:
    return datetime.now(UTC).replace(tzinfo=None)


def _option_limit_price(bid: float | None, ask: float | None) -> float | None:
    """Mid-price rounded to a valid options tick.

    OCC tick rules: penny increments below $3, nickel increments at/above $3.
    Alpaca enforces this at submission time — sending 3-decimal mids triggers
    `limit price must be limited to 2 decimal places`. Tastytrade is more
    forgiving but still rejects sub-penny on cheap options.

    Returns None when bid/ask aren't both present.
    """
    if bid is None or ask is None:
        return None
    mid = (bid + ask) / 2
    if mid <= 0:
        return None
    if mid >= 3.0:
        # Nearest $0.05.
        snapped = round(mid * 20) / 20
    else:
        snapped = round(mid, 2)
    # Floor at $0.01 — Alpaca rejects $0.00 limit on a SELL.
    return max(snapped, 0.01)


def _multi_leg_client_order_id(
    proposal: MultiLegProposal, today: date | None = None
) -> str:
    """Deterministic per-(strategy, underlying, legs, day) so retries collide."""
    today = today or datetime.now(UTC).date()
    leg_key = "+".join(
        f"{leg.contract_symbol}:{leg.action.value if hasattr(leg.action, 'value') else leg.action}:{leg.ratio_qty}"
        for leg in proposal.legs
    )
    raw = (
        f"{proposal.strategy_id}|mleg|{proposal.symbol}|{leg_key}|"
        f"{proposal.quantity}|{today.isoformat()}"
    )
    digest = hashlib.sha1(raw.encode()).hexdigest()[:16]
    return f"wb-{digest}"


def _client_order_id(proposal: Proposal, today: date | None = None) -> str:
    """Deterministic per-(proposal, day, strategy) so a retry of the same proposal
    on the same day collides with the prior attempt — broker idempotency does
    the rest. Strategy is part of the key so two strategies trading the same
    symbol/contract on the same day get distinct client_order_ids."""
    today = today or datetime.now(UTC).date()
    raw = (
        f"{proposal.strategy_id}|{proposal.symbol}|{proposal.contract.occ_symbol}|"
        f"{proposal.order_type.value}|{proposal.quantity}|{today.isoformat()}"
    )
    digest = hashlib.sha1(raw.encode()).hexdigest()[:16]
    return f"wb-{digest}"


def _router_config(config: dict[str, Any]) -> RouterConfig:
    section = config.get("execution", {}) or {}
    intel = config.get("intelligence", {}) or {}
    return RouterConfig(
        dry_run=bool(section.get("dry_run", False)),
        retry_max_attempts=int(section.get("retry_max_attempts", 5)),
        retry_initial_backoff_seconds=float(section.get("retry_initial_backoff_seconds", 1.0)),
        retry_max_backoff_seconds=float(section.get("retry_max_backoff_seconds", 60.0)),
        stale_pending_minutes=float(section.get("stale_pending_minutes", 15.0)),
        open_slippage=float(section.get("open_slippage", 0.05)),
        close_slippage=float(section.get("close_slippage", 0.05)),
        entry_cutoff_minutes_before_close=int(
            section.get("entry_cutoff_minutes_before_close", 15)
        ),
        news_check_advisory=bool(intel.get("news_check_advisory", False)),
    )


def _pending_state_for(
    order_type: OrderType,
    contract_is_put: bool,
    strategy_id: str | None = None,
    *,
    is_swing: bool = False,
) -> PositionState:
    # Sub-sprint 2.2b: directional swing — a single deep-ITM long. BUY_TO_OPEN →
    # SWING_PENDING (reconciler upgrades to SWING_OPEN on the fill). SELL_TO_CLOSE
    # holds SWING_OPEN until the reconciler observes the close fill → IDLE.
    if is_swing:
        if order_type == OrderType.BUY_TO_OPEN:
            return PositionState.SWING_PENDING
        return PositionState.SWING_OPEN
    # TICKET-015: PMCC has its own pending states. long=BUY_TO_OPEN call,
    # short=SELL_TO_OPEN call, long-close=SELL_TO_CLOSE call (transient
    # CLOSING), short-close=BUY_TO_CLOSE (no state change; reconciler drives
    # PMCC_BOTH_OPEN → PMCC_LONG_OPEN on the observed fill).
    if strategy_id == "pmcc":
        if order_type == OrderType.BUY_TO_OPEN:
            return PositionState.PMCC_LONG_PENDING
        if order_type == OrderType.SELL_TO_OPEN:
            return PositionState.PMCC_SHORT_PENDING
        if order_type == OrderType.SELL_TO_CLOSE:
            return PositionState.PMCC_CLOSING
        return PositionState.PMCC_BOTH_OPEN  # BUY_TO_CLOSE short — held until fill
    # SELL_TO_OPEN of a put → CSP_PENDING; of a call → CC_PENDING.
    # BUY_TO_CLOSE returns the position to its prior state on fill, so we don't
    # change state here for closes; we keep it CSP_OPEN/CC_OPEN until reconciler
    # observes the close.
    if order_type == OrderType.SELL_TO_OPEN:
        return PositionState.CSP_PENDING if contract_is_put else PositionState.CC_PENDING
    if order_type == OrderType.SELL_TO_CLOSE:
        return PositionState.SHARES_HELD  # no-op effectively; reconciler will overwrite
    return PositionState.CSP_OPEN if contract_is_put else PositionState.CC_OPEN


class OrderRouter:
    def __init__(
        self,
        broker: Broker,
        repos: Repos,
        config: dict[str, Any],
        universe: dict[str, Any],
        *,
        news_checker: Any = None,
    ) -> None:
        self._broker = broker
        self._repos = repos
        self._config = config
        self._universe = universe
        self._cfg = _router_config(config)
        self._gate = RiskGate(broker, repos, config, universe)
        # Sub-sprint 2.2b: ids of strategies whose type is "swing" — these route
        # to the SWING_* pending states (single deep-ITM long lifecycle).
        from core.strategies import load_strategies
        self._swing_ids = {s.id for s in load_strategies(config) if s.type == "swing"}
        # News checker is an awaitable that takes (symbol) and returns a
        # NewsCheckResult-shaped object with .decision / .rationale. Optional —
        # tests pass a stub or None to disable.
        self._news_checker = news_checker

    async def _replace_stale_pending(
        self, client_id: str, proposal_label: str
    ) -> tuple[str, str | None]:
        """Pre-submission stale-order handler.

        Looks up any existing local order with the deterministic
        `client_id`. If it's still active (PENDING / PARTIAL), decides
        whether to skip the new submission (order is fresh enough that the
        broker may still fill the prior limit) or cancel-and-replace
        (order has been stale longer than `stale_pending_minutes`).

        Returns (action, replacement_id):
          "proceed"  → submit normally with the original client_id
          "skip"     → caller returns a RouteResult marked skipped_duplicate_pending
          "replace"  → submit with the returned (unique) replacement client_id;
                       prior order already cancelled at broker + marked CANCELLED locally.
        """
        existing = await self._repos.orders.get_by_client_id(client_id)
        if existing is None or existing.status not in (
            OrderStatus.PENDING, OrderStatus.PARTIAL,
        ):
            return ("proceed", None)
        now = _utcnow()
        age_min = (now - existing.placed_at).total_seconds() / 60.0
        if age_min < self._cfg.stale_pending_minutes:
            log_checkpoint(
                "router_skip_duplicate_pending",
                status="skip",
                client_order_id=client_id,
                proposal=proposal_label,
                age_min=round(age_min, 1),
                threshold_min=self._cfg.stale_pending_minutes,
            )
            return ("skip", None)
        # Stale — cancel at broker, mark CANCELLED locally, replace with new id.
        if existing.broker_order_id:
            try:
                await self._broker.cancel_order(existing.broker_order_id)
            except Exception as exc:
                log_checkpoint(
                    "router_cancel_stale_fail",
                    status="fail",
                    client_order_id=client_id,
                    broker_order_id=existing.broker_order_id,
                    error=str(exc),
                )
                # Cancel failed — bail rather than risk a duplicate at the broker.
                return ("skip", None)
        if existing.id is not None:
            status_val = OrderStatus.CANCELLED.value if hasattr(
                OrderStatus.CANCELLED, "value"
            ) else "CANCELLED"
            await self._repos.orders.update(existing.id, status=status_val)
        new_id = f"{client_id}-r{int(now.timestamp())}"
        log_checkpoint(
            "router_replace_stale_pending",
            status="ok",
            old_client_order_id=client_id,
            new_client_order_id=new_id,
            broker_order_id_cancelled=existing.broker_order_id,
            age_min=round(age_min, 1),
        )
        return ("replace", new_id)

    async def place(
        self,
        proposal: Proposal,
        *,
        sleep: Any = asyncio.sleep,
        today: date | None = None,
    ) -> RouteResult:
        # Entry-window gate: don't OPEN new positions near/after the close.
        if proposal.order_type in _OPEN_ORDER_TYPES and not within_entry_window(
            minutes_before_close=self._cfg.entry_cutoff_minutes_before_close,
        ):
            log_checkpoint(
                "router_skip_entry_window",
                status="skip",
                symbol=proposal.symbol,
                order_type=proposal.order_type.value,
            )
            return RouteResult(
                proposal=proposal, placed=None, skipped_outside_entry_window=True,
            )

        # Risk gates first. Cheap to check, expensive to fail late.
        try:
            await self._gate.evaluate(proposal, today=today)
        except RiskCheckFailed as exc:
            log_checkpoint(
                "router_risk_fail",
                status="fail",
                symbol=proposal.symbol,
                rule=exc.rule,
                detail=exc.detail,
            )
            return RouteResult(
                proposal=proposal,
                placed=None,
                risk_failed=True,
                risk_failure_rule=exc.rule,
                risk_failure_detail=exc.detail,
            )

        # News check (spec §9.2: "Before placing any new CSP" — puts only).
        # CCs are about closing existing exposure on already-held shares; a
        # news catalyst is a different decision class. Closes/rolls also skip.
        news_decision: str | None = None
        news_rationale: str | None = None
        effective_qty = proposal.quantity
        from core.models import OptionType

        # Wheel CSP path is byte-identical (SELL_TO_OPEN + PUT, no profile).
        # TICKET-015 adds a disjoint branch: a single-leg OPEN that carries a
        # news_check_profile (PMCC long BUY_TO_OPEN) fires news_check with that
        # profile. The two conditions can never both be true — a wheel CSP is
        # SELL_TO_OPEN+PUT with profile=None; a PMCC long is BUY_TO_OPEN+CALL
        # with profile set.
        _profile = getattr(proposal, "news_check_profile", None)
        _is_wheel_csp = (
            proposal.order_type == OrderType.SELL_TO_OPEN
            and proposal.contract.option_type == OptionType.PUT
        )
        _is_profiled_open = (
            _profile is not None and proposal.order_type in _OPEN_ORDER_TYPES
        )
        if self._news_checker is not None and (_is_wheel_csp or _is_profiled_open):
            if _is_wheel_csp:
                check = await self._news_checker(proposal.symbol)
            else:
                check = await self._news_checker(proposal.symbol, _profile)
            news_decision = getattr(check, "decision", None)
            news_rationale = getattr(check, "rationale", None)
            log_checkpoint(
                "router_news_check",
                status="ok",
                symbol=proposal.symbol,
                decision=news_decision,
            )
            if news_decision == "block":
                return RouteResult(
                    proposal=proposal,
                    placed=None,
                    news_decision="block",
                    news_rationale=news_rationale,
                )
            if news_decision == "caution" and self._cfg.news_check_advisory:
                # Advisory mode (paper testing): record the caution but proceed
                # at full size. Only a hard "block" stops the order.
                log_checkpoint(
                    "router_news_caution_advisory",
                    status="ok",
                    symbol=proposal.symbol,
                    note="advisory mode — proceeding at full size",
                    rationale=news_rationale,
                )
            elif news_decision == "caution":
                halved = effective_qty // 2
                if halved == 0:
                    # Spec-stretch: caution + qty=1 → block (halving a single
                    # contract isn't possible).
                    log_checkpoint(
                        "router_news_caution_block",
                        status="ok",
                        symbol=proposal.symbol,
                        original_qty=effective_qty,
                    )
                    return RouteResult(
                        proposal=proposal,
                        placed=None,
                        news_decision="caution",
                        news_rationale=(news_rationale or "")
                        + " | qty=1 cannot be halved — treated as block",
                    )
                effective_qty = halved
                # Preserve EVERY other field (strategy_id especially — rebuilding
                # the Proposal by hand previously dropped it, silently re-tagging
                # a halved weekly_wheel CSP as the default monthly_wheel and
                # corrupting its idempotency key + position lookup).
                proposal = replace(
                    proposal,
                    quantity=halved,
                    rationale=proposal.rationale + " (size halved by news_check)",
                )

        if self._cfg.dry_run:
            log_checkpoint(
                "router_dry_run",
                status="ok",
                symbol=proposal.symbol,
                occ=proposal.contract.occ_symbol,
                qty=proposal.quantity,
            )
            return RouteResult(
                proposal=proposal,
                placed=None,
                dry_run=True,
                news_decision=news_decision,
                news_rationale=news_rationale,
                quantity_adjusted=effective_qty if news_decision == "caution" else None,
            )

        # Pre-submission: handle a stale PENDING order with the same
        # deterministic client_order_id (Sprint 14 — Option B). Either skip
        # (still fresh) or cancel + replace with a unique id.
        base_client_id = _client_order_id(proposal, today)
        action, replacement_id = await self._replace_stale_pending(
            base_client_id, f"{proposal.symbol} {proposal.order_type.value}",
        )
        if action == "skip":
            return RouteResult(
                proposal=proposal,
                placed=None,
                skipped_duplicate_pending=True,
                news_decision=news_decision,
                news_rationale=news_rationale,
            )
        effective_client_id = replacement_id if action == "replace" else base_client_id

        order = self._build_order(proposal, today=today, client_order_id=effective_client_id)
        placed = await self._submit_with_retry(order, sleep=sleep)

        # DB writes. Order first; position state second.
        await self._persist_order(placed)
        await self._upsert_position_pending(proposal, placed)

        return RouteResult(
            proposal=proposal,
            placed=placed,
            news_decision=news_decision,
            news_rationale=news_rationale,
            quantity_adjusted=effective_qty if news_decision == "caution" else None,
        )

    async def place_multi_leg(
        self,
        proposal: MultiLegProposal,
        *,
        sleep: Any = asyncio.sleep,
        today: date | None = None,
    ) -> RouteResult:
        """Place a defined-risk multi-leg package atomically.

        Differences vs `place(Proposal)`:
          - News check is OPT-IN per strategy via `proposal.news_check_profile`.
            When None (default for put_spread/bear_call_spread — preserves
            existing behavior), the gate is skipped. When set (iron_condor uses
            `neutral_range`), news_check fires with that profile's prompt and
            applies the standard block/caution handling. PMCC long-call opens
            and calendar opens will follow the same pattern.
          - Limit price is the proposal's net credit, snapped to a $0.01 tick.
          - Position state goes to SPREAD_PENDING; reconciler upgrades to
            SPREAD_OPEN on observed fill.
        """
        # Entry-window gate: spread opens are entries — don't place near close.
        if proposal.order_type == OrderType.MULTI_LEG_OPEN and not within_entry_window(
            minutes_before_close=self._cfg.entry_cutoff_minutes_before_close,
        ):
            log_checkpoint(
                "router_skip_entry_window",
                status="skip",
                symbol=proposal.symbol,
                strategy=proposal.strategy_id,
                order_type=proposal.order_type.value,
            )
            return RouteResult(
                proposal=proposal, placed=None, skipped_outside_entry_window=True,
            )

        try:
            await self._gate.evaluate(proposal, today=today)
        except RiskCheckFailed as exc:
            log_checkpoint(
                "router_risk_fail",
                status="fail",
                symbol=proposal.symbol,
                strategy=proposal.strategy_id,
                rule=exc.rule,
                detail=exc.detail,
            )
            return RouteResult(
                proposal=proposal,
                placed=None,
                risk_failed=True,
                risk_failure_rule=exc.rule,
                risk_failure_detail=exc.detail,
            )

        # TICKET-014: per-strategy news_check on MULTI_LEG_OPEN. Opt-in via
        # proposal.news_check_profile. Mirrors the single-leg path's
        # block/caution/advisory handling. Caution + qty=1 → skip (can't half
        # a structured package). Closes (MULTI_LEG_CLOSE) bypass.
        news_decision: str | None = None
        news_rationale: str | None = None
        if (
            self._news_checker is not None
            and proposal.order_type == OrderType.MULTI_LEG_OPEN
            and proposal.news_check_profile is not None
        ):
            check = await self._news_checker(
                proposal.symbol, proposal.news_check_profile,
            )
            news_decision = getattr(check, "decision", None)
            news_rationale = getattr(check, "rationale", None)
            log_checkpoint(
                "router_news_check",
                status="ok",
                symbol=proposal.symbol,
                strategy=proposal.strategy_id,
                profile=proposal.news_check_profile,
                decision=news_decision,
            )
            if news_decision == "block":
                log_checkpoint(
                    "router_news_block",
                    status="skip",
                    symbol=proposal.symbol,
                    strategy=proposal.strategy_id,
                    rationale=news_rationale,
                )
                return RouteResult(
                    proposal=proposal,
                    placed=None,
                    news_decision="block",
                    news_rationale=news_rationale,
                )
            if news_decision == "caution" and self._cfg.news_check_advisory:
                log_checkpoint(
                    "router_news_caution_advisory",
                    status="ok",
                    symbol=proposal.symbol,
                    strategy=proposal.strategy_id,
                    note="advisory mode — proceeding at full size",
                    rationale=news_rationale,
                )
            elif news_decision == "caution":
                halved = proposal.quantity // 2
                if halved == 0:
                    # qty=1 caution → skip. Halving a structured package
                    # below 1 contract isn't meaningful; the conservative
                    # call is to not enter.
                    log_checkpoint(
                        "router_news_caution_block",
                        status="skip",
                        symbol=proposal.symbol,
                        strategy=proposal.strategy_id,
                        original_qty=proposal.quantity,
                        note="qty=1 cannot be halved — treated as block",
                    )
                    return RouteResult(
                        proposal=proposal,
                        placed=None,
                        news_decision="caution",
                        news_rationale=(news_rationale or "")
                        + " | qty=1 cannot be halved — treated as block",
                    )
                proposal = replace(
                    proposal,
                    quantity=halved,
                    rationale=proposal.rationale + " (size halved by news_check)",
                )

        if self._cfg.dry_run:
            log_checkpoint(
                "router_dry_run_mleg",
                status="ok",
                symbol=proposal.symbol,
                strategy=proposal.strategy_id,
                qty=proposal.quantity,
                n_legs=len(proposal.legs),
                net_credit=proposal.net_credit_per_spread,
            )
            return RouteResult(proposal=proposal, placed=None, dry_run=True)

        base_client_id = _multi_leg_client_order_id(proposal, today)
        action, replacement_id = await self._replace_stale_pending(
            base_client_id, f"{proposal.symbol} {proposal.order_type.value}",
        )
        if action == "skip":
            return RouteResult(
                proposal=proposal,
                placed=None,
                skipped_duplicate_pending=True,
            )
        client_id = replacement_id if action == "replace" else base_client_id
        # Concede slippage so the package is marketable enough to fill —
        # mid-to-mid often won't cross. OPENS give up a little credit
        # (net_credit − open_slippage); CLOSES (net_credit < 0, a debit) pay a
        # little more (net_credit − close_slippage, i.e. more negative). Same
        # subtraction direction for both.
        if proposal.order_type == OrderType.MULTI_LEG_OPEN:
            limit_price = round(proposal.net_credit_per_spread - self._cfg.open_slippage, 2)
        else:
            limit_price = round(proposal.net_credit_per_spread - self._cfg.close_slippage, 2)

        last_exc: Exception | None = None
        backoff = self._cfg.retry_initial_backoff_seconds
        placed: Order | None = None
        for attempt in range(1, self._cfg.retry_max_attempts + 1):
            with checkpoint(
                "router_submit_mleg",
                attempt=attempt,
                symbol=proposal.symbol,
                strategy=proposal.strategy_id,
            ) as ctx:
                try:
                    placed = await self._broker.place_multi_leg_order(
                        underlying=proposal.symbol,
                        legs=list(proposal.legs),
                        quantity=proposal.quantity,
                        limit_price=limit_price,
                        client_order_id=client_id,
                        strategy_id=proposal.strategy_id,
                        account_id=self._config.get("account", {}).get("id", "primary"),
                    )
                    ctx["broker_order_id"] = placed.broker_order_id
                    ctx["status"] = (
                        placed.status.value
                        if hasattr(placed.status, "value")
                        else str(placed.status)
                    )
                    break
                except OrderRejected:
                    raise
                except BrokerUnavailable as exc:
                    last_exc = exc
                    ctx["error"] = str(exc)
                    if attempt >= self._cfg.retry_max_attempts:
                        raise
            await sleep(min(backoff, self._cfg.retry_max_backoff_seconds))
            backoff = min(backoff * 2, self._cfg.retry_max_backoff_seconds)
        if placed is None:
            raise BrokerUnavailable(str(last_exc) if last_exc else "exhausted retries")

        # TICKET-014 precursor #1: stamp the proposal's authoritative
        # width_dollars onto raw_request so the reconciler can read it
        # verbatim (see _open_cycle_for_spread). The old computed-from-legs
        # path (max-strike − min-strike) returns the outer span for a 4-leg
        # iron condor — e.g. 20 instead of 5 for a 90/95P + 105/110C — and
        # overstates initial_capital_at_risk ~4×. Verticals still produce
        # the same number from both paths, so the reconciler falls back
        # to the leg-derived width when this field is absent.
        if placed.raw_request is not None and proposal.width_dollars:
            placed.raw_request["width_dollars"] = float(proposal.width_dollars)

        await self._persist_order(placed)
        await self._upsert_position_spread_pending(proposal, placed)
        return RouteResult(
            proposal=proposal,
            placed=placed,
            news_decision=news_decision,
            news_rationale=news_rationale,
        )

    # -- Internals ------------------------------------------------------------

    def _build_order(
        self,
        proposal: Proposal,
        *,
        today: date | None,
        client_order_id: str | None = None,
    ) -> Order:
        contract = proposal.contract
        return Order(
            account_id=self._config.get("account", {}).get("id", "primary"),
            symbol=proposal.symbol,
            strategy_id=proposal.strategy_id,
            order_type=proposal.order_type,
            contract_symbol=contract.occ_symbol,
            strike=contract.strike,
            expiration=contract.expiration,
            option_type=contract.option_type,
            quantity=proposal.quantity,
            limit_price=_option_limit_price(contract.bid, contract.ask),
            status=OrderStatus.PENDING,
            placed_at=_utcnow(),
            client_order_id=client_order_id or _client_order_id(proposal, today),
            # TICKET-005: plumb the structured close-reason from the proposer
            # into orders.trigger_reason so the dashboard + post-hoc analysis
            # can read it without parsing the rationale string.
            trigger_reason=proposal.trigger_reason,
        )

    async def _submit_with_retry(
        self,
        order: Order,
        *,
        sleep: Any,
    ) -> Order:
        cfg = self._cfg
        backoff = cfg.retry_initial_backoff_seconds
        last_exc: Exception | None = None
        for attempt in range(1, cfg.retry_max_attempts + 1):
            with checkpoint("router_submit", attempt=attempt, occ=order.contract_symbol) as ctx:
                try:
                    placed = await self._broker.place_order(order)
                    ctx["broker_order_id"] = placed.broker_order_id
                    ctx["status"] = placed.status.value if hasattr(placed.status, "value") else str(placed.status)
                    return placed
                except OrderRejected:
                    # Rejection is broker-side validation; retrying won't help.
                    raise
                except BrokerUnavailable as exc:
                    last_exc = exc
                    ctx["error"] = str(exc)
                    if attempt >= cfg.retry_max_attempts:
                        raise
            await sleep(min(backoff, cfg.retry_max_backoff_seconds))
            backoff = min(backoff * 2, cfg.retry_max_backoff_seconds)
        # unreachable; loop either returns or raises
        raise BrokerUnavailable(str(last_exc) if last_exc else "exhausted retries")

    async def _persist_order(self, order: Order) -> None:
        existing = (
            await self._repos.orders.get_by_client_id(order.client_order_id)
            if order.client_order_id
            else None
        )
        if existing is not None:
            # Idempotent re-submit landed; update broker_order_id/status if newer.
            updates: dict[str, Any] = {}
            if order.broker_order_id and order.broker_order_id != existing.broker_order_id:
                updates["broker_order_id"] = order.broker_order_id
            if order.status != existing.status:
                updates["status"] = order.status.value if hasattr(order.status, "value") else str(order.status)
            if order.raw_response is not None:
                updates["raw_response"] = order.raw_response
            if updates and existing.id is not None:
                await self._repos.orders.update(existing.id, **updates)
            return
        await self._repos.orders.insert(order)

    async def _upsert_position_pending(self, proposal: Proposal, placed: Order) -> None:
        from core.models import OptionType

        account_id = self._config.get("account", {}).get("id", "primary")
        contract_is_put = proposal.contract.option_type == OptionType.PUT
        new_state = _pending_state_for(
            proposal.order_type, contract_is_put, proposal.strategy_id,
            is_swing=proposal.strategy_id in self._swing_ids,
        )
        existing = await self._repos.positions.get_by_symbol(
            account_id, proposal.symbol, strategy_id=proposal.strategy_id
        )
        now = _utcnow()
        # TICKET-015: PMCC short-close (BUY_TO_CLOSE) leaves state unchanged —
        # the reconciler drives PMCC_BOTH_OPEN → PMCC_LONG_OPEN on the fill.
        if proposal.strategy_id == "pmcc" and proposal.order_type == OrderType.BUY_TO_CLOSE:
            return
        if proposal.strategy_id == "pmcc":
            if existing is None:
                await self._repos.positions.insert(
                    Position(
                        account_id=account_id,
                        symbol=proposal.symbol,
                        strategy_id=proposal.strategy_id,
                        state=new_state,
                        shares=0,
                        state_changed_at=now,
                        state_change_reason=f"router_pending:{placed.client_order_id}",
                    )
                )
            elif existing.id is not None:
                await self._repos.positions.update_state(
                    existing.id, new_state,
                    f"router_pending:{placed.client_order_id}", when=now,
                )
            return
        if existing is None:
            await self._repos.positions.insert(
                Position(
                    account_id=account_id,
                    symbol=proposal.symbol,
                    strategy_id=proposal.strategy_id,
                    state=new_state,
                    shares=0,
                    state_changed_at=now,
                    state_change_reason=f"router_pending:{placed.client_order_id}",
                )
            )
            return
        # Don't downgrade SHARES_HELD when placing a CC.
        if proposal.order_type == OrderType.SELL_TO_OPEN and not contract_is_put:
            # CC pending — keep shares but mark intent.
            if existing.id is not None:
                await self._repos.positions.update_state(
                    existing.id,
                    PositionState.CC_PENDING,
                    f"router_pending:{placed.client_order_id}",
                    when=now,
                )
            return
        if existing.id is not None:
            await self._repos.positions.update_state(
                existing.id,
                new_state,
                f"router_pending:{placed.client_order_id}",
                when=now,
            )

    async def _upsert_position_spread_pending(
        self, proposal: MultiLegProposal, placed: Order
    ) -> None:
        account_id = self._config.get("account", {}).get("id", "primary")
        existing = await self._repos.positions.get_by_symbol(
            account_id, proposal.symbol, strategy_id=proposal.strategy_id
        )
        now = _utcnow()
        reason = f"router_pending:{placed.client_order_id}"
        if existing is None:
            await self._repos.positions.insert(
                Position(
                    account_id=account_id,
                    symbol=proposal.symbol,
                    strategy_id=proposal.strategy_id,
                    state=PositionState.SPREAD_PENDING,
                    shares=0,
                    state_changed_at=now,
                    state_change_reason=reason,
                )
            )
            return
        if existing.id is not None:
            await self._repos.positions.update_state(
                existing.id,
                PositionState.SPREAD_PENDING,
                reason,
                when=now,
            )
