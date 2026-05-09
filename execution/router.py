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
from dataclasses import dataclass
from datetime import UTC, date, datetime
from typing import Any

from core.broker import Broker, BrokerUnavailable, OrderRejected
from core.checkpoint import checkpoint, log_checkpoint
from core.models import Order, OrderStatus, OrderType, Position, PositionState
from db.repo import Repos
from risk.limits import RiskCheckFailed, RiskGate
from strategies.wheel import Proposal


@dataclass(frozen=True, slots=True)
class RouterConfig:
    dry_run: bool = False
    retry_max_attempts: int = 5
    retry_initial_backoff_seconds: float = 1.0
    retry_max_backoff_seconds: float = 60.0


@dataclass(frozen=True, slots=True)
class RouteResult:
    proposal: Proposal
    placed: Order | None  # None when dry-run, risk-failed, or news-blocked
    risk_failed: bool = False
    risk_failure_rule: str | None = None
    risk_failure_detail: str | None = None
    dry_run: bool = False
    news_decision: str | None = None  # "proceed" | "caution" | "block" | None
    news_rationale: str | None = None
    quantity_adjusted: int | None = None  # final qty if news_check forced a halve


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
    return RouterConfig(
        dry_run=bool(section.get("dry_run", False)),
        retry_max_attempts=int(section.get("retry_max_attempts", 5)),
        retry_initial_backoff_seconds=float(section.get("retry_initial_backoff_seconds", 1.0)),
        retry_max_backoff_seconds=float(section.get("retry_max_backoff_seconds", 60.0)),
    )


def _pending_state_for(order_type: OrderType, contract_is_put: bool) -> PositionState:
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
        # News checker is an awaitable that takes (symbol) and returns a
        # NewsCheckResult-shaped object with .decision / .rationale. Optional —
        # tests pass a stub or None to disable.
        self._news_checker = news_checker

    async def place(
        self,
        proposal: Proposal,
        *,
        sleep: Any = asyncio.sleep,
        today: date | None = None,
    ) -> RouteResult:
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

        if (
            self._news_checker is not None
            and proposal.order_type == OrderType.SELL_TO_OPEN
            and proposal.contract.option_type == OptionType.PUT
        ):
            check = await self._news_checker(proposal.symbol)
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
            if news_decision == "caution":
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
                proposal = Proposal(
                    symbol=proposal.symbol,
                    contract=proposal.contract,
                    order_type=proposal.order_type,
                    quantity=halved,
                    rationale=proposal.rationale + " (size halved by news_check)",
                    requires_screen=proposal.requires_screen,
                    requires_human=proposal.requires_human,
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

        order = self._build_order(proposal, today=today)
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

    # -- Internals ------------------------------------------------------------

    def _build_order(self, proposal: Proposal, *, today: date | None) -> Order:
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
            client_order_id=_client_order_id(proposal, today),
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
        new_state = _pending_state_for(proposal.order_type, contract_is_put)
        existing = await self._repos.positions.get_by_symbol(
            account_id, proposal.symbol, strategy_id=proposal.strategy_id
        )
        now = _utcnow()
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
