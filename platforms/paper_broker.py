"""In-memory paper broker.

Deterministic, async-safe enough for unit tests. All Sprint 3+ unit tests for the
selectors, router, reconciler, and risk gates run against this — that's why it
exposes test helpers (`fill_order`, `assign`, `expire`, `seed_chain`, `seed_quote`)
that have no analogue on a real broker.

Conventions:
- Stock positions are keyed by underlying symbol; short option legs are keyed by
  OCC symbol with negative `shares` representing contract count × -100 *would* be
  noisy, so we store option positions in a separate map and surface them as
  Position rows with `shares=0` and a synthetic state. The reconciler in Sprint 4
  is what actually diffs broker-shape against our local schema; this layer's job
  is just to round-trip data.
- Times are UTC.
"""

from __future__ import annotations

import asyncio
import uuid
from datetime import UTC, date, datetime
from typing import Any

from core.broker import Broker, BrokerUnavailable, OrderRejected
from core.models import (
    Account,
    OptionContract,
    OptionType,
    Order,
    OrderStatus,
    Position,
    PositionState,
    Quote,
)


def _utcnow() -> datetime:
    return datetime.now(UTC).replace(tzinfo=None)


class PaperBroker(Broker):
    def __init__(
        self,
        account_id: str = "PAPER-1",
        cash: float = 10_000.0,
    ) -> None:
        self._account_id = account_id
        self._cash = cash
        self._initial_cash = cash
        # Stock positions: symbol -> (shares, cost_basis)
        self._stock: dict[str, tuple[int, float]] = {}
        # Open option short legs: occ_symbol -> Order (the SELL_TO_OPEN that opened it)
        self._open_options: dict[str, Order] = {}
        # All orders ever, by broker_order_id
        self._orders: dict[str, Order] = {}
        # Seeded market data
        self._quotes: dict[str, Quote] = {}
        self._chains: dict[str, list[OptionContract]] = {}
        self._lock = asyncio.Lock()
        self._unavailable = False

    # -- Broker interface -----------------------------------------------------

    @property
    def name(self) -> str:
        return "paper"

    async def get_account(self) -> Account:
        self._raise_if_unavailable()
        equity = self._cash + sum(
            shares * self._mark(symbol) for symbol, (shares, _) in self._stock.items()
        )
        return Account(
            account_id=self._account_id,
            cash=self._cash,
            buying_power=self._cash,
            equity=equity,
            options_buying_power=self._cash,
        )

    async def get_positions(self) -> list[Position]:
        self._raise_if_unavailable()
        now = _utcnow()
        out: list[Position] = []
        for symbol, (shares, cost_basis) in self._stock.items():
            out.append(
                Position(
                    account_id=self._account_id,
                    symbol=symbol,
                    state=PositionState.SHARES_HELD if shares else PositionState.IDLE,
                    shares=shares,
                    cost_basis=cost_basis,
                    state_changed_at=now,
                )
            )
        # Surface open short option legs as positions on the underlying. The reconciler
        # is the component that interprets these; we just report broker truth.
        for occ, order in self._open_options.items():
            if order.symbol in self._stock:
                continue  # already represented by a stock row
            out.append(
                Position(
                    account_id=self._account_id,
                    symbol=order.symbol,
                    state=PositionState.CSP_OPEN
                    if order.option_type == OptionType.PUT
                    else PositionState.CC_OPEN,
                    shares=0,
                    cost_basis=None,
                    state_changed_at=now,
                    state_change_reason=f"open_short:{occ}",
                )
            )
        return out

    async def get_quote(self, symbol: str) -> Quote:
        self._raise_if_unavailable()
        if symbol not in self._quotes:
            raise BrokerUnavailable(f"no quote seeded for {symbol}")
        return self._quotes[symbol]

    async def get_option_chain(
        self,
        underlying: str,
        expiration: date | None = None,
        option_type: OptionType | None = None,
    ) -> list[OptionContract]:
        self._raise_if_unavailable()
        chain = self._chains.get(underlying.upper(), [])
        if expiration is not None:
            chain = [c for c in chain if c.expiration == expiration]
        if option_type is not None:
            chain = [c for c in chain if c.option_type == option_type]
        return list(chain)

    async def place_order(self, order: Order) -> Order:
        self._raise_if_unavailable()
        async with self._lock:
            if order.client_order_id and any(
                o.client_order_id == order.client_order_id for o in self._orders.values()
            ):
                # Idempotency: caller retried with same client id, return the prior result.
                for existing in self._orders.values():
                    if existing.client_order_id == order.client_order_id:
                        return existing.model_copy(deep=True)

            if order.quantity <= 0:
                raise OrderRejected(f"quantity must be positive, got {order.quantity}")

            broker_id = f"paper-{uuid.uuid4().hex[:12]}"
            stored = order.model_copy(
                update={
                    "broker_order_id": broker_id,
                    "status": OrderStatus.PENDING,
                    "placed_at": order.placed_at or _utcnow(),
                    "raw_request": order.model_dump(mode="json"),
                    "raw_response": {"broker_order_id": broker_id, "status": "PENDING"},
                }
            )
            self._orders[broker_id] = stored
            return stored.model_copy(deep=True)

    async def cancel_order(self, broker_order_id: str) -> None:
        self._raise_if_unavailable()
        async with self._lock:
            order = self._orders.get(broker_order_id)
            if order is None:
                return
            if order.status not in (OrderStatus.PENDING, OrderStatus.PARTIAL):
                return
            self._orders[broker_order_id] = order.model_copy(
                update={"status": OrderStatus.CANCELLED}
            )

    async def get_orders_since(self, since: datetime) -> list[Order]:
        self._raise_if_unavailable()
        out: list[Order] = []
        for o in self._orders.values():
            stamp = o.filled_at or o.placed_at
            if stamp is None or stamp >= since:
                out.append(o.model_copy(deep=True))
        return out

    # -- Test helpers ---------------------------------------------------------

    def seed_quote(self, quote: Quote) -> None:
        self._quotes[quote.symbol] = quote

    def seed_chain(self, underlying: str, contracts: list[OptionContract]) -> None:
        self._chains[underlying.upper()] = list(contracts)

    def set_unavailable(self, value: bool) -> None:
        """Force the next call to raise BrokerUnavailable. Used for retry tests."""
        self._unavailable = value

    async def fill_order(self, broker_order_id: str, fill_price: float | None = None) -> Order:
        """Mark an open order as filled and apply its effect to cash + positions."""
        async with self._lock:
            order = self._orders[broker_order_id]
            if order.status not in (OrderStatus.PENDING, OrderStatus.PARTIAL):
                raise ValueError(f"order {broker_order_id} not fillable, status={order.status}")
            price = fill_price if fill_price is not None else (order.limit_price or 0.0)
            self._apply_fill(order, price)
            filled = order.model_copy(
                update={
                    "status": OrderStatus.FILLED,
                    "fill_price": price,
                    "filled_at": _utcnow(),
                }
            )
            self._orders[broker_order_id] = filled
            return filled.model_copy(deep=True)

    async def assign(self, occ_symbol: str) -> None:
        """Simulate option assignment.

        For a short put: deliver shares at strike, debit cash, drop the short leg.
        For a short call: remove shares, credit cash at strike, drop the short leg.
        """
        async with self._lock:
            opener = self._open_options.get(occ_symbol)
            if opener is None:
                raise ValueError(f"no open short leg {occ_symbol}")
            assert opener.strike is not None
            shares = 100 * opener.quantity
            if opener.option_type == OptionType.PUT:
                self._cash -= opener.strike * shares
                self._add_shares(opener.symbol, shares, opener.strike)
            else:
                self._cash += opener.strike * shares
                self._remove_shares(opener.symbol, shares)
            del self._open_options[occ_symbol]

    async def expire(self, occ_symbol: str) -> None:
        """Worthless expiration of an open short leg. Premium is already booked."""
        async with self._lock:
            self._open_options.pop(occ_symbol, None)

    # -- Internals ------------------------------------------------------------

    def _raise_if_unavailable(self) -> None:
        if self._unavailable:
            raise BrokerUnavailable("paper broker forced unavailable")

    def _mark(self, symbol: str) -> float:
        q = self._quotes.get(symbol)
        if q is None:
            return 0.0
        if q.last is not None:
            return q.last
        if q.bid is not None and q.ask is not None:
            return (q.bid + q.ask) / 2
        return q.bid or q.ask or 0.0

    def _apply_fill(self, order: Order, price: float) -> None:
        from core.models import OrderType

        contracts = order.quantity
        shares = 100 * contracts
        if order.order_type == OrderType.SELL_TO_OPEN and order.contract_symbol:
            self._cash += price * shares
            self._open_options[order.contract_symbol] = order
        elif order.order_type == OrderType.BUY_TO_CLOSE and order.contract_symbol:
            self._cash -= price * shares
            self._open_options.pop(order.contract_symbol, None)
        elif order.order_type == OrderType.BUY_TO_OPEN:
            self._cash -= price * contracts
            self._add_shares(order.symbol, contracts, price)
        elif order.order_type == OrderType.SELL_TO_CLOSE:
            self._cash += price * contracts
            self._remove_shares(order.symbol, contracts)

    def _add_shares(self, symbol: str, shares: int, price: float) -> None:
        existing_shares, existing_basis = self._stock.get(symbol, (0, 0.0))
        new_shares = existing_shares + shares
        if new_shares == 0:
            self._stock.pop(symbol, None)
            return
        new_basis = (existing_shares * existing_basis + shares * price) / new_shares
        self._stock[symbol] = (new_shares, new_basis)

    def _remove_shares(self, symbol: str, shares: int) -> None:
        existing_shares, existing_basis = self._stock.get(symbol, (0, 0.0))
        new_shares = existing_shares - shares
        if new_shares <= 0:
            self._stock.pop(symbol, None)
        else:
            self._stock[symbol] = (new_shares, existing_basis)
