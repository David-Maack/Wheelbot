"""Alpaca paper-trading adapter.

Wraps the (synchronous) alpaca-py SDK behind our async Broker ABC by dispatching
each call through `asyncio.to_thread`. That keeps the strategy layer fully async
without us having to maintain a custom HTTP client.

Scope: paper-only in Sprint 2. Live Tastytrade is Sprint 6. Alpaca options approval
varies by account; this adapter assumes the caller's paper account is options-
enabled (Level 2 minimum to sell CSPs / CCs — same bar as Tastytrade).

Env vars (loaded by `core.config.load_secrets`):
    ALPACA_API_KEY
    ALPACA_API_SECRET
    ALPACA_PAPER       — informational; this adapter always uses paper=True
"""

from __future__ import annotations

import asyncio
import os
from datetime import UTC, date, datetime
from typing import TYPE_CHECKING, Any

from core.broker import Broker, BrokerUnavailable, OrderRejected
from core.models import (
    Account,
    OptionContract,
    OptionType,
    Order,
    OrderLeg,
    OrderStatus,
    OrderType as WheelOrderType,
    Position,
    PositionState,
    Quote,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from alpaca.trading.client import TradingClient
    from alpaca.data.historical.option import OptionHistoricalDataClient
    from alpaca.data.historical.stock import StockHistoricalDataClient


# Alpaca order statuses → our OrderStatus.
_PENDING = {"new", "pending_new", "accepted", "accepted_for_bidding", "pending_review", "held"}
_CANCELLED = {"canceled", "pending_cancel", "expired", "replaced", "pending_replace"}
_REJECTED = {"rejected", "suspended", "stopped", "done_for_day", "calculated"}


def _utcnow() -> datetime:
    return datetime.now(UTC).replace(tzinfo=None)


def _strip_tz(dt: datetime | None) -> datetime | None:
    if dt is None:
        return None
    if dt.tzinfo is not None:
        return dt.astimezone(UTC).replace(tzinfo=None)
    return dt


def _map_status(alpaca_status: Any) -> OrderStatus:
    """Map alpaca-py OrderStatus to our OrderStatus.

    alpaca-py's OrderStatus is `class OrderStatus(str, Enum)`. The instance
    IS a string with content like "filled", but `str(instance)` returns
    `"OrderStatus.FILLED"` (Enum's __str__ override). We need the value, not
    the qualname — `.value` is the canonical accessor; `str(.value)` is
    defensive against future changes.
    """
    if hasattr(alpaca_status, "value"):
        s = str(alpaca_status.value).lower()
    elif isinstance(alpaca_status, str):
        s = alpaca_status.lower()
    else:
        s = str(alpaca_status).lower()
    if s == "filled":
        return OrderStatus.FILLED
    if s == "partially_filled":
        return OrderStatus.PARTIAL
    if s in _PENDING:
        return OrderStatus.PENDING
    if s in _CANCELLED:
        return OrderStatus.CANCELLED
    if s in _REJECTED:
        return OrderStatus.REJECTED
    return OrderStatus.PENDING


def _parse_occ(occ: str) -> tuple[str, date, OptionType, float] | None:
    """OCC21 symbol → (underlying, expiration, type, strike). None if not an option."""
    if len(occ) < 15:
        return None
    # Underlying is the prefix before the 6-digit YYMMDD; pad varies, so search.
    for i in range(1, 7):
        prefix = occ[:i].rstrip()
        rest = occ[i:]
        if len(rest) == 15 and rest[:6].isdigit() and rest[6] in "PC" and rest[7:].isdigit():
            yy, mm, dd = int(rest[:2]), int(rest[2:4]), int(rest[4:6])
            year = 2000 + yy
            try:
                exp = date(year, mm, dd)
            except ValueError:
                return None
            opt = OptionType.PUT if rest[6] == "P" else OptionType.CALL
            strike = int(rest[7:]) / 1000.0
            return prefix, exp, opt, strike
    return None


class AlpacaBroker(Broker):
    """alpaca-py adapter, paper endpoints only."""

    def __init__(
        self,
        api_key: str | None = None,
        api_secret: str | None = None,
        account_id: str = "alpaca-paper",
    ) -> None:
        # Imported here so importing this module never requires alpaca-py to be
        # installed — only constructing it does.
        from alpaca.data.historical.option import OptionHistoricalDataClient
        from alpaca.data.historical.stock import StockHistoricalDataClient
        from alpaca.trading.client import TradingClient

        key = api_key or os.environ.get("ALPACA_API_KEY")
        secret = api_secret or os.environ.get("ALPACA_API_SECRET")
        if not key or not secret:
            raise BrokerUnavailable(
                "ALPACA_API_KEY / ALPACA_API_SECRET not set; can't construct AlpacaBroker"
            )
        self._account_id = account_id
        self._trading: TradingClient = TradingClient(key, secret, paper=True)
        self._option_data: OptionHistoricalDataClient = OptionHistoricalDataClient(key, secret)
        self._stock_data: StockHistoricalDataClient = StockHistoricalDataClient(key, secret)

    # -- Broker interface -----------------------------------------------------

    @property
    def name(self) -> str:
        return "alpaca_paper"

    async def get_account(self) -> Account:
        a = await asyncio.to_thread(self._trading.get_account)
        return Account(
            account_id=self._account_id,
            cash=float(a.cash or 0),
            buying_power=float(a.buying_power or 0),
            equity=float(a.equity or 0),
            options_buying_power=float(getattr(a, "options_buying_power", None) or a.cash or 0),
            maintenance_margin=float(getattr(a, "maintenance_margin", 0) or 0) or None,
            pattern_day_trader=bool(a.pattern_day_trader)
            if a.pattern_day_trader is not None
            else None,
        )

    async def get_positions(self) -> list[Position]:
        from alpaca.trading.enums import AssetClass

        raw = await asyncio.to_thread(self._trading.get_all_positions)
        now = _utcnow()
        out: list[Position] = []
        for p in raw:
            qty = int(float(p.qty or 0))
            asset_class = getattr(p, "asset_class", None)
            if asset_class == AssetClass.US_OPTION:
                parsed = _parse_occ(p.symbol)
                underlying = parsed[0] if parsed else p.symbol
                opt_type = parsed[2] if parsed else None
                state = (
                    PositionState.CSP_OPEN
                    if (qty < 0 and opt_type == OptionType.PUT)
                    else PositionState.CC_OPEN
                    if (qty < 0 and opt_type == OptionType.CALL)
                    else PositionState.MANUAL_INTERVENTION
                )
                out.append(
                    Position(
                        account_id=self._account_id,
                        symbol=underlying,
                        state=state,
                        shares=0,
                        cost_basis=None,
                        state_changed_at=now,
                        state_change_reason=f"alpaca_open_short:{p.symbol}",
                    )
                )
            else:
                out.append(
                    Position(
                        account_id=self._account_id,
                        symbol=p.symbol,
                        state=PositionState.SHARES_HELD if qty else PositionState.IDLE,
                        shares=qty,
                        cost_basis=float(p.avg_entry_price) if p.avg_entry_price else None,
                        state_changed_at=now,
                    )
                )
        return out

    async def get_quote(self, symbol: str) -> Quote:
        from alpaca.data.requests import OptionLatestQuoteRequest, StockLatestQuoteRequest

        is_option = _parse_occ(symbol) is not None
        try:
            if is_option:
                req = OptionLatestQuoteRequest(symbol_or_symbols=symbol)
                resp = await asyncio.to_thread(self._option_data.get_option_latest_quote, req)
            else:
                req = StockLatestQuoteRequest(symbol_or_symbols=symbol)
                resp = await asyncio.to_thread(self._stock_data.get_stock_latest_quote, req)
        except Exception as exc:  # network/auth/etc
            raise BrokerUnavailable(f"alpaca quote fetch failed: {exc}") from exc

        q = resp.get(symbol) if isinstance(resp, dict) else None
        if q is None:
            raise BrokerUnavailable(f"no quote returned for {symbol}")
        return Quote(
            symbol=symbol,
            bid=float(q.bid_price) if q.bid_price is not None else None,
            ask=float(q.ask_price) if q.ask_price is not None else None,
            bid_size=int(q.bid_size) if q.bid_size is not None else None,
            ask_size=int(q.ask_size) if q.ask_size is not None else None,
            timestamp=_strip_tz(q.timestamp),
        )

    async def get_option_chain(
        self,
        underlying: str,
        expiration: date | None = None,
        option_type: OptionType | None = None,
    ) -> list[OptionContract]:
        from alpaca.data.requests import OptionChainRequest
        from alpaca.trading.enums import ContractType

        req_kwargs: dict[str, Any] = {"underlying_symbol": underlying}
        if expiration is not None:
            req_kwargs["expiration_date"] = expiration
        if option_type is not None:
            req_kwargs["type"] = (
                ContractType.PUT if option_type == OptionType.PUT else ContractType.CALL
            )
        req = OptionChainRequest(**req_kwargs)

        try:
            chain = await asyncio.to_thread(self._option_data.get_option_chain, req)
        except Exception as exc:
            raise BrokerUnavailable(f"alpaca chain fetch failed: {exc}") from exc

        out: list[OptionContract] = []
        for occ, snap in chain.items():
            parsed = _parse_occ(occ)
            if parsed is None:
                continue
            _, exp, opt, strike = parsed
            quote = getattr(snap, "latest_quote", None)
            trade = getattr(snap, "latest_trade", None)
            greeks = getattr(snap, "greeks", None)
            out.append(
                OptionContract(
                    underlying=underlying.upper(),
                    occ_symbol=occ,
                    strike=strike,
                    expiration=exp,
                    option_type=opt,
                    bid=float(quote.bid_price) if quote and quote.bid_price is not None else None,
                    ask=float(quote.ask_price) if quote and quote.ask_price is not None else None,
                    last=float(trade.price) if trade and trade.price is not None else None,
                    delta=float(greeks.delta) if greeks and greeks.delta is not None else None,
                    theta=float(greeks.theta) if greeks and greeks.theta is not None else None,
                    vega=float(greeks.vega) if greeks and greeks.vega is not None else None,
                    iv=float(snap.implied_volatility)
                    if snap.implied_volatility is not None
                    else None,
                )
            )
        return out

    async def place_order(self, order: Order) -> Order:
        from alpaca.common.exceptions import APIError
        from alpaca.trading.enums import OrderSide, TimeInForce
        from alpaca.trading.requests import LimitOrderRequest, MarketOrderRequest

        side = self._side(order.order_type)
        symbol = order.contract_symbol or order.symbol
        common = {
            "symbol": symbol,
            "qty": order.quantity,
            "side": side,
            "time_in_force": TimeInForce.DAY,
            "client_order_id": order.client_order_id,
        }
        if order.limit_price is not None:
            req = LimitOrderRequest(limit_price=order.limit_price, **common)
        else:
            req = MarketOrderRequest(**common)

        try:
            placed = await asyncio.to_thread(self._trading.submit_order, req)
        except APIError as exc:
            raise OrderRejected(f"alpaca rejected order: {exc}") from exc
        except Exception as exc:
            raise BrokerUnavailable(f"alpaca submit_order failed: {exc}") from exc

        return order.model_copy(
            update={
                "broker_order_id": str(placed.id),
                "status": _map_status(placed.status),
                "limit_price": float(placed.limit_price) if placed.limit_price else order.limit_price,
                "fill_price": float(placed.filled_avg_price) if placed.filled_avg_price else None,
                "placed_at": _strip_tz(placed.submitted_at) or order.placed_at or _utcnow(),
                "filled_at": _strip_tz(placed.filled_at),
                "raw_request": req.model_dump(mode="json"),
                "raw_response": placed.model_dump(mode="json"),
            }
        )

    async def place_multi_leg_order(
        self,
        *,
        underlying: str,
        legs: list[OrderLeg],
        quantity: int,
        limit_price: float | None,
        client_order_id: str | None = None,
        strategy_id: str | None = None,
        account_id: str = "primary",
    ) -> Order:
        """Submit an atomic multi-leg options package via Alpaca MLEG.

        Net package price semantics: Alpaca's `limit_price` for an MLEG order
        is the NET debit paid (positive = debit, you pay; negative = credit,
        you receive). Our convention is the opposite for clarity (positive =
        credit), so we negate when sending to Alpaca.
        """
        from alpaca.common.exceptions import APIError
        from alpaca.trading.enums import (
            OrderClass,
            OrderSide,
            PositionIntent,
            TimeInForce,
        )
        from alpaca.trading.requests import (
            LimitOrderRequest,
            MarketOrderRequest,
            OptionLegRequest,
        )

        if not legs:
            raise OrderRejected("multi-leg order requires at least one leg")
        if quantity <= 0:
            raise OrderRejected(f"quantity must be positive, got {quantity}")

        alpaca_legs = [
            OptionLegRequest(
                symbol=leg.contract_symbol,
                ratio_qty=leg.ratio_qty,
                side=self._side(WheelOrderType(leg.action)),
                position_intent=self._position_intent(WheelOrderType(leg.action)),
            )
            for leg in legs
        ]

        common: dict[str, Any] = {
            "qty": quantity,
            "order_class": OrderClass.MLEG,
            "legs": alpaca_legs,
            "time_in_force": TimeInForce.DAY,
            "client_order_id": client_order_id,
        }
        if limit_price is not None:
            req = LimitOrderRequest(limit_price=-limit_price, **common)
        else:
            req = MarketOrderRequest(**common)

        try:
            placed = await asyncio.to_thread(self._trading.submit_order, req)
        except APIError as exc:
            raise OrderRejected(f"alpaca rejected multi-leg order: {exc}") from exc
        except Exception as exc:
            raise BrokerUnavailable(f"alpaca submit_order (mleg) failed: {exc}") from exc

        all_close = all(
            leg.action in (WheelOrderType.SELL_TO_CLOSE, WheelOrderType.BUY_TO_CLOSE)
            for leg in legs
        )
        order_type = (
            WheelOrderType.MULTI_LEG_CLOSE if all_close else WheelOrderType.MULTI_LEG_OPEN
        )

        return Order(
            account_id=account_id,
            symbol=underlying,
            strategy_id=strategy_id,
            broker_order_id=str(placed.id),
            client_order_id=client_order_id,
            order_type=order_type,
            contract_symbol=None,
            strike=None,
            expiration=None,
            option_type=None,
            quantity=quantity,
            limit_price=limit_price,
            fill_price=float(placed.filled_avg_price) if placed.filled_avg_price else None,
            status=_map_status(placed.status),
            placed_at=_strip_tz(placed.submitted_at) or _utcnow(),
            filled_at=_strip_tz(placed.filled_at),
            raw_request={
                "underlying": underlying,
                "legs": [leg.model_dump(mode="json") for leg in legs],
                "quantity": quantity,
                "limit_price": limit_price,
                "alpaca_request": req.model_dump(mode="json"),
            },
            raw_response=placed.model_dump(mode="json"),
        )

    async def cancel_order(self, broker_order_id: str) -> None:
        from alpaca.common.exceptions import APIError

        try:
            await asyncio.to_thread(self._trading.cancel_order_by_id, broker_order_id)
        except APIError:
            # Already filled / cancelled / unknown — broker contract says no-op.
            return
        except Exception as exc:
            raise BrokerUnavailable(f"alpaca cancel_order failed: {exc}") from exc

    async def get_orders_since(self, since: datetime) -> list[Order]:
        from alpaca.trading.enums import QueryOrderStatus
        from alpaca.trading.requests import GetOrdersRequest

        # Alpaca's `after` filter expects an aware UTC datetime; if naive, treat as UTC.
        after = since.replace(tzinfo=UTC) if since.tzinfo is None else since
        req = GetOrdersRequest(status=QueryOrderStatus.ALL, after=after, limit=500)
        try:
            raw = await asyncio.to_thread(self._trading.get_orders, req)
        except Exception as exc:
            raise BrokerUnavailable(f"alpaca get_orders failed: {exc}") from exc
        return [self._to_order(o) for o in raw]

    # -- Internals ------------------------------------------------------------

    @staticmethod
    def _side(order_type: WheelOrderType) -> Any:
        from alpaca.trading.enums import OrderSide

        if order_type in (WheelOrderType.SELL_TO_OPEN, WheelOrderType.SELL_TO_CLOSE):
            return OrderSide.SELL
        return OrderSide.BUY

    @staticmethod
    def _position_intent(order_type: WheelOrderType) -> Any:
        from alpaca.trading.enums import PositionIntent

        if order_type == WheelOrderType.SELL_TO_OPEN:
            return PositionIntent.SELL_TO_OPEN
        if order_type == WheelOrderType.SELL_TO_CLOSE:
            return PositionIntent.SELL_TO_CLOSE
        if order_type == WheelOrderType.BUY_TO_OPEN:
            return PositionIntent.BUY_TO_OPEN
        if order_type == WheelOrderType.BUY_TO_CLOSE:
            return PositionIntent.BUY_TO_CLOSE
        raise ValueError(f"position_intent: unsupported order_type {order_type}")

    def _to_order(self, raw: Any) -> Order:
        """Best-effort mapping; the local DB carries raw_response for the audit trail."""
        from alpaca.trading.enums import AssetClass

        is_option = getattr(raw, "asset_class", None) == AssetClass.US_OPTION
        parsed = _parse_occ(raw.symbol) if is_option else None
        side_str = str(raw.side).split(".")[-1].lower() if raw.side else ""
        # We can't always recover SELL_TO_OPEN vs SELL_TO_CLOSE from Alpaca's payload
        # alone — position_intent helps when set; otherwise the reconciler infers from
        # state. Best-effort default keeps the audit row useful.
        intent = str(getattr(raw, "position_intent", "") or "").lower()
        if side_str == "sell":
            order_type = (
                WheelOrderType.SELL_TO_OPEN
                if "open" in intent
                else WheelOrderType.SELL_TO_CLOSE
                if "close" in intent
                else WheelOrderType.SELL_TO_OPEN
            )
        else:
            order_type = (
                WheelOrderType.BUY_TO_CLOSE
                if "close" in intent
                else WheelOrderType.BUY_TO_OPEN
                if "open" in intent
                else WheelOrderType.BUY_TO_CLOSE
            )

        return Order(
            account_id=self._account_id,
            symbol=parsed[0] if parsed else raw.symbol,
            broker_order_id=str(raw.id),
            client_order_id=raw.client_order_id,
            order_type=order_type,
            contract_symbol=raw.symbol if is_option else None,
            strike=parsed[3] if parsed else None,
            expiration=parsed[1] if parsed else None,
            option_type=parsed[2] if parsed else None,
            quantity=int(float(raw.qty or 0)),
            limit_price=float(raw.limit_price) if raw.limit_price else None,
            fill_price=float(raw.filled_avg_price) if raw.filled_avg_price else None,
            status=_map_status(raw.status),
            placed_at=_strip_tz(raw.submitted_at) or _utcnow(),
            filled_at=_strip_tz(raw.filled_at),
            raw_request=None,
            raw_response=raw.model_dump(mode="json") if hasattr(raw, "model_dump") else None,
        )
