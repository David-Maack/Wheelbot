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


def _is_leg_child(raw: Any) -> bool:
    """True when the order is a child leg of a multi-leg parent.

    Children carry a reference to the parent through one of several fields
    depending on alpaca-py version: legs/order_class, hwm, or an explicit
    parent identifier. We check several to stay version-resilient. The
    canonical signal in current alpaca-py is the order_class enum.
    """
    # If alpaca-py reports the order_class as something like
    # OrderClass.MLEG_LEG or simply tags the leg, we'd see it here.
    # Empirically the safest discriminator is: a non-MLEG order whose
    # symbol is an OCC option AND that doesn't carry a client_order_id
    # we set. Client_order_id is OUR string ("wb-...") on parents; legs
    # get broker-generated IDs.
    order_class = getattr(raw, "order_class", None)
    order_class_str = str(getattr(order_class, "value", order_class) or "").lower()
    if "mleg" in order_class_str and "leg" in order_class_str.replace("mleg", ""):
        # e.g. "mleg_leg" or "leg" suffix
        return True
    # Heuristic fallback for SDK versions that don't tag children explicitly:
    # leg orders have a symbol but no client_order_id we recognize. We DO
    # set client_order_id on every order we place via place_order or
    # place_multi_leg_order, so any order whose client_order_id is None
    # is either a leg or a manual broker-side order — treat both as "not
    # ours, skip".
    cid = getattr(raw, "client_order_id", None)
    if cid is None or not str(cid).startswith("wb-"):
        return True
    return False


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
                "limit_price": float(placed.limit_price) if placed.limit_price is not None else order.limit_price,
                "fill_price": float(placed.filled_avg_price) if placed.filled_avg_price is not None else None,
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
            # Alpaca's filled_avg_price for an MLEG package is the net debit
            # paid (positive). Our convention is positive=credit, so negate on
            # the way out to stay consistent with _to_order_multi_leg (the
            # read-back path) and the cycle PnL math. `is not None` guard so a
            # genuine 0.0 fill isn't silently dropped to None.
            fill_price=-float(placed.filled_avg_price) if placed.filled_avg_price is not None else None,
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
        # `nested=True` rolls multi-leg children under their parent's `.legs`
        # instead of returning them as flat top-level orders. Without this,
        # the reconciler sees each leg as an "unknown order" and flags the
        # affected position MANUAL_INTERVENTION every tick.
        req = GetOrdersRequest(
            status=QueryOrderStatus.ALL, after=after, limit=500, nested=True
        )
        try:
            raw = await asyncio.to_thread(self._trading.get_orders, req)
        except Exception as exc:
            raise BrokerUnavailable(f"alpaca get_orders failed: {exc}") from exc
        # Defensive belt-and-suspenders: even with nested=True, skip any order
        # that carries a non-null parent ID (i.e. is a child of a multi-leg).
        # Different alpaca-py versions surface this field under different
        # names; we check the common ones.
        return [
            self._to_order(o)
            for o in raw
            if not _is_leg_child(o)
        ]

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

    def _to_order_multi_leg(self, raw: Any, legs: list[Any]) -> Order:
        """Map an Alpaca MLEG parent order back into our single Order row.

        Convention matches `place_multi_leg_order`: order_type is inferred
        from leg position_intents (all CLOSE → MULTI_LEG_CLOSE; any OPEN →
        MULTI_LEG_OPEN), underlying is parsed from the first leg's OCC
        symbol. fill_price is the parent's filled_avg_price which Alpaca
        reports as the *signed* net price per package — positive = credit
        we received, negative = debit we paid. (Alpaca's display sign for
        MLEG is the inverse of our submit-time convention, which we already
        negated in place_multi_leg_order.)
        """
        first_leg_symbol = getattr(legs[0], "symbol", None) or ""
        parsed_first = _parse_occ(first_leg_symbol)
        underlying = parsed_first[0] if parsed_first else first_leg_symbol

        # Infer OPEN vs CLOSE from leg position_intents. Alpaca's
        # PositionIntent.value is a kebab-style string like "sell_to_close".
        all_close = True
        for leg in legs:
            intent = str(getattr(leg, "position_intent", "") or "").lower()
            if "close" not in intent:
                all_close = False
                break
        order_type = (
            WheelOrderType.MULTI_LEG_CLOSE if all_close else WheelOrderType.MULTI_LEG_OPEN
        )

        # Alpaca's filled_avg_price for an MLEG package is the net debit
        # paid (positive). Our convention is positive=credit; flip the sign
        # on the way back in so the cycle PnL math (which adds fill_price
        # × qty × 100) gives the right number.
        raw_fill = (
            float(raw.filled_avg_price) if raw.filled_avg_price is not None else None
        )
        fill_price = -raw_fill if raw_fill is not None else None
        raw_limit = (
            float(raw.limit_price) if raw.limit_price is not None else None
        )
        limit_price = -raw_limit if raw_limit is not None else None

        return Order(
            account_id=self._account_id,
            symbol=underlying,
            broker_order_id=str(raw.id),
            client_order_id=raw.client_order_id,
            order_type=order_type,
            contract_symbol=None,
            strike=None,
            expiration=None,
            option_type=None,
            quantity=int(float(raw.qty or 0)),
            limit_price=limit_price,
            fill_price=fill_price,
            status=_map_status(raw.status),
            placed_at=_strip_tz(raw.submitted_at) or _utcnow(),
            filled_at=_strip_tz(raw.filled_at),
            raw_request=None,
            raw_response=raw.model_dump(mode="json") if hasattr(raw, "model_dump") else None,
        )

    def _to_order(self, raw: Any) -> Order:
        """Best-effort mapping; the local DB carries raw_response for the audit trail.

        Multi-leg packages (OrderClass.MLEG) come back from Alpaca with
        `symbol=None` on the parent order — the symbols live on `raw.legs`.
        We detect that case and synthesize a single Order row tagged with the
        underlying from the first leg, matching how place_multi_leg_order
        writes them on the way out.
        """
        from alpaca.trading.enums import AssetClass, OrderClass

        order_class = getattr(raw, "order_class", None)
        legs = getattr(raw, "legs", None) or []
        if order_class == OrderClass.MLEG and legs:
            return self._to_order_multi_leg(raw, legs)

        is_option = getattr(raw, "asset_class", None) == AssetClass.US_OPTION
        parsed = _parse_occ(raw.symbol) if is_option and raw.symbol else None
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
            limit_price=float(raw.limit_price) if raw.limit_price is not None else None,
            fill_price=float(raw.filled_avg_price) if raw.filled_avg_price is not None else None,
            status=_map_status(raw.status),
            placed_at=_strip_tz(raw.submitted_at) or _utcnow(),
            filled_at=_strip_tz(raw.filled_at),
            raw_request=None,
            raw_response=raw.model_dump(mode="json") if hasattr(raw, "model_dump") else None,
        )
