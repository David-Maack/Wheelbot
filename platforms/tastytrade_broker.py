"""Tastytrade adapter — production broker.

Wraps `tastyware/tastytrade` (SDK 12.x), which is fully async over httpx.
Same Broker ABC contract as PaperBroker / AlpacaBroker.

Auth: SDK 12.x uses OAuth2. The constructor expects:
  - `provider_secret` — your developer-app's client secret (from
    https://developer.tastytrade.com).
  - `refresh_token`   — the long-lived token captured by
    `scripts/bootstrap_tastytrade.py`.
  - `is_test`         — True for sandbox (cert.tastyworks.com), False for prod.

The SDK Session is an async context manager. We open it lazily on first use
and close it via `aclose()`. Tests construct adapters with a stub Session.

Symbol formats:
  - We pass our canonical OCC symbol (`F250620P00010000`) to the SDK; the
    SDK accepts both the dense 21-char and the space-padded form via the
    REST endpoint, so we normalize to dense on the way in and on the way back.
  - DXLink streamer symbols (`.F250620P10`) are not used here — quotes go
    through the HTTP `get_market_data` snapshot endpoint instead.
"""

from __future__ import annotations

import os
from datetime import UTC, date, datetime
from decimal import Decimal
from typing import TYPE_CHECKING, Any

from core.broker import Broker, BrokerUnavailable, OrderRejected
from core.checkpoint import log_checkpoint
from core.models import (
    Account,
    OptionContract,
    OptionType,
    Order,
    OrderStatus,
    OrderType as WheelOrderType,
    Position,
    PositionState,
    Quote,
)

if TYPE_CHECKING:  # pragma: no cover
    from tastytrade import Account as TtAccount, Session


# Tastytrade order statuses → ours.
_PENDING = {"Received", "Live", "Routed", "In Flight", "Contingent"}
_CANCELLED = {"Cancelled", "Cancel Requested", "Removed", "Expired"}
_REJECTED = {"Rejected"}


def _utcnow() -> datetime:
    return datetime.now(UTC).replace(tzinfo=None)


def _strip_tz(dt: datetime | None) -> datetime | None:
    if dt is None:
        return None
    if dt.tzinfo is not None:
        return dt.astimezone(UTC).replace(tzinfo=None)
    return dt


def _occ_dense(occ: str) -> str:
    """Return the dense 15-char-suffix OCC form (no padding spaces)."""
    return occ.replace(" ", "")


def _parse_occ(occ: str) -> tuple[str, date, OptionType, float] | None:
    """Dense OCC → (underlying, expiration, type, strike). Tolerates Tastytrade
    space-padded form by collapsing spaces first."""
    s = _occ_dense(occ)
    if len(s) < 15:
        return None
    for i in range(1, 7):
        prefix = s[:i]
        rest = s[i:]
        if len(rest) == 15 and rest[:6].isdigit() and rest[6] in "PC" and rest[7:].isdigit():
            yy, mm, dd = int(rest[:2]), int(rest[2:4]), int(rest[4:6])
            try:
                exp = date(2000 + yy, mm, dd)
            except ValueError:
                return None
            opt = OptionType.PUT if rest[6] == "P" else OptionType.CALL
            strike = int(rest[7:]) / 1000.0
            return prefix, exp, opt, strike
    return None


def _map_status(tt_status: str) -> OrderStatus:
    s = str(tt_status).strip()
    if s == "Filled":
        return OrderStatus.FILLED
    if s in _PENDING:
        return OrderStatus.PENDING
    if s == "Partially Removed":
        return OrderStatus.PARTIAL
    if s in _CANCELLED:
        return OrderStatus.CANCELLED
    if s in _REJECTED:
        return OrderStatus.REJECTED
    return OrderStatus.PENDING


def _wheel_action_to_tt(order_type: WheelOrderType) -> Any:
    from tastytrade.order import OrderAction

    return {
        WheelOrderType.SELL_TO_OPEN: OrderAction.SELL_TO_OPEN,
        WheelOrderType.BUY_TO_CLOSE: OrderAction.BUY_TO_CLOSE,
        WheelOrderType.BUY_TO_OPEN: OrderAction.BUY_TO_OPEN,
        WheelOrderType.SELL_TO_CLOSE: OrderAction.SELL_TO_CLOSE,
    }[order_type]


def _tt_action_to_wheel(action: Any) -> WheelOrderType:
    s = str(action.value if hasattr(action, "value") else action)
    return {
        "Sell to Open": WheelOrderType.SELL_TO_OPEN,
        "Buy to Close": WheelOrderType.BUY_TO_CLOSE,
        "Buy to Open": WheelOrderType.BUY_TO_OPEN,
        "Sell to Close": WheelOrderType.SELL_TO_CLOSE,
    }.get(s, WheelOrderType.SELL_TO_OPEN)


class TastytradeBroker(Broker):
    """Real broker. Defaults to sandbox; prod is opt-in via `is_test=False`."""

    def __init__(
        self,
        provider_secret: str | None = None,
        refresh_token: str | None = None,
        is_test: bool = True,
        account_number: str | None = None,
        account_id: str = "tastytrade",
    ) -> None:
        # Imported here so importing this module never requires the SDK.
        from tastytrade import Session  # noqa: PLC0415

        self._provider_secret = provider_secret or os.environ.get("TASTYTRADE_PROVIDER_SECRET")
        self._refresh_token = refresh_token or os.environ.get("TASTYTRADE_REMEMBER_TOKEN")
        self._is_test = is_test
        self._account_id = account_id
        self._account_number = account_number or os.environ.get("TASTYTRADE_ACCOUNT_NUMBER")
        if not self._provider_secret or not self._refresh_token:
            raise BrokerUnavailable(
                "TASTYTRADE_PROVIDER_SECRET / TASTYTRADE_REMEMBER_TOKEN not set"
            )
        self._session_factory: Any = lambda: Session(
            provider_secret=self._provider_secret,
            refresh_token=self._refresh_token,
            is_test=self._is_test,
        )
        self._session: Session | None = None
        self._account: TtAccount | None = None
        if not is_test:
            log_checkpoint(
                "tastytrade_broker_init",
                status="ok",
                mode="production",
                account=self._account_number,
            )

    # -- Broker interface -----------------------------------------------------

    @property
    def name(self) -> str:
        return "tastytrade_sandbox" if self._is_test else "tastytrade"

    async def get_account(self) -> Account:
        session, account = await self._ensure_session()
        try:
            balances = await account.get_balances(session)
        except Exception as exc:
            raise BrokerUnavailable(f"tastytrade get_balances failed: {exc}") from exc
        return Account(
            account_id=self._account_id,
            cash=float(balances.cash_balance or 0),
            buying_power=float(balances.derivative_buying_power or balances.equity_buying_power or 0),
            equity=float(balances.net_liquidating_value or 0),
            options_buying_power=float(balances.derivative_buying_power or 0) or None,
            maintenance_margin=float(balances.maintenance_requirement or 0) or None,
            pattern_day_trader=None,
        )

    async def get_positions(self) -> list[Position]:
        from tastytrade.order import InstrumentType

        session, account = await self._ensure_session()
        try:
            positions = await account.get_positions(session)
        except Exception as exc:
            raise BrokerUnavailable(f"tastytrade get_positions failed: {exc}") from exc

        out: list[Position] = []
        now = _utcnow()
        for p in positions:
            qty = int(float(p.quantity or 0))
            direction = str(p.quantity_direction or "")
            signed_qty = -qty if direction == "Short" else qty
            if p.instrument_type == InstrumentType.EQUITY_OPTION:
                parsed = _parse_occ(p.symbol)
                opt_type = parsed[2] if parsed else None
                state = (
                    PositionState.CSP_OPEN
                    if (signed_qty < 0 and opt_type == OptionType.PUT)
                    else PositionState.CC_OPEN
                    if (signed_qty < 0 and opt_type == OptionType.CALL)
                    else PositionState.MANUAL_INTERVENTION
                )
                out.append(
                    Position(
                        account_id=self._account_id,
                        symbol=parsed[0] if parsed else (p.underlying_symbol or p.symbol),
                        state=state,
                        shares=0,
                        cost_basis=None,
                        state_changed_at=now,
                        state_change_reason=f"tastytrade_short:{p.symbol}",
                    )
                )
            elif p.instrument_type == InstrumentType.EQUITY:
                out.append(
                    Position(
                        account_id=self._account_id,
                        symbol=p.symbol,
                        state=PositionState.SHARES_HELD if signed_qty else PositionState.IDLE,
                        shares=signed_qty,
                        cost_basis=float(p.average_open_price) if p.average_open_price else None,
                        state_changed_at=now,
                    )
                )
        return out

    async def get_quote(self, symbol: str) -> Quote:
        from tastytrade.market_data import get_market_data
        from tastytrade.order import InstrumentType

        session, _account = await self._ensure_session()
        is_option = _parse_occ(symbol) is not None
        instrument_type = (
            InstrumentType.EQUITY_OPTION if is_option else InstrumentType.EQUITY
        )
        try:
            md = await get_market_data(session, _occ_dense(symbol), instrument_type)
        except Exception as exc:
            raise BrokerUnavailable(f"tastytrade quote fetch failed: {exc}") from exc
        return Quote(
            symbol=symbol,
            bid=float(md.bid) if md.bid is not None else None,
            ask=float(md.ask) if md.ask is not None else None,
            last=float(md.last) if md.last is not None else None,
            bid_size=int(md.bid_size) if md.bid_size is not None else None,
            ask_size=int(md.ask_size) if md.ask_size is not None else None,
            timestamp=_strip_tz(md.updated_at),
        )

    async def get_option_chain(
        self,
        underlying: str,
        expiration: date | None = None,
        option_type: OptionType | None = None,
    ) -> list[OptionContract]:
        from tastytrade.instruments import NestedOptionChain

        session, _account = await self._ensure_session()
        try:
            chains = await NestedOptionChain.get(session, underlying)
        except Exception as exc:
            raise BrokerUnavailable(f"tastytrade chain fetch failed: {exc}") from exc

        out: list[OptionContract] = []
        for chain in chains:
            for exp in chain.expirations:
                if expiration is not None and exp.expiration_date != expiration:
                    continue
                for strike in exp.strikes:
                    for side, occ_field in (
                        (OptionType.PUT, "put"),
                        (OptionType.CALL, "call"),
                    ):
                        if option_type is not None and side != option_type:
                            continue
                        occ = getattr(strike, occ_field, None)
                        if not occ:
                            continue
                        out.append(
                            OptionContract(
                                underlying=underlying.upper(),
                                occ_symbol=_occ_dense(occ),
                                strike=float(strike.strike_price),
                                expiration=exp.expiration_date,
                                option_type=side,
                            )
                        )
        return out

    async def place_order(self, order: Order) -> Order:
        from tastytrade.order import (
            InstrumentType,
            Leg,
            NewOrder,
            OrderTimeInForce,
        )
        from tastytrade.order import OrderType as TtOrderType
        from tastytrade.utils import TastytradeError

        session, account = await self._ensure_session()

        leg = Leg(
            instrument_type=InstrumentType.EQUITY_OPTION
            if order.option_type is not None
            else InstrumentType.EQUITY,
            symbol=_occ_dense(order.contract_symbol or order.symbol),
            action=_wheel_action_to_tt(order.order_type),
            quantity=Decimal(order.quantity),
        )
        new_order = NewOrder(
            time_in_force=OrderTimeInForce.DAY,
            order_type=(
                TtOrderType.LIMIT if order.limit_price is not None else TtOrderType.MARKET
            ),
            price=Decimal(str(order.limit_price)) if order.limit_price is not None else None,
            legs=[leg],
            external_identifier=order.client_order_id,
        )

        try:
            placed = await account.place_order(session, new_order, dry_run=False)
        except TastytradeError as exc:
            raise OrderRejected(f"tastytrade rejected order: {exc}") from exc
        except Exception as exc:
            raise BrokerUnavailable(f"tastytrade place_order failed: {exc}") from exc

        po = placed.order
        return order.model_copy(
            update={
                "broker_order_id": str(po.id),
                "status": _map_status(str(po.status)),
                "limit_price": float(po.price) if po.price is not None else order.limit_price,
                "placed_at": _strip_tz(po.received_at) or order.placed_at or _utcnow(),
                "filled_at": _strip_tz(po.terminal_at)
                if str(po.status) == "Filled"
                else None,
                "raw_request": new_order.model_dump(mode="json"),
                "raw_response": placed.model_dump(mode="json"),
            }
        )

    async def cancel_order(self, broker_order_id: str) -> None:
        from tastytrade.utils import TastytradeError

        session, account = await self._ensure_session()
        try:
            order_id = int(broker_order_id)
        except ValueError:
            return
        try:
            await account.delete_order(session, order_id)
        except TastytradeError:
            return
        except Exception as exc:
            raise BrokerUnavailable(f"tastytrade cancel_order failed: {exc}") from exc

    async def get_orders_since(self, since: datetime) -> list[Order]:
        session, account = await self._ensure_session()
        start_at = since if since.tzinfo else since.replace(tzinfo=UTC)
        try:
            placed = await account.get_order_history(
                session, start_at=start_at, per_page=200
            )
        except Exception as exc:
            raise BrokerUnavailable(f"tastytrade get_orders failed: {exc}") from exc
        return [self._to_order(p) for p in placed]

    async def aclose(self) -> None:
        if self._session is not None:
            await self._session.__aexit__(None, None, None)
            self._session = None
            self._account = None

    # -- Internals ------------------------------------------------------------

    async def _ensure_session(self) -> tuple[Any, Any]:
        from tastytrade import Account as TtAccount

        if self._session is None:
            self._session = self._session_factory()
            await self._session.__aenter__()
        if self._account is None:
            if self._account_number:
                self._account = await TtAccount.get(self._session, self._account_number)
            else:
                accounts = await TtAccount.get(self._session)
                if not accounts:
                    raise BrokerUnavailable("no Tastytrade accounts on this user")
                self._account = accounts[0]
        return self._session, self._account

    def _to_order(self, raw: Any) -> Order:
        leg = raw.legs[0] if raw.legs else None
        is_option = (
            str(leg.instrument_type) == "Equity Option" if leg is not None else False
        )
        symbol = leg.symbol if leg is not None else raw.underlying_symbol
        parsed = _parse_occ(symbol or "") if is_option else None
        action = leg.action if leg is not None else None
        wheel_type = _tt_action_to_wheel(action) if action else WheelOrderType.SELL_TO_OPEN
        qty = int(float(leg.quantity if leg else raw.size or 0))
        return Order(
            account_id=self._account_id,
            symbol=parsed[0] if parsed else (raw.underlying_symbol or symbol or ""),
            broker_order_id=str(raw.id),
            client_order_id=raw.external_identifier,
            order_type=wheel_type,
            contract_symbol=symbol if is_option else None,
            strike=parsed[3] if parsed else None,
            expiration=parsed[1] if parsed else None,
            option_type=parsed[2] if parsed else None,
            quantity=qty,
            limit_price=float(raw.price) if raw.price is not None else None,
            fill_price=None,  # SDK doesn't expose a single fill price here; reconciler treats this best-effort
            status=_map_status(str(raw.status)),
            placed_at=_strip_tz(raw.received_at) or _utcnow(),
            filled_at=_strip_tz(raw.terminal_at) if str(raw.status) == "Filled" else None,
            raw_request=None,
            raw_response=raw.model_dump(mode="json") if hasattr(raw, "model_dump") else None,
        )
