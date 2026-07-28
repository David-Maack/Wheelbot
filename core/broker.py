"""Broker abstract base class.

Per spec §3: strategy code never imports a concrete broker. Everything goes through
this ABC so swapping platforms (Alpaca paper → Tastytrade live) is one line in
config.yaml plus a factory mapping.

Concrete implementations live in `platforms/`:

    paper_broker.py     in-memory mock for tests (Sprint 2)
    alpaca_broker.py    alpaca-py against paper endpoints (Sprint 2)
    tastytrade_broker.py against the production tastytrade SDK (Sprint 6)

All methods are async because real brokers are network-bound; the paper broker
implements them as no-op coroutines for symmetry.

Returned types are the Pydantic models in `core/models.py`. Adapters are
responsible for translating broker-native objects into these.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import date, datetime

from core.models import (
    Account,
    OptionContract,
    OptionType,
    Order,
    OrderLeg,
    Position,
    Quote,
)


class BrokerError(Exception):
    """Base for any broker-layer failure surfaced to the strategy layer."""


class OrderRejected(BrokerError):
    """Broker accepted the request but rejected the order (validation, BP, etc.)."""


class BrokerUnavailable(BrokerError):
    """Transport/HTTP failure or auth problem. The router decides on retry."""


class OrderNotCancelable(BrokerError):
    """2026-07-23: the broker refused the cancel — the order is (or may be)
    already filled/terminal. Callers must NOT mark the order cancelled
    locally: swallowing this and writing CANCELLED let a fill that raced
    the cancel become a permanently untracked position."""


class Broker(ABC):
    """Async broker interface. See class docstring for usage rules."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Short identifier used in logs and the dashboard, e.g. "alpaca_paper"."""

    @abstractmethod
    async def get_account(self) -> Account:
        """Cash, buying power, equity. Called by sizing and risk gates."""

    @abstractmethod
    async def get_positions(self) -> list[Position]:
        """All current broker-side positions (stock + open option short legs).

        These are the broker's view, not our DB's. The reconciler diffs the two.
        """

    @abstractmethod
    async def get_quote(self, symbol: str) -> Quote:
        """Top-of-book for a stock ticker or OCC option symbol."""

    @abstractmethod
    async def get_option_chain(
        self,
        underlying: str,
        expiration: date | None = None,
        option_type: OptionType | None = None,
    ) -> list[OptionContract]:
        """Return contracts for `underlying`, optionally narrowed by expiry/type.

        Filtering by DTE / delta / OI / spread is the chain layer's job (Sprint 3),
        not the broker's. Brokers may return Greeks; if they don't, callers fall
        back to data/greeks.py.
        """

    async def populate_quotes(
        self, contracts: list[OptionContract]
    ) -> list[OptionContract]:
        """Return `contracts` with bid/ask/mid/last/volume/open_interest filled.

        Default no-op: adapters whose `get_option_chain` already returns quoted
        contracts (Alpaca, Paper) inherit this untouched. Tastytrade returns
        structure-only chains and overrides this to batch-fetch quotes for just
        the DTE-windowed contracts the chain layer evaluates (bounded cost).
        """
        return contracts

    @abstractmethod
    async def place_order(self, order: Order) -> Order:
        """Submit `order` to the broker.

        Caller sets `client_order_id` for idempotency (Sprint 4 router owns this).
        Returned Order has `broker_order_id`, updated `status`, and the raw
        request/response captured for the audit trail.

        Raises OrderRejected on broker validation failures.
        Raises BrokerUnavailable on transport failures.
        """

    @abstractmethod
    async def cancel_order(self, broker_order_id: str) -> None:
        """Best-effort cancel. No-op if already filled/cancelled — brokers vary."""

    @abstractmethod
    async def get_orders_since(self, since: datetime) -> list[Order]:
        """All orders placed or updated at/after `since`. Reconciler input."""

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
        """Submit a multi-leg options package atomically.

        Used for verticals (2 legs), iron condors (4 legs), strangles (2 legs).
        The broker fills all legs together or none — that's the whole point
        vs leg-by-leg execution.

        Args:
            underlying: ticker symbol (e.g. "F"); stored on the resulting Order.
            legs: per-leg contract + action + ratio_qty.
            quantity: package quantity (e.g. 3 means 3 verticals = 6 contracts
                across 2 legs at ratio_qty=1 each).
            limit_price: NET price for the package. Positive = credit (we
                receive), negative = debit (we pay), None = market order.
            client_order_id: caller-supplied idempotency key.
            strategy_id: which strategy submitted this; written onto the Order.
            account_id: account this order belongs to.

        Returns an Order with order_type=MULTI_LEG_OPEN/MULTI_LEG_CLOSE
        (inferred from the legs' actions) and the legs preserved in
        raw_request/raw_response. The reconciler treats package fill as a
        single state-change event.

        Raises OrderRejected on broker validation failures.
        Raises BrokerUnavailable on transport failures.
        Raises NotImplementedError if the broker doesn't support multi-leg
        (default Broker base class behavior — concrete adapters override).
        """
        raise NotImplementedError(
            f"{self.name} broker does not support multi-leg orders"
        )
