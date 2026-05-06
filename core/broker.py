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
    Position,
    Quote,
)


class BrokerError(Exception):
    """Base for any broker-layer failure surfaced to the strategy layer."""


class OrderRejected(BrokerError):
    """Broker accepted the request but rejected the order (validation, BP, etc.)."""


class BrokerUnavailable(BrokerError):
    """Transport/HTTP failure or auth problem. The router decides on retry."""


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
