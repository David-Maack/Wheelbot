"""Pydantic models mirroring db/schema.sql.

Models are the in-process source of truth for shape; schema.sql is the on-disk source of
truth. db/repo.py is responsible for the round-trip and must keep these aligned.
"""

from __future__ import annotations

from datetime import date, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class PositionState(StrEnum):
    IDLE = "IDLE"
    SCANNING = "SCANNING"
    CSP_PENDING = "CSP_PENDING"
    CSP_OPEN = "CSP_OPEN"
    ROLL_EVAL = "ROLL_EVAL"
    CSP_CLOSED = "CSP_CLOSED"
    ASSIGNED = "ASSIGNED"
    SHARES_HELD = "SHARES_HELD"
    CC_PENDING = "CC_PENDING"
    CC_OPEN = "CC_OPEN"
    CC_CLOSED = "CC_CLOSED"
    CALLED_AWAY = "CALLED_AWAY"
    # Vertical credit-spread states (sub-sprint 3). The package is opened as
    # a single multi-leg order; reconciler treats a fill as one state event.
    SPREAD_PENDING = "SPREAD_PENDING"
    SPREAD_OPEN = "SPREAD_OPEN"
    SPREAD_CLOSED = "SPREAD_CLOSED"
    # Both legs ITM at expiry — defined max loss already realized.
    SPREAD_ASSIGNED = "SPREAD_ASSIGNED"
    BROKER_DOWN = "BROKER_DOWN"
    MANUAL_INTERVENTION = "MANUAL_INTERVENTION"
    KILLED = "KILLED"


class OrderType(StrEnum):
    SELL_TO_OPEN = "SELL_TO_OPEN"
    BUY_TO_CLOSE = "BUY_TO_CLOSE"
    BUY_TO_OPEN = "BUY_TO_OPEN"
    SELL_TO_CLOSE = "SELL_TO_CLOSE"
    # Multi-leg packages — verticals, condors, strangles. The Order.quantity
    # is the package quantity (e.g. 3 verticals); each leg has its own
    # ratio_qty applied as a multiplier. Per-leg detail lives in raw_request.
    MULTI_LEG_OPEN = "MULTI_LEG_OPEN"
    MULTI_LEG_CLOSE = "MULTI_LEG_CLOSE"


class OptionType(StrEnum):
    PUT = "PUT"
    CALL = "CALL"


class OrderStatus(StrEnum):
    PENDING = "PENDING"
    FILLED = "FILLED"
    PARTIAL = "PARTIAL"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"


class CycleOutcome(StrEnum):
    CSP_EXPIRED = "CSP_EXPIRED"
    CSP_CLOSED_PROFIT = "CSP_CLOSED_PROFIT"
    CC_EXPIRED = "CC_EXPIRED"
    CC_CLOSED_PROFIT = "CC_CLOSED_PROFIT"
    CC_CALLED_AWAY = "CC_CALLED_AWAY"
    SPREAD_EXPIRED_PROFIT = "SPREAD_EXPIRED_PROFIT"   # both legs OTM at expiry
    SPREAD_CLOSED_PROFIT = "SPREAD_CLOSED_PROFIT"     # closed at profit_close_pct
    SPREAD_MAX_LOSS = "SPREAD_MAX_LOSS"               # both legs ITM at expiry
    MANUAL_CLOSE = "MANUAL_CLOSE"


class StateLogTrigger(StrEnum):
    ORDER_FILL = "ORDER_FILL"
    RECONCILER = "RECONCILER"
    MANUAL = "MANUAL"
    KILL_SWITCH = "KILL_SWITCH"
    STRATEGY = "STRATEGY"
    SYSTEM = "SYSTEM"


class LlmDecisionType(StrEnum):
    SCREEN = "SCREEN"
    NEWS_CHECK = "NEWS_CHECK"
    ROLL_ADVISE = "ROLL_ADVISE"


class Regime(StrEnum):
    BULL_TREND = "BULL_TREND"
    NEUTRAL = "NEUTRAL"
    BEAR_TREND = "BEAR_TREND"
    HIGH_VOL = "HIGH_VOL"


class _Base(BaseModel):
    model_config = ConfigDict(extra="forbid", use_enum_values=True)


class Position(_Base):
    id: int | None = None
    account_id: str
    symbol: str
    strategy_id: str = "monthly_wheel"
    state: PositionState
    shares: int = 0
    cost_basis: float | None = None
    current_cycle_id: int | None = None
    state_changed_at: datetime
    state_change_reason: str | None = None


class Order(_Base):
    id: int | None = None
    account_id: str
    symbol: str
    strategy_id: str | None = None
    cycle_id: int | None = None
    broker_order_id: str | None = None
    client_order_id: str | None = None
    order_type: OrderType
    contract_symbol: str | None = None
    strike: float | None = None
    expiration: date | None = None
    option_type: OptionType | None = None
    quantity: int
    limit_price: float | None = None
    fill_price: float | None = None
    status: OrderStatus
    placed_at: datetime
    filled_at: datetime | None = None
    raw_request: dict[str, Any] | None = None
    raw_response: dict[str, Any] | None = None


class WheelCycle(_Base):
    id: int | None = None
    account_id: str
    symbol: str
    strategy_id: str | None = None
    started_at: datetime
    ended_at: datetime | None = None
    initial_csp_strike: float | None = None
    initial_csp_premium: float | None = None
    initial_capital_at_risk: float | None = None
    final_pnl: float | None = None
    final_pnl_pct: float | None = None
    cycle_outcome: CycleOutcome | None = None
    days_held: int | None = None
    n_orders: int | None = None


class StateLog(_Base):
    id: int | None = None
    position_id: int
    from_state: PositionState | None = None
    to_state: PositionState
    reason: str | None = None
    triggered_by: StateLogTrigger | None = None
    metadata: dict[str, Any] | None = None
    created_at: datetime


class Candidate(_Base):
    id: int | None = None
    run_date: date
    symbol: str
    score: float | None = None
    rank: int | None = None
    rationale: str | None = None
    ivr: float | None = None
    iv_pct: float | None = None
    price: float | None = None
    bp_required: float | None = None
    suggested_strike: float | None = None
    suggested_dte: int | None = None
    raw_llm_response: dict[str, Any] | None = None
    acted_on: bool = False


class RegimeSnapshot(_Base):
    id: int | None = None
    snapshot_date: date
    spy_close: float | None = None
    spy_sma_200: float | None = None
    spy_above_sma: bool | None = None
    vix_close: float | None = None
    vix_change_pct: float | None = None
    choppiness: float | None = None
    regime: Regime | None = None
    csps_allowed: bool | None = None
    notes: str | None = None


class LlmDecision(_Base):
    id: int | None = None
    decision_type: LlmDecisionType
    context: dict[str, Any] | None = None
    model: str | None = None
    response: dict[str, Any] | None = None
    decision: str | None = None
    confidence: float | None = None
    acted_on: bool | None = None
    outcome: str | None = None
    created_at: datetime
    tokens_in: int | None = None
    tokens_out: int | None = None
    cost_usd: float | None = None


class IvHistory(_Base):
    id: int | None = None
    symbol: str
    snapshot_date: date
    iv_30d: float | None = None


class DailyState(_Base):
    """Per-day per-account anchor row used by execution/kill_switch.py."""

    id: int | None = None
    account_id: str
    snapshot_date: date
    session_open_equity: float | None = None
    consecutive_losses: int = 0
    kill_switch_armed: bool = False
    kill_switch_reason: str | None = None


class ChainSnapshot(_Base):
    """Captured option chain at a decision point. Used by the cycle backtester."""

    id: int | None = None
    captured_at: datetime
    symbol: str
    strategy_id: str | None = None
    side: str  # "put" | "call"
    underlying_price: float | None = None
    contracts: list[dict[str, Any]]
    decision_id: int | None = None
    cycle_id: int | None = None
    notes: str | None = None


class OrderLeg(_Base):
    """One leg of a multi-leg options order.

    Used by Broker.place_multi_leg_order. The leg specifies which contract
    to trade and which side; the parent order's `quantity` multiplies each
    leg's `ratio_qty` to determine how many contracts of that leg to fill.
    """

    contract_symbol: str           # OCC option symbol (e.g. F250620P00010000)
    underlying: str                # underlying ticker
    option_type: OptionType
    strike: float
    expiration: date
    action: OrderType              # SELL_TO_OPEN / BUY_TO_OPEN / SELL_TO_CLOSE / BUY_TO_CLOSE
    ratio_qty: int = 1             # multiplier on the parent order's quantity


class OptionContract(_Base):
    """Pure data — not persisted. Used by chain selectors and broker adapters."""

    underlying: str
    occ_symbol: str
    strike: float
    expiration: date
    option_type: OptionType
    bid: float | None = None
    ask: float | None = None
    mid: float | None = None
    last: float | None = None
    volume: int | None = None
    open_interest: int | None = None
    delta: float | None = None
    theta: float | None = None
    vega: float | None = None
    iv: float | None = None
    underlying_price: float | None = None


class Account(_Base):
    """Snapshot of broker-reported account state. Not persisted."""

    account_id: str
    cash: float
    buying_power: float
    equity: float
    options_buying_power: float | None = None
    maintenance_margin: float | None = None
    pattern_day_trader: bool | None = None


class Quote(_Base):
    """Top-of-book snapshot for a stock or option. Not persisted."""

    symbol: str
    bid: float | None = None
    ask: float | None = None
    last: float | None = None
    bid_size: int | None = None
    ask_size: int | None = None
    timestamp: datetime | None = None

    @property
    def mid(self) -> float | None:
        if self.bid is None or self.ask is None:
            return None
        return (self.bid + self.ask) / 2


class UniverseEntry(_Base):
    symbol: str
    name: str | None = None
    tier: int = Field(ge=1, le=3)
    strategies: list[str] = Field(default_factory=lambda: ["monthly_wheel"])
    notes: str | None = None
    overrides: dict[str, Any] = Field(default_factory=dict)
