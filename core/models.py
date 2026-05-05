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
    BROKER_DOWN = "BROKER_DOWN"
    MANUAL_INTERVENTION = "MANUAL_INTERVENTION"
    KILLED = "KILLED"


class OrderType(StrEnum):
    SELL_TO_OPEN = "SELL_TO_OPEN"
    BUY_TO_CLOSE = "BUY_TO_CLOSE"
    BUY_TO_OPEN = "BUY_TO_OPEN"
    SELL_TO_CLOSE = "SELL_TO_CLOSE"


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


class IvHistory(_Base):
    id: int | None = None
    symbol: str
    snapshot_date: date
    iv_30d: float | None = None


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


class UniverseEntry(_Base):
    symbol: str
    name: str | None = None
    tier: int = Field(ge=1, le=3)
    notes: str | None = None
    overrides: dict[str, Any] = Field(default_factory=dict)
