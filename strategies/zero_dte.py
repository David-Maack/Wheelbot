"""0DTE paper-test strategy (2026-07 research sprint).

Evidence base: docs/research/0dte_research_2026-07.md. The two structures
implemented are the research doc's recommended first paper tests (§5b):

  - ``narrow_vertical`` (default): trend-filtered directional credit spread —
    bull put when SPY trades above its session VWAP (fallback: the day's
    opening price), bear call below. $1-2 wide, short delta ~0.20, and THE
    WING IS THE STOP: no stop orders at all. Narrow width makes max loss
    structural ("no reliance on fill quality" — zerodte.com via research §3.3),
    which eliminates the stop-fill slippage gap that is the single largest
    paper-vs-live divergence (FlashAlpha: realistic fills cut CAGR 30-60%).
  - ``iron_condor``: the consensus morning baseline — 10-15 delta shorts,
    $2-5 wings, profit-take at 25% of credit via the normal close path.

Entry only inside a 10:00-10:15 ET window (Option Alpha's profitable cohort
entered ~10:15, after the opening chaos; research §1.1/§5a). Same-day
expiration (dte_min = dte_max = 0) on SPY.

HARD FLATTEN at 15:00 ET regardless of P&L: on expiration day Alpaca REJECTS
option orders after ~15:15 ET (15:30 for broad-based ETFs like SPY) and
force-liquidates expiring positions at 15:30/15:45 ET — a position still
open past our flatten deadline forfeits exit control to Alpaca's liquidation
engine (research §3.4; alpaca.markets/learn/how-to-trade-0dte-options-on-alpaca).

Dual-ledger realism (research §4): Alpaca paper fills are systematically
optimistic (NBBO touch fills, no slippage, no size constraint), so every
entry/exit writes BOTH raw and penalized figures to ``zero_dte_ledger``:
penalized concedes half the summed leg bid-ask spreads on each side, plus a
fixed ``stop_slippage_adder`` on flatten exits. Go/no-go reads penalized
P&L only. Phase 1 writes ledger rows from the propose paths using quotes at
proposal time; phase 2 will reconcile them against actual broker fills via
the reconciler flow.

Out of scope (deliberately): stop orders of any kind, staggered multi-entry
condors (research §5b variant 3 — test only after the slippage model is
calibrated), QQQ/XSP underlyings, holding into the final hour.
"""

from __future__ import annotations

from datetime import UTC, date, datetime, time
from typing import Any
from zoneinfo import ZoneInfo

from core.broker import Broker
from core.checkpoint import log_checkpoint
from core.models import OptionType, OrderLeg, OrderType, PositionState, Quote
from core.strategies import StrategyDefinition
from data.chain import ChainFilters, fetch_filtered_chain
from db.repo import Repos
from strategies.iron_condor import (
    _build_legs as _build_condor_legs,
)
from strategies.iron_condor import (
    _find_protective_long,
    _pick_short_at_delta,
    select_iron_condor,
)
from strategies.spread_selector import SpreadCandidate
from strategies.spreads import (
    DIRECTION_BEAR_CALL,
    DIRECTION_BULL_PUT,
    DIRECTION_IRON_CONDOR,
    MultiLegProposal,
    _build_legs,
    _flip_action,
    _leg_quote,
    _make_chain_recorder,
    _open_order_for_position,
    _tier_flags,
)

ET = ZoneInfo("America/New_York")

STRUCTURE_NARROW_VERTICAL = "narrow_vertical"
STRUCTURE_IRON_CONDOR = "iron_condor"

# No news_check on zero_dte opens: the position lives ~5 hours inside a single
# session and the macro-blackout risk rule already blocks FOMC/CPI/NFP days
# (the event-day gate the research recommends). A headline read adds latency
# inside a 15-minute entry window for little signal.
NEWS_CHECK_PROFILE: str | None = None


# -- time gates --------------------------------------------------------------


def _parse_et(hhmm: str) -> time:
    """'10:00' -> time(10, 0). Raises ValueError on garbage — a typo'd window
    should fail loudly at first use, not silently never trade."""
    hh, mm = hhmm.strip().split(":")
    return time(int(hh), int(mm))


def _now_et(now: datetime | None = None) -> datetime:
    moment = now if now is not None else datetime.now(UTC)
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=UTC)
    return moment.astimezone(ET)


def in_entry_window(params: dict[str, Any], now: datetime | None = None) -> bool:
    """True on a weekday between entry_start_et and entry_end_et (inclusive
    start, inclusive end — the window is only 15 minutes wide at defaults, a
    single 5-minute tick either lands in it or not)."""
    et = _now_et(now)
    if et.weekday() >= 5:
        return False
    start = _parse_et(str(params.get("entry_start_et", "10:00")))
    end = _parse_et(str(params.get("entry_end_et", "10:15")))
    return start <= et.time() <= end


def past_flatten(params: dict[str, Any], now: datetime | None = None) -> bool:
    """True at/after flatten_et. Alpaca rejects expiry-day option orders after
    ~15:15 ET (15:30 broad-based ETFs) and force-liquidates at 15:30/15:45 —
    the 15:00 default leaves a full tick + repricing headroom before that."""
    return _now_et(now).time() >= _parse_et(str(params.get("flatten_et", "15:00")))


# -- trend rule --------------------------------------------------------------


def pick_direction(spot: float, session_ref: float) -> str:
    """Narrow-vertical direction: sell the put spread (bullish premium) when
    SPY is at/above the session reference (VWAP, falling back to the day's
    open), sell the call spread below it. The simple trend filter is where
    most of the documented edge lives (Option Alpha SMA5-filtered setups
    >75% WR, PF > 2.0 — research §1.1/§5b)."""
    return DIRECTION_BULL_PUT if spot >= session_ref else DIRECTION_BEAR_CALL


def _session_vwap_or_open(symbol: str, today: date) -> float | None:
    """Session VWAP from today's completed 5-min Alpaca bars; falls back to
    the day's opening price (first bar's open); None on any failure — the
    caller skips the entry rather than guessing a direction."""
    try:
        # Lazy import — pandas/alpaca stack only loads when the strategy is
        # actually evaluating an entry.
        from strategies.swing import fetch_swing_frames

        bars5m, _daily, _weekly, _vix = fetch_swing_frames(symbol)
        if not len(bars5m):
            return None
        session = bars5m[[d.date() == today for d in bars5m.index]]
        if not len(session):
            return None
        vol = session["volume"]
        if float(vol.sum()) > 0:
            typical = (session["high"] + session["low"] + session["close"]) / 3.0
            return float((typical * vol).sum() / vol.sum())
        return float(session["open"].iloc[0])
    except Exception as exc:  # noqa: BLE001 — data outage must not break the tick
        log_checkpoint("zero_dte_session_ref_fail", status="fail", symbol=symbol,
                       error=str(exc))
        return None


# -- selectors ---------------------------------------------------------------


def _mid(bid: float | None, ask: float | None) -> float | None:
    if bid is None or ask is None:
        return None
    m = (bid + ask) / 2
    return m if m > 0 else None


async def select_zero_dte_vertical(
    broker: Broker,
    symbol: str,
    params: dict[str, Any],
    direction: str,
    *,
    today: date | None = None,
    record_chain: Any | None = None,
) -> SpreadCandidate | None:
    """Same-day-expiry narrow vertical at ~0.20 short delta.

    The existing vertical selectors (spread_selector / call_spread_selector)
    rank shorts by annualized_yield, which is undefined at DTE 0 (divides by
    DTE) — every candidate would be dropped. This selector adapts the iron
    condor's delta-target approach instead: pick the short closest to the
    midpoint of [short_delta_min, short_delta_max], wing it at
    spread_width_dollars via the same protective-long finder.
    """
    today = today or date.today()
    delta_min = float(params.get("short_delta_min", 0.15))
    delta_max = float(params.get("short_delta_max", 0.25))
    target = (delta_min + delta_max) / 2.0
    width = float(params.get("spread_width_dollars", 2.0))
    min_credit_pct = float(params.get("min_credit_pct_of_width", 15.0))

    side = "put" if direction == DIRECTION_BULL_PUT else "call"
    opt_type = OptionType.PUT if side == "put" else OptionType.CALL
    filters = ChainFilters(
        dte_min=0, dte_max=0,
        delta_min=delta_min, delta_max=delta_max,
        open_interest_min=int(params.get("open_interest_min", 0)),
        volume_min=int(params.get("volume_min", 0)),
        bid_ask_spread_max_pct=float(params.get("bid_ask_spread_max_pct", 100.0)),
    )
    candidates = await fetch_filtered_chain(broker, symbol, side, filters, today=today)
    if record_chain is not None:
        await record_chain(symbol, side, candidates)
    if not candidates:
        log_checkpoint("zero_dte_no_short_candidates", status="ok",
                       symbol=symbol, direction=direction)
        return None
    short = _pick_short_at_delta(candidates, target_abs_delta=target)
    if short is None:
        log_checkpoint("zero_dte_no_short", status="ok", symbol=symbol,
                       direction=direction)
        return None

    wide = ChainFilters(
        dte_min=0, dte_max=0, delta_min=0.0, delta_max=1.0,
        open_interest_min=0, volume_min=0, bid_ask_spread_max_pct=100.0,
    )
    full_chain = await fetch_filtered_chain(broker, symbol, side, wide, today=today)
    long = _find_protective_long(full_chain, short, wing_width=width, side=opt_type)
    if long is None:
        log_checkpoint("zero_dte_no_long_wing", status="ok", symbol=symbol,
                       direction=direction, short_strike=short.strike)
        return None

    short_mid = _mid(short.bid, short.ask)
    long_mid = _mid(long.bid, long.ask)
    if short_mid is None or long_mid is None:
        log_checkpoint("zero_dte_missing_mids", status="ok", symbol=symbol)
        return None
    net_credit = short_mid - long_mid
    if net_credit <= 0:
        log_checkpoint("zero_dte_non_positive_credit", status="ok", symbol=symbol,
                       net_credit=net_credit)
        return None
    actual_width = abs(short.strike - long.strike)
    if actual_width <= 0:
        return None
    credit_pct = (net_credit / actual_width) * 100.0
    if credit_pct < min_credit_pct:
        log_checkpoint(
            "zero_dte_skip_low_credit", status="ok", symbol=symbol,
            direction=direction, credit=net_credit, width=actual_width,
            credit_pct=round(credit_pct, 1), min_credit_pct=min_credit_pct,
        )
        return None
    max_loss = (actual_width - net_credit) * 100.0
    log_checkpoint(
        "zero_dte_vertical_selected", status="ok", symbol=symbol,
        direction=direction, short_occ=short.occ_symbol, long_occ=long.occ_symbol,
        credit=net_credit, width=actual_width, max_loss=max_loss,
        short_delta=short.delta,
    )
    return SpreadCandidate(
        short=short, long=long,
        net_credit_per_spread=net_credit,
        max_loss_per_spread=max_loss,
        width_dollars=actual_width,
        short_yield=0.0,  # yield-ranking N/A at DTE 0
    )


# -- ledger writer -----------------------------------------------------------


def entry_penalized_credit(credit_raw: float, spread_width_quotes: float) -> float:
    """Penalized entry credit: concede half the summed leg bid-ask spreads
    (research §4.2 — Vilkov et al.'s half-spread cost model)."""
    return credit_raw - spread_width_quotes / 2.0


def exit_penalized_debit(
    debit_raw: float,
    spread_width_quotes: float,
    *,
    flatten: bool,
    slippage_adder: float,
) -> float:
    """Penalized exit debit: pay half the summed leg spreads on the way out;
    a forced flatten additionally pays stop_slippage_adder per share — the
    practitioner convention for forced/urgent exits (research §4.2)."""
    out = debit_raw + spread_width_quotes / 2.0
    if flatten:
        out += slippage_adder
    return out


def _legs_spread_width_quotes(quotes: list[tuple[float | None, float | None]]) -> float:
    """Sum of (ask - bid) across legs, $/share. Missing sides count 0 —
    conservative would be rejecting, but entry/close paths already reject
    quoteless legs before reaching here."""
    total = 0.0
    for bid, ask in quotes:
        if bid is not None and ask is not None and ask >= bid:
            total += ask - bid
    return total


async def _write_entry_ledger(
    repos: Repos,
    *,
    symbol: str,
    structure: str,
    quantity: int,
    credit_raw: float,
    spread_width_quotes: float,
    now: datetime | None = None,
) -> None:
    """Insert the entry half of a dual-ledger row at proposal time. Phase 2
    will link cycle_id + true fill prices via the reconciler; until then the
    proposal-time quotes are the ledger's entry basis (noted in the research
    doc's §4.3 requirement list). Never raises — ledger failures must not
    block the trade itself."""
    try:
        ts = (now or datetime.now(UTC)).astimezone(UTC).replace(tzinfo=None)
        penalized = entry_penalized_credit(credit_raw, spread_width_quotes)
        row_id = await repos.zero_dte_ledger.insert_entry({
            "symbol": symbol,
            "structure": structure,
            "quantity": quantity,
            "entry_ts": ts.isoformat(),
            "entry_credit_raw": credit_raw,
            "entry_spread_width_quotes": spread_width_quotes,
            "entry_credit_penalized": penalized,
        })
        log_checkpoint(
            "zero_dte_ledger", status="ok", event="entry", row_id=row_id,
            symbol=symbol, structure=structure, quantity=quantity,
            credit_raw=round(credit_raw, 4),
            spread_width_quotes=round(spread_width_quotes, 4),
            credit_penalized=round(penalized, 4),
        )
    except Exception as exc:  # noqa: BLE001 — ledger must never block trading
        log_checkpoint("zero_dte_ledger", status="fail", event="entry",
                       symbol=symbol, error=str(exc))


async def _write_exit_ledger(
    repos: Repos,
    *,
    symbol: str,
    quantity: int,
    debit_raw: float,
    spread_width_quotes: float,
    exit_reason: str,
    slippage_adder: float,
    now: datetime | None = None,
) -> None:
    """Update the latest ledger row for `symbol` with exit figures. A close
    that doesn't fill gets re-proposed next tick and simply refreshes these
    columns with newer quotes — last write wins. Never raises."""
    try:
        row = await repos.zero_dte_ledger.latest_for_symbol(symbol)
        if row is None:
            log_checkpoint("zero_dte_ledger", status="fail", event="exit",
                           symbol=symbol, error="no ledger row to close")
            return
        ts = (now or datetime.now(UTC)).astimezone(UTC).replace(tzinfo=None)
        flatten = exit_reason == "flatten"
        debit_penalized = exit_penalized_debit(
            debit_raw, spread_width_quotes,
            flatten=flatten, slippage_adder=slippage_adder,
        )
        entry_raw = float(row.get("entry_credit_raw") or 0.0)
        entry_pen = float(row.get("entry_credit_penalized") or 0.0)
        pnl_raw = (entry_raw - debit_raw) * 100.0 * quantity
        pnl_penalized = (entry_pen - debit_penalized) * 100.0 * quantity
        await repos.zero_dte_ledger.record_exit(
            int(row["id"]),
            exit_ts=ts.isoformat(),
            exit_debit_raw=debit_raw,
            exit_debit_penalized=debit_penalized,
            pnl_raw=pnl_raw,
            pnl_penalized=pnl_penalized,
            exit_reason=exit_reason,
        )
        log_checkpoint(
            "zero_dte_ledger", status="ok", event="exit", row_id=row["id"],
            symbol=symbol, exit_reason=exit_reason,
            debit_raw=round(debit_raw, 4),
            debit_penalized=round(debit_penalized, 4),
            pnl_raw=round(pnl_raw, 2), pnl_penalized=round(pnl_penalized, 2),
        )
    except Exception as exc:  # noqa: BLE001 — ledger must never block trading
        log_checkpoint("zero_dte_ledger", status="fail", event="exit",
                       symbol=symbol, error=str(exc))


# -- entry orchestrator ------------------------------------------------------


def _sizing_quantity(
    strategy: StrategyDefinition,
    max_loss_per_spread: float,
    *,
    size_multiplier: float = 1.0,
) -> int:
    """floor(cap / max_loss); 0 = skip (never over-risk — audit #13 rule).
    Reads max_capital_per_spread_usd for BOTH structures (the condor variant
    shares the same $200 defined-risk cap)."""
    cap = float(strategy.params.get("max_capital_per_spread_usd", 0) or 0)
    if cap <= 0 or max_loss_per_spread <= 0:
        return 1
    return int((cap * float(size_multiplier)) // max_loss_per_spread)


async def propose_for_symbol(
    broker: Broker,
    repos: Repos,
    symbol: str,
    config: dict[str, Any],
    universe: dict[str, Any],
    *,
    today: date | None = None,
    strategy: StrategyDefinition | None = None,
    size_multiplier: float = 1.0,
    now: datetime | None = None,
    session_ref: float | None = None,
) -> MultiLegProposal | None:
    """Build a same-day-expiry MultiLegProposal, or None.

    Gate order: entry window -> position state -> max-positions-per-day ->
    daily loss cap -> trend/selector. `now` and `session_ref` are injectable
    for tests; production leaves them None (wall clock + live VWAP).
    """
    if strategy is None:
        log_checkpoint("zero_dte_skip_no_strategy", status="fail", symbol=symbol)
        return None
    params = strategy.params
    et = _now_et(now)
    today = today or et.date()

    if not in_entry_window(params, now):
        log_checkpoint(
            "zero_dte_skip_window", status="ok", symbol=symbol,
            strategy=strategy.id, now_et=et.strftime("%H:%M"),
            window=f"{params.get('entry_start_et', '10:00')}-{params.get('entry_end_et', '10:15')}",
        )
        return None

    account_id = config.get("account", {}).get("id", "primary")
    position = await repos.positions.get_by_symbol(
        account_id, symbol, strategy_id=strategy.id,
    )
    state = position.state if position else PositionState.IDLE
    if state not in (PositionState.IDLE, PositionState.SPREAD_CLOSED):
        log_checkpoint("zero_dte_skip_state", status="ok", symbol=symbol,
                       strategy=strategy.id, state=str(state))
        return None

    # One round-trip per day by default. Cycles are created on fill, so a
    # filled-and-flattened morning trade blocks a second entry even though
    # the position is back to SPREAD_CLOSED.
    max_per_day = int(params.get("max_positions_per_day", 1))
    n_today = await repos.cycles.count_started_on(account_id, strategy.id, today)
    if n_today >= max_per_day:
        log_checkpoint("zero_dte_skip_max_per_day", status="ok", symbol=symbol,
                       strategy=strategy.id, n_today=n_today, cap=max_per_day)
        return None

    # Strategy-local daily loss cap, read from the PENALIZED ledger (the
    # honest number). Independent of the account-level kill switch.
    loss_cap = float(params.get("zero_dte_daily_loss_usd", 250.0))
    if loss_cap > 0:
        realized = await repos.zero_dte_ledger.realized_pnl_on(today)
        if realized <= -loss_cap:
            log_checkpoint("zero_dte_skip_daily_loss_cap", status="skip",
                           symbol=symbol, strategy=strategy.id,
                           realized_penalized=round(realized, 2), cap=loss_cap)
            return None

    record = _make_chain_recorder(
        repos,
        cycle_id=position.current_cycle_id if position else None,
        strategy_id=strategy.id,
    )
    structure = str(params.get("structure", STRUCTURE_NARROW_VERTICAL))

    if structure == STRUCTURE_IRON_CONDOR:
        # 10-15Δ shorts, $2-5 wings, 25% profit-take (research §5b variant 1).
        condor_params = {
            **params,
            "dte_min": 0,
            "dte_max": 0,
            "short_put_delta_target": float(params.get("condor_short_delta_target", 0.12)),
            "short_call_delta_target": float(params.get("condor_short_delta_target", 0.12)),
            "wing_width": float(params.get("condor_wing_width_dollars", 3.0)),
            "min_credit_pct": float(params.get("condor_min_credit_pct", 10.0)),
        }
        cand = await select_iron_condor(
            broker, symbol, condor_params, today=today, record_chain=record,
        )
        if cand is None:
            return None
        quantity = _sizing_quantity(
            strategy, cand.max_loss_per_spread, size_multiplier=size_multiplier,
        )
        if quantity <= 0:
            log_checkpoint("zero_dte_skip_over_cap", status="skip", symbol=symbol,
                           strategy=strategy.id, max_loss=cand.max_loss_per_spread)
            return None
        legs = _build_condor_legs(cand)
        direction = DIRECTION_IRON_CONDOR
        net_credit = cand.net_credit_per_spread
        max_loss = cand.max_loss_per_spread
        width = cand.wing_width_dollars
        leg_quotes = [
            (c.bid, c.ask)
            for c in (cand.long_put, cand.short_put, cand.short_call, cand.long_call)
        ]
        rationale = (
            f"zero_dte_condor[{strategy.id}] "
            f"LP/SP={cand.long_put.strike}/{cand.short_put.strike} "
            f"SC/LC={cand.short_call.strike}/{cand.long_call.strike} "
            f"width={width:.2f} credit={net_credit:.2f} "
            f"max_loss={max_loss:.2f} qty={quantity}"
        )
    elif structure == STRUCTURE_NARROW_VERTICAL:
        # Trend rule: SPY vs session VWAP (fallback: day's open).
        try:
            q: Quote = await broker.get_quote(symbol)
            spot = q.mid if q.mid is not None else (q.last or q.bid or q.ask)
        except Exception:
            spot = None
        if spot is None:
            log_checkpoint("zero_dte_skip_no_spot", status="ok", symbol=symbol)
            return None
        ref = session_ref if session_ref is not None else _session_vwap_or_open(symbol, today)
        if ref is None:
            log_checkpoint("zero_dte_skip_no_session_ref", status="ok", symbol=symbol)
            return None
        direction = pick_direction(float(spot), float(ref))
        cand_v = await select_zero_dte_vertical(
            broker, symbol, params, direction, today=today, record_chain=record,
        )
        if cand_v is None:
            return None
        quantity = _sizing_quantity(
            strategy, cand_v.max_loss_per_spread, size_multiplier=size_multiplier,
        )
        if quantity <= 0:
            log_checkpoint("zero_dte_skip_over_cap", status="skip", symbol=symbol,
                           strategy=strategy.id, max_loss=cand_v.max_loss_per_spread)
            return None
        legs = _build_legs(cand_v)
        net_credit = cand_v.net_credit_per_spread
        max_loss = cand_v.max_loss_per_spread
        width = cand_v.width_dollars
        leg_quotes = [(cand_v.short.bid, cand_v.short.ask), (cand_v.long.bid, cand_v.long.ask)]
        rationale = (
            f"zero_dte_{direction}[{strategy.id}] spot={float(spot):.2f} "
            f"ref={float(ref):.2f} short={cand_v.short.strike} "
            f"long={cand_v.long.strike} width={width:.2f} "
            f"credit={net_credit:.2f} max_loss={max_loss:.2f} qty={quantity} "
            f"(wing-is-the-stop)"
        )
    else:
        log_checkpoint("zero_dte_unknown_structure", status="fail",
                       symbol=symbol, strategy=strategy.id, structure=structure)
        return None

    # Dual-ledger entry write at proposal-time quotes (phase 2: reconcile
    # against the actual fill via the reconciler flow).
    await _write_entry_ledger(
        repos,
        symbol=symbol,
        structure=structure,
        quantity=quantity,
        credit_raw=net_credit,
        spread_width_quotes=_legs_spread_width_quotes(leg_quotes),
        now=now,
    )

    needs_screen, needs_human = _tier_flags(symbol, universe)
    return MultiLegProposal(
        symbol=symbol,
        legs=legs,
        net_credit_per_spread=net_credit,
        max_loss_per_spread=max_loss,
        width_dollars=width,
        quantity=quantity,
        rationale=rationale,
        strategy_id=strategy.id,
        requires_screen=needs_screen,
        requires_human=needs_human,
        direction=direction,
        news_check_profile=NEWS_CHECK_PROFILE,
    )


async def propose_all(
    broker: Broker,
    repos: Repos,
    config: dict[str, Any],
    universe: dict[str, Any],
    *,
    today: date | None = None,
    strategy: StrategyDefinition | None = None,
    size_multiplier: float = 1.0,
    now: datetime | None = None,
) -> list[MultiLegProposal]:
    out: list[MultiLegProposal] = []
    for entry in universe["tickers"]:
        proposal = await propose_for_symbol(
            broker, repos, entry.symbol, config, universe,
            today=today, strategy=strategy,
            size_multiplier=size_multiplier, now=now,
        )
        if proposal is not None:
            out.append(proposal)
    log_checkpoint(
        "zero_dte_propose_all", status="ok",
        strategy=strategy.id if strategy else "zero_dte",
        n_proposals=len(out),
    )
    return out


# -- close orchestrator ------------------------------------------------------


async def propose_close_for_symbol(
    broker: Broker,
    repos: Repos,
    symbol: str,
    config: dict[str, Any],
    *,
    strategy: StrategyDefinition | None = None,
    now: datetime | None = None,
) -> MultiLegProposal | None:
    """Close triggers, in priority order:

    1. HARD FLATTEN at/after flatten_et (15:00 ET default) REGARDLESS of P&L.
       Alpaca rejects expiry-day option orders after ~15:15 ET (15:30 for
       broad-based ETFs like SPY) and force-liquidates expiring positions at
       15:30/15:45 ET — missing our own flatten hands the exit to Alpaca's
       liquidation engine at whatever the market gives (research §3.4).
    2. Profit take at profit_close_pct of the original credit (condor 25 /
       vertical 50 defaults).

    NO stop trigger by design: for the narrow vertical THE WING IS THE STOP —
    max loss is structural, which is precisely what makes this paper test
    credible (no simulated stop fills to flatter, research §3.3/§5b).
    """
    if strategy is None:
        return None
    params = strategy.params
    account_id = config.get("account", {}).get("id", "primary")
    position = await repos.positions.get_by_symbol(
        account_id, symbol, strategy_id=strategy.id,
    )
    if position is None or position.state != PositionState.SPREAD_OPEN:
        return None
    if position.id is None:
        return None
    open_order = await _open_order_for_position(repos, position.id)
    if open_order is None or not open_order.raw_request:
        return None
    legs_raw = open_order.raw_request.get("legs") or []
    if not legs_raw:
        return None

    structure = str(params.get("structure", STRUCTURE_NARROW_VERTICAL))
    close_legs: list[OrderLeg] = []
    has_put = False
    has_call = False
    for leg in legs_raw:
        ol = OrderLeg(
            contract_symbol=leg["contract_symbol"],
            underlying=leg["underlying"],
            option_type=leg["option_type"],
            strike=leg["strike"],
            expiration=leg["expiration"],
            action=_flip_action(OrderType(leg["action"])),
            ratio_qty=leg.get("ratio_qty", 1),
        )
        close_legs.append(ol)
        if str(leg["option_type"]) == OptionType.PUT.value:
            has_put = True
        else:
            has_call = True
    if has_put and has_call:
        direction = DIRECTION_IRON_CONDOR
    elif has_call:
        direction = DIRECTION_BEAR_CALL
    else:
        direction = DIRECTION_BULL_PUT

    # Re-quote each leg: MID drives the DECISION, BID/ASK drive the ORDER
    # price so the close is marketable and fills (same convention as
    # strategies/spreads.py — a mid limit on a fast 0DTE tape would cancel-
    # loop while gamma runs).
    leg_quotes: list[tuple[OrderLeg, Quote | None]] = []
    for leg in close_legs:
        leg_quotes.append((leg, await _leg_quote(broker, leg.contract_symbol)))
    if any(q is None or q.mid is None for _, q in leg_quotes):
        log_checkpoint("zero_dte_close_skip_no_quote", status="skip",
                       symbol=symbol, strategy=strategy.id)
        return None

    debit_to_close = 0.0
    marketable_debit = 0.0
    quote_spread_sum = 0.0
    for leg, q in leg_quotes:
        assert q is not None and q.mid is not None
        mid = q.mid
        ask = q.ask if q.ask is not None else mid
        bid = q.bid if q.bid is not None else mid
        if ask >= bid:
            quote_spread_sum += ask - bid
        if leg.action == OrderType.BUY_TO_CLOSE:
            debit_to_close += mid
            marketable_debit += ask
        elif leg.action == OrderType.SELL_TO_CLOSE:
            debit_to_close -= mid
            marketable_debit -= bid

    # Hard-clamp the close debit at the wing width (2026-07-23 TSLA lesson —
    # paying more than the structure's worst settlement value is irrational).
    put_strikes = sorted(
        float(ol.strike) for ol in close_legs
        if str(ol.option_type) == OptionType.PUT.value
    )
    call_strikes = sorted(
        float(ol.strike) for ol in close_legs
        if str(ol.option_type) == OptionType.CALL.value
    )
    widths = []
    if len(put_strikes) >= 2:
        widths.append(put_strikes[-1] - put_strikes[0])
    if len(call_strikes) >= 2:
        widths.append(call_strikes[-1] - call_strikes[0])
    max_close_debit = max(widths) if widths else None
    if max_close_debit is not None and marketable_debit > max_close_debit:
        log_checkpoint("zero_dte_close_debit_clamped", status="ok",
                       symbol=symbol, strategy=strategy.id,
                       marketable_debit=round(marketable_debit, 2),
                       clamped_to=round(max_close_debit, 2))
        marketable_debit = max_close_debit

    original_credit = open_order.fill_price or 0.0
    default_profit_pct = 25.0 if structure == STRUCTURE_IRON_CONDOR else 50.0
    profit_close_pct = float(params.get("profit_close_pct", default_profit_pct))
    target_max_debit = (1 - profit_close_pct / 100.0) * original_credit

    flatten_trigger = past_flatten(params, now)
    profit_trigger = original_credit > 0 and debit_to_close <= target_max_debit
    if not (flatten_trigger or profit_trigger):
        return None

    exit_reason = "flatten" if flatten_trigger else "profit_target"
    quantity = open_order.quantity or 1
    await _write_exit_ledger(
        repos,
        symbol=symbol,
        quantity=quantity,
        debit_raw=debit_to_close,
        spread_width_quotes=quote_spread_sum,
        exit_reason=exit_reason,
        slippage_adder=float(params.get("stop_slippage_adder", 0.05)),
        now=now,
    )

    rationale_parts = []
    if flatten_trigger:
        rationale_parts.append(
            f"HARD FLATTEN at/after {params.get('flatten_et', '15:00')} ET "
            f"(Alpaca expiry-day cutoff ~15:15/15:30, auto-liq 15:30/15:45) "
            f"debit={debit_to_close:.2f}"
        )
    if profit_trigger:
        rationale_parts.append(
            f"profit_close at debit={debit_to_close:.2f} <= target "
            f"{target_max_debit:.2f} (orig credit={original_credit:.2f}, "
            f"pct={profit_close_pct})"
        )
    rationale = f"zero_dte_close[{strategy.id}] " + "; ".join(rationale_parts)
    log_checkpoint(
        "zero_dte_close_fired", status="ok", symbol=symbol,
        strategy=strategy.id, exit_reason=exit_reason,
        debit=round(debit_to_close, 2), marketable=round(marketable_debit, 2),
    )
    return MultiLegProposal(
        symbol=symbol,
        legs=close_legs,
        net_credit_per_spread=-marketable_debit,
        max_loss_per_spread=0.0,  # closing — no incremental risk
        width_dollars=0.0,
        quantity=quantity,
        rationale=rationale,
        strategy_id=strategy.id,
        order_type=OrderType.MULTI_LEG_CLOSE,
        direction=direction,
    )


async def propose_all_closes(
    broker: Broker,
    repos: Repos,
    config: dict[str, Any],
    *,
    strategy: StrategyDefinition | None = None,
    now: datetime | None = None,
) -> list[MultiLegProposal]:
    """Walk SPREAD_OPEN zero_dte positions; propose closes. Runs every tick
    regardless of gates (closes are never entry-gated), so the 15:00 flatten
    fires on the first tick at/after the deadline."""
    if strategy is None:
        return []
    account_id = config.get("account", {}).get("id", "primary")
    active = await repos.positions.list_active(account_id, strategy_id=strategy.id)
    out: list[MultiLegProposal] = []
    for pos in active:
        if pos.state != PositionState.SPREAD_OPEN:
            continue
        proposal = await propose_close_for_symbol(
            broker, repos, pos.symbol, config, strategy=strategy, now=now,
        )
        if proposal is not None:
            out.append(proposal)
    log_checkpoint(
        "zero_dte_propose_closes", status="ok",
        strategy=strategy.id, n_proposals=len(out),
    )
    return out
