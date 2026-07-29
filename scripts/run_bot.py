# =====================================================================
# DEPLOY NOTE — DB MIGRATIONS
# =====================================================================
# Migrations in db/migrations/ are baked into the Docker image at build
# time. A `git pull` on the LXC host does NOT apply new migrations — the
# running container still has the old image.
#
# To deploy a migration:
#   1. cd /opt/wheelbot && git pull
#   2. docker compose build wheelbot
#   3. docker compose up -d wheelbot
#   4. Verify:  docker exec wheelbot python -m scripts.db_health
#                docker exec wheelbot bash scripts/migrate_check.sh
#
# The bot's first reconcile tick after restart re-fetches broker state, so
# any old PENDING orders the cursor lost track of get re-evaluated. Watch
# for transient MANUAL_INTERVENTION flags in the minute after restart and
# clear them with scripts.reenable_strategy / manual_close once verified.
# =====================================================================

"""Long-running WheelBot entrypoint.

This is the main process the LXC container runs. It:

  - Loads config + secrets, sets up structured logging.
  - Constructs broker, repos, news source, optional Anthropic client.
  - Wires the Discord notifier as the global dispatcher.
  - Wires `news_check` into the OrderRouter.
  - Wires `roll_orchestrator` into the Reconciler.
  - Each tick:
      1. Reconcile (kill-switch primed, broker polled, state updates written).
      2. If kill-switch is armed → skip new entries this tick.
      3. Run wheel orchestrator over the universe → produce Proposals.
      4. Place each Proposal through the router (risk gates + idempotency).
      5. (Reconciler's roll-trigger scan already ran inside step 1; the wired
         roll evaluator handles ROLL/LET_ASSIGN/CLOSE actions.)
  - Cadence: 5 min in market hours, 30 min off.
  - Consecutive broker failures → BROKER_DOWN per ReconcilerLoop.

Run with:
    python -m scripts.run_bot
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from core.broker import Broker
from core.broker_factory import make_broker
from core.checkpoint import log_checkpoint
from core.config import load_config, load_universe
from core.logs import setup_logging
from core.models import OptionType, Order, OrderType, Position
from core.notify import make_notifier, notify, set_dispatcher
from core.strategies import StrategyDefinition, load_strategies, universe_for_strategy
from core.watchlists import effective_universe
from data.ivr import IVRProvider
from db.repo import Database, Repos
from execution.kill_switch import KillSwitchResult
from execution.loop import ReconcilerLoop
from execution.reconciler import Reconciler
from execution.router import OrderRouter
from risk.earnings_recheck import check_open_positions_for_new_earnings
from intelligence.anthropic_client import AnthropicClient
from intelligence.budget import BudgetTracker
from intelligence.news import make_news_source
from intelligence.news_check import news_check as run_news_check
from intelligence.position_news_sentry import run_position_news_sentry
from strategies import roll_orchestrator
from strategies.roll_advisor import RollAction, RollContext
from strategies.iron_condor import propose_all as propose_all_iron_condor
from strategies.calendar import propose_all as propose_all_calendar
from strategies.calendar_close import propose_all_closes as propose_all_calendar_closes
from strategies.pmcc import propose_all as propose_all_pmcc
from strategies.pmcc_close import propose_all_closes as propose_all_pmcc_closes
from strategies.spreads import (
    MultiLegProposal,
    propose_all as propose_all_spreads,
    propose_all_closes as propose_all_spread_closes,
)
from risk import auto_disable, expectancy_floor, win_rate_floor
from strategies.swing import propose_all_swing_closes, propose_all_swings
from strategies.zero_dte import (
    propose_all as propose_all_zero_dte,
    propose_all_closes as propose_all_zero_dte_closes,
)
from strategies.wheel import propose_all
from strategies.wheel_close import propose_all_closes as propose_all_wheel_closes


def _utcnow() -> datetime:
    return datetime.now(UTC).replace(tzinfo=None)


async def _make_news_check_callable(news, anthropic, config):
    """Returns the callable the OrderRouter expects, or None when disabled.

    TICKET-014: the inner closure now accepts an optional `profile` kwarg so
    the router can dispatch per-strategy prompts (bullish_csp / bullish_long
    / neutral_range / neutral_pin). Default = bullish_csp preserves the
    wheel CSP path's existing prompt byte-for-byte (locked by
    test_wheel_csp_*_unchanged regression tests)."""
    intel = config.get("intelligence", {}) or {}
    if not bool(intel.get("llm_news_check_enabled", True)):
        return None
    if anthropic is None or news is None:
        return None

    async def _check(symbol: str, profile: str = "bullish_csp"):
        return await run_news_check(
            symbol=symbol, news=news, anthropic=anthropic, config=config,
            profile=profile,
        )

    return _check


async def _make_roll_evaluator(
    broker: Broker,
    repos: Repos,
    anthropic,
    config,
    universe,
    *,
    router: OrderRouter | None = None,
):
    """Returns the callable the Reconciler's roll-trigger scan invokes.

    When `router` is provided, ROLL/CLOSE outcomes are *executed* leg-by-leg
    through the same router (full risk gates + idempotency). LET_ASSIGN is a
    no-op by design — wait for the broker to assign. Disagreement halts the
    position via the orchestrator and never reaches here.
    """

    async def _eval(position: Position, short: Order, current_mid: float):
        if short.contract_symbol is None or short.option_type is None or short.strike is None:
            return None
        # Underlying spot for the RollContext.
        try:
            quote = await broker.get_quote(position.symbol)
            spot = quote.mid if quote.mid is not None else (quote.last or quote.bid or quote.ask)
        except Exception:
            spot = None
        if spot is None:
            return None

        # Reconstruct an OptionContract with the current delta. We pull a fresh
        # quote on the option to populate delta; if Greeks aren't available the
        # roll advisor will see delta=None and the trigger won't fire — safe.
        try:
            opt_quote = await broker.get_quote(short.contract_symbol)
        except Exception:
            opt_quote = None
        from core.models import OptionContract

        short_contract = OptionContract(
            underlying=position.symbol,
            occ_symbol=short.contract_symbol,
            strike=short.strike,
            expiration=short.expiration or datetime.now(UTC).date(),
            option_type=short.option_type,
            bid=opt_quote.bid if opt_quote else None,
            ask=opt_quote.ask if opt_quote else None,
            delta=None,  # populated below if we can compute it
        )
        # Best-effort delta from BS using current spot + short premium mid.
        from data.greeks import fill_greeks

        if opt_quote is not None and opt_quote.mid is not None:
            greeks = fill_greeks(
                underlying_price=spot,
                strike=short.strike,
                expiration=short_contract.expiration,
                option_type=short.option_type,
                market_price=opt_quote.mid,
                today=datetime.now(UTC).date(),
            )
            if greeks is not None:
                short_contract = short_contract.model_copy(update={"delta": greeks.delta})

        ctx = RollContext(
            symbol=position.symbol,
            short_contract=short_contract,
            short_quantity=max(short.quantity, 1),
            short_premium_collected_per_share=short.fill_price or 0.0,
            current_short_mid=current_mid,
            underlying_price=spot,
            cycle_id=position.current_cycle_id,
        )
        outcome = await roll_orchestrator.evaluate(
            broker=broker,
            repos=repos,
            anthropic=anthropic,
            ctx=ctx,
            position_id=position.id,
            config=config,
            universe=universe,
        )
        # Execute the action through the router (leg-by-leg; no multi-leg
        # primitive in the broker ABC by design).
        if router is not None and outcome.action is not None and not outcome.halted:
            await _execute_roll_action(
                router=router,
                repos=repos,
                config=config,
                outcome=outcome,
                position=position,
                short=short,
                short_contract=short_contract,
            )
        return outcome

    return _eval


async def _execute_roll_action(
    *,
    router: OrderRouter,
    repos: Repos,
    config: dict[str, Any],
    outcome,
    position: Position,
    short: Order,
    short_contract,
):
    """Place the BTC (and optional STO) implied by the roll outcome.

    2026-07-23 review fix: the roll evaluator runs INSIDE reconcile_once,
    BEFORE loop.tick() checks the kill switch — so this path used to open a
    brand-new short (the STO leg) while trading was supposed to be halted.
    The BTC leg is always allowed (risk-reducing); the STO leg now checks
    the stop file + the (Sprint-9-latched) daily_state armed flag itself.
    """
    from strategies.roll_advisor import RollAction
    from strategies.wheel import Proposal

    qty = max(short.quantity, 1)
    if outcome.action in (RollAction.ROLL, RollAction.CLOSE):
        btc = Proposal(
            symbol=position.symbol,
            contract=short_contract,
            order_type=OrderType.BUY_TO_CLOSE,
            quantity=qty,
            rationale=f"roll_advisor:{outcome.action.value}",
            strategy_id=position.strategy_id,
        )
        try:
            btc_result = await router.place(btc)
            log_checkpoint(
                "roll_btc",
                status="ok",
                symbol=position.symbol,
                action=outcome.action.value,
                placed=btc_result.placed is not None,
            )
        except Exception as exc:
            log_checkpoint(
                "roll_btc_fail", status="fail", symbol=position.symbol, error=str(exc)
            )
            return

    if outcome.action == RollAction.ROLL and outcome.rule and outcome.rule.new_contract:
        ks_reason: str | None = None
        risk_cfg = config.get("risk", {}) or {}
        stop_file = risk_cfg.get("stop_file_path")
        if stop_file and Path(stop_file).expanduser().exists():
            ks_reason = f"stop file present: {stop_file}"
        else:
            account_id = (config.get("account") or {}).get("id", "primary")
            row = await repos.daily_state.get(account_id, datetime.now(UTC).date())
            if row is not None and row.kill_switch_armed:
                ks_reason = row.kill_switch_reason or "kill switch armed"
        if ks_reason:
            log_checkpoint(
                "roll_sto_skipped_kill_switch",
                status="skip",
                symbol=position.symbol,
                strategy=position.strategy_id,
                reason=ks_reason,
                note="BTC leg placed; new short NOT opened while halted",
            )
            return

        sto = Proposal(
            symbol=position.symbol,
            contract=outcome.rule.new_contract,
            order_type=OrderType.SELL_TO_OPEN,
            quantity=qty,
            rationale=f"roll_advisor:ROLL credit {outcome.rule.expected_credit_per_share}",
            strategy_id=position.strategy_id,
        )
        try:
            sto_result = await router.place(sto)
            log_checkpoint(
                "roll_sto",
                status="ok",
                symbol=position.symbol,
                placed=sto_result.placed is not None,
                new_strike=outcome.rule.new_contract.strike,
            )
        except Exception as exc:
            log_checkpoint(
                "roll_sto_fail", status="fail", symbol=position.symbol, error=str(exc)
            )


async def _propose_and_route(
    *,
    broker: Broker,
    repos: Repos,
    router: OrderRouter,
    ivr: IVRProvider,
    config: dict[str, Any],
    universe: dict[str, Any],
    strategies: list[StrategyDefinition],
    delta_unavailable_counters: dict[int, int],
):
    """Iterate enabled strategies; dispatch by type to the right orchestrator;
    route each proposal through the router. Per-strategy summary lines so the
    dashboard can show which strategy produced what."""
    total_proposals = 0
    total_placed = 0
    total_blocked = 0
    # 2026-07-01 AI audit item #2: daily LLM exception veto (reduce-only). If
    # today's regime snapshot carries llm_risk_veto, halve ALL strategies' entry
    # sizes for the day (closes unaffected, like every other gate). Read once
    # per tick; any failure reads as no-veto.
    veto_multiplier = 1.0
    regime_snap = None
    try:
        from intelligence.regime_veto import veto_size_multiplier
        regime_snap = await repos.regime.latest()
        veto_multiplier = veto_size_multiplier(regime_snap)
        if veto_multiplier < 1.0:
            log_checkpoint("bot_llm_risk_veto_active", status="ok",
                           size_multiplier=veto_multiplier)
    except Exception:  # noqa: BLE001 — reduce-only: never let the veto read break the loop
        veto_multiplier = 1.0
    # 2026-07-06: regime-aware IVR floor. Hand the latest VIX close to the
    # provider so the selectors' effective_ivr_min can relax the floor when
    # the whole market is rich (IVR reads "low" in sustained high-vol —
    # the rank compares against an elevated 52w high). A missing snapshot
    # reads as no-modulation; calm-regime behavior is unchanged either way.
    try:
        ivr.set_regime_vix(regime_snap.vix_close if regime_snap else None)
    except Exception:  # noqa: BLE001 — context-only: never let the read break the loop
        ivr.set_regime_vix(None)
    for strategy in strategies:
        # TICKET-029: existing positions are MANAGED (closes / stops / rolls)
        # on EVERY tick regardless of the gates below — disabling a strategy, or
        # the drawdown breaker / win-rate floor tripping, must NEVER orphan its
        # open positions (that left a disabled-wheel PLTR CSP unmanaged to
        # -300%). Only NEW ENTRIES are gated. `opens_allowed` aggregates every
        # entry-suppressing gate; closes always run further down.
        opens_allowed = True
        size_multiplier = 1.0
        strategy_universe = universe_for_strategy(strategy, universe)
        if not strategy.enabled:
            opens_allowed = False
        else:
            # Sprint 13 sub-sprint 1 + TICKET-008: tri-state drawdown breaker.
            # DISABLED → no new entries (closes still run). WARNING → size
            # spreads at half (wheels stay qty=1). NORMAL → unchanged.
            drawdown_state = await auto_disable.current_state(repos, strategy.id)
            if drawdown_state is auto_disable.DrawdownState.DISABLED:
                log_checkpoint(
                    "bot_strategy_runtime_disabled",
                    status="skip",
                    strategy=strategy.id,
                    state=drawdown_state.value,
                )
                opens_allowed = False
            else:
                # TICKET-009 + expectancy floor (2026-07-06): independent
                # pause gate. ANY non-NULL pause_state (LOW_WIN_RATE from the
                # win-rate floor, NEGATIVE_EXPECTANCY from the expectancy
                # floor) suppresses entries while drawdown is NORMAL or
                # WARNING. Notify fires once on the NORMAL→PAUSED transition
                # inside the owning floor's check_and_apply (24h limit).
                runtime_row = await repos.strategy_runtime.get(strategy.id) or {}
                pause_state = runtime_row.get("pause_state")
                if pause_state:
                    log_checkpoint(
                        "bot_strategy_runtime_paused",
                        status="skip",
                        strategy=strategy.id,
                        state=pause_state,
                    )
                    opens_allowed = False
                else:
                    size_multiplier = drawdown_state.size_multiplier * veto_multiplier
                    if drawdown_state is auto_disable.DrawdownState.WARNING:
                        log_checkpoint(
                            "bot_strategy_warning_size_halved",
                            status="ok",
                            strategy=strategy.id,
                            size_multiplier=size_multiplier,
                        )
        # A non-empty universe is required to OPEN; closes never need it.
        if opens_allowed and not strategy_universe["tickers"]:
            log_checkpoint(
                "bot_strategy_empty_universe",
                status="skip",
                strategy=strategy.id,
            )
            opens_allowed = False
        # 2026-07-23 review fix: EXCEPTION-ISOLATE the per-strategy dispatch.
        # A single unwrapped raiser (e.g. broker.get_option_chain inside one
        # strategy's open pass) used to abort the whole tick — skipping every
        # DOWNSTREAM strategy's stops, force-closes, and rolls (swing is last
        # in config order, so it lost the most). One strategy's failure now
        # logs + notifies and the loop moves on.
        try:
            if strategy.type == "wheel":
                # Profit-closes first — free up capital before considering new
                # entries. Closes are unaffected by drawdown WARNING; only new
                # entries scale.
                close_proposals = await propose_all_wheel_closes(
                    broker, repos, config, strategy=strategy,
                    delta_unavailable_counters=delta_unavailable_counters,
                )
                open_proposals = []
                if opens_allowed:
                    open_proposals = await propose_all(
                        broker, repos, config, strategy_universe, ivr, strategy=strategy,
                        size_multiplier=size_multiplier,
                    )
                proposals = close_proposals + open_proposals
            elif strategy.type == "vertical_spread":
                # Closes first — free up capital before considering new entries.
                close_proposals = await propose_all_spread_closes(
                    broker, repos, config, strategy=strategy,
                )
                open_proposals = []
                if opens_allowed:
                    open_proposals = await propose_all_spreads(
                        broker, repos, config, strategy_universe,
                        strategy=strategy, ivr=ivr,
                        size_multiplier=size_multiplier,
                    )
                proposals = close_proposals + open_proposals
            elif strategy.type == "iron_condor":
                # TICKET-014: closes reuse propose_all_spread_closes — the close
                # orchestrator detects iron_condor by strategy.id and tags
                # rationale + direction correctly (precursor fix #3). Opens go
                # through the dedicated iron_condor selector.
                close_proposals = await propose_all_spread_closes(
                    broker, repos, config, strategy=strategy,
                )
                open_proposals = []
                if opens_allowed:
                    open_proposals = await propose_all_iron_condor(
                        broker, repos, config, strategy_universe,
                        strategy=strategy, ivr=ivr,
                        size_multiplier=size_multiplier,
                    )
                proposals = close_proposals + open_proposals
            elif strategy.type == "pmcc":
                # TICKET-015: PMCC. Closes first (short profit/time, long roll),
                # then opens (long when IDLE, short when LONG_OPEN). All single-leg
                # Proposals routed via place(). size_multiplier does not apply —
                # PMCC is sized at 1 long + 1 short per position.
                close_proposals = await propose_all_pmcc_closes(
                    broker, repos, config, strategy=strategy,
                )
                open_proposals = []
                if opens_allowed:
                    open_proposals = await propose_all_pmcc(
                        broker, repos, config, strategy_universe,
                        strategy=strategy, ivr=ivr,
                    )
                proposals = close_proposals + open_proposals
            elif strategy.type == "calendar":
                # TICKET-016: Calendar. 2-leg MLEG (same strike, two expirations),
                # net DEBIT. Closes first (25% profit / 2-DTE force-close), then
                # opens (ATM, INVERTED IVR gate — enter only on LOW IV). Both are
                # MultiLegProposals → place_multi_leg. size_multiplier N/A.
                close_proposals = await propose_all_calendar_closes(
                    broker, repos, config, strategy=strategy,
                )
                open_proposals = []
                if opens_allowed:
                    open_proposals = await propose_all_calendar(
                        broker, repos, config, strategy_universe,
                        strategy=strategy, ivr=ivr,
                    )
                proposals = close_proposals + open_proposals
            elif strategy.type == "zero_dte":
                # 0DTE paper test (docs/research/0dte_research_2026-07.md).
                # Closes FIRST and unconditionally — the 15:00 ET hard flatten
                # must fire even when opens are gated (Alpaca rejects expiry-
                # day option orders after ~15:15/15:30 ET and force-liquidates
                # at 15:30/15:45). Opens are additionally gated inside the
                # module by the 10:00-10:15 ET entry window, max-positions-
                # per-day, and the strategy's own penalized daily loss cap.
                close_proposals = await propose_all_zero_dte_closes(
                    broker, repos, config, strategy=strategy,
                )
                open_proposals = []
                if opens_allowed:
                    open_proposals = await propose_all_zero_dte(
                        broker, repos, config, strategy_universe,
                        strategy=strategy, size_multiplier=size_multiplier,
                    )
                proposals = close_proposals + open_proposals
            elif strategy.type == "swing":
                # Directional SPY swing (sub-sprint 2.1: DRY-RUN — evaluates and logs
                # the live signal, places nothing. 2.2 adds deep-ITM long-option entry).
                close_proposals = await propose_all_swing_closes(
                    broker, repos, config, strategy=strategy,
                )
                open_proposals = []
                if opens_allowed:
                    open_proposals = await propose_all_swings(
                        broker, repos, config, strategy_universe,
                        strategy=strategy, size_multiplier=size_multiplier,
                    )
                proposals = close_proposals + open_proposals
            else:
                log_checkpoint(
                    "bot_strategy_unknown_type",
                    status="fail",
                    strategy=strategy.id,
                    type=strategy.type,
                )
                continue
        except Exception as exc:  # noqa: BLE001 — isolate; see comment above
            log_checkpoint(
                "bot_strategy_exception",
                status="fail",
                strategy=strategy.id,
                error=f"{type(exc).__name__}: {exc}",
            )
            try:
                await notify(
                    "bot.strategy_exception",
                    f"{strategy.id} propose pass raised — other strategies unaffected",
                    strategy=strategy.id,
                    error=f"{type(exc).__name__}: {exc}",
                )
            except Exception:  # noqa: BLE001 — never let notify break isolation
                pass
            continue

        placed = 0
        blocked = 0
        for p in proposals:
            try:
                if isinstance(p, MultiLegProposal):
                    result = await router.place_multi_leg(p)
                else:
                    result = await router.place(p)
            except Exception as exc:
                log_checkpoint(
                    "bot_route_exception",
                    status="fail",
                    symbol=p.symbol,
                    strategy=strategy.id,
                    error=str(exc),
                )
                continue
            if result.placed is not None:
                placed += 1
            elif result.risk_failed or result.news_decision == "block":
                blocked += 1
        log_checkpoint(
            "bot_strategy_summary",
            status="ok",
            strategy=strategy.id,
            n_proposals=len(proposals),
            placed=placed,
            blocked=blocked,
        )
        total_proposals += len(proposals)
        total_placed += placed
        total_blocked += blocked

        # Evaluate the drawdown circuit breaker after this strategy's iteration
        # completes. Any cycles closed during this tick are now in the rolling
        # window, so a blow-out trade trips the breaker on the very next tick.
        try:
            await auto_disable.check_and_apply(repos, strategy, config)
        except Exception as exc:  # defensive — never let monitoring break trading
            log_checkpoint(
                "auto_disable_check_failed",
                status="fail",
                strategy=strategy.id,
                error=str(exc),
            )
        # TICKET-009: evaluate the win-rate floor right after drawdown so
        # both signals are re-derived from the freshest cycle data. Independent
        # of drawdown — a single tick can trip drawdown WARNING and win-rate
        # PAUSE in the same iteration if both thresholds breach. Defensive
        # catch — never let monitoring break trading.
        try:
            await win_rate_floor.check_and_apply(repos, strategy, config)
        except Exception as exc:
            log_checkpoint(
                "win_rate_floor_check_failed",
                status="fail",
                strategy=strategy.id,
                error=str(exc),
            )
        # 2026-07-06: expectancy floor — same cadence, same defensive catch.
        # Runs after the win-rate floor; whichever trips first owns the pause
        # (neither overwrites the other).
        try:
            await expectancy_floor.check_and_apply(repos, strategy, config)
        except Exception as exc:
            log_checkpoint(
                "expectancy_floor_check_failed",
                status="fail",
                strategy=strategy.id,
                error=str(exc),
            )
    log_checkpoint(
        "bot_propose_route_summary",
        status="ok",
        n_proposals=total_proposals,
        placed=total_placed,
        blocked=total_blocked,
    )


async def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--once",
        action="store_true",
        help="Run a single tick and exit (smoke-testing).",
    )
    args = parser.parse_args(argv)

    config = load_config()
    setup_logging(config)
    universe = load_universe()

    db_path = Path(config.get("database", {}).get("path", "wheelbot.db")).expanduser()

    # Apply any pending migrations BEFORE we open the long-lived connection.
    # `docker compose up -d --build` historically required a follow-up
    # `python -m scripts.run_migration --all-pending` step that was easy to
    # forget — this auto-heals it. apply_pending() raises on any per-migration
    # failure (transactions are atomic), and we let that propagate so the
    # container exits non-zero rather than running against a partial schema.
    if db_path.exists():
        try:
            from scripts.run_migration import apply_pending
            applied = apply_pending(db_path)
            if applied:
                log_checkpoint(
                    "bot_migrations_applied",
                    status="ok",
                    versions=[m.version for m in applied],
                )
            else:
                log_checkpoint("bot_migrations_up_to_date", status="ok")
        except Exception as exc:
            log_checkpoint("bot_migrations_fail", status="fail", error=str(exc))
            raise
    else:
        log_checkpoint(
            "bot_migrations_skip_no_db",
            status="skip",
            db_path=str(db_path),
            note="DB file missing — run scripts.bootstrap_db before first start",
        )

    db = Database(db_path)
    await db.connect()
    repos = Repos(db)

    broker = make_broker(config)

    # Notifier first so any wiring failures still log loudly.
    notifier = make_notifier(config)
    set_dispatcher(notifier)
    log_checkpoint("bot_notifier", status="ok", name=notifier.name)

    # Anthropic — optional. If the key isn't set, news_check + roll LLM degrade
    # to fail-open / rule-only; the bot still runs.
    anthropic: AnthropicClient | None = None
    if os.environ.get("ANTHROPIC_API_KEY"):
        # strict=True refuses to price unknown models so a typo'd *_model
        # config value or a deprecated model surfaces loudly instead of
        # silently bypassing the daily cap at $0/tok.
        budget = BudgetTracker(repos.llm_decisions, config, strict=True)
        anthropic = AnthropicClient(repos.llm_decisions, budget)
        log_checkpoint(
            "bot_anthropic", status="ok", daily_budget_usd=budget.daily_cap_usd
        )
    else:
        log_checkpoint("bot_anthropic", status="skip", detail="ANTHROPIC_API_KEY unset")

    news = make_news_source(config)
    # 2026-07-06: regime-aware IVR floor knobs. Absent block → delta 0 → the
    # floor never modulates (new-knob convention). Per-strategy
    # ivr_relax_vix / ivr_relax_delta params override these defaults.
    ivr_relax = (config.get("risk") or {}).get("ivr_regime_relax") or {}
    ivr = IVRProvider(
        repos.iv_history,
        relax_vix_threshold=ivr_relax.get("vix_threshold"),
        relax_floor_delta=float(ivr_relax.get("floor_delta", 0.0)),
    )

    news_check_callable = await _make_news_check_callable(news, anthropic, config)
    router = OrderRouter(broker, repos, config, universe, news_checker=news_check_callable)
    roll_evaluator = await _make_roll_evaluator(
        broker, repos, anthropic, config, universe, router=router
    )

    reconciler = Reconciler(
        broker, repos, config, roll_evaluator=roll_evaluator, universe=universe
    )

    strategies = load_strategies(config)
    enabled_strategies = [s for s in strategies if s.enabled]
    log_checkpoint(
        "bot_strategies_loaded",
        status="ok",
        enabled=[s.id for s in enabled_strategies],
        disabled=[s.id for s in strategies if not s.enabled],
    )

    # TICKET-005: process-local counters for the delta-stop "stuck unavailable"
    # detector. dict[position_id, consecutive_failures]. Resets on process
    # restart (acceptable — restart implies someone's looking).
    delta_unavailable_counters: dict[int, int] = {}
    # TICKET-006: rate-limit state for the mid-cycle earnings recheck. Single
    # counter incremented on every _post_tick, only does the per-position
    # work when it crosses risk.earnings_recheck.check_interval_ticks.
    earnings_recheck_state: dict[str, int] = {"ticks_since_check": 0}
    # 2026-07-09: same pattern for the position news sentry.
    news_sentry_state: dict[str, int] = {"ticks_since_check": 0}

    async def _post_tick(_loop: ReconcilerLoop, ks: KillSwitchResult | None):
        kill_switch_tripped = ks is not None and ks.tripped
        # Universe-refresh overlay: when enabled and a watchlist run is APPLIED,
        # refresh strategy membership each tick. Mutates the SHARED universe
        # dict in place — router, risk gate, and reconciler all hold this same
        # object, so membership stays consistent everywhere. Fail-open: any
        # error leaves the previous tick's membership untouched.
        if bool((config.get("universe_refresh") or {}).get("enabled", False)):
            try:
                refreshed = await effective_universe(repos, config)
                universe["tickers"] = refreshed["tickers"]
            except Exception as exc:  # noqa: BLE001 — overlay must never break the loop
                log_checkpoint("bot_universe_overlay_fail", status="fail", error=str(exc))
        # TICKET-006: earnings recheck runs even when the kill switch is
        # tripped — for action='flag_manual' it MUST run (state annotation
        # the operator needs to see), and for action='close' the function
        # itself logs earnings_recheck_close_skipped_kill_switch and skips.
        # Always before _propose_and_route so any flag_manual writes are
        # visible to the proposers (they'd skip MANUAL_INTERVENTION anyway).
        try:
            await check_open_positions_for_new_earnings(
                repos=repos, router=router, config=config,
                kill_switch_tripped=kill_switch_tripped,
                recheck_state=earnings_recheck_state,
            )
        except Exception as exc:  # defensive — recheck must not block trading
            log_checkpoint("earnings_recheck_fail", status="fail", error=str(exc))

        # 2026-07-08: stale-open sweep. Entry orders that never fill leave
        # their positions *_PENDING all day (no re-proposal → the router's
        # stale-replace path never fires for opens). Cancel them at the
        # broker after stale_pending_minutes; the next reconcile restores
        # the position and the entry re-prices from fresh quotes. Runs even
        # under the kill switch — cancelling resting orders reduces risk.
        try:
            await router.cancel_stale_pending_opens()
        except Exception as exc:  # noqa: BLE001 — sweep must never break the tick
            log_checkpoint("stale_open_sweep_fail", status="fail", error=str(exc))

        # 2026-07-09: position news sentry — hourly Haiku read of each open
        # position's headlines. NOTIFY-ONLY (Discord + llm_decisions); never
        # places or closes. Runs even under the kill switch — the operator
        # needs the signal most exactly when trading is halted.
        try:
            await run_position_news_sentry(
                repos=repos, news=news, anthropic=anthropic, config=config,
                sentry_state=news_sentry_state,
            )
        except Exception as exc:  # noqa: BLE001 — sentry must never break the tick
            log_checkpoint("news_sentry_pass_fail", status="fail", error=str(exc))

        if kill_switch_tripped:
            log_checkpoint("bot_skip_kill_switch", status="ok", reason=ks.reason)
            return
        await _propose_and_route(
            broker=broker, repos=repos, router=router, ivr=ivr,
            config=config, universe=universe, strategies=strategies,
            delta_unavailable_counters=delta_unavailable_counters,
        )

    loop = ReconcilerLoop(
        broker, repos, config, reconciler=reconciler, post_tick=_post_tick
    )

    log_checkpoint(
        "bot_started",
        status="ok",
        broker=broker.name,
        notifier=notifier.name,
        anthropic=bool(anthropic),
        news=bool(news),
    )

    try:
        if args.once:
            await loop.tick()
        else:
            await loop.run_forever()
    finally:
        if hasattr(broker, "aclose"):
            await broker.aclose()
        await db.close()
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
