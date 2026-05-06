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
from core.notify import make_notifier, set_dispatcher
from data.ivr import IVRProvider
from db.repo import Database, Repos
from execution.kill_switch import KillSwitchResult
from execution.loop import ReconcilerLoop
from execution.reconciler import Reconciler
from execution.router import OrderRouter
from intelligence.anthropic_client import AnthropicClient
from intelligence.budget import BudgetTracker
from intelligence.news import make_news_source
from intelligence.news_check import news_check as run_news_check
from strategies import roll_orchestrator
from strategies.roll_advisor import RollAction, RollContext
from strategies.wheel import propose_all


def _utcnow() -> datetime:
    return datetime.now(UTC).replace(tzinfo=None)


async def _make_news_check_callable(news, anthropic, config):
    """Returns the callable the OrderRouter expects, or None when disabled."""
    intel = config.get("intelligence", {}) or {}
    if not bool(intel.get("llm_news_check_enabled", True)):
        return None
    if anthropic is None or news is None:
        return None

    async def _check(symbol: str):
        return await run_news_check(symbol=symbol, news=news, anthropic=anthropic, config=config)

    return _check


async def _make_roll_evaluator(
    broker: Broker, repos: Repos, anthropic, config, universe
):
    """Returns the callable the Reconciler's roll-trigger scan invokes."""

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
        return outcome

    return _eval


async def _execute_roll_outcomes(
    *,
    broker: Broker,
    repos: Repos,
    router: OrderRouter,
    config: dict[str, Any],
):
    """After a reconcile pass, walk active positions whose roll_evaluator
    returned a non-None action and place the corresponding orders. Today the
    evaluator runs inside the reconciler and only stores the outcome via
    log_checkpoint + state_log; this hook is where future multi-leg roll
    execution would land. Punt-marker: leg-by-leg execution is deliberately
    not wired here because the orchestrator's halt-on-disagreement path
    already serves the common case (rules + LLM both off → rule-only)."""
    return  # see docstring


async def _propose_and_route(
    *,
    broker: Broker,
    repos: Repos,
    router: OrderRouter,
    ivr: IVRProvider,
    config: dict[str, Any],
    universe: dict[str, Any],
):
    proposals = await propose_all(broker, repos, config, universe, ivr)
    placed = 0
    blocked = 0
    for p in proposals:
        try:
            result = await router.place(p)
        except Exception as exc:
            log_checkpoint(
                "bot_route_exception",
                status="fail",
                symbol=p.symbol,
                error=str(exc),
            )
            continue
        if result.placed is not None:
            placed += 1
        elif result.risk_failed or result.news_decision == "block":
            blocked += 1
    log_checkpoint(
        "bot_propose_route_summary",
        status="ok",
        n_proposals=len(proposals),
        placed=placed,
        blocked=blocked,
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
        budget = BudgetTracker(repos.llm_decisions, config)
        anthropic = AnthropicClient(repos.llm_decisions, budget)
        log_checkpoint(
            "bot_anthropic", status="ok", daily_budget_usd=budget.daily_cap_usd
        )
    else:
        log_checkpoint("bot_anthropic", status="skip", detail="ANTHROPIC_API_KEY unset")

    news = make_news_source(config)
    ivr = IVRProvider(repos.iv_history)

    news_check_callable = await _make_news_check_callable(news, anthropic, config)
    roll_evaluator = await _make_roll_evaluator(broker, repos, anthropic, config, universe)

    router = OrderRouter(broker, repos, config, universe, news_checker=news_check_callable)

    reconciler = Reconciler(
        broker, repos, config, roll_evaluator=roll_evaluator, universe=universe
    )

    async def _post_tick(_loop: ReconcilerLoop, ks: KillSwitchResult | None):
        if ks is not None and ks.tripped:
            log_checkpoint("bot_skip_kill_switch", status="ok", reason=ks.reason)
            return
        await _propose_and_route(
            broker=broker, repos=repos, router=router, ivr=ivr,
            config=config, universe=universe,
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
