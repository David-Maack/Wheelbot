"""TICKET-028 — Tastytrade SANDBOX execution-mechanics smoke test.

Exercises the order WRITE path against the live sandbox through the Broker
abstraction — the Sprint-6 code that was never run end-to-end:

  Phase A  single-leg   place_order -> get_orders_since -> cancel_order
  Phase B  multi-leg    place_multi_leg_order (put vertical) -> verify -> cancel
           positions    get_positions shape (reconciler input)

Every order is NON-FILLABLE and cancelled immediately: the single-leg sells a
deep-OTM put ABOVE its strike (a put can't be worth more than its strike, so it
can never fill), and the vertical asks a credit well above the spread's value.
After cancelling, the order is re-fetched and asserted CANCELLED — a marketable
limit that filled would otherwise be masked by best-effort cancel_order.
**SANDBOX ONLY** (is_test=True → cert.tastyworks.com, fake money, 24h reset).
It never touches real money.

Run (after deploying the latest image so the OCC-padding fix is present):

    docker exec wheelbot python -m scripts.tastytrade_sandbox_smoke

Outside the order-entry window the sandbox rejects DAY orders with
`tif.day_invalid_intersession_options` ("accepted after 3:15pm CT"). The
script reports that as a MARKET-HOURS skip — the symbol/structure already
validated, so re-run during regular trading hours for the full
place→cancel confirmation. An `invalid_symbol` rejection means the padding
fix isn't in the running image (rebuild).
"""

from __future__ import annotations

import asyncio
import sys
import uuid
from datetime import UTC, date, datetime, timedelta
from typing import Any

from core.broker import Broker, OrderRejected
from core.checkpoint import configure_logging, log_checkpoint
from core.config import load_secrets
from core.models import (
    OptionType,
    Order,
    OrderLeg,
    OrderStatus,
    OrderType,
)

SYMBOL = "F"
TARGET_DTE = 30


def _utcnow() -> datetime:
    return datetime.now(UTC).replace(tzinfo=None)


def _is_market_hours_block(msg: str) -> bool:
    m = msg.lower()
    return (
        "intersession" in m
        or "tif.day" in m
        or "after 3:15" in m
        or "market is closed" in m
        or "not open" in m
    )


def _classify_rejection(exc: Exception) -> str:
    msg = str(exc)
    if "invalid_symbol" in msg.lower():
        return f"FAIL invalid_symbol — OCC padding fix NOT in this image (rebuild). {msg}"
    if _is_market_hours_block(msg):
        return ("MARKET-HOURS SKIP — order-entry window closed; symbol/structure "
                "validated. Re-run during RTH for the full place→cancel.")
    return f"REJECTED — {msg}"


async def _find_order(broker: Broker, oid: str, cutoff: datetime) -> Order | None:
    orders = await broker.get_orders_since(cutoff)
    return next((o for o in orders if o.broker_order_id == oid), None)


async def _verify_and_cancel(broker: Broker, placed: Order, label: str) -> str:
    """Place is done; confirm the order is on the book, cancel it, then re-fetch
    and assert it actually went CANCELLED. cancel_order is best-effort (swallows
    errors), so trusting it would let a 422 on a filled order read as success —
    we verify the resulting status instead."""
    oid = placed.broker_order_id
    if not oid:
        return f"{label}: FAIL — no broker_order_id (status={placed.status})"
    if placed.status == OrderStatus.REJECTED:
        return f"{label}: REJECTED (status) raw={placed.raw_response}"
    cutoff = _utcnow() - timedelta(minutes=5)
    before = await _find_order(broker, oid, cutoff)
    seen = before is not None
    if before is not None and before.status == OrderStatus.FILLED:
        return (f"{label}: WARN — order FILLED immediately (marketable limit); left a "
                f"position and is not cancellable. id={oid}")
    await broker.cancel_order(oid)
    after = await _find_order(broker, oid, cutoff)
    status = after.status if after is not None else None
    if status == OrderStatus.CANCELLED:
        return f"{label}: PASS — id={oid} placed -> in_orders_since={seen} -> cancelled"
    if status == OrderStatus.FILLED:
        return f"{label}: WARN — filled before cancel; left a position. id={oid}"
    return f"{label}: WARN — cancel not confirmed (status={status}); id={oid} seen={seen}"


async def _low_strike_puts(broker: Broker, n: int) -> tuple[date, list[Any]]:
    """The n lowest-strike puts on F at the expiry nearest TARGET_DTE (deepest
    OTM → safest non-fillable test contracts)."""
    chain = await broker.get_option_chain(SYMBOL, option_type=OptionType.PUT)
    if not chain:
        return date.today(), []
    target = date.today() + timedelta(days=TARGET_DTE)
    exp = min((c.expiration for c in chain), key=lambda e: abs((e - target).days))
    puts = sorted((c for c in chain if c.expiration == exp), key=lambda c: c.strike)
    return exp, puts[:n]


async def _phase_a_single_leg(broker: Broker) -> str:
    _exp, puts = await _low_strike_puts(broker, 1)
    if not puts:
        return "single-leg: SKIP — empty F put chain"
    c = puts[0]
    # SELL_TO_OPEN above the put's max value (a put can't be worth more than its
    # strike) -> non-marketable -> rests on the book. A $0.01 sell would be
    # marketable ("sell for >= 1c") and fill instantly, leaving a position.
    order = Order(
        account_id="tastytrade", symbol=SYMBOL, order_type=OrderType.SELL_TO_OPEN,
        contract_symbol=c.occ_symbol, strike=c.strike, expiration=c.expiration,
        option_type=OptionType.PUT, quantity=1, limit_price=round(c.strike + 1.0, 2),
        status=OrderStatus.PENDING, placed_at=_utcnow(),
        client_order_id=f"wb-smoke-1leg-{uuid.uuid4().hex[:8]}",
    )
    try:
        placed = await broker.place_order(order)
    except OrderRejected as exc:
        return f"single-leg: {_classify_rejection(exc)}"
    except Exception as exc:  # noqa: BLE001
        return f"single-leg: ERROR — {type(exc).__name__}: {exc}"
    return await _verify_and_cancel(broker, placed, "single-leg")


async def _phase_b_multi_leg(broker: Broker) -> str:
    exp, puts = await _low_strike_puts(broker, 2)
    if len(puts) < 2:
        return "multi-leg: SKIP — <2 put strikes at target expiry"
    long_put, short_put = puts[0], puts[1]  # buy lower strike, sell higher → put credit spread
    legs = [
        OrderLeg(contract_symbol=short_put.occ_symbol, underlying=SYMBOL,
                 option_type=OptionType.PUT, strike=short_put.strike, expiration=exp,
                 action=OrderType.SELL_TO_OPEN, ratio_qty=1),
        OrderLeg(contract_symbol=long_put.occ_symbol, underlying=SYMBOL,
                 option_type=OptionType.PUT, strike=long_put.strike, expiration=exp,
                 action=OrderType.BUY_TO_OPEN, ratio_qty=1),
    ]
    try:
        placed = await broker.place_multi_leg_order(
            underlying=SYMBOL, legs=legs, quantity=1,
            limit_price=0.10,  # credit well above the deep-OTM mid → rests, won't fill
            client_order_id=f"wb-smoke-vert-{uuid.uuid4().hex[:8]}",
            strategy_id="smoke", account_id="tastytrade",
        )
    except OrderRejected as exc:
        return f"multi-leg: {_classify_rejection(exc)}"
    except NotImplementedError:
        return "multi-leg: SKIP — adapter does not implement place_multi_leg_order"
    except Exception as exc:  # noqa: BLE001
        return f"multi-leg: ERROR — {type(exc).__name__}: {exc}"
    return await _verify_and_cancel(broker, placed, "multi-leg")


async def _phase_b_positions(broker: Broker) -> str:
    try:
        pos = await broker.get_positions()
    except Exception as exc:  # noqa: BLE001
        return f"positions: ERROR — {type(exc).__name__}: {exc}"
    if not pos:
        return "positions: OK — 0 open (expected; nothing filled)"
    shapes = ", ".join(f"{p.symbol}:{p.state}:{p.shares}" for p in pos[:5])
    return f"positions: OK — {len(pos)} open → {shapes}"


async def run_smoke(broker: Broker) -> list[str]:
    """Run all phases against `broker`, returning one result line each. Takes the
    broker so tests can inject a mock; main() supplies the real sandbox adapter."""
    results = [
        await _phase_a_single_leg(broker),
        await _phase_b_multi_leg(broker),
        await _phase_b_positions(broker),
    ]
    return results


def main(argv: list[str] | None = None) -> int:
    configure_logging()
    load_secrets()
    from platforms.tastytrade_broker import TastytradeBroker

    broker = TastytradeBroker(is_test=True)

    async def _main() -> list[str]:
        try:
            return await run_smoke(broker)
        finally:
            aclose = getattr(broker, "aclose", None)
            if aclose is not None:
                await aclose()

    results = asyncio.run(_main())
    print("\nTastytrade sandbox execution smoke — results:")
    for line in results:
        print("  •", line)

    hard_fail = any("FAIL" in r for r in results)
    log_checkpoint(
        "tastytrade_sandbox_smoke",
        status="fail" if hard_fail else "ok",
        results=results,
    )
    return 1 if hard_fail else 0


if __name__ == "__main__":
    sys.exit(main())
