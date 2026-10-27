"""Fill-quality report for the mid-fill test (docs/MID_FILL_TEST.md).

    docker exec wheelbot python -m scripts.fill_report
    docker exec wheelbot python -m scripts.fill_report --since 2026-11-02 --json
    docker exec wheelbot python -m scripts.fill_report --probe-feed

Reads the orders table and the quote tape (order_quote_log) read-only and
summarises three order classes for one strategy (default monthly_wheel):

  entries   SELL_TO_OPEN at mid, re-pegged every 15 min
  resting   the resting 50% profit-take (client ids "wb-rest-...")
  urgent    every other BUY_TO_CLOSE (stops, time / earnings / delta, rolls)

Per class: fill rate (per order and, for entries, per symbol-day), time to
fill, where in the spread fills landed, how often the market touched our
limit while it worked, paper fills with no touch (simulator artefacts) and
the mid's drift in the 30 minutes after a fill (adverse selection).

--probe-feed asks Alpaca for one wheel contract's quote on the OPRA feed:
refused means the account sees the free "indicative" feed, so touch rates
measure an approximation of the NBBO, not the NBBO.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import statistics
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

from core.config import load_config

EPS = 1e-9


def _ts(s: str | None) -> datetime | None:
    return datetime.fromisoformat(s) if s else None


def classify(order: dict[str, Any]) -> str | None:
    if order["order_type"] == "SELL_TO_OPEN":
        return "entries"
    if order["order_type"] == "BUY_TO_CLOSE":
        return (
            "resting"
            if str(order.get("client_order_id") or "").startswith("wb-rest-")
            else "urgent"
        )
    return None


def _spread_position(order: dict[str, Any], quote: dict[str, Any] | None) -> float | None:
    """Where a fill landed in the spread it was priced from: 0 = paid the
    whole spread (sold at bid / bought at ask), 0.5 = mid, 1 = the far side."""
    if not quote or order.get("fill_price") is None:
        return None
    bid, ask = quote.get("bid"), quote.get("ask")
    if bid is None or ask is None or ask - bid <= EPS:
        return None
    fill = float(order["fill_price"])
    if order["order_type"] == "SELL_TO_OPEN":
        return (fill - bid) / (ask - bid)
    return (ask - fill) / (ask - bid)


def _touched(order: dict[str, Any], rows: list[dict[str, Any]]) -> bool:
    limit = order.get("limit_price")
    if limit is None:
        return False
    for r in rows:
        if (
            order["order_type"] == "SELL_TO_OPEN"
            and r["bid"] is not None
            and r["bid"] >= limit - EPS
        ):
            return True
        if (
            order["order_type"] == "BUY_TO_CLOSE"
            and r["ask"] is not None
            and r["ask"] <= limit + EPS
        ):
            return True
    return False


def _drift(order: dict[str, Any], rows: list[dict[str, Any]]) -> float | None:
    """$/contract the mid moved against us by the last post-fill tape row
    (positive = adverse: the option rose after we sold it, or fell after we
    bought it back)."""
    after = [r for r in rows if r["bid"] is not None and r["ask"] is not None]
    if not after or order.get("fill_price") is None:
        return None
    mid = (after[-1]["bid"] + after[-1]["ask"]) / 2
    fill = float(order["fill_price"])
    move = mid - fill if order["order_type"] == "SELL_TO_OPEN" else fill - mid
    return move * 100


def _pct(xs: list[float], q: float) -> float | None:
    if not xs:
        return None
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * (len(xs) - 1) + 0.5))]


def summarise(orders: list[dict[str, Any]], tape: list[dict[str, Any]]) -> dict[str, Any]:
    by_order: dict[int, dict[str, list[dict[str, Any]]]] = defaultdict(
        lambda: {"working": [], "post_fill": []}
    )
    for r in tape:
        by_order[r["order_id"]][r["phase"]].append(r)
    out: dict[str, Any] = {}
    for cls in ("entries", "resting", "urgent"):
        rows = [o for o in orders if classify(o) == cls]
        filled = [o for o in rows if o["status"] == "FILLED"]
        ttf = [
            (_ts(o["filled_at"]) - _ts(o["placed_at"])).total_seconds() / 60
            for o in filled
            if o.get("filled_at") and o.get("placed_at")
        ]
        taped = [o for o in rows if by_order[o["id"]]["working"]]
        touched = [o for o in taped if _touched(o, by_order[o["id"]]["working"])]
        filled_taped = [o for o in filled if by_order[o["id"]]["working"]]
        no_touch_fills = [o for o in filled_taped if not _touched(o, by_order[o["id"]]["working"])]
        pos = [
            p
            for o in filled
            if (p := _spread_position(o, (o.get("raw_request") or {}).get("wb_quote"))) is not None
        ]
        drift = [d for o in filled if (d := _drift(o, by_order[o["id"]]["post_fill"])) is not None]
        s: dict[str, Any] = {
            "orders": len(rows),
            "filled": len(filled),
            "fill_rate_per_order": len(filled) / len(rows) if rows else None,
            "minutes_to_fill_median": statistics.median(ttf) if ttf else None,
            "minutes_to_fill_p90": _pct(ttf, 0.9),
            "spread_position_median": statistics.median(pos) if pos else None,
            "orders_with_tape": len(taped),
            "touch_rate": len(touched) / len(taped) if taped else None,
            "fills_without_touch": len(no_touch_fills),
            "fills_with_tape": len(filled_taped),
            "post_fill_drift_usd_mean": statistics.mean(drift) if drift else None,
            "n_drift": len(drift),
        }
        if cls == "entries":
            days = {(o["symbol"], o["placed_at"][:10]) for o in rows}
            filled_days = {(o["symbol"], o["placed_at"][:10]) for o in filled}
            s["symbol_days"] = len(days)
            s["fill_rate_per_symbol_day"] = len(filled_days) / len(days) if days else None
        if cls == "resting":
            shorts = {o["contract_symbol"] for o in rows}
            filled_shorts = {o["contract_symbol"] for o in filled}
            s["positions"] = len(shorts)
            s["positions_closed_by_resting_fill"] = len(filled_shorts)
            s["cancelled_for_urgent_close"] = sum(1 for o in rows if o["status"] == "CANCELLED")
        if cls == "urgent":
            cost = [
                (float(o["fill_price"]) - q["mid"]) * 100
                for o in filled
                if (q := (o.get("raw_request") or {}).get("wb_quote")) and q.get("mid") is not None
            ]
            s["cost_vs_mid_usd_mean"] = statistics.mean(cost) if cost else None
            s["by_trigger"] = dict(
                sorted(
                    (
                        (t, sum(1 for o in rows if (o.get("trigger_reason") or "?") == t))
                        for t in {o.get("trigger_reason") or "?" for o in rows}
                    ),
                )
            )
        out[cls] = s
    return out


def load(db_path: Path, strategy: str, since: str | None) -> tuple[list[dict], list[dict]]:
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        sql = (
            "SELECT * FROM orders WHERE strategy_id = ? "
            "AND order_type IN ('SELL_TO_OPEN','BUY_TO_CLOSE')"
        )
        args: list[Any] = [strategy]
        if since:
            sql += " AND placed_at >= ?"
            args.append(since)
        orders = []
        for r in conn.execute(sql, args):
            d = dict(r)
            d["raw_request"] = json.loads(d["raw_request"]) if d.get("raw_request") else None
            orders.append(d)
        ids = [o["id"] for o in orders]
        tape: list[dict] = []
        if ids:
            marks = ",".join("?" * len(ids))
            tape = [
                dict(r)
                for r in conn.execute(
                    f"SELECT * FROM order_quote_log WHERE order_id IN ({marks}) "
                    "ORDER BY order_id, logged_at",
                    ids,
                )
            ]
        return orders, tape
    finally:
        conn.close()


def probe_feed(contract: str) -> str:
    import os

    from alpaca.data.enums import OptionsFeed
    from alpaca.data.historical.option import OptionHistoricalDataClient
    from alpaca.data.requests import OptionLatestQuoteRequest

    client = OptionHistoricalDataClient(
        os.environ["ALPACA_API_KEY"], os.environ["ALPACA_API_SECRET"]
    )
    lines = []
    for feed in (OptionsFeed.INDICATIVE, OptionsFeed.OPRA):
        try:
            q = client.get_option_latest_quote(
                OptionLatestQuoteRequest(symbol_or_symbols=contract, feed=feed)
            )[contract]
            lines.append(
                f"{feed.value:<10} ok   bid {q.bid_price} ask {q.ask_price} at {q.timestamp}"
            )
        except Exception as exc:  # noqa: BLE001 — the refusal IS the answer
            lines.append(f"{feed.value:<10} refused: {exc}")
    return "\n".join(lines)


def _fmt(v: Any) -> str:
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:.2f}"
    return str(v)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--strategy", default="monthly_wheel")
    ap.add_argument("--since", help="ISO date; only orders placed on/after it")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--probe-feed", action="store_true")
    ap.add_argument("--contract", help="OCC symbol for --probe-feed (default: latest wheel order)")
    args = ap.parse_args(argv)
    config = load_config()
    db_path = Path(config["database"]["path"])
    orders, tape = load(db_path, args.strategy, args.since)
    if args.probe_feed:
        contract = args.contract or next(
            (
                o["contract_symbol"]
                for o in sorted(orders, key=lambda o: o["placed_at"], reverse=True)
                if o.get("contract_symbol")
            ),
            None,
        )
        if not contract:
            print("no contract to probe; pass --contract", file=sys.stderr)
            return 2
        print(f"probing {contract}\n{probe_feed(contract)}")
        return 0
    report = summarise(orders, tape)
    if args.json:
        print(json.dumps(report, indent=2))
        return 0
    print(f"fill report — {args.strategy}" + (f" since {args.since}" if args.since else ""))
    for cls, s in report.items():
        print(f"\n[{cls}]")
        for k, v in s.items():
            print(f"  {k:<34} {_fmt(v)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
