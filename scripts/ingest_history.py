"""Daily ATM 30-day IV snapshot → iv_history table.

Run once per trading day (cron at e.g. 16:30 ET). Two sources:

    --source=broker     Preferred. Pulls today's option chain, picks the contract
                        nearest to 30 DTE and to the underlying spot, uses its
                        IV (broker-supplied or BS-solved from mid).

    --source=realized   Bootstrap fallback. Computes 30-day realized vol from
                        yfinance daily closes. Lower fidelity — IV is forward-
                        looking, realized vol is backward — but lets us seed
                        history quickly while we're paper-trading.

Idempotent: ON CONFLICT(symbol, snapshot_date) DO UPDATE in IvHistoryRepo.

Usage:
    python -m scripts.ingest_history --source=broker
    python -m scripts.ingest_history --source=realized --backfill-days=60
"""

from __future__ import annotations

import argparse
import asyncio
import math
from datetime import UTC, date, datetime, timedelta

from core.broker import Broker
from core.broker_factory import make_broker
from core.checkpoint import checkpoint, configure_logging, log_checkpoint
from core.config import load_config, load_universe
from core.models import IvHistory, OptionType
from data.greeks import fill_greeks
from db.repo import Database, IvHistoryRepo


async def _atm_30d_iv_from_broker(broker: Broker, symbol: str, today: date) -> float | None:
    target = today + timedelta(days=30)
    try:
        quote = await broker.get_quote(symbol)
    except Exception as exc:
        log_checkpoint("ingest_quote_fail", status="fail", symbol=symbol, error=str(exc))
        return None
    spot = quote.mid if quote.mid is not None else (quote.last or quote.bid or quote.ask)
    if spot is None:
        return None

    chain = await broker.get_option_chain(symbol, option_type=OptionType.PUT)
    if not chain:
        return None

    # Closest expiry to 30d, then strike closest to spot.
    chain.sort(
        key=lambda c: (abs((c.expiration - target).days), abs(c.strike - spot))
    )
    contract = chain[0]
    if contract.iv is not None:
        return contract.iv

    market_price = contract.last
    if market_price is None and contract.bid is not None and contract.ask is not None:
        market_price = (contract.bid + contract.ask) / 2
    if market_price is None or market_price <= 0:
        return None
    result = fill_greeks(
        underlying_price=spot,
        strike=contract.strike,
        expiration=contract.expiration,
        option_type=OptionType.PUT,
        market_price=market_price,
        today=today,
    )
    return result.iv if result else None


def _realized_vol_from_yfinance(symbol: str, end: date, window_days: int = 30) -> float | None:
    try:
        import yfinance as yf
    except ImportError:
        log_checkpoint("ingest_yfinance_missing", status="fail")
        return None
    # yfinance is patchy — pull a slightly larger buffer to ensure window_days closes.
    start = end - timedelta(days=window_days * 3)
    df = yf.download(symbol, start=start.isoformat(), end=end.isoformat(), progress=False)
    if df is None or df.empty or "Close" not in df.columns:
        return None
    closes = df["Close"].dropna().tail(window_days + 1)
    if len(closes) < window_days // 2:
        return None
    log_returns = (closes / closes.shift(1)).dropna().apply(math.log)
    daily_std = float(log_returns.std())
    return daily_std * math.sqrt(252)


async def _ingest_broker_source(repos_db: Database, config: dict, symbols: list[str]) -> int:
    broker = make_broker(config)
    repo = IvHistoryRepo(repos_db)
    today = datetime.now(UTC).date()
    written = 0
    for symbol in symbols:
        with checkpoint("ingest_iv", symbol=symbol) as ctx:
            iv = await _atm_30d_iv_from_broker(broker, symbol, today)
            if iv is None:
                ctx["skipped"] = "no_iv"
                continue
            await repo.upsert(
                IvHistory(symbol=symbol, snapshot_date=today, iv_30d=iv)
            )
            ctx["iv"] = round(iv, 4)
            written += 1
    return written


async def _ingest_realized_source(
    repos_db: Database, symbols: list[str], backfill_days: int
) -> int:
    repo = IvHistoryRepo(repos_db)
    today = date.today()
    written = 0
    for symbol in symbols:
        with checkpoint("ingest_iv_realized", symbol=symbol) as ctx:
            for offset in range(backfill_days):
                target = today - timedelta(days=offset)
                iv = _realized_vol_from_yfinance(symbol, target)
                if iv is None:
                    continue
                await repo.upsert(
                    IvHistory(symbol=symbol, snapshot_date=target, iv_30d=iv)
                )
                written += 1
            ctx["written"] = written
    return written


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=["broker", "realized"], default="broker")
    parser.add_argument(
        "--backfill-days",
        type=int,
        default=1,
        help="Realized-vol mode only: how many days to backfill.",
    )
    args = parser.parse_args()

    configure_logging()
    config = load_config()
    universe = load_universe()
    symbols = [t.symbol for t in universe["tickers"]]
    db_path = config.get("database", {}).get("path", "wheelbot.db")

    async with Database(db_path) as db:
        if args.source == "broker":
            written = await _ingest_broker_source(db, config, symbols)
        else:
            written = await _ingest_realized_source(db, symbols, args.backfill_days)
    log_checkpoint("ingest_history_done", status="ok", source=args.source, n_written=written)


if __name__ == "__main__":
    asyncio.run(main())
