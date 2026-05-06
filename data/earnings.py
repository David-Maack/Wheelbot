"""Earnings calendar lookup.

Used by `risk/limits.py` rule 4 (earnings blackout) — refuses to enter a CSP
whose expiration falls within the blackout window around an earnings event.

This is a deliberate spec stretch (no §13 ticket); without it, rule 4 has no
data source. Behavior:

- Best-effort yfinance lookup. yfinance is patchy — earnings dates are missing
  for many tickers, scraped from a brittle Yahoo endpoint, and sometimes wrong.
- **Fail-open** when no date is returned. Logged with `status=skip`.
- Result is cached in-process for `cache_ttl_seconds` (default 6h) so a screener
  pass over the universe doesn't hammer yfinance.

Swap this module for a Finnhub-backed implementation when we get a key —
spec §14 #4 names Finnhub as the preferred source. The function signature is
the contract.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import date, datetime, timezone

from core.checkpoint import log_checkpoint


@dataclass(frozen=True, slots=True)
class EarningsLookup:
    symbol: str
    next_date: date | None
    source: str  # "yfinance" | "none"


_CACHE: dict[str, tuple[float, EarningsLookup]] = {}


def _coerce_date(raw: object) -> date | None:
    if raw is None:
        return None
    if isinstance(raw, date) and not isinstance(raw, datetime):
        return raw
    if isinstance(raw, datetime):
        return raw.date()
    try:
        s = str(raw)
        return datetime.fromisoformat(s.split(" ")[0]).date()
    except Exception:
        return None


def _yfinance_next(symbol: str) -> date | None:
    try:
        import yfinance as yf  # noqa: PLC0415 — optional dep; loaded on use
    except ImportError:
        log_checkpoint("earnings_yfinance_missing", status="skip", symbol=symbol)
        return None

    try:
        ticker = yf.Ticker(symbol)
        # yfinance >= 0.2.30 exposes .calendar (DataFrame) and .get_earnings_dates
        try:
            cal = ticker.calendar
        except Exception:
            cal = None
        candidates: list[date] = []
        if cal is not None and hasattr(cal, "get"):
            value = cal.get("Earnings Date")
            if isinstance(value, list):
                for v in value:
                    d = _coerce_date(v)
                    if d:
                        candidates.append(d)
            else:
                d = _coerce_date(value)
                if d:
                    candidates.append(d)

        try:
            df = ticker.get_earnings_dates(limit=8)
        except Exception:
            df = None
        if df is not None and not df.empty:
            for idx in df.index:
                d = _coerce_date(idx)
                if d:
                    candidates.append(d)

        today = datetime.now(timezone.utc).date()
        future = sorted([d for d in candidates if d >= today])
        return future[0] if future else None
    except Exception as exc:
        log_checkpoint(
            "earnings_yfinance_fail", status="fail", symbol=symbol, error=str(exc)
        )
        return None


def next_earnings(
    symbol: str,
    *,
    cache_ttl_seconds: int = 6 * 3600,
    now: float | None = None,
) -> EarningsLookup:
    """Return the next earnings date for `symbol`, or `next_date=None` if unknown.

    Callers MUST treat `next_date is None` as "no data" and fail-open in their
    blackout rule, otherwise we'd halt trading every time yfinance hiccupped.
    """
    now = now if now is not None else time.time()
    cached = _CACHE.get(symbol)
    if cached and now - cached[0] < cache_ttl_seconds:
        return cached[1]

    next_date = _yfinance_next(symbol)
    result = EarningsLookup(
        symbol=symbol,
        next_date=next_date,
        source="yfinance" if next_date else "none",
    )
    _CACHE[symbol] = (now, result)
    log_checkpoint(
        "earnings_lookup",
        status="ok" if next_date else "skip",
        symbol=symbol,
        next_date=str(next_date) if next_date else None,
    )
    return result


def in_blackout(
    symbol: str,
    expiration: date,
    *,
    days_before: int,
    days_after: int,
    today: date | None = None,
) -> bool | None:
    """True if expiration falls inside the blackout window around earnings.

    Returns None when we can't determine the next earnings date — caller decides
    fail-open vs fail-closed. The risk gate uses fail-open (skip rule).
    """
    lookup = next_earnings(symbol)
    if lookup.next_date is None:
        return None
    today = today or datetime.now(timezone.utc).date()
    earnings = lookup.next_date
    # Blackout if expiration is within days_before before or days_after after the event.
    return (earnings - expiration).days <= days_before and (
        expiration - earnings
    ).days <= days_after


def _clear_cache() -> None:
    """Test helper — drop the in-process cache."""
    _CACHE.clear()
