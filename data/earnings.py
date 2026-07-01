"""Earnings calendar lookup with pluggable backends.

Used by `risk/limits.py` rule 4 (earnings blackout) and the LLM screener
(intelligence/screener.py).

Source preference order:
    1. Finnhub (if FINNHUB_API_KEY present) — paid-grade reliability on the
       free tier; rate-limited to 60 req/min.
    2. yfinance — last-resort fallback. Brittle scraping endpoint; many
       tickers return nothing.

Both fail-open: if the source returns nothing, `next_earnings()` returns
`next_date=None` and callers treat it as "no data" (skip the rule, not block).

Result is cached in-process for `cache_ttl_seconds` (default 6h) so a screener
pass over the universe doesn't hammer the upstream.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone

import httpx

from core.checkpoint import log_checkpoint


@dataclass(frozen=True, slots=True)
class EarningsLookup:
    symbol: str
    next_date: date | None
    source: str  # "finnhub" | "yfinance" | "none"


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


def _finnhub_next(symbol: str) -> date | None:
    api_key = os.environ.get("FINNHUB_API_KEY")
    if not api_key:
        return None
    today = datetime.now(timezone.utc).date()
    end = today.replace(year=today.year + 1) if today.month != 12 else date(today.year + 1, 12, 31)
    params = {
        "from": today.isoformat(),
        "to": end.isoformat(),
        "symbol": symbol.upper(),
        "token": api_key,
    }
    try:
        resp = httpx.get("https://finnhub.io/api/v1/calendar/earnings", params=params, timeout=8.0)
        if resp.status_code in (401, 403):
            log_checkpoint("earnings_finnhub_auth", status="fail", symbol=symbol)
            return None
        if resp.status_code == 429:
            log_checkpoint("earnings_finnhub_rate_limit", status="skip", symbol=symbol)
            return None
        resp.raise_for_status()
        payload = resp.json()
    except httpx.HTTPError as exc:
        log_checkpoint("earnings_finnhub_fail", status="fail", symbol=symbol, error=str(exc))
        return None

    earnings_list = payload.get("earningsCalendar") or []
    candidates: list[date] = []
    for item in earnings_list:
        raw = item.get("date")
        d = _coerce_date(raw)
        if d and d >= today:
            candidates.append(d)
    return min(candidates) if candidates else None


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

    # Source preference: Finnhub if key is present, else yfinance.
    next_date = _finnhub_next(symbol)
    source = "finnhub" if next_date else None
    if next_date is None:
        next_date = _yfinance_next(symbol)
        source = "yfinance" if next_date else "none"
    assert source is not None
    result = EarningsLookup(symbol=symbol, next_date=next_date, source=source)
    _CACHE[symbol] = (now, result)
    log_checkpoint(
        "earnings_lookup",
        status="ok" if next_date else "skip",
        symbol=symbol,
        next_date=str(next_date) if next_date else None,
        source=source,
    )
    return result


def in_blackout(
    symbol: str,
    expiration: date,
    *,
    days_before: int,
    days_after: int,
    today: date | None = None,
    block_if_spans: bool = True,
    entry_avoid_days: int | None = None,
) -> bool | None:
    """True if opening a position expiring on `expiration` conflicts with the
    symbol's next earnings date. Two independent triggers (either one → True):

      1. Expiration sits inside the window AROUND the event:
         earnings - days_before  <=  expiration  <=  earnings + days_after.
      2. `block_if_spans` and the position would be HELD THROUGH earnings —
         the event falls in (today, expiration]. This catches a position that
         expires well AFTER earnings (outside window #1) but is short gamma
         across the event, e.g. earnings 2026-07-27 with a 2026-07-31 expiry.
         Pre-TICKET-030 only trigger #1 ran, so earnings-spanning spreads
         slipped through entry and were caught (too late) by the mid-cycle
         recheck → MANUAL_INTERVENTION churn.

    Returns None when we can't determine the next earnings date — caller decides
    fail-open vs fail-closed. The risk gate uses fail-open (skip rule).
    """
    lookup = next_earnings(symbol)
    if lookup.next_date is None:
        return None
    today = today or datetime.now(timezone.utc).date()
    earnings = lookup.next_date
    # Entry-proximity mode (when entry_avoid_days is set): only avoid OPENING
    # when earnings is imminent from TODAY. The 21-DTE time-close means defined-
    # risk spreads exit well before an expiry-date earnings, so the expiry-based
    # triggers below are over-cautious in earnings season (they block nearly
    # every 30-45 DTE spread whose life merely reaches late-July earnings). Here
    # we block only when earnings lands within the next `entry_avoid_days` days.
    # `today <= earnings`, NOT `<`: a same-day report (e.g. AMC tonight) must
    # block — the strict comparison let the bot sell premium hours before the gap.
    if entry_avoid_days is not None:
        return today <= earnings <= today + timedelta(days=entry_avoid_days)
    near_expiry = (earnings - expiration).days <= days_before and (
        expiration - earnings
    ).days <= days_after
    spans = block_if_spans and today <= earnings <= expiration
    return near_expiry or spans


def _clear_cache() -> None:
    """Test helper — drop the in-process cache."""
    _CACHE.clear()
