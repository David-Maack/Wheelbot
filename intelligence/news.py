"""News sources for the pre-trade news check.

`NewsSource` ABC + concrete implementations. The news_check module asks a
NewsSource for headlines for a ticker over the last N hours; the source is
free to fail-open (return empty list) on rate limits or transport errors.

Implemented:
    FinnhubNewsSource      — primary; uses /api/v1/company-news endpoint.

Easy to add later (signature stays the same):
    NewsApiOrgNewsSource   — newsapi.org, 100 req/day free.
    AlpacaNewsSource       — uses your existing Alpaca creds, financial focus.
    MarketauxNewsSource    — finance-tagged, 100 req/day free.
    PolygonNewsSource      — financial, ticker-pretagged.

Selection happens at construction time via `make_news_source(config)`.
"""

from __future__ import annotations

import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

import httpx

from core.checkpoint import log_checkpoint


class NewsSourceUnavailable(Exception):
    """Transport / auth / rate-limit failure. Caller should fail-open."""


@dataclass(frozen=True, slots=True)
class Headline:
    symbol: str
    headline: str
    summary: str | None
    url: str | None
    published_at: datetime
    source: str


class NewsSource(ABC):
    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    async def recent(self, symbol: str, *, hours: int = 48) -> list[Headline]: ...


class FinnhubNewsSource(NewsSource):
    """Finnhub /company-news endpoint. Free tier: 60 req/min, plenty for our use."""

    BASE = "https://finnhub.io/api/v1"

    def __init__(
        self,
        api_key: str | None = None,
        *,
        cache_ttl_seconds: int = 1800,
        timeout_seconds: float = 8.0,
    ) -> None:
        self._key = api_key or os.environ.get("FINNHUB_API_KEY")
        if not self._key:
            raise NewsSourceUnavailable("FINNHUB_API_KEY not set")
        self._ttl = cache_ttl_seconds
        self._timeout = timeout_seconds
        self._cache: dict[tuple[str, int], tuple[float, list[Headline]]] = {}

    @property
    def name(self) -> str:
        return "finnhub"

    async def recent(self, symbol: str, *, hours: int = 48) -> list[Headline]:
        key = (symbol.upper(), hours)
        now = time.time()
        cached = self._cache.get(key)
        if cached and now - cached[0] < self._ttl:
            return cached[1]

        end = datetime.now(timezone.utc).date()
        start = end - timedelta(days=max(1, (hours + 23) // 24))
        params = {
            "symbol": symbol.upper(),
            "from": start.isoformat(),
            "to": end.isoformat(),
            "token": self._key,
        }
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                resp = await client.get(f"{self.BASE}/company-news", params=params)
            if resp.status_code in (401, 403):
                raise NewsSourceUnavailable(f"finnhub auth failed: {resp.status_code}")
            if resp.status_code == 429:
                raise NewsSourceUnavailable("finnhub rate-limited (429)")
            resp.raise_for_status()
            payload = resp.json()
        except NewsSourceUnavailable:
            raise
        except httpx.HTTPError as exc:
            raise NewsSourceUnavailable(f"finnhub transport: {exc}") from exc

        cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
        out: list[Headline] = []
        for item in payload or []:
            ts = item.get("datetime")
            if ts is None:
                continue
            published = datetime.fromtimestamp(int(ts), tz=timezone.utc)
            if published < cutoff:
                continue
            out.append(
                Headline(
                    symbol=symbol.upper(),
                    headline=item.get("headline", "")[:240],
                    summary=item.get("summary") or None,
                    url=item.get("url") or None,
                    published_at=published.replace(tzinfo=None),
                    source=item.get("source") or "finnhub",
                )
            )
        out.sort(key=lambda h: h.published_at, reverse=True)
        self._cache[key] = (now, out)
        log_checkpoint(
            "news_recent",
            status="ok",
            source="finnhub",
            symbol=symbol,
            n_headlines=len(out),
        )
        return out


def make_news_source(config: dict[str, Any]) -> NewsSource | None:
    """Pick a configured NewsSource. None if no source can be built — caller
    fails-open."""
    intel = config.get("intelligence", {}) or {}
    name = (intel.get("news_source") or "finnhub").lower()
    if name == "finnhub":
        try:
            return FinnhubNewsSource()
        except NewsSourceUnavailable:
            log_checkpoint("news_source_unavailable", status="skip", source="finnhub")
            return None
    log_checkpoint("news_source_unknown", status="skip", source=name)
    return None
