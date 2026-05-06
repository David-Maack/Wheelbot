"""intelligence/news — FinnhubNewsSource against fixture HTTP responses."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

import httpx
import pytest

from intelligence.news import FinnhubNewsSource, NewsSourceUnavailable


def _payload(timestamps: list[datetime]) -> list[dict[str, Any]]:
    return [
        {
            "datetime": int(ts.timestamp()),
            "headline": f"News at {ts.isoformat()}",
            "summary": "summary text",
            "url": "https://example.com",
            "source": "TestSource",
        }
        for ts in timestamps
    ]


def _make_transport(handler):
    return httpx.MockTransport(handler)


@pytest.mark.asyncio
async def test_returns_recent_headlines(monkeypatch):
    now = datetime.now(UTC)
    fresh_ts = [now - timedelta(hours=2), now - timedelta(hours=10)]
    stale_ts = [now - timedelta(hours=72)]

    def handler(request: httpx.Request):
        assert request.url.params.get("symbol") == "F"
        return httpx.Response(200, json=_payload(fresh_ts + stale_ts))

    real_client = httpx.AsyncClient
    monkeypatch.setattr(
        httpx,
        "AsyncClient",
        lambda **kw: real_client(transport=_make_transport(handler), **{k: v for k, v in kw.items() if k != "transport"}),
    )
    src = FinnhubNewsSource(api_key="test")
    headlines = await src.recent("F", hours=24)
    assert len(headlines) == 2  # stale 72h dropped
    assert all(h.symbol == "F" for h in headlines)


@pytest.mark.asyncio
async def test_429_raises_news_source_unavailable(monkeypatch):
    def handler(request):
        return httpx.Response(429, text="rate limited")

    real_client = httpx.AsyncClient
    monkeypatch.setattr(
        httpx,
        "AsyncClient",
        lambda **kw: real_client(transport=_make_transport(handler), **{k: v for k, v in kw.items() if k != "transport"}),
    )
    src = FinnhubNewsSource(api_key="test")
    with pytest.raises(NewsSourceUnavailable):
        await src.recent("F")


@pytest.mark.asyncio
async def test_401_raises_unavailable(monkeypatch):
    def handler(request):
        return httpx.Response(401, json={"error": "Invalid API key"})

    real_client = httpx.AsyncClient
    monkeypatch.setattr(
        httpx,
        "AsyncClient",
        lambda **kw: real_client(transport=_make_transport(handler), **{k: v for k, v in kw.items() if k != "transport"}),
    )
    src = FinnhubNewsSource(api_key="bad")
    with pytest.raises(NewsSourceUnavailable):
        await src.recent("F")


def test_constructor_requires_api_key(monkeypatch):
    monkeypatch.delenv("FINNHUB_API_KEY", raising=False)
    with pytest.raises(NewsSourceUnavailable):
        FinnhubNewsSource()
