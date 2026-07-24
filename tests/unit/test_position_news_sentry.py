"""intelligence/position_news_sentry — notify-only mid-cycle news check."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pytest

from core.models import Position, PositionState
from intelligence.news import Headline, NewsSourceUnavailable
from intelligence.position_news_sentry import run_position_news_sentry


def _utc() -> datetime:
    return datetime.now(UTC).replace(tzinfo=None)


def _headline(symbol: str = "TSLA", text: str = "TSLA cuts guidance") -> Headline:
    return Headline(symbol=symbol, headline=text, summary="details",
                    url=None, published_at=_utc(), source="finnhub")


class _StubNews:
    def __init__(self, headlines: dict[str, list[Headline]] | None = None,
                 unavailable: bool = False):
        self._headlines = headlines or {}
        self._unavailable = unavailable
        self.calls: list[str] = []

    @property
    def name(self) -> str:
        return "stub"

    async def recent(self, symbol: str, *, hours: int = 48) -> list[Headline]:
        self.calls.append(symbol)
        if self._unavailable:
            raise NewsSourceUnavailable("stub down")
        return self._headlines.get(symbol, [])


class _StubAnthropic:
    """Returns a fixed parsed verdict; records call contexts. `fail_symbols`
    raise to exercise the per-position fail-open."""

    def __init__(self, verdict: str = "hold", confidence: float = 0.8,
                 fail_symbols: set[str] | None = None):
        self._verdict = verdict
        self._confidence = confidence
        self._fail = fail_symbols or set()
        self.calls: list[dict[str, Any]] = []

    async def call(self, *, decision_type, model, system, user_payload,
                   max_output_tokens, context) -> dict[str, Any]:
        if context.get("symbol") in self._fail:
            raise RuntimeError("stub LLM boom")
        self.calls.append({"context": context, "payload": user_payload})
        return {"parsed": {"verdict": self._verdict,
                           "confidence": self._confidence,
                           "rationale": "stub rationale"}}


def _config(**overrides: Any) -> dict[str, Any]:
    base = {
        "account": {"id": "test"},
        "intelligence": {
            "position_news_sentry": {
                "enabled": True,
                "check_interval_ticks": 1,   # tests: every call does work
                "lookback_hours": 24,
                "notify_cooldown_hours": 4,
            },
        },
    }
    base["intelligence"]["position_news_sentry"].update(overrides)
    return base


async def _seed_open(db_repos, *, symbol: str,
                     state: PositionState = PositionState.SPREAD_OPEN,
                     strategy_id: str = "put_spread") -> int:
    return await db_repos.positions.insert(
        Position(account_id="test", symbol=symbol, strategy_id=strategy_id,
                 state=state, shares=0, state_changed_at=_utc())
    )


async def _no_notify(*args, **kwargs):
    return None


@pytest.mark.asyncio
async def test_disabled_is_noop(db_repos):
    news = _StubNews({"TSLA": [_headline()]})
    llm = _StubAnthropic()
    await _seed_open(db_repos, symbol="TSLA")
    results = await run_position_news_sentry(
        repos=db_repos, news=news, anthropic=llm,
        config=_config(enabled=False), sentry_state={"ticks_since_check": 0},
    )
    assert results == []
    assert news.calls == [] and llm.calls == []


@pytest.mark.asyncio
async def test_interval_gating(db_repos):
    news = _StubNews({"TSLA": [_headline()]})
    llm = _StubAnthropic()
    await _seed_open(db_repos, symbol="TSLA")
    state = {"ticks_since_check": 0}
    cfg = _config(check_interval_ticks=3)
    for _ in range(2):  # ticks 1-2: below the interval — no work
        assert await run_position_news_sentry(
            repos=db_repos, news=news, anthropic=llm, config=cfg, sentry_state=state,
        ) == []
    assert news.calls == []
    results = await run_position_news_sentry(  # tick 3 fires
        repos=db_repos, news=news, anthropic=llm, config=cfg, sentry_state=state,
    )
    assert len(results) == 1
    assert state["ticks_since_check"] == 0  # reset for the next window


@pytest.mark.asyncio
async def test_hold_verdict_no_notify(db_repos, monkeypatch):
    notified = []

    async def _capture(*args, **kw):
        notified.append(kw)

    monkeypatch.setattr("intelligence.position_news_sentry.notify", _capture)
    news = _StubNews({"TSLA": [_headline()]})
    llm = _StubAnthropic(verdict="hold")
    await _seed_open(db_repos, symbol="TSLA")
    results = await run_position_news_sentry(
        repos=db_repos, news=news, anthropic=llm,
        config=_config(), sentry_state={"ticks_since_check": 0},
    )
    assert len(results) == 1 and results[0].verdict == "hold"
    assert notified == []


@pytest.mark.asyncio
async def test_alert_notifies_once_within_cooldown(db_repos, monkeypatch):
    notified = []

    async def _capture(*args, **kw):
        notified.append(kw)

    monkeypatch.setattr("intelligence.position_news_sentry.notify", _capture)
    news = _StubNews({"TSLA": [_headline()]})
    llm = _StubAnthropic(verdict="exit_advisory", confidence=0.9)
    await _seed_open(db_repos, symbol="TSLA")
    cfg = _config()
    state = {"ticks_since_check": 0}
    r1 = await run_position_news_sentry(
        repos=db_repos, news=news, anthropic=llm, config=cfg, sentry_state=state,
    )
    r2 = await run_position_news_sentry(
        repos=db_repos, news=news, anthropic=llm, config=cfg, sentry_state=state,
    )
    assert r1[0].verdict == "exit_advisory" and r2[0].verdict == "exit_advisory"
    assert len(notified) == 1  # second pass inside the 4h per-position cooldown
    assert notified[0]["verdict"] == "exit_advisory"


@pytest.mark.asyncio
async def test_no_headlines_no_llm_spend(db_repos):
    news = _StubNews({"TSLA": []})
    llm = _StubAnthropic()
    await _seed_open(db_repos, symbol="TSLA")
    results = await run_position_news_sentry(
        repos=db_repos, news=news, anthropic=llm,
        config=_config(), sentry_state={"ticks_since_check": 0},
    )
    assert results == []
    assert llm.calls == []


@pytest.mark.asyncio
async def test_news_source_down_fails_open(db_repos, monkeypatch):
    monkeypatch.setattr("intelligence.position_news_sentry.notify", _no_notify)
    news = _StubNews(unavailable=True)
    llm = _StubAnthropic()
    await _seed_open(db_repos, symbol="TSLA")
    results = await run_position_news_sentry(
        repos=db_repos, news=news, anthropic=llm,
        config=_config(), sentry_state={"ticks_since_check": 0},
    )
    assert results == [] and llm.calls == []


@pytest.mark.asyncio
async def test_one_bad_symbol_does_not_kill_the_pass(db_repos, monkeypatch):
    monkeypatch.setattr("intelligence.position_news_sentry.notify", _no_notify)
    news = _StubNews({"TSLA": [_headline("TSLA")], "AAPL": [_headline("AAPL")]})
    llm = _StubAnthropic(verdict="hold", fail_symbols={"TSLA"})
    await _seed_open(db_repos, symbol="AAPL")
    await _seed_open(db_repos, symbol="TSLA")
    results = await run_position_news_sentry(
        repos=db_repos, news=news, anthropic=llm,
        config=_config(), sentry_state={"ticks_since_check": 0},
    )
    assert [r.symbol for r in results] == ["AAPL"]


@pytest.mark.asyncio
async def test_unknown_verdict_normalizes_to_hold(db_repos, monkeypatch):
    monkeypatch.setattr("intelligence.position_news_sentry.notify", _no_notify)
    news = _StubNews({"TSLA": [_headline()]})
    llm = _StubAnthropic(verdict="PANIC SELL EVERYTHING")
    await _seed_open(db_repos, symbol="TSLA")
    results = await run_position_news_sentry(
        repos=db_repos, news=news, anthropic=llm,
        config=_config(), sentry_state={"ticks_since_check": 0},
    )
    assert results[0].verdict == "hold"


@pytest.mark.asyncio
async def test_only_open_states_watched(db_repos):
    news = _StubNews({"TSLA": [_headline("TSLA")], "AAPL": [_headline("AAPL")],
                      "SOFI": [_headline("SOFI")]})
    llm = _StubAnthropic()
    await _seed_open(db_repos, symbol="TSLA", state=PositionState.SPREAD_OPEN)
    await _seed_open(db_repos, symbol="AAPL", state=PositionState.SPREAD_PENDING)
    await _seed_open(db_repos, symbol="SOFI", state=PositionState.MANUAL_INTERVENTION)
    await run_position_news_sentry(
        repos=db_repos, news=news, anthropic=llm,
        config=_config(), sentry_state={"ticks_since_check": 0},
    )
    assert news.calls == ["TSLA"]  # pendings and flagged positions skipped


@pytest.mark.asyncio
async def test_context_carries_cycle_id_for_hit_rate_join(db_repos, monkeypatch):
    monkeypatch.setattr("intelligence.position_news_sentry.notify", _no_notify)
    from core.models import WheelCycle
    cycle_id = await db_repos.cycles.insert(
        WheelCycle(account_id="test", symbol="TSLA", strategy_id="put_spread",
                   started_at=_utc())
    )
    await db_repos.positions.insert(
        Position(account_id="test", symbol="TSLA", strategy_id="put_spread",
                 state=PositionState.SPREAD_OPEN, shares=0,
                 current_cycle_id=cycle_id, state_changed_at=_utc())
    )
    news = _StubNews({"TSLA": [_headline()]})
    llm = _StubAnthropic(verdict="alert")
    await run_position_news_sentry(
        repos=db_repos, news=news, anthropic=llm,
        config=_config(), sentry_state={"ticks_since_check": 0},
    )
    assert llm.calls[0]["context"]["cycle_id"] == cycle_id
    assert llm.calls[0]["payload"]["position"]["symbol"] == "TSLA"
