"""Daily LLM regime exception veto — reduce-only by construction."""

from __future__ import annotations

from datetime import date

import pytest

from core.models import RegimeSnapshot
from intelligence.regime_veto import (
    build_dossier,
    parse_veto,
    run_regime_veto,
    veto_size_multiplier,
)


def _snap(**over):
    base = dict(snapshot_date=date(2026, 7, 1), regime="BULL_TREND",
                vix_close=17.0, spy_above_sma=True, choppiness=45.0)
    base.update(over)
    return RegimeSnapshot(**base)


def test_parse_veto_fails_open_on_everything_but_explicit_true():
    assert parse_veto({"veto": True, "reason": "bank failure"}) == (True, "bank failure")
    assert parse_veto({"veto": "true"})[0] is True
    assert parse_veto({"veto": False, "reason": "calm"}) == (False, None)
    assert parse_veto({"veto": "maybe"}) == (False, None)
    assert parse_veto({"text": "unparseable prose"}) == (False, None)
    assert parse_veto(None) == (False, None)
    assert parse_veto({}) == (False, None)


def test_veto_size_multiplier_reduce_only():
    assert veto_size_multiplier(None) == 1.0
    assert veto_size_multiplier(_snap()) == 1.0                    # no veto field
    assert veto_size_multiplier(_snap(llm_risk_veto=False)) == 1.0
    assert veto_size_multiplier(_snap(llm_risk_veto=True)) == 0.5


def test_build_dossier_shape():
    d = build_dossier(_snap(), events=[], headlines=[{"headline": "x", "source": "y"}])
    assert d["numeric_regime"]["regime"] == "BULL_TREND"
    assert d["numeric_regime"]["vix_close"] == 17.0
    assert d["headlines"] == [{"headline": "x", "source": "y"}]


@pytest.mark.asyncio
async def test_run_regime_veto_stamps_snapshot(db_repos, monkeypatch):
    """Full path with a stubbed LLM client + stubbed headlines: the verdict is
    written onto today's snapshot row and read back by the sizing hook."""
    today = date(2026, 7, 1)
    await db_repos.regime.insert(_snap(snapshot_date=today, csps_allowed=True))
    monkeypatch.setenv("FINNHUB_API_KEY", "test-key")
    monkeypatch.setattr(
        "intelligence.regime_veto.fetch_general_news",
        lambda *a, **k: [{"headline": "Major bank fails overnight", "source": "t"}],
    )

    class _StubClient:
        async def call(self, **kwargs):
            return {"veto": True, "reason": "bank failure in headlines"}

    verdict = await run_regime_veto(db_repos, {"intelligence": {}}, _StubClient(), today=today)
    assert verdict is True
    snap = await db_repos.regime.get_for_date(today)
    assert snap.llm_risk_veto is True
    assert "bank failure" in (snap.llm_veto_reason or "")
    assert veto_size_multiplier(snap) == 0.5


@pytest.mark.asyncio
async def test_run_regime_veto_fails_open_on_client_error(db_repos, monkeypatch):
    today = date(2026, 7, 1)
    await db_repos.regime.insert(_snap(snapshot_date=today, csps_allowed=True))
    monkeypatch.setenv("FINNHUB_API_KEY", "test-key")
    monkeypatch.setattr(
        "intelligence.regime_veto.fetch_general_news",
        lambda *a, **k: [{"headline": "calm day", "source": "t"}],
    )

    class _BoomClient:
        async def call(self, **kwargs):
            raise RuntimeError("api down")

    verdict = await run_regime_veto(db_repos, {"intelligence": {}}, _BoomClient(), today=today)
    assert verdict is False  # reduce-only: failure = no veto
    snap = await db_repos.regime.get_for_date(today)
    assert veto_size_multiplier(snap) == 1.0
