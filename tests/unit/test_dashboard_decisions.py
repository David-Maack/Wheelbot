"""dashboard /decisions view."""

from __future__ import annotations

import base64
from datetime import UTC, datetime
from typing import Any

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from core.models import LlmDecision, LlmDecisionType
from dashboard.app import DashboardDeps, build_app
from platforms.paper_broker import PaperBroker


def _auth() -> dict[str, str]:
    return {"Authorization": "Basic " + base64.b64encode(b"wheelbot:hunter2").decode()}


@pytest_asyncio.fixture
async def app_client(db_repos, tmp_path):
    deps = DashboardDeps(
        repos=db_repos,
        broker=PaperBroker(cash=20_000),
        config={
            "account": {"id": "test"},
            "dashboard": {"basic_auth_user": "wheelbot"},
            "intelligence": {"daily_budget_usd": 1.0},
            "risk": {"stop_file_path": str(tmp_path / "STOP")},
        },
        auth_user="wheelbot",
        auth_password="hunter2",
    )
    app = build_app(deps)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        yield client, deps


@pytest.mark.asyncio
async def test_decisions_empty_view(app_client):
    client, _deps = app_client
    resp = await client.get("/decisions", headers=_auth())
    assert resp.status_code == 200
    assert "No LLM decisions yet" in resp.text


@pytest.mark.asyncio
async def test_decisions_lists_recent(app_client):
    client, deps = app_client
    now = datetime.now(UTC).replace(tzinfo=None)
    await deps.repos.llm_decisions.insert(
        LlmDecision(
            decision_type=LlmDecisionType.NEWS_CHECK,
            model="claude-haiku-4-5",
            decision="block",
            cost_usd=0.05,
            tokens_in=200,
            tokens_out=50,
            created_at=now,
        )
    )
    resp = await client.get("/decisions", headers=_auth())
    assert resp.status_code == 200
    assert "block" in resp.text
    assert "claude-haiku-4-5" in resp.text


@pytest.mark.asyncio
async def test_decisions_filter_by_decision(app_client):
    client, deps = app_client
    now = datetime.now(UTC).replace(tzinfo=None)
    for verdict in ("proceed", "block", "caution"):
        await deps.repos.llm_decisions.insert(
            LlmDecision(
                decision_type=LlmDecisionType.NEWS_CHECK,
                model="claude-haiku-4-5",
                decision=verdict,
                cost_usd=0.01,
                created_at=now,
            )
        )
    resp = await client.get("/decisions?decision=block", headers=_auth())
    assert resp.status_code == 200
    assert "block" in resp.text
    # The other two shouldn't appear in their own row.
    assert resp.text.lower().count(">proceed<") == 1  # the dropdown option only


@pytest.mark.asyncio
async def test_decisions_shows_today_spend(app_client):
    client, deps = app_client
    now = datetime.now(UTC).replace(tzinfo=None)
    await deps.repos.llm_decisions.insert(
        LlmDecision(
            decision_type=LlmDecisionType.SCREEN,
            model="claude-opus-4-7",
            decision="screen_complete",
            cost_usd=0.30,
            created_at=now,
        )
    )
    resp = await client.get("/decisions", headers=_auth())
    assert "0.3000" in resp.text
