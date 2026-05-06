"""core/notify — Discord webhook + null fallback."""

from __future__ import annotations

from typing import Any

import httpx
import pytest

from core.notify import (
    DiscordNotifier,
    Event,
    NullNotifier,
    get_dispatcher,
    make_notifier,
    notify,
    set_dispatcher,
)


def _mock_transport(handler):
    return httpx.MockTransport(handler)


@pytest.fixture(autouse=True)
def _reset_dispatcher():
    set_dispatcher(NullNotifier())
    yield
    set_dispatcher(NullNotifier())


@pytest.mark.asyncio
async def test_null_notifier_is_a_noop():
    n = NullNotifier()
    await n.send(Event(event_type="position.assigned", title="F"))
    # No exception means pass.


@pytest.mark.asyncio
async def test_discord_payload_shape(monkeypatch):
    seen: dict[str, Any] = {}

    def handler(request: httpx.Request):
        seen["url"] = str(request.url)
        seen["body"] = request.read().decode()
        return httpx.Response(204)

    real_client = httpx.AsyncClient
    monkeypatch.setattr(
        httpx,
        "AsyncClient",
        lambda **kw: real_client(transport=_mock_transport(handler), **{k: v for k, v in kw.items() if k != "transport"}),
    )
    n = DiscordNotifier(webhook_url="https://discord.test/hook")
    await n.send(
        Event(
            event_type="position.assigned",
            title="F assigned",
            payload={"symbol": "F", "cycle_id": 7},
        )
    )
    assert "discord.test" in seen["url"]
    assert "F assigned" in seen["body"]
    assert "position.assigned" in seen["body"]


@pytest.mark.asyncio
async def test_discord_5xx_retries_once(monkeypatch):
    attempts = {"n": 0}

    def handler(request):
        attempts["n"] += 1
        return httpx.Response(503 if attempts["n"] == 1 else 204)

    real_client = httpx.AsyncClient
    monkeypatch.setattr(
        httpx,
        "AsyncClient",
        lambda **kw: real_client(transport=_mock_transport(handler), **{k: v for k, v in kw.items() if k != "transport"}),
    )
    n = DiscordNotifier(webhook_url="https://discord.test/hook")
    await n.send(Event(event_type="risk.kill_switch_armed", title="armed"))
    assert attempts["n"] == 2


@pytest.mark.asyncio
async def test_discord_4xx_does_not_retry(monkeypatch):
    attempts = {"n": 0}

    def handler(request):
        attempts["n"] += 1
        return httpx.Response(400, json={"error": "bad webhook"})

    real_client = httpx.AsyncClient
    monkeypatch.setattr(
        httpx,
        "AsyncClient",
        lambda **kw: real_client(transport=_mock_transport(handler), **{k: v for k, v in kw.items() if k != "transport"}),
    )
    n = DiscordNotifier(webhook_url="https://discord.test/hook")
    await n.send(Event(event_type="cycle.closed_loss", title="lost"))
    assert attempts["n"] == 1


def test_make_notifier_falls_back_to_null_without_webhook(monkeypatch):
    monkeypatch.delenv("DISCORD_WEBHOOK_URL", raising=False)
    n = make_notifier({})
    assert isinstance(n, NullNotifier)


def test_make_notifier_returns_discord_when_url_set(monkeypatch):
    monkeypatch.setenv("DISCORD_WEBHOOK_URL", "https://discord.test/hook")
    n = make_notifier({})
    assert isinstance(n, DiscordNotifier)


@pytest.mark.asyncio
async def test_dispatch_helper_routes_to_set_dispatcher():
    sent: list[Event] = []

    class _Capture(NullNotifier):
        async def send(self, event):
            sent.append(event)

    set_dispatcher(_Capture())
    await notify("position.assigned", "F assigned", symbol="F")
    assert len(sent) == 1
    assert sent[0].event_type == "position.assigned"
    assert sent[0].payload == {"symbol": "F"}


def test_event_severity_mapping():
    assert Event("position.assigned", "x").severity == "low"
    assert Event("cycle.closed_loss", "x").severity == "medium"
    assert Event("risk.kill_switch_armed", "x").severity == "high"
    assert Event("position.broker_down", "x").severity == "high"
