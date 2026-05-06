"""Outbound notifications — Discord webhooks for state-change and risk events.

Spec §13 #36 says "Slack/Discord webhook." David's chosen Discord; this module
ships only the Discord adapter for now. Adding a Slack adapter later is a
single class implementing the same `Notifier` ABC.

Design:

  - One `Notifier` ABC + `DiscordNotifier` + `NullNotifier` (the default when
    no webhook URL is configured).
  - `make_notifier(config)` returns the right one.
  - `Event` dataclass carries event_type + title + payload.
  - Best-effort delivery: one retry on 5xx, fail-silent on 4xx (logged).
  - Notifications are advisory — never block trading.

Event types (centralized so reconciler/kill_switch/router/anthropic can
import without circular deps):

    position.assigned
    position.called_away
    position.manual_intervention
    position.broker_down
    risk.kill_switch_armed
    risk.budget_exceeded
    roll.disagreement
    cycle.closed_loss            (loss-only — David's preference)
"""

from __future__ import annotations

import asyncio
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import httpx

from core.checkpoint import log_checkpoint


@dataclass(frozen=True, slots=True)
class Event:
    event_type: str
    title: str
    payload: dict[str, Any] = field(default_factory=dict)

    @property
    def severity(self) -> str:
        if self.event_type.startswith("risk.") or self.event_type == "position.broker_down":
            return "high"
        if self.event_type in ("position.manual_intervention", "roll.disagreement"):
            return "high"
        if self.event_type == "cycle.closed_loss":
            return "medium"
        return "low"


class Notifier(ABC):
    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    async def send(self, event: Event) -> None: ...


class NullNotifier(Notifier):
    """No-op when no webhook is configured. Logs every drop so the operator
    knows what they would have been told."""

    @property
    def name(self) -> str:
        return "null"

    async def send(self, event: Event) -> None:
        log_checkpoint(
            "notify_skipped",
            status="skip",
            event_type=event.event_type,
            severity=event.severity,
            title=event.title,
        )


_SEVERITY_COLOR = {
    "high": 0xB22222,
    "medium": 0xC97700,
    "low": 0x128A30,
}


class DiscordNotifier(Notifier):
    """Discord webhook with embeds. One retry on 5xx; logs on 4xx without raising."""

    def __init__(
        self,
        webhook_url: str | None = None,
        *,
        timeout_seconds: float = 5.0,
    ) -> None:
        self._url = webhook_url or os.environ.get("DISCORD_WEBHOOK_URL")
        if not self._url:
            raise ValueError("DISCORD_WEBHOOK_URL not set")
        self._timeout = timeout_seconds

    @property
    def name(self) -> str:
        return "discord"

    async def send(self, event: Event) -> None:
        body = self._payload(event)
        for attempt in (1, 2):
            try:
                async with httpx.AsyncClient(timeout=self._timeout) as client:
                    resp = await client.post(self._url, json=body)
                if 200 <= resp.status_code < 300:
                    log_checkpoint(
                        "notify_ok",
                        status="ok",
                        event_type=event.event_type,
                        attempt=attempt,
                    )
                    return
                if 400 <= resp.status_code < 500:
                    log_checkpoint(
                        "notify_4xx",
                        status="fail",
                        event_type=event.event_type,
                        status_code=resp.status_code,
                    )
                    return
                # 5xx → retry once
                log_checkpoint(
                    "notify_5xx",
                    status="fail",
                    event_type=event.event_type,
                    attempt=attempt,
                    status_code=resp.status_code,
                )
            except httpx.HTTPError as exc:
                log_checkpoint(
                    "notify_transport_fail",
                    status="fail",
                    event_type=event.event_type,
                    attempt=attempt,
                    error=str(exc),
                )
            if attempt < 2:
                await asyncio.sleep(0.5)

    def _payload(self, event: Event) -> dict[str, Any]:
        fields = [
            {"name": k, "value": str(v)[:1024], "inline": True}
            for k, v in (event.payload or {}).items()
            if v is not None
        ][:24]
        return {
            "content": None,
            "embeds": [
                {
                    "title": event.title[:256],
                    "description": f"`{event.event_type}`",
                    "color": _SEVERITY_COLOR.get(event.severity, 0x444444),
                    "fields": fields,
                }
            ],
        }


def make_notifier(config: dict[str, Any]) -> Notifier:
    """Pick a notifier from config + env. Falls back to NullNotifier."""
    notifications = config.get("notifications", {}) or {}
    if not bool(notifications.get("enabled", True)):
        return NullNotifier()
    discord_enabled = bool(notifications.get("discord_enabled", True))
    if discord_enabled and os.environ.get("DISCORD_WEBHOOK_URL"):
        try:
            return DiscordNotifier()
        except ValueError:
            return NullNotifier()
    return NullNotifier()


_dispatcher: Notifier | None = None


def set_dispatcher(notifier: Notifier) -> None:
    """Tests / runtime wiring set the process-wide notifier."""
    global _dispatcher
    _dispatcher = notifier


def get_dispatcher() -> Notifier:
    return _dispatcher or NullNotifier()


async def notify(event_type: str, title: str, **payload: Any) -> None:
    """Convenience: emit an event through the current dispatcher."""
    await get_dispatcher().send(Event(event_type=event_type, title=title, payload=payload))
