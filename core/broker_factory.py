"""Broker factory.

Strategy code calls `make_broker(config)` and gets back a `Broker` instance —
nothing in `strategies/` or `execution/` should ever import a concrete adapter.

The `account.broker` key in config.yaml selects the implementation:

    paper              → platforms.paper_broker.PaperBroker
    alpaca_paper       → platforms.alpaca_broker.AlpacaBroker
    tastytrade_sandbox → (Sprint 6)
    tastytrade         → (Sprint 6)
"""

from __future__ import annotations

from typing import Any

from core.broker import Broker


def make_broker(config: dict[str, Any]) -> Broker:
    name = config.get("account", {}).get("broker", "paper")
    account_id = config.get("account", {}).get("id", "primary")

    if name == "paper":
        from platforms.paper_broker import PaperBroker

        return PaperBroker(account_id=account_id)

    if name == "alpaca_paper":
        from platforms.alpaca_broker import AlpacaBroker

        return AlpacaBroker(account_id=account_id)

    if name in ("tastytrade", "tastytrade_sandbox"):
        raise NotImplementedError(f"{name} broker arrives in Sprint 6")

    raise ValueError(f"unknown broker {name!r}; check config.account.broker")
