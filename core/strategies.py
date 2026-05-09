"""Strategy registry and loader.

A strategy is a config-defined trading approach with its own parameter set,
universe slice, and concurrent-position cap. The bot iterates registered
strategies each tick and dispatches by `type` to the appropriate orchestrator
(currently `wheel`; `vertical_spread` arrives in sub-sprint 3).

config.yaml format:

    strategies:
      - id: monthly_wheel        # unique key; used as positions.strategy_id
        display_name: "Monthly Wheel"
        type: wheel              # dispatcher key
        enabled: true
        max_concurrent: 4
        params:
          dte_min: 30
          dte_max: 45
          csp_delta_min: 0.20
          csp_delta_max: 0.30
          ...

Universe membership is determined by `UniverseEntry.strategies` — a ticker
appears in a strategy's universe iff its strategies list contains that
strategy's id.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from core.checkpoint import log_checkpoint


@dataclass(frozen=True, slots=True)
class StrategyDefinition:
    """One row of the strategies registry."""

    id: str
    display_name: str
    type: str  # "wheel" | "vertical_spread"
    enabled: bool
    max_concurrent: int
    params: dict[str, Any] = field(default_factory=dict)

    def merged_wheel_params(self, base_wheel: dict[str, Any]) -> dict[str, Any]:
        """For type=wheel strategies: merge base wheel config with this
        strategy's overrides. The wheel orchestrator's selectors expect a
        config dict shaped like the legacy `wheel` section, so we synthesize
        one per strategy. `max_concurrent_positions` is exposed as part of
        this so the risk gate's concurrent-cap rule reads the per-strategy
        cap, not the legacy account-wide one."""
        merged = dict(base_wheel)
        merged.update(self.params)
        merged["max_concurrent_positions"] = self.max_concurrent
        return merged


def load_strategies(config: dict[str, Any]) -> list[StrategyDefinition]:
    """Parse the strategies section of config.yaml. Falls back to a single
    monthly_wheel strategy from the legacy `wheel` section if no strategies
    block is present (backward compat for pre-multi-strategy configs)."""
    raw = config.get("strategies")
    if not raw:
        # Legacy config — synthesize a single strategy from the wheel block.
        wheel = config.get("wheel", {}) or {}
        return [
            StrategyDefinition(
                id="monthly_wheel",
                display_name="Monthly Wheel",
                type="wheel",
                enabled=True,
                max_concurrent=int(wheel.get("max_concurrent_positions", 4)),
                params={k: v for k, v in wheel.items() if k != "max_concurrent_positions"},
            )
        ]

    disabled = set(config.get("disabled_strategies", []) or [])
    out: list[StrategyDefinition] = []
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        sid = str(entry.get("id", "")).strip()
        if not sid:
            continue
        enabled = bool(entry.get("enabled", True)) and sid not in disabled
        out.append(
            StrategyDefinition(
                id=sid,
                display_name=str(entry.get("display_name", sid)),
                type=str(entry.get("type", "wheel")),
                enabled=enabled,
                max_concurrent=int(entry.get("max_concurrent", 4)),
                params=dict(entry.get("params", {}) or {}),
            )
        )
    log_checkpoint(
        "strategies_loaded",
        status="ok",
        count=len(out),
        ids=[s.id for s in out],
    )
    return out


def universe_for_strategy(
    strategy: StrategyDefinition, universe: dict[str, Any]
) -> dict[str, Any]:
    """Return a universe-shaped dict containing only tickers tagged with
    this strategy. Preserves banned/banned_rules so risk gates still apply."""
    tickers = [
        t for t in universe.get("tickers", []) if strategy.id in t.strategies
    ]
    return {
        "tickers": tickers,
        "banned": universe.get("banned", []),
        "banned_rules": universe.get("banned_rules", []),
    }
