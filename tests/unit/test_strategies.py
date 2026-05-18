"""core/strategies — registry loader + universe partition."""

from __future__ import annotations

from core.models import UniverseEntry
from core.strategies import (
    StrategyDefinition,
    load_strategies,
    universe_for_strategy,
)


def test_load_legacy_config_synthesizes_single_strategy():
    """Configs without a strategies block should still work — fall back to a
    single monthly_wheel strategy reading from the legacy wheel section."""
    cfg = {
        "wheel": {"max_concurrent_positions": 4, "dte_min": 30, "dte_max": 45},
    }
    strategies = load_strategies(cfg)
    assert len(strategies) == 1
    assert strategies[0].id == "monthly_wheel"
    assert strategies[0].max_concurrent == 4
    assert strategies[0].enabled is True
    assert strategies[0].params["dte_min"] == 30


def test_load_explicit_strategies_block():
    cfg = {
        "strategies": [
            {
                "id": "monthly_wheel",
                "display_name": "Monthly",
                "type": "wheel",
                "enabled": True,
                "max_concurrent": 4,
                "params": {"dte_min": 30, "dte_max": 45},
            },
            {
                "id": "weekly_wheel",
                "display_name": "Weekly",
                "type": "wheel",
                "enabled": True,
                "max_concurrent": 4,
                "params": {"dte_min": 7, "dte_max": 14},
            },
        ],
    }
    strategies = load_strategies(cfg)
    assert [s.id for s in strategies] == ["monthly_wheel", "weekly_wheel"]
    assert strategies[1].params["dte_min"] == 7


def test_disabled_strategies_list_overrides_enabled_flag():
    cfg = {
        "strategies": [
            {"id": "monthly_wheel", "enabled": True, "max_concurrent": 4, "params": {}},
            {"id": "put_spread", "enabled": True, "max_concurrent": 4, "params": {}},
        ],
        "disabled_strategies": ["put_spread"],
    }
    strategies = load_strategies(cfg)
    by_id = {s.id: s for s in strategies}
    assert by_id["monthly_wheel"].enabled is True
    assert by_id["put_spread"].enabled is False


def test_merged_wheel_params_overlays_strategy_params_on_base():
    s = StrategyDefinition(
        id="weekly_wheel",
        display_name="Weekly",
        type="wheel",
        enabled=True,
        max_concurrent=4,
        params={"dte_min": 7, "dte_max": 14, "csp_delta_max": 0.25},
    )
    base = {
        "dte_min": 30, "dte_max": 45,
        "csp_delta_min": 0.20, "csp_delta_max": 0.30,
        "open_interest_min": 500,
    }
    merged = s.merged_wheel_params(base)
    # Strategy values win
    assert merged["dte_min"] == 7
    assert merged["dte_max"] == 14
    assert merged["csp_delta_max"] == 0.25
    # Base values flow through where strategy doesn't override
    assert merged["csp_delta_min"] == 0.20
    assert merged["open_interest_min"] == 500
    # max_concurrent_positions exposed for risk gate
    assert merged["max_concurrent_positions"] == 4


def test_universe_for_strategy_filters_by_strategy_tag():
    s = StrategyDefinition(
        id="weekly_wheel", display_name="Weekly", type="wheel",
        enabled=True, max_concurrent=4, params={},
    )
    universe = {
        "tickers": [
            UniverseEntry(symbol="F", tier=1, strategies=["monthly_wheel"]),
            UniverseEntry(symbol="HOOD", tier=2, strategies=["weekly_wheel"]),
            UniverseEntry(symbol="AAPL", tier=1, strategies=["put_spread"]),
            UniverseEntry(symbol="PLTR", tier=2, strategies=["weekly_wheel", "put_spread"]),
        ],
        "banned": ["GME"],
        "banned_rules": ["biotech under $20"],
    }
    result = universe_for_strategy(s, universe)
    syms = sorted(t.symbol for t in result["tickers"])
    assert syms == ["HOOD", "PLTR"]
    # banned passes through
    assert result["banned"] == ["GME"]


# -- Live config / universe smoke (config/config.yaml + config/universe.yaml) --


def _load_real_config_and_universe():
    """Load the actual on-disk config + universe shipped with the repo."""
    from pathlib import Path
    from core.config import load_config, load_universe

    config_dir = Path(__file__).resolve().parents[2] / "config"
    return load_config(config_dir=config_dir), load_universe(config_dir=config_dir)


def test_real_config_registers_bear_call_spread():
    """config.yaml must declare bear_call_spread with direction=bear_call."""
    cfg, _ = _load_real_config_and_universe()
    strategies = load_strategies(cfg)
    by_id = {s.id: s for s in strategies}
    assert "bear_call_spread" in by_id, "bear_call_spread missing from strategies"
    bcs = by_id["bear_call_spread"]
    assert bcs.type == "vertical_spread"
    assert bcs.params.get("direction") == "bear_call"
    # Ships disabled — flipped via a separate commit once smoke-tested live.
    assert bcs.enabled is False
    # Both spread strategies share the same sizing for cross-strategy comparability.
    assert bcs.params.get("spread_width_dollars") == 5.0
    assert bcs.params.get("max_capital_per_spread_usd") == 500


def test_bear_call_and_put_spread_universes_are_disjoint():
    """Avoid accidental short-strangle exposure on the same ticker."""
    cfg, universe = _load_real_config_and_universe()
    strategies = load_strategies(cfg)
    by_id = {s.id: s for s in strategies}

    put_universe = universe_for_strategy(by_id["put_spread"], universe)
    call_universe = universe_for_strategy(by_id["bear_call_spread"], universe)
    put_syms = {t.symbol for t in put_universe["tickers"]}
    call_syms = {t.symbol for t in call_universe["tickers"]}
    overlap = put_syms & call_syms
    assert not overlap, f"put_spread and bear_call_spread share tickers: {overlap}"
    # Sanity: both universes are non-trivial.
    assert len(put_syms) >= 5
    assert len(call_syms) >= 5
