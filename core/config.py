"""Config loader.

Three layers, last write wins, merged per top-level section:

1. config/config.yaml         — committed defaults
2. config/config.local.yaml   — gitignored local overrides
3. config/secrets.env         — gitignored API keys, loaded into os.environ

Override config dir with WHEELBOT_CONFIG_DIR. Override DB path with WHEELBOT_DB_PATH.
"""

from __future__ import annotations

import os
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

from core.models import UniverseEntry

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_DIR = ROOT / "config"


def _config_dir() -> Path:
    return Path(os.environ.get("WHEELBOT_CONFIG_DIR", DEFAULT_CONFIG_DIR))


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a YAML mapping at the top level")
    return data


def _merge_sections(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Per-spec §5: overrides apply per-section. Top-level keys are merged shallowly:
    if override provides section X, X merges into base[X] one level deep."""
    out = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            merged = deepcopy(out[key])
            merged.update(value)
            out[key] = merged
        else:
            out[key] = deepcopy(value)
    return out


def load_secrets(config_dir: Path | None = None) -> None:
    """Load secrets.env into os.environ. Idempotent. Safe to call early."""
    cdir = config_dir or _config_dir()
    secrets_path = cdir / "secrets.env"
    if secrets_path.exists():
        load_dotenv(secrets_path, override=False)


def load_config(config_dir: Path | None = None) -> dict[str, Any]:
    cdir = config_dir or _config_dir()
    base = _read_yaml(cdir / "config.yaml")
    overrides = _read_yaml(cdir / "config.local.yaml")
    cfg = _merge_sections(base, overrides)

    load_secrets(cdir)

    # Env-var overrides for the handful of paths that need to differ between
    # docker / bare-metal without editing YAML.
    db_path_env = os.environ.get("WHEELBOT_DB_PATH")
    if db_path_env:
        cfg.setdefault("database", {})["path"] = db_path_env

    return cfg


def load_universe(config_dir: Path | None = None) -> dict[str, Any]:
    cdir = config_dir or _config_dir()
    raw = _read_yaml(cdir / "universe.yaml")
    tickers = [UniverseEntry(**t) for t in raw.get("tickers", [])]
    return {
        "tickers": tickers,
        "banned": list(raw.get("banned", [])),
        "banned_rules": list(raw.get("banned_rules", [])),
    }


def get_universe_entry(symbol: str, universe: dict[str, Any]) -> UniverseEntry | None:
    for entry in universe["tickers"]:
        if entry.symbol.upper() == symbol.upper():
            return entry
    return None


def effective_wheel_params(
    symbol: str, config: dict[str, Any], universe: dict[str, Any]
) -> dict[str, Any]:
    """Return wheel params with per-ticker overrides applied (spec §6)."""
    base = deepcopy(config.get("wheel", {}))
    entry = get_universe_entry(symbol, universe)
    if entry and entry.overrides:
        base.update(entry.overrides)
    return base
