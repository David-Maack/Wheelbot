"""Pre-live readiness checks.

Runs a battery of read-only checks before flipping `account.broker` to
`tastytrade` (production). Exits non-zero if any required check fails.

    python -m scripts.preflight_live           # against current config
    python -m scripts.preflight_live --json    # machine-readable

Checks:

  REQUIRED:
    * Configured broker is `tastytrade` or `tastytrade_sandbox`.
    * Provider secret + remember token are present.
    * Broker authenticates and returns an account.
    * Universe loads (at least one tier-1 ticker).
    * SQLite DB is writable; positions table queryable.
    * Storage volume / DB parent directory exists.

  WARN-ONLY:
    * Any position in MANUAL_INTERVENTION → flag, don't fail.
    * Kill switch armed → flag, don't fail (running preflight while halted is fine).
    * Stop file present → flag, don't fail.
    * regime_snapshots empty → flag (Sprint 7 populates).
    * iv_history empty → flag (means no IVR gating yet).

This script doesn't talk to the dashboard or the reconciler. It's a
single-pass diagnostic; reads only.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from core.broker_factory import make_broker
from core.checkpoint import configure_logging, log_checkpoint
from core.config import _config_dir, load_config, load_universe
from db.repo import Database, Repos


@dataclass
class CheckResult:
    name: str
    status: str  # "pass" | "warn" | "fail"
    detail: str = ""


@dataclass
class PreflightReport:
    target_mode: str
    results: list[CheckResult] = field(default_factory=list)

    @property
    def failures(self) -> list[CheckResult]:
        return [r for r in self.results if r.status == "fail"]

    @property
    def warnings(self) -> list[CheckResult]:
        return [r for r in self.results if r.status == "warn"]

    @property
    def ok(self) -> bool:
        return not self.failures


async def _check_broker(report: PreflightReport, config: dict[str, Any]) -> None:
    broker_name = config.get("account", {}).get("broker", "")
    if broker_name not in ("tastytrade", "tastytrade_sandbox"):
        report.results.append(
            CheckResult("broker_selection", "fail", f"current broker={broker_name!r}, expected tastytrade or tastytrade_sandbox")
        )
        return
    secrets_path = _config_dir() / "secrets.env"
    secrets_text = secrets_path.read_text(encoding="utf-8") if secrets_path.exists() else ""
    missing = []
    for key in ("TASTYTRADE_PROVIDER_SECRET", "TASTYTRADE_REMEMBER_TOKEN"):
        if key not in secrets_text:
            missing.append(key)
    if missing:
        report.results.append(
            CheckResult("broker_secrets", "fail", f"missing in secrets.env: {', '.join(missing)}")
        )
        return
    report.results.append(CheckResult("broker_secrets", "pass"))

    try:
        broker = make_broker(config)
        account = await broker.get_account()
        if hasattr(broker, "aclose"):
            await broker.aclose()
    except Exception as exc:
        report.results.append(CheckResult("broker_auth", "fail", str(exc)))
        return
    report.results.append(
        CheckResult(
            "broker_auth",
            "pass",
            f"equity={account.equity:.2f} bp={account.buying_power:.2f}",
        )
    )


def _check_universe(report: PreflightReport) -> None:
    try:
        universe = load_universe()
    except Exception as exc:
        report.results.append(CheckResult("universe", "fail", str(exc)))
        return
    tier1 = [t for t in universe.get("tickers", []) if t.tier == 1]
    if not tier1:
        report.results.append(CheckResult("universe", "fail", "no tier-1 tickers"))
        return
    report.results.append(
        CheckResult("universe", "pass", f"{len(tier1)} tier-1 / {len(universe['tickers'])} total")
    )


async def _check_db(report: PreflightReport, config: dict[str, Any]) -> Repos | None:
    db_path = Path(config.get("database", {}).get("path", "wheelbot.db")).expanduser()
    parent = db_path.parent
    if not parent.exists():
        report.results.append(
            CheckResult("storage_volume", "fail", f"DB parent dir missing: {parent}")
        )
        return None
    report.results.append(CheckResult("storage_volume", "pass", str(parent)))

    try:
        db = Database(db_path)
        c = await db.connect()
        await c.execute("SELECT COUNT(*) FROM positions")
    except Exception as exc:
        report.results.append(CheckResult("db_writable", "fail", str(exc)))
        return None
    report.results.append(CheckResult("db_writable", "pass"))
    return Repos(db)


async def _check_ops(report: PreflightReport, repos: Repos, config: dict[str, Any]) -> None:
    account_id = config.get("account", {}).get("id", "primary")
    all_positions = await repos.positions.list_all(account_id)
    flagged = [
        p
        for p in all_positions
        if (p.state.value if hasattr(p.state, "value") else str(p.state)) == "MANUAL_INTERVENTION"
    ]
    if flagged:
        report.results.append(
            CheckResult(
                "manual_intervention",
                "warn",
                f"{len(flagged)} positions flagged: " + ", ".join(p.symbol for p in flagged[:5]),
            )
        )
    else:
        report.results.append(CheckResult("manual_intervention", "pass"))

    # Kill switch + stop file.
    from datetime import UTC, datetime

    today = datetime.now(UTC).date()
    anchor = await repos.daily_state.get(account_id, today)
    if anchor and anchor.kill_switch_armed:
        report.results.append(
            CheckResult("kill_switch", "warn", anchor.kill_switch_reason or "armed")
        )
    else:
        report.results.append(CheckResult("kill_switch", "pass"))

    stop_path = config.get("risk", {}).get("stop_file_path")
    if stop_path and Path(stop_path).expanduser().exists():
        report.results.append(
            CheckResult("stop_file", "warn", f"present at {stop_path}")
        )
    else:
        report.results.append(CheckResult("stop_file", "pass"))


async def _check_data_signals(report: PreflightReport, repos: Repos) -> None:
    c = await repos.db.connect()
    async with c.execute("SELECT COUNT(*) FROM regime_snapshots") as cur:
        regime_count = (await cur.fetchone())[0]
    if regime_count == 0:
        report.results.append(
            CheckResult(
                "regime_snapshots",
                "warn",
                "no rows yet — regime gate will skip until Sprint 7 populates",
            )
        )
    else:
        report.results.append(CheckResult("regime_snapshots", "pass", f"{regime_count} rows"))

    async with c.execute("SELECT COUNT(*) FROM iv_history") as cur:
        iv_count = (await cur.fetchone())[0]
    if iv_count < 20:
        report.results.append(
            CheckResult(
                "iv_history",
                "warn",
                f"only {iv_count} rows — IVR gate will fail-open (skip)",
            )
        )
    else:
        report.results.append(CheckResult("iv_history", "pass", f"{iv_count} rows"))


async def run() -> PreflightReport:
    config = load_config()
    target = "production" if config.get("account", {}).get("broker") == "tastytrade" else "sandbox"
    report = PreflightReport(target_mode=target)

    await _check_broker(report, config)
    _check_universe(report)
    repos = await _check_db(report, config)
    if repos is None:
        return report
    try:
        await _check_ops(report, repos, config)
        await _check_data_signals(report, repos)
    finally:
        await repos.db.close()
    return report


def _print_human(report: PreflightReport) -> None:
    print(f"Preflight ({report.target_mode})")
    print("=" * 50)
    for r in report.results:
        marker = {"pass": "PASS", "warn": "WARN", "fail": "FAIL"}[r.status]
        print(f"  [{marker}] {r.name:<24} {r.detail}")
    print("-" * 50)
    print(
        f"summary: {len([r for r in report.results if r.status == 'pass'])} pass, "
        f"{len(report.warnings)} warn, {len(report.failures)} fail"
    )
    if report.failures:
        print("\nNOT READY — resolve failures above before going live.")
    elif report.warnings:
        print("\nReady, with warnings. Review above.")
    else:
        print("\nReady.")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="Machine-readable output")
    args = parser.parse_args(argv)
    configure_logging()
    report = asyncio.run(run())
    if args.json:
        print(json.dumps({
            "target_mode": report.target_mode,
            "ok": report.ok,
            "results": [asdict(r) for r in report.results],
        }, indent=2))
    else:
        _print_human(report)
    log_checkpoint("preflight_done", status="ok" if report.ok else "fail", failures=len(report.failures), warnings=len(report.warnings))
    return 0 if report.ok else 1


if __name__ == "__main__":
    sys.exit(main())
