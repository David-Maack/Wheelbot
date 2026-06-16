"""TICKET-022 — Alpaca paper vs Tastytrade sandbox option-pricing parity.

Stage 2 of the go-live runbook (docs/GO_LIVE_RUNBOOK.md). For each ticker
with at least one active strategy, fetches the option chain from BOTH
brokers, normalizes OCC symbols on both sides, joins on the canonical
form, computes per-contract mid-price diff, and writes one row per
matched contract to broker_parity_log.

Recommended cron (NYSE hours, 6 samples/day at :35 UTC 14-19):

    35 14-19 * * 1-5  cd /opt/wheelbot && docker exec wheelbot \\
        python -m scripts.parity_run

Acceptance criteria the harness reports against (per runbook §2 entry to
Stage 3 and runbook Stage 2 acceptance):

  - mean abs(mid_diff_pct) per symbol  <  2%
  - worst abs(mid_diff_pct) per symbol <  5%
  - zero contracts where Alpaca has bid>0 AND ask>0 but Tastytrade
    returns neither (asymmetric_liquidity flag)

Failure modes (see runbook Appendix → Stage 2):
  - Symbol-format mismatch: Alpaca's raw _parse_occ-friendly form vs
    Tastytrade's _occ_dense form. This script routes BOTH sides through
    _normalize_occ() before insert. The unit test asserts equality on
    matched contracts.
  - Sandbox creds missing: hard-fail with an actionable message pointing
    at scripts.bootstrap_tastytrade and runbook §5 Part A.
  - Rate limits: a startup get_account() ping verifies both brokers
    respond. Per-symbol loop sleeps INTER_SYMBOL_DELAY_S between fetches
    (default 0.5s → ~120 calls/min/broker, under both documented limits).
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from collections import defaultdict
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

from core.broker import Broker, BrokerError, BrokerUnavailable
from core.broker_factory import make_broker
from core.checkpoint import configure_logging, log_checkpoint
from core.config import load_config, load_universe
from core.models import OptionContract
from db.repo import Database, Repos

# Defensive throttle. Alpaca paper allows 200/min, Tastytrade allows
# 120/min documented. 0.5s/symbol × 1 chain call/broker = ~120/min/broker.
INTER_SYMBOL_DELAY_S = 0.5

# Shared near-the-money window the harness quotes + compares per symbol. Both
# brokers price the SAME contracts: the nearest expiration at/after
# WINDOW_MIN_DTE, strikes within WINDOW_STRIKE_PCT of spot (calls and puts),
# capped at MAX_CONTRACTS_PER_SYMBOL by distance to spot. This bounds the
# quote-fetch cost and — unlike a per-broker first-N slice — guarantees overlap.
MAX_CONTRACTS_PER_SYMBOL = 50
WINDOW_MIN_DTE = 7
WINDOW_STRIKE_PCT = 0.10

# Window for the daily report.
REPORT_DAYS = 1

# Default location for daily markdown reports — same volume the bot uses.
DEFAULT_REPORT_DIR = Path("/mnt/wheelbot-storage/parity_reports")

# Actionable error text when Tastytrade sandbox creds are missing.
MISSING_CREDS_MESSAGE = """\
Tastytrade sandbox credentials missing — TICKET-022 parity harness cannot run.

To provision:
  (1) From a host with browser access to cert.tastyworks.com, run:
        python -m scripts.bootstrap_tastytrade --env sandbox
  (2) Copy the captured TASTYTRADE_PROVIDER_SECRET and
      TASTYTRADE_REMEMBER_TOKEN into config/secrets.env on the LXC.
  (3) Restart wheelbot:
        cd /opt/wheelbot && docker compose restart wheelbot
  (4) Re-run preflight to confirm:
        docker exec wheelbot python -m scripts.preflight_live

See docs/GO_LIVE_RUNBOOK.md §5 Part A for the full verification path.
"""


def _normalize_occ(occ: str) -> str:
    """Single canonical form for OCC symbols. Tastytrade SDK may emit
    space-padded form; Alpaca uses dense 21-char. Strip whitespace and
    upper-case — both feeds route through this before insert."""
    return (occ or "").replace(" ", "").upper()


def _mid(contract: OptionContract) -> float | None:
    """Best-effort mid: explicit .mid if populated, else (bid+ask)/2 when
    both sides present. None when either side is missing."""
    if contract.mid is not None:
        return float(contract.mid)
    if contract.bid is not None and contract.ask is not None:
        return (float(contract.bid) + float(contract.ask)) / 2.0
    return None


def _alpaca_has_liquidity(contract: OptionContract) -> bool:
    """Per Q3(a): both bid > 0 AND ask > 0 on Alpaca."""
    return (
        contract.bid is not None and contract.bid > 0
        and contract.ask is not None and contract.ask > 0
    )


def _tasty_has_liquidity(contract: OptionContract | None) -> bool:
    if contract is None:
        return False
    return (
        contract.bid is not None and contract.bid > 0
        and contract.ask is not None and contract.ask > 0
    )


async def _ping_broker(broker: Broker, label: str) -> None:
    """Pre-flight liveness + auth check. Aborts on any failure with a
    label-tagged message so the operator can see which broker is unhappy."""
    try:
        await broker.get_account()
    except BrokerUnavailable as exc:
        raise SystemExit(
            f"{label} unavailable: {exc}. Rate limit, network, or auth issue. "
            f"Will retry on the next cron tick."
        ) from exc
    except Exception as exc:
        raise SystemExit(f"{label} failed: {type(exc).__name__}: {exc}") from exc


def _select_symbols(universe: dict[str, Any]) -> list[str]:
    """Q1(b)-dynamic: every ticker with a non-empty strategies list."""
    out: list[str] = []
    for entry in universe.get("tickers", []):
        strategies = getattr(entry, "strategies", None) or []
        if strategies:
            out.append(entry.symbol.upper())
    return sorted(set(out))


async def _fetch_chain(
    broker: Broker, symbol: str, label: str,
) -> list[OptionContract]:
    """Fetch the full chain. Adapter-side filters (DTE, type, expiration)
    are NOT applied here — we want the brokers' natural views to compare.
    Returns empty list on broker failure; the caller logs and continues."""
    try:
        contracts = await broker.get_option_chain(symbol)
    except BrokerUnavailable as exc:
        log_checkpoint(
            "parity_chain_unavailable",
            status="skip", broker=label, symbol=symbol, error=str(exc),
        )
        return []
    except Exception as exc:
        log_checkpoint(
            "parity_chain_failed",
            status="fail", broker=label, symbol=symbol,
            error=f"{type(exc).__name__}: {exc}",
        )
        return []
    return list(contracts)


def _join_and_compute(
    alpaca: list[OptionContract], tasty: list[OptionContract],
) -> tuple[list[OptionContract], list[OptionContract | None]]:
    """Join two chains by canonical OCC. Returns (alpaca_ordered, tasty_aligned)
    so index i in tasty_aligned matches index i in alpaca_ordered (None if
    Tastytrade has no match for that Alpaca contract)."""
    tasty_by_key: dict[str, OptionContract] = {}
    for c in tasty:
        key = _normalize_occ(c.occ_symbol)
        tasty_by_key[key] = c
    aligned: list[OptionContract | None] = []
    ordered: list[OptionContract] = []
    for ac in alpaca:
        key = _normalize_occ(ac.occ_symbol)
        ordered.append(ac)
        aligned.append(tasty_by_key.get(key))
    return ordered, aligned


def _build_rows(
    fetched_at: datetime,
    symbol: str,
    alpaca: list[OptionContract],
    tasty_aligned: list[OptionContract | None],
) -> list[dict[str, Any]]:
    """Per-contract row dicts ready for broker_parity_log.insert_many."""
    rows: list[dict[str, Any]] = []
    for ac, tc in zip(alpaca, tasty_aligned):
        a_mid = _mid(ac)
        t_mid = _mid(tc) if tc is not None else None
        diff_pct: float | None = None
        if a_mid is not None and t_mid is not None and a_mid > 0:
            diff_pct = (t_mid - a_mid) / a_mid * 100.0
        asym = int(_alpaca_has_liquidity(ac) and not _tasty_has_liquidity(tc))
        rows.append({
            "fetched_at": fetched_at.isoformat(),
            "symbol": symbol,
            "contract_symbol": _normalize_occ(ac.occ_symbol),
            "alpaca_mid": a_mid,
            "tasty_mid": t_mid,
            "mid_diff_pct": diff_pct,
            "alpaca_bid": ac.bid,
            "alpaca_ask": ac.ask,
            "tasty_bid": tc.bid if tc is not None else None,
            "tasty_ask": tc.ask if tc is not None else None,
            "asymmetric_liquidity": asym,
        })
    return rows


def _render_daily_report(
    *,
    report_date: datetime,
    stats: list[dict[str, Any]],
    outliers: list[dict[str, Any]],
    tasty_label: str = "sandbox",
) -> str:
    """Markdown for /mnt/wheelbot-storage/parity_reports/YYYY-MM-DD.md.

    Sections: header, per-symbol summary table, acceptance section,
    top-10 outliers. Acceptance section flags PASS / FAIL per the
    runbook's three Stage 2 criteria."""
    lines: list[str] = []
    lines.append(f"# Broker parity report — {report_date.date().isoformat()}")
    lines.append("")
    lines.append(f"Comparison: Alpaca paper vs Tastytrade {tasty_label}.")
    lines.append("")
    total_samples = sum(int(r.get("samples") or 0) for r in stats)
    total_asym = sum(int(r.get("asymmetric_count") or 0) for r in stats)
    lines.append(f"Total samples: **{total_samples}** across **{len(stats)}** symbols.")
    lines.append(f"Total asymmetric-liquidity rows: **{total_asym}**.")
    lines.append("")

    # Per-symbol table.
    lines.append("## Per-symbol summary")
    lines.append("")
    lines.append("| Symbol | Samples | Avg |Δmid|% | Worst |Δmid|% | Asymmetric Liquidity |")
    lines.append("|--------|---------|--------------|----------------|----------------------|")
    for r in stats:
        avg = float(r["avg_abs_diff_pct"] or 0.0)
        worst = float(r["worst_abs_diff_pct"] or 0.0)
        lines.append(
            f"| {r['symbol']} | {r['samples']} | "
            f"{avg:.2f}% | {worst:.2f}% | {r['asymmetric_count'] or 0} |"
        )
    lines.append("")

    # Acceptance — Stage 2 criteria, all must hold.
    avg_ok = all(float(r["avg_abs_diff_pct"] or 0.0) < 2.0 for r in stats) if stats else False
    worst_ok = all(float(r["worst_abs_diff_pct"] or 0.0) < 5.0 for r in stats) if stats else False
    asym_ok = total_asym == 0
    lines.append("## Stage 2 acceptance check")
    lines.append("")
    lines.append(f"- Mean |Δmid|% per symbol < 2%: **{'PASS' if avg_ok else 'FAIL'}**")
    lines.append(f"- Worst |Δmid|% per symbol < 5%: **{'PASS' if worst_ok else 'FAIL'}**")
    lines.append(f"- Zero asymmetric-liquidity rows: **{'PASS' if asym_ok else 'FAIL'}**")
    if not (avg_ok and worst_ok and asym_ok):
        lines.append("")
        lines.append("⚠ Acceptance not met today. Do not advance to Stage 3.")
    lines.append("")

    # Top-10 outliers.
    if outliers:
        lines.append("## Top outliers")
        lines.append("")
        lines.append("| Symbol | Contract | Alpaca mid | Tasty mid | Δmid % |")
        lines.append("|--------|----------|------------|-----------|--------|")
        for r in outliers:
            a = "—" if r["alpaca_mid"] is None else f"{r['alpaca_mid']:.2f}"
            t = "—" if r["tasty_mid"] is None else f"{r['tasty_mid']:.2f}"
            d = "—" if r["mid_diff_pct"] is None else f"{r['mid_diff_pct']:+.2f}%"
            lines.append(
                f"| {r['symbol']} | `{r['contract_symbol']}` | {a} | {t} | {d} |"
            )
        lines.append("")
    return "\n".join(lines) + "\n"


def _verify_creds(config: dict[str, Any], *, use_prod: bool = False) -> None:
    """Q2(a)-actionable: hard-fail with operator-readable message when the
    Tastytrade env is incomplete. Prod parity uses a SEPARATE read-only cred
    set (TASTYTRADE_PROD_*) so it never clobbers the sandbox creds."""
    required = (
        ("TASTYTRADE_PROD_PROVIDER_SECRET", "TASTYTRADE_PROD_REMEMBER_TOKEN")
        if use_prod
        else ("TASTYTRADE_PROVIDER_SECRET", "TASTYTRADE_REMEMBER_TOKEN")
    )
    missing = [k for k in required if not os.environ.get(k)]
    if missing:
        sys.stderr.write(MISSING_CREDS_MESSAGE)
        sys.stderr.write(f"\nMissing env vars: {', '.join(missing)}\n")
        raise SystemExit(2)


async def _spot(broker: Broker, symbol: str) -> float | None:
    """Underlying mid for centering the near-the-money window. Best-effort —
    None if the quote fails (the window then falls back to strike order)."""
    try:
        q = await broker.get_quote(symbol)
    except Exception:
        return None
    if q.mid is not None:
        return float(q.mid)
    return float(q.last) if q.last is not None else None


def _select_window_keys(
    alpaca: list[OptionContract],
    tasty: list[OptionContract],
    spot: float | None,
    *,
    today: date,
    min_dte: int,
    pct: float,
    max_contracts: int,
) -> set[str]:
    """Canonical OCC keys present in BOTH chains, restricted to a shared
    near-the-money window: the nearest expiration at/after ``min_dte``, strikes
    within +/-``pct`` of ``spot`` (calls and puts), capped at ``max_contracts``
    by distance to spot. Guarantees the two brokers price the SAME contracts and
    bounds quote-fetch cost. Empty when there's no overlapping expiry in range.
    Falls back to strike order when spot is unknown, and ignores the pct band if
    it would leave nothing (keeps the nearest strikes instead)."""
    a_by_key = {_normalize_occ(c.occ_symbol): c for c in alpaca}
    t_keys = {_normalize_occ(c.occ_symbol) for c in tasty}
    common = [c for k, c in a_by_key.items() if k in t_keys]
    if not common:
        return set()
    exps = sorted({c.expiration for c in common if (c.expiration - today).days >= min_dte})
    if not exps:
        return set()
    near = [c for c in common if c.expiration == exps[0]]
    if spot is not None and spot > 0:
        band = [c for c in near if abs(c.strike - spot) <= pct * spot]
        near = band or near  # fall back to nearest strikes if the band is empty
        near.sort(key=lambda c: abs(c.strike - spot))
    else:
        near.sort(key=lambda c: c.strike)
    return {_normalize_occ(c.occ_symbol) for c in near[:max_contracts]}


class _ReadOnlyBroker(Broker):
    """Wraps a broker so the prod parity path can only read. Every order/cancel
    method hard-raises — belt-and-suspenders on top of the read-only OAuth
    scope. Used only for ``--prod`` runs against the real account."""

    def __init__(self, inner: Broker):
        self._inner = inner

    @property
    def name(self) -> str:
        return self._inner.name

    async def get_account(self):
        return await self._inner.get_account()

    async def get_positions(self):
        return await self._inner.get_positions()

    async def get_quote(self, symbol: str):
        return await self._inner.get_quote(symbol)

    async def get_option_chain(self, underlying, expiration=None, option_type=None):
        return await self._inner.get_option_chain(underlying, expiration, option_type)

    async def populate_quotes(self, contracts):
        return await self._inner.populate_quotes(contracts)

    async def get_orders_since(self, since):
        return await self._inner.get_orders_since(since)

    async def place_order(self, order):
        raise BrokerError("parity harness is read-only; place_order is blocked")

    async def cancel_order(self, broker_order_id):
        raise BrokerError("parity harness is read-only; cancel_order is blocked")

    async def place_multi_leg_order(self, **kwargs):
        raise BrokerError("parity harness is read-only; place_multi_leg_order is blocked")

    async def aclose(self):
        aclose = getattr(self._inner, "aclose", None)
        if aclose is not None:
            await aclose()


def _make_tasty_broker(config: dict[str, Any], *, use_prod: bool) -> Broker:
    """Sandbox via the factory, or a read-only prod adapter from TASTYTRADE_PROD_*."""
    if not use_prod:
        tasty_cfg = {**config, "account": {**config.get("account", {}), "broker": "tastytrade_sandbox"}}
        return make_broker(tasty_cfg)
    from platforms.tastytrade_broker import TastytradeBroker

    inner = TastytradeBroker(
        is_test=False,
        provider_secret=os.environ.get("TASTYTRADE_PROD_PROVIDER_SECRET"),
        refresh_token=os.environ.get("TASTYTRADE_PROD_REMEMBER_TOKEN"),
        account_number=os.environ.get("TASTYTRADE_PROD_ACCOUNT_NUMBER"),
    )
    return _ReadOnlyBroker(inner)


async def run(
    *,
    config: dict[str, Any] | None = None,
    universe: dict[str, Any] | None = None,
    report_dir: Path | None = None,
    now: datetime | None = None,
    alpaca_broker: Broker | None = None,
    tasty_broker: Broker | None = None,
    use_prod: bool = False,
) -> dict[str, Any]:
    """Single-pass parity run. Returns a small summary dict (symbol count,
    rows written, report path). Reusable from tests with injected brokers.

    use_prod=True prices the Tastytrade side against PRODUCTION (read-only)
    from TASTYTRADE_PROD_* creds — the sandbox serves no option quotes."""
    cfg = config if config is not None else load_config()
    uni = universe if universe is not None else load_universe()
    now = now or datetime.now(UTC).replace(tzinfo=None)
    report_dir = report_dir or DEFAULT_REPORT_DIR

    symbols = _select_symbols(uni)
    if not symbols:
        log_checkpoint("parity_no_symbols", status="skip")
        return {"symbols": 0, "rows": 0, "report_path": None}

    # Pre-flight: build (or accept injected) brokers and ping both. Injected
    # brokers skip the creds check — tests use PaperBroker for both sides.
    tasty_name = "tastytrade" if use_prod else "tastytrade_sandbox"
    if alpaca_broker is None or tasty_broker is None:
        _verify_creds(cfg, use_prod=use_prod)
        alpaca_cfg = {**cfg, "account": {**cfg.get("account", {}), "broker": "alpaca_paper"}}
        alpaca_broker = alpaca_broker or make_broker(alpaca_cfg)
        tasty_broker = tasty_broker or _make_tasty_broker(cfg, use_prod=use_prod)
    await _ping_broker(alpaca_broker, "alpaca_paper")
    await _ping_broker(tasty_broker, tasty_name)
    log_checkpoint(
        "parity_preflight_ok",
        status="ok", symbols=len(symbols),
        estimated_calls=len(symbols) * 2,
    )

    # Fetch + join + collect rows.
    all_rows: list[dict[str, Any]] = []
    db_path = Path(cfg.get("database", {}).get("path", "wheelbot.db")).expanduser()
    async with Database(db_path) as db:
        repos = Repos(db)
        for sym in symbols:
            alpaca_chain = await _fetch_chain(alpaca_broker, sym, "alpaca_paper")
            tasty_chain = await _fetch_chain(tasty_broker, sym, tasty_name)
            if not alpaca_chain or not tasty_chain:
                log_checkpoint(
                    "parity_skip_empty_chain",
                    status="skip", symbol=sym,
                    alpaca_n=len(alpaca_chain), tasty_n=len(tasty_chain),
                )
                await asyncio.sleep(INTER_SYMBOL_DELAY_S)
                continue
            # Shared near-the-money window, then quote ONLY those contracts on
            # each side (no-op for Alpaca, batched snapshot for Tastytrade).
            spot = await _spot(alpaca_broker, sym)
            keys = _select_window_keys(
                alpaca_chain, tasty_chain, spot,
                today=now.date(), min_dte=WINDOW_MIN_DTE,
                pct=WINDOW_STRIKE_PCT, max_contracts=MAX_CONTRACTS_PER_SYMBOL,
            )
            if not keys:
                log_checkpoint(
                    "parity_skip_no_window",
                    status="skip", symbol=sym,
                    alpaca_n=len(alpaca_chain), tasty_n=len(tasty_chain),
                )
                await asyncio.sleep(INTER_SYMBOL_DELAY_S)
                continue
            sel_alpaca = [c for c in alpaca_chain if _normalize_occ(c.occ_symbol) in keys]
            sel_tasty = [c for c in tasty_chain if _normalize_occ(c.occ_symbol) in keys]
            sel_alpaca = await alpaca_broker.populate_quotes(sel_alpaca)
            sel_tasty = await tasty_broker.populate_quotes(sel_tasty)
            alpaca_ordered, tasty_aligned = _join_and_compute(sel_alpaca, sel_tasty)
            rows = _build_rows(now, sym, alpaca_ordered, tasty_aligned)
            all_rows.extend(rows)
            await asyncio.sleep(INTER_SYMBOL_DELAY_S)

        written = await repos.broker_parity_log.insert_many(all_rows)

        # Daily report covers today's window — keep it scoped so a cron at
        # 19:35 includes all six samples of the day.
        report_since = now.replace(hour=0, minute=0, second=0, microsecond=0)
        report_until = report_since + timedelta(days=REPORT_DAYS)
        stats = await repos.broker_parity_log.stats_by_symbol(report_since, until=report_until)
        outliers = await repos.broker_parity_log.top_outliers(
            report_since, until=report_until, limit=10,
        )

    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / f"{now.date().isoformat()}.md"
    report_text = _render_daily_report(
        report_date=now, stats=stats, outliers=outliers,
        tasty_label="production" if use_prod else "sandbox",
    )
    report_path.write_text(report_text, encoding="utf-8")

    log_checkpoint(
        "parity_run_done",
        status="ok", symbols=len(symbols), rows=written,
        report_path=str(report_path),
    )

    # Best-effort close on adapters that own SDK sessions (Tastytrade).
    for b in (alpaca_broker, tasty_broker):
        aclose = getattr(b, "aclose", None)
        if callable(aclose):
            try:
                await aclose()
            except Exception:
                pass

    return {
        "symbols": len(symbols),
        "rows": written,
        "report_path": str(report_path),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--report-dir", type=Path, default=None,
        help=f"Where to write daily report (default {DEFAULT_REPORT_DIR}).",
    )
    parser.add_argument(
        "--prod", action="store_true",
        help="Price the Tastytrade side against PRODUCTION (read-only) using "
             "TASTYTRADE_PROD_* creds instead of the sandbox. The live bot is "
             "unaffected; the prod adapter is wrapped read-only.",
    )
    args = parser.parse_args(argv)
    configure_logging()
    result = asyncio.run(run(report_dir=args.report_dir, use_prod=args.prod))
    print(
        f"parity_run done — {result['symbols']} symbols, "
        f"{result['rows']} rows, report at {result['report_path']}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
