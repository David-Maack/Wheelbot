"""Weekly universe refresh — two-tier dynamic watchlists.

Tier 1 (quant, deterministic, zero LLM cost): build the candidate pool =
universe.yaml tickers (including parked) + `universe_refresh.candidate_pool`,
minus banned; gather price / median volume / IVR / earnings-distance per
symbol; gate ADD-eligibility on hard filters (price band, spec §6's >5M median
volume, earnings distance).

Tier 2 (LLM, one call/week): hand the model each strategy's spec, its current
membership, and the candidate metrics; it returns per-strategy watchlists as
add/keep/drop actions with scores and rationales.

Code-enforced guardrails (never trust the model):
  - symbols with an open position / active cycle can NEVER be dropped
  - pinned symbols (`universe_refresh.pinned`) can never be dropped
  - adds must have passed the quant gate; banned symbols never pass
  - churn caps: at most max_adds/max_drops per strategy per run
  - membership never shrinks below min_symbols_per_strategy (drops revert)
  - any failure → no new run applied → the bot keeps its last-good universe

Runs are persisted as PROPOSED (spec §6: never auto-add without human review).
Apply via the MCP `approve_watchlist` tool, or set `universe_refresh.auto_apply:
true` once trusted. Watchlist membership only controls what a strategy LOOKS AT
— every entry still passes the full risk gate (tier-2 screener score for
auto-added names, liquidity, IVR, earnings, capital caps) at trade time.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from typing import Any

from core.broker import Broker
from core.checkpoint import checkpoint, log_checkpoint
from core.config import load_universe
from core.models import LlmDecisionType, WatchlistEntry, WatchlistRun, WatchlistRunStatus
from core.notify import notify
from core.strategies import load_strategies, universe_for_strategy
from core.watchlists import effective_universe
from data.discovery import discover_candidates, has_tradable_chain
from data.earnings import next_earnings
from data.ivr import IVRProvider
from data.yf_helpers import safe_history
from db.repo import Repos
from intelligence.anthropic_client import AnthropicClient
from intelligence.budget import BudgetExceeded

SYSTEM_PROMPT = """You are the universe curator for a multi-strategy options
income bot. Once a week you review each strategy's watchlist against a
quant-prefiltered candidate pool and propose measured changes.

For EACH strategy you receive: its spec (structure, DTE band, delta band,
capital caps, IV-rank gates) and its CURRENT watchlist. For each candidate
symbol you receive: price, median daily volume, IV rank / percentile, days to
next earnings, and whether it passed the quant ADD gate.

Propose the best watchlist for each strategy. Your core job is MATCHING
candidates to strategy profiles: rich-but-not-broken IV and collateral that
fits the capital caps for premium sellers; cheap underlyings for the low-price
strategies; liquid, affordable LEAPs for PMCC; range-bound liquid names for
iron condors; LOW IV rank for calendars. Think "which strategy, if any, is
this symbol FOR?" — a great put-spread name is usually a terrible calendar
name, and vice versa.

Candidates with newly_discovered=true came from a market-wide most-actives
scan, not the curated universe. Their ivr is often null (no IV history yet) —
judge them on price, volume, and strategy fit, and be selective: a discovered
add must clearly beat an incumbent, not merely match it.

Rules:
- Respect the churn limits in `limits` — a watchlist should evolve, not churn.
- Only propose "add" for candidates with add_eligible=true.
- Never propose dropping a symbol listed in that strategy's protected list.
- Prefer keeping incumbents on a tie — changes need a reason, stability doesn't.
- Include EVERY current member of every strategy in your answer with action
  "keep" or "drop" — never silently omit one.

Return ONLY a JSON object, no markdown fences, exactly this shape:

{
  "decision": "refresh_complete",
  "summary": "<= 400 chars, what changed and why",
  "watchlists": [
    {
      "strategy_id": "put_spread",
      "symbols": [
        {"symbol": "AAPL", "action": "keep|add|drop", "score": 0-100,
         "rationale": "<= 200 chars"}
      ]
    }
  ]
}

`score` is your conviction the symbol belongs on that strategy's watchlist."""


@dataclass(slots=True)
class CandidateSnapshot:
    symbol: str
    price: float | None = None
    median_volume: float | None = None
    ivr: float | None = None
    iv_pct: float | None = None
    earnings_in_days: int | None = None
    add_eligible: bool = False
    ineligible_reason: str | None = None
    member_of: list[str] = field(default_factory=list)
    # True = came from the market-wide most-actives scan, not universe.yaml
    # or the manual candidate_pool. These get an extra option-chain
    # tradability gate and a per-run cap.
    newly_discovered: bool = False


async def _fetch_candidate(
    broker: Broker, ivr: IVRProvider, symbol: str
) -> CandidateSnapshot:
    snap = CandidateSnapshot(symbol=symbol)
    try:
        quote = await broker.get_quote(symbol)
        snap.price = quote.mid if quote.mid is not None else (quote.last or quote.bid or quote.ask)
    except Exception:
        snap.price = None

    df = await asyncio.to_thread(safe_history, symbol, period="3mo")
    if not df.empty and "Volume" in df:
        vol = df["Volume"].dropna()
        if len(vol) >= 20:
            snap.median_volume = float(vol.median())

    stats = await ivr.stats(symbol)
    if stats is not None:
        snap.ivr = stats.rank
        snap.iv_pct = stats.percentile

    try:
        earnings = next_earnings(symbol)
        if earnings.next_date is not None:
            snap.earnings_in_days = (earnings.next_date - datetime.now(UTC).date()).days
    except Exception:
        snap.earnings_in_days = None
    return snap


def _apply_quant_gate(snap: CandidateSnapshot, ur: dict[str, Any]) -> None:
    """Tier 1: hard ADD-eligibility gates. Incumbents stay in the payload even
    when ineligible (the LLM may still keep them; only ADDs are gated)."""
    price_min = float(ur.get("min_price", 5.0))
    price_max = float(ur.get("max_price", 1200.0))
    vol_min = float(ur.get("min_median_volume", 5_000_000))
    earn_min = int(ur.get("earnings_min_days", 7))

    if snap.price is None:
        snap.ineligible_reason = "no price"
    elif not price_min <= snap.price <= price_max:
        snap.ineligible_reason = f"price {snap.price:.2f} outside [{price_min}, {price_max}]"
    elif snap.median_volume is None:
        snap.ineligible_reason = "no volume history"
    elif snap.median_volume < vol_min:
        snap.ineligible_reason = f"median volume {snap.median_volume:,.0f} < {vol_min:,.0f}"
    elif snap.earnings_in_days is not None and 0 <= snap.earnings_in_days < earn_min:
        snap.ineligible_reason = f"earnings in {snap.earnings_in_days}d < {earn_min}d"
    else:
        snap.add_eligible = True


def _strategy_payload(
    config: dict[str, Any],
    current: dict[str, list[str]],
    protected: dict[str, list[str]],
    exclude: set[str],
) -> list[dict[str, Any]]:
    out = []
    interesting = (
        "dte_min", "dte_max", "csp_delta_min", "csp_delta_max", "short_delta_min",
        "short_delta_max", "short_delta_target", "long_delta_target", "spread_width_dollars",
        "max_capital_per_spread_usd", "max_capital_per_condor_usd",
        "max_capital_per_position_usd", "max_debit_per_spread_usd", "ivr_min", "ivr_max",
        "direction", "wing_width",
    )
    for s in load_strategies(config):
        if s.id in exclude or not s.enabled:
            continue
        out.append({
            "strategy_id": s.id,
            "display_name": s.display_name,
            "type": s.type,
            "params": {k: v for k, v in s.params.items() if k in interesting},
            "current_watchlist": current.get(s.id, []),
            "protected": protected.get(s.id, []),
        })
    return out


def enforce_guardrails(
    parsed: dict[str, Any],
    *,
    current: dict[str, list[str]],
    protected: dict[str, set[str]],
    eligible_adds: set[str],
    known_strategies: set[str],
    max_adds: int,
    max_drops: int,
    min_symbols: int,
) -> tuple[dict[str, list[dict[str, Any]]], list[str]]:
    """Post-parse enforcement of every guardrail. Pure so it's unit-testable.

    Returns ({strategy_id: [entry dicts]}, guardrail_notes). Every current
    member of every known strategy appears in the output exactly once — the
    entries are a complete diff, not a delta."""
    notes: list[str] = []
    by_strategy: dict[str, dict[str, dict[str, Any]]] = {}

    for wl in parsed.get("watchlists") or []:
        if not isinstance(wl, dict):
            continue
        sid = str(wl.get("strategy_id", "")).strip()
        if sid not in known_strategies:
            if sid:
                notes.append(f"{sid}: unknown/excluded strategy — proposal ignored")
            continue
        rows = by_strategy.setdefault(sid, {})
        for item in wl.get("symbols") or []:
            if not isinstance(item, dict) or not item.get("symbol"):
                continue
            sym = str(item["symbol"]).upper()
            action = str(item.get("action", "keep")).lower()
            if action not in ("add", "keep", "drop"):
                action = "keep"
            try:
                raw_score = item.get("score")
                score = max(0.0, min(100.0, float(raw_score))) if raw_score is not None else None
            except (TypeError, ValueError):
                score = None
            rows[sym] = {
                "symbol": sym,
                "action": action,
                "score": score,
                "rationale": str(item.get("rationale") or "")[:200],
            }

    result: dict[str, list[dict[str, Any]]] = {}
    for sid in sorted(known_strategies):
        members = {s.upper() for s in current.get(sid, [])}
        rows = by_strategy.get(sid, {})
        prot = protected.get(sid, set())

        # Current members the model omitted are kept — silence is not consent.
        for sym in members:
            if sym not in rows:
                rows[sym] = {"symbol": sym, "action": "keep", "score": None,
                             "rationale": "not mentioned by model — kept"}
                if by_strategy.get(sid):
                    notes.append(f"{sid}: model omitted {sym} — kept")

        adds, drops, keeps = [], [], []
        for sym, row in rows.items():
            action = row["action"]
            if action == "add":
                if sym in members:
                    row["action"] = "keep"
                    row["rationale"] = f"already a member | {row['rationale']}"[:200]
                    keeps.append(row)
                elif sym not in eligible_adds:
                    notes.append(f"{sid}: add {sym} rejected — failed quant gate or banned")
                else:
                    adds.append(row)
            elif action == "drop":
                if sym not in members:
                    notes.append(f"{sid}: drop {sym} ignored — not a member")
                elif sym in prot:
                    row["action"] = "keep"
                    protected_note = f"protected (open position or pinned) | {row['rationale']}"
                    row["rationale"] = protected_note[:200]
                    keeps.append(row)
                    notes.append(f"{sid}: drop {sym} vetoed — protected")
                else:
                    drops.append(row)
            else:
                if sym in members:
                    keeps.append(row)
                # "keep" for a non-member is a malformed add — ignore silently.

        adds.sort(key=lambda r: r["score"] if r["score"] is not None else -1.0, reverse=True)
        if len(adds) > max_adds:
            for row in adds[max_adds:]:
                notes.append(f"{sid}: add {row['symbol']} clamped — churn cap {max_adds}")
            adds = adds[:max_adds]
        drops.sort(key=lambda r: r["score"] if r["score"] is not None else 101.0)
        if len(drops) > max_drops:
            for row in drops[max_drops:]:
                row["action"] = "keep"
                row["rationale"] = f"drop clamped by churn cap | {row['rationale']}"[:200]
                keeps.append(row)
                notes.append(f"{sid}: drop {row['symbol']} clamped — churn cap {max_drops}")
            drops = drops[:max_drops]

        # Floor: membership never shrinks below min_symbols. Revert the
        # least-confident drops (highest score) first.
        while drops and len(members) - len(drops) + len(adds) < min_symbols:
            row = drops.pop()
            row["action"] = "keep"
            row["rationale"] = f"drop reverted — min size {min_symbols} | {row['rationale']}"[:200]
            keeps.append(row)
            notes.append(f"{sid}: drop {row['symbol']} reverted — min size {min_symbols}")

        result[sid] = sorted(adds + keeps + drops, key=lambda r: (r["action"], r["symbol"]))
    return result, notes


async def run_universe_refresh(
    *,
    broker: Broker,
    repos: Repos,
    ivr: IVRProvider,
    anthropic: AnthropicClient,
    config: dict[str, Any],
    run_date: date | None = None,
) -> dict[str, Any]:
    ur = config.get("universe_refresh", {}) or {}
    if not bool(ur.get("enabled", False)):
        log_checkpoint("universe_refresh_disabled", status="skip")
        return {"skipped": "disabled"}

    run_date = run_date or datetime.now(UTC).date()
    account_id = config.get("account", {}).get("id", "primary")
    exclude = set(ur.get("exclude_strategies", ["spy_swing_opt"]) or [])

    with checkpoint("universe_refresh_run") as ctx:
        # Current EFFECTIVE membership (yaml + any applied run) is the diff base.
        universe = await effective_universe(repos, config)
        yaml_universe = load_universe()
        banned = {str(b).upper() for b in yaml_universe.get("banned", [])}

        strategies = [s for s in load_strategies(config) if s.id not in exclude and s.enabled]
        known = {s.id for s in strategies}
        current: dict[str, list[str]] = {
            s.id: [t.symbol.upper() for t in universe_for_strategy(s, universe)["tickers"]]
            for s in strategies
        }

        # Protected = open position/cycle symbols + operator pins. Never droppable.
        open_by_strategy: dict[str, set[str]] = {}
        for p in await repos.positions.list_active(account_id):
            open_by_strategy.setdefault(p.strategy_id or "", set()).add(p.symbol.upper())
        pinned_cfg = ur.get("pinned", {}) or {}
        protected: dict[str, set[str]] = {
            sid: open_by_strategy.get(sid, set())
            | {str(s).upper() for s in (pinned_cfg.get(sid) or [])}
            for sid in known
        }

        # Candidate pool: universe.yaml tier-1/2 tickers (parked-but-not-demoted
        # included) + the configured extra pool, minus banned. Tier 3 is the
        # banned tier (spec §6) — never a candidate, and the overlay refuses it
        # too (core/watchlists.py).
        tier3 = {t.symbol.upper() for t in yaml_universe.get("tickers", []) if t.tier == 3}
        pool = {t.symbol.upper() for t in yaml_universe.get("tickers", []) if t.tier in (1, 2)}
        pool |= {str(s).upper() for s in (ur.get("candidate_pool") or [])}
        pool -= banned | tier3

        # Tier 0 — market discovery: fold in the top most-active US stocks so
        # the LLM can match fresh names to strategy profiles, not just shuffle
        # the hand-curated set. Fail-open: a broken screener shrinks the pool
        # back to hand-curated, never blocks the refresh.
        disc = ur.get("discovery", {}) or {}
        discovered_new: set[str] = set()
        if bool(disc.get("enabled", False)):
            found = {s for s in await discover_candidates(config)}
            discovered_new = found - pool - banned - tier3
            pool |= discovered_new
            ctx["n_discovered_new"] = len(discovered_new)
        ctx["n_candidates"] = len(pool)

        snapshots = await asyncio.gather(*(_fetch_candidate(broker, ivr, s) for s in sorted(pool)))
        member_of: dict[str, list[str]] = {}
        for sid, symbols in current.items():
            for sym in symbols:
                member_of.setdefault(sym, []).append(sid)
        for snap in snapshots:
            snap.member_of = sorted(member_of.get(snap.symbol, []))
            snap.newly_discovered = snap.symbol in discovered_new
            _apply_quant_gate(snap, ur)

        # Discovered names face two extra hurdles: an options market must
        # actually exist (one chain fetch each — per-contract liquidity is
        # still enforced at entry time), and only the top max_new_candidates
        # by dollar volume go to the LLM. Everything that fails is dropped
        # from the payload entirely — no point spending Opus tokens on names
        # that can never be added.
        if discovered_new:
            spread_max = float(disc.get("chain_spread_max_pct", 15.0))
            max_new = int(disc.get("max_new_candidates", 25))
            new_snaps = [s for s in snapshots if s.newly_discovered and s.add_eligible]
            tradable = await asyncio.gather(
                *(has_tradable_chain(broker, s.symbol, spread_max) for s in new_snaps)
            )
            for snap, ok in zip(new_snaps, tradable, strict=True):
                if not ok:
                    snap.add_eligible = False
                    snap.ineligible_reason = "no tradable option chain"
            survivors = sorted(
                (s for s in new_snaps if s.add_eligible),
                key=lambda s: (s.price or 0.0) * (s.median_volume or 0.0),
                reverse=True,
            )
            for snap in survivors[max_new:]:
                snap.add_eligible = False
                snap.ineligible_reason = f"discovery cap {max_new}"
            snapshots = [s for s in snapshots if not s.newly_discovered or s.add_eligible]
            ctx["n_discovered_kept"] = sum(1 for s in snapshots if s.newly_discovered)

        eligible_adds = {s.symbol for s in snapshots if s.add_eligible}
        ctx["n_add_eligible"] = len(eligible_adds)

        max_adds = int(ur.get("max_adds_per_strategy", 2))
        max_drops = int(ur.get("max_drops_per_strategy", 2))
        min_symbols = int(ur.get("min_symbols_per_strategy", 3))
        payload = {
            "run_date": str(run_date),
            "limits": {"max_adds_per_strategy": max_adds, "max_drops_per_strategy": max_drops,
                       "min_symbols_per_strategy": min_symbols},
            "strategies": _strategy_payload(
                config, current, {k: sorted(v) for k, v in protected.items()}, exclude
            ),
            "candidates": [
                {
                    "symbol": s.symbol,
                    "price": s.price,
                    "median_daily_volume": s.median_volume,
                    "ivr": s.ivr,
                    "iv_percentile": s.iv_pct,
                    "earnings_in_days": s.earnings_in_days,
                    "add_eligible": s.add_eligible,
                    "ineligible_reason": s.ineligible_reason,
                    "currently_on": s.member_of,
                    "newly_discovered": s.newly_discovered,
                }
                for s in snapshots
            ],
        }

        try:
            result = await anthropic.call(
                decision_type=LlmDecisionType.UNIVERSE_REFRESH,
                model=str(ur.get("model", "claude-opus-4-7")),
                system=SYSTEM_PROMPT,
                user_payload=payload,
                max_output_tokens=int(ur.get("max_output_tokens", 4096)),
                context={"run_date": str(run_date)},
            )
        except BudgetExceeded as exc:
            log_checkpoint("universe_refresh_budget_skip", status="skip", error=str(exc))
            return {"skipped": "budget_exceeded"}

        parsed = result.get("parsed") or {}
        if not parsed.get("watchlists"):
            # Non-JSON / truncated output. Record a FAILED run so the gap is
            # visible; the bot keeps its last-good membership (fail-open).
            run_id = await repos.watchlists.insert_run(WatchlistRun(
                run_date=run_date, status=WatchlistRunStatus.FAILED,
                llm_decision_id=result.get("decision_id"), cost_usd=result.get("cost_usd"),
                summary="model returned no parseable watchlists",
                created_at=datetime.now(UTC),
            ))
            log_checkpoint("universe_refresh_parse_fail", status="fail", run_id=run_id,
                           raw_preview=str(result.get("raw_text", ""))[:200])
            return {"run_id": run_id, "status": "failed", "cost_usd": result.get("cost_usd")}

        entries_by_strategy, guardrail_notes = enforce_guardrails(
            parsed,
            current=current,
            protected=protected,
            eligible_adds=eligible_adds,
            known_strategies=known,
            max_adds=max_adds,
            max_drops=max_drops,
            min_symbols=min_symbols,
        )

        summary = str(parsed.get("summary") or "")[:400]
        run_id = await repos.watchlists.insert_run(WatchlistRun(
            run_date=run_date, status=WatchlistRunStatus.PROPOSED,
            llm_decision_id=result.get("decision_id"), cost_usd=result.get("cost_usd"),
            summary=summary, created_at=datetime.now(UTC),
        ))
        n_adds = n_drops = 0
        for sid, rows in entries_by_strategy.items():
            for row in rows:
                n_adds += row["action"] == "add"
                n_drops += row["action"] == "drop"
                await repos.watchlists.insert_entry(WatchlistEntry(
                    run_id=run_id, strategy_id=sid, symbol=row["symbol"],
                    action=row["action"], score=row["score"], rationale=row["rationale"],
                ))
        for note in guardrail_notes:
            log_checkpoint("universe_refresh_guardrail", status="ok", run_id=run_id, note=note)
        ctx["run_id"] = run_id
        ctx["adds"] = n_adds
        ctx["drops"] = n_drops
        ctx["cost_usd"] = round(result.get("cost_usd") or 0.0, 4)

        status = WatchlistRunStatus.PROPOSED.value
        if n_adds == 0 and n_drops == 0:
            # Nothing changed — auto-close the run so proposals don't pile up.
            await repos.watchlists.set_status(run_id, WatchlistRunStatus.REJECTED.value)
            status = "no_changes"
        elif bool(ur.get("auto_apply", False)):
            await repos.watchlists.apply_run(run_id, applied_by="auto")
            status = WatchlistRunStatus.APPLIED.value

        await notify(
            "universe_refresh",
            f"Universe refresh: {n_adds} adds / {n_drops} drops ({status})",
            run_id=run_id,
            summary=summary,
            guardrail_notes=guardrail_notes[:10],
            action=("applied automatically" if status == WatchlistRunStatus.APPLIED.value
                    else "review via MCP get_watchlists / approve_watchlist"),
        )
        return {"run_id": run_id, "status": status, "adds": n_adds, "drops": n_drops,
                "cost_usd": result.get("cost_usd"), "guardrail_notes": guardrail_notes}
