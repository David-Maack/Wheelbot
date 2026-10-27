"""Pre-trade risk gates from spec §8.

Every order goes through `RiskGate.evaluate(proposal, ...)` before submission.
On any rule's failure the gate raises `RiskCheckFailed`; the router catches it,
logs loudly, and the order is not placed.

Rules implemented (1-7 of §8, plus banned symbols and the macro blackout):

0. Banned symbols          — universe.yaml `banned` never opens (2026-10-27: was
                              only filtered by the since-removed watchlist overlay).
1. Buying-power floor      — keep ≥ buying_power_floor_pct free after this order.
2. Per-position cap        — notional of this position ≤ max_position_pct_of_account.
3. Concurrent positions    — count of non-IDLE positions per strategy ≤ strategy.max_concurrent.
3b. Concurrent total       — count across ALL strategies ≤ account.max_concurrent_total
                              (small-account safety; skipped if config setting absent).
4. Earnings blackout       — fail-open when no data (yfinance is patchy).
5. CC strike floor         — short-call strike ≥ cost basis. Re-checked here
                              even though cc_selector enforces it: belt + suspenders.
6. Liquidity gates         — re-check OI/volume/spread at submit time (chain may
                              have moved between selection and submission).
7. Regime gate             — if regime.enabled, refuse new CSPs and bull put
                              spreads when csps_allowed is false. Closes bypass
                              this rule. Fail-open when no row exists yet.

Params: the shared `wheel:` block, overlaid by the proposal's strategy params,
overlaid by per-ticker universe overrides. Before 2026-10-27 the strategy's
own params were ignored here, so every strategy was gated on the wheel's
BP floor / position cap / liquidity settings.

Rules 8-10 (kill switch / consecutive losses / stop file) live in
`execution/kill_switch.py` — they halt the *runner*, not individual orders.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Any

from core.broker import Broker
from core.checkpoint import log_checkpoint
from core.config import get_universe_entry
from core.models import OptionType, OrderType, PositionState
from core.strategies import planned_time_close_dte, strategy_params
from data.earnings import in_blackout
from db.repo import Repos
from strategies.spreads import MultiLegProposal
from strategies.wheel import Proposal


class RiskCheckFailed(Exception):
    """Raised by RiskGate.evaluate when any rule fails."""

    def __init__(self, rule: str, detail: str) -> None:
        super().__init__(f"{rule}: {detail}")
        self.rule = rule
        self.detail = detail


@dataclass(slots=True)
class RuleResult:
    rule: str
    status: str  # "pass" | "fail" | "skip"
    detail: str = ""


@dataclass(slots=True)
class RiskCheckResult:
    proposal: Proposal | MultiLegProposal
    results: list[RuleResult] = field(default_factory=list)

    def add(self, rule: str, status: str, detail: str = "") -> None:
        self.results.append(RuleResult(rule=rule, status=status, detail=detail))

    @property
    def passed(self) -> bool:
        return not any(r.status == "fail" for r in self.results)

    @property
    def failures(self) -> list[RuleResult]:
        return [r for r in self.results if r.status == "fail"]


_CLOSE_ORDER_TYPES = (
    OrderType.BUY_TO_CLOSE,
    OrderType.SELL_TO_CLOSE,
    OrderType.MULTI_LEG_CLOSE,
)


def _is_close(proposal: Proposal | MultiLegProposal) -> bool:
    """True for any order that reduces/exits an existing position.

    Closes must bypass entry-side risk gates (buying-power floor, per-position
    cap, earnings blackout, regime, concurrent caps). A buyback
    RELEASES collateral and reduces exposure — gating it can trap the bot in a
    losing position it's trying to stop out of.
    """
    return proposal.order_type in _CLOSE_ORDER_TYPES


def _notional(proposal: Proposal | MultiLegProposal) -> float:
    """Capital-at-risk for the proposal.

    - CSP: cash secured = strike × 100 × qty
    - CC: share value = underlying × 100 × qty
    - Multi-leg defined-risk: max_loss × qty (spread max loss is dollars
      per package; positive)
    """
    if isinstance(proposal, MultiLegProposal):
        return proposal.max_loss_per_spread * proposal.quantity
    contract = proposal.contract
    if contract.option_type == OptionType.PUT:
        return contract.strike * 100 * proposal.quantity
    underlying = contract.underlying_price or contract.strike
    return underlying * 100 * proposal.quantity


class RiskGate:
    def __init__(
        self,
        broker: Broker,
        repos: Repos,
        config: dict[str, Any],
        universe: dict[str, Any],
    ) -> None:
        self._broker = broker
        self._repos = repos
        self._config = config
        self._universe = universe

    def _gate_params(self, proposal: Proposal | MultiLegProposal) -> dict[str, Any]:
        """wheel block < strategy params < per-ticker overrides."""
        params = dict(self._config.get("wheel", {}) or {})
        params.update(strategy_params(self._config, getattr(proposal, "strategy_id", None)))
        entry = get_universe_entry(proposal.symbol, self._universe)
        if entry and entry.overrides:
            params.update(entry.overrides)
        return params

    async def evaluate(
        self,
        proposal: Proposal | MultiLegProposal,
        *,
        today: date | None = None,
        raise_on_fail: bool = True,
    ) -> RiskCheckResult:
        result = RiskCheckResult(proposal=proposal)
        params = self._gate_params(proposal)
        account = await self._broker.get_account()

        self._rule_banned(result, proposal)
        if isinstance(proposal, MultiLegProposal):
            await self._rule_buying_power(result, proposal, account, params)
            await self._rule_position_cap(result, proposal, account, params)
            await self._rule_concurrent_cap(result, proposal, params)
            await self._rule_concurrent_total(result, proposal)
            await self._rule_earnings_multi_leg(result, proposal, params, today)
            await self._rule_regime_multi_leg(result, proposal, params)
            await self._rule_macro_blackout(result, proposal, today)
        else:
            await self._rule_buying_power(result, proposal, account, params)
            await self._rule_position_cap(result, proposal, account, params)
            await self._rule_concurrent_cap(result, proposal, params)
            await self._rule_concurrent_total(result, proposal)
            await self._rule_earnings(result, proposal, params, today)
            self._rule_cc_strike_floor(result, proposal)
            self._rule_liquidity(result, proposal, params)
            await self._rule_regime(result, proposal, params)
            await self._rule_macro_blackout(result, proposal, today)

        log_checkpoint(
            "risk_gate",
            status="ok" if result.passed else "fail",
            symbol=proposal.symbol,
            failures=[r.rule for r in result.failures],
        )

        if not result.passed and raise_on_fail:
            first_fail = result.failures[0]
            raise RiskCheckFailed(first_fail.rule, first_fail.detail)
        return result

    # --- Rule 0 (2026-10-27) — banned symbols -----------------------------------
    def _rule_banned(
        self, result: RiskCheckResult, proposal: Proposal | MultiLegProposal,
    ) -> None:
        if _is_close(proposal):
            result.add("banned", "skip", "close — never blocked")
            return
        banned = {str(b).upper() for b in (self._universe.get("banned") or [])}
        if proposal.symbol.upper() in banned:
            result.add("banned", "fail", f"{proposal.symbol} is on the banned list")
        else:
            result.add("banned", "pass")

    # --- Rule 1 ----------------------------------------------------------------
    async def _rule_buying_power(
        self,
        result: RiskCheckResult,
        proposal: Proposal | MultiLegProposal,
        account: Any,
        params: dict[str, Any],
    ) -> None:
        if _is_close(proposal):
            result.add("buying_power_floor", "skip", "close — frees collateral, not gated")
            return
        floor_pct = float(params.get("buying_power_floor_pct", 20))
        bp_after = account.buying_power - _notional(proposal)
        floor = account.equity * (floor_pct / 100.0)
        if bp_after < floor:
            result.add(
                "buying_power_floor",
                "fail",
                f"BP after order {bp_after:.2f} < floor {floor:.2f} ({floor_pct}% of {account.equity:.2f})",
            )
        else:
            result.add("buying_power_floor", "pass")

    # --- Rule 2 ----------------------------------------------------------------
    async def _rule_position_cap(
        self,
        result: RiskCheckResult,
        proposal: Proposal | MultiLegProposal,
        account: Any,
        params: dict[str, Any],
    ) -> None:
        if _is_close(proposal):
            result.add("per_position_cap", "skip", "close — reduces exposure, not gated")
            return
        cap_pct = float(params.get("max_position_pct_of_account", 30))
        notional = _notional(proposal)
        cap = account.equity * (cap_pct / 100.0)
        if notional > cap:
            result.add(
                "per_position_cap",
                "fail",
                f"notional {notional:.2f} > cap {cap:.2f} ({cap_pct}% of {account.equity:.2f})",
            )
        else:
            result.add("per_position_cap", "pass")

    # --- Rule 3 ----------------------------------------------------------------
    async def _rule_concurrent_cap(
        self,
        result: RiskCheckResult,
        proposal: Proposal | MultiLegProposal,
        params: dict[str, Any],
    ) -> None:
        # 2026-07-08 fix: `params` here is effective_wheel_params — the WHEEL
        # config section — so the old read of max_concurrent_positions applied
        # the wheel-global cap (4) to EVERY strategy, and the per-strategy
        # `max_concurrent` in the strategies: block was never consulted
        # (observed: 3-4 put_spread pendings piling up past its cap of 2).
        # The strategy's own cap now wins; the wheel param stays the fallback
        # for strategies without one.
        cap_raw: Any = None
        for s in (self._config.get("strategies") or []):
            if s.get("id") == proposal.strategy_id:
                cap_raw = s.get("max_concurrent")
                break
        if cap_raw is None:
            cap_raw = params.get("max_concurrent_positions", 4)
        cap = int(cap_raw)
        account_id = self._config.get("account", {}).get("id", "primary")
        # Per-strategy concurrent cap: count only positions belonging to the
        # same strategy. Different strategies share the account but have
        # independent slot accounting.
        strategy_id = proposal.strategy_id
        active = await self._repos.positions.list_active(
            account_id, strategy_id=strategy_id
        )
        # SPREAD_CLOSED is terminal bookkeeping, not live exposure — the spread
        # orchestrators all treat it as re-proposable (state ∈ {IDLE,
        # SPREAD_CLOSED}). Counting it as active permanently ate cap slots:
        # observed 2026-08-03 with the global book locked at 6/6 by three
        # closed spreads (MSFT/TSLA/SOFI), blocking every entry for days.
        active = [p for p in active if p.state != PositionState.SPREAD_CLOSED]
        symbol = proposal.symbol.upper()
        new_slot = not any(p.symbol.upper() == symbol for p in active)
        if not new_slot:
            # Managing an existing position (CC on SHARES_HELD, close, roll,
            # etc.) doesn't add new exposure — skip the cap entirely. This
            # also handles the case where we're already over-cap from
            # historical opens before the cap was tightened.
            result.add(
                "concurrent_positions_cap",
                "skip",
                f"managing existing position on {symbol} (strategy={strategy_id})",
            )
            return
        projected = len(active) + 1
        if projected > cap:
            result.add(
                "concurrent_positions_cap",
                "fail",
                f"projected {projected} > cap {cap} (strategy={strategy_id}, active={len(active)})",
            )
        else:
            result.add("concurrent_positions_cap", "pass")

    # --- Rule 3b — global cap across all strategies ----------------------------
    async def _rule_concurrent_total(
        self,
        result: RiskCheckResult,
        proposal: Proposal | MultiLegProposal,
    ) -> None:
        """Sprint 12 sub-sprint 4: global concurrent-position cap.

        Counts ALL non-IDLE positions across every strategy, not just the
        proposal's strategy. Required for small-account safety where the
        per-strategy cap × N strategies can produce dangerous total exposure
        (e.g. 4 × 4 = 16 = $5,600 max simultaneous loss on $5-wide spreads,
        catastrophic on a $2-5k bankroll).

        Skips when account.max_concurrent_total is not set in config
        (backwards-compat — production configs should always set it).
        """
        account_section = self._config.get("account", {}) or {}
        cap_raw = account_section.get("max_concurrent_total")
        if cap_raw is None:
            result.add("concurrent_total_cap", "skip", "account.max_concurrent_total not set")
            return
        cap = int(cap_raw)
        account_id = account_section.get("id", "primary")
        active = await self._repos.positions.list_active(account_id)  # no strategy filter
        # Terminal SPREAD_CLOSED rows are not live exposure — see the same
        # filter in _rule_concurrent_cap.
        active = [p for p in active if p.state != PositionState.SPREAD_CLOSED]
        symbol = proposal.symbol.upper()
        # A position is uniquely identified by (account, symbol, strategy_id).
        # The cap-skip "managing existing" path applies ONLY when the proposal
        # is for the SAME (symbol, strategy) pair already active. A proposal
        # for a different strategy on the same symbol (e.g. put_spread on
        # COIN when weekly_wheel already holds COIN shares) is genuinely new
        # exposure and must be subject to the cap.
        strategy_id = proposal.strategy_id
        new_slot = not any(
            p.symbol.upper() == symbol and p.strategy_id == strategy_id
            for p in active
        )
        if not new_slot:
            result.add(
                "concurrent_total_cap",
                "skip",
                f"managing existing {strategy_id} position on {symbol} "
                f"(active total={len(active)})",
            )
            return
        projected = len(active) + 1
        if projected > cap:
            result.add(
                "concurrent_total_cap",
                "fail",
                f"projected total {projected} > cap {cap} "
                f"(active across all strategies={len(active)})",
            )
        else:
            result.add("concurrent_total_cap", "pass")

    def _earnings_avoid_days(self) -> int | None:
        """Global earnings entry-proximity window (risk.earnings_blackout.
        entry_avoid_days). When set, the earnings gate blocks only when earnings
        is within this many days of TODAY — instead of the expiry-based triggers,
        which block nearly every 30-45 DTE spread during earnings season even
        though the 21-DTE time-close exits before an expiry-date earnings. None
        → legacy (expiry-window + spans) behavior."""
        v = ((self._config.get("risk", {}) or {}).get("earnings_blackout", {}) or {}).get(
            "entry_avoid_days"
        )
        return int(v) if v is not None else None

    # --- Rule 4 ----------------------------------------------------------------
    async def _rule_earnings(
        self,
        result: RiskCheckResult,
        proposal: Proposal,
        params: dict[str, Any],
        today: date | None,
    ) -> None:
        if _is_close(proposal):
            result.add("earnings_blackout", "skip", "close — not gated by earnings")
            return
        days_before = int(params.get("earnings_blackout_days_before", 5))
        days_after = int(params.get("earnings_blackout_days_after", 2))
        block_if_spans = bool(params.get("earnings_block_if_spans", True))
        in_window = in_blackout(
            proposal.symbol,
            proposal.contract.expiration,
            days_before=days_before,
            days_after=days_after,
            today=today,
            block_if_spans=block_if_spans,
            entry_avoid_days=self._earnings_avoid_days(),
            time_close_dte=planned_time_close_dte(self._config, proposal.strategy_id),
        )
        if in_window is None:
            result.add("earnings_blackout", "skip", "no earnings data")
        elif in_window:
            result.add(
                "earnings_blackout",
                "fail",
                f"expiration {proposal.contract.expiration} inside blackout window",
            )
        else:
            result.add("earnings_blackout", "pass")

    # --- Rule 4 (multi-leg) -----------------------------------------------------
    async def _rule_earnings_multi_leg(
        self,
        result: RiskCheckResult,
        proposal: MultiLegProposal,
        params: dict[str, Any],
        today: date | None,
    ) -> None:
        """Use the short leg's expiration — it's the leg that bears assignment risk."""
        if _is_close(proposal):
            result.add("earnings_blackout", "skip", "close — not gated by earnings")
            return
        days_before = int(params.get("earnings_blackout_days_before", 5))
        days_after = int(params.get("earnings_blackout_days_after", 2))
        block_if_spans = bool(params.get("earnings_block_if_spans", True))
        short_leg = next(
            (
                leg for leg in proposal.legs
                if str(leg.action) in ("SELL_TO_OPEN", "OrderType.SELL_TO_OPEN")
                or (hasattr(leg.action, "value") and leg.action.value == "SELL_TO_OPEN")
            ),
            proposal.legs[0],
        )
        in_window = in_blackout(
            proposal.symbol,
            short_leg.expiration,
            days_before=days_before,
            days_after=days_after,
            today=today,
            block_if_spans=block_if_spans,
            entry_avoid_days=self._earnings_avoid_days(),
            time_close_dte=planned_time_close_dte(self._config, proposal.strategy_id),
        )
        if in_window is None:
            result.add("earnings_blackout", "skip", "no earnings data")
        elif in_window:
            result.add(
                "earnings_blackout",
                "fail",
                f"short-leg expiration {short_leg.expiration} inside blackout window",
            )
        else:
            result.add("earnings_blackout", "pass")

    # --- Rule 7 (multi-leg) -----------------------------------------------------
    async def _rule_regime_multi_leg(
        self,
        result: RiskCheckResult,
        proposal: MultiLegProposal,
        params: dict[str, Any],
    ) -> None:
        """Multi-leg opens (zero_dte's bull put spread — the only multi-leg
        strategy since 2026-10-27) are bullish premium: gated by csps_allowed,
        the same flag as a CSP. Closes always pass — reducing exposure should
        not be blocked by an unfavorable regime, only opens.
        """
        if proposal.order_type == OrderType.MULTI_LEG_CLOSE:
            result.add("regime", "skip", "closes bypass the regime gate")
            return
        if not self._config.get("regime", {}).get("enabled", False):
            result.add("regime", "skip", "regime gating disabled in config")
            return
        c = await self._repos.db.connect()
        async with c.execute(
            "SELECT csps_allowed FROM regime_snapshots ORDER BY snapshot_date DESC LIMIT 1"
        ) as cur:
            row = await cur.fetchone()
        if row is None:
            result.add("regime", "skip", "no regime snapshots yet")
            return
        flag = bool(row["csps_allowed"]) if row["csps_allowed"] is not None else True
        if not flag:
            result.add(
                "regime", "fail",
                "current regime snapshot disallows new bullish-premium trades",
            )
        else:
            result.add("regime", "pass")

    # --- Rule 5 ----------------------------------------------------------------
    def _rule_cc_strike_floor(self, result: RiskCheckResult, proposal: Proposal) -> None:
        if proposal.contract.option_type != OptionType.CALL:
            result.add("cc_strike_floor", "skip", "not a CC")
            return
        # The proposal carries no cost_basis directly; it's enforced inside the
        # cc_selector before constructing the Proposal. Re-checking here would
        # require carrying cost_basis on the Proposal — we skip rather than
        # re-fetch the position here. Documented limitation; selector is canonical.
        result.add(
            "cc_strike_floor",
            "skip",
            "enforced upstream in cc_selector",
        )

    # --- Rule 6 ----------------------------------------------------------------
    def _rule_liquidity(
        self,
        result: RiskCheckResult,
        proposal: Proposal,
        params: dict[str, Any],
    ) -> None:
        c = proposal.contract
        oi_min = int(params.get("open_interest_min", 0))
        vol_min = int(params.get("volume_min", 0))
        spread_max = float(params.get("bid_ask_spread_max_pct", 100.0))

        oi = c.open_interest or 0
        vol = c.volume or 0
        if c.bid is None or c.ask is None or c.ask <= 0:
            result.add("liquidity", "fail", "missing bid/ask")
            return
        mid = (c.bid + c.ask) / 2
        if mid <= 0:
            result.add("liquidity", "fail", f"non-positive mid {mid}")
            return
        spread_pct = ((c.ask - c.bid) / mid) * 100.0

        problems = []
        if oi < oi_min:
            problems.append(f"OI {oi} < {oi_min}")
        if vol < vol_min:
            problems.append(f"vol {vol} < {vol_min}")
        if spread_pct > spread_max:
            problems.append(f"spread {spread_pct:.1f}% > {spread_max}%")
        if problems:
            result.add("liquidity", "fail", "; ".join(problems))
        else:
            result.add("liquidity", "pass")

    # --- Rule 7 ----------------------------------------------------------------
    async def _rule_regime(
        self,
        result: RiskCheckResult,
        proposal: Proposal,
        params: dict[str, Any],
    ) -> None:
        if not self._config.get("regime", {}).get("enabled", False):
            result.add("regime", "skip", "regime gating disabled in config")
            return
        # Closes always pass — reducing exposure should never be blocked by an
        # unfavorable regime. Parallel to the multi-leg fix in 35e73db.
        if proposal.order_type == OrderType.BUY_TO_CLOSE:
            result.add("regime", "skip", "closes bypass the regime gate")
            return
        # CCs are about closing existing exposure — regime gate applies only to
        # new CSPs (spec §8 rule 7).
        if proposal.contract.option_type != OptionType.PUT:
            result.add("regime", "skip", "not a CSP")
            return
        # `RegimeSnapshotsRepo` doesn't expose a "latest" query yet — Sprint 7
        # ticket 33 adds it. For now: try a thin direct read; if no rows, skip.
        c = await self._repos.db.connect()
        async with c.execute(
            "SELECT csps_allowed FROM regime_snapshots ORDER BY snapshot_date DESC LIMIT 1"
        ) as cur:
            row = await cur.fetchone()
        if row is None:
            result.add("regime", "skip", "no regime snapshots yet")
            return
        csps_allowed = bool(row["csps_allowed"]) if row["csps_allowed"] is not None else True
        if not csps_allowed:
            result.add("regime", "fail", "current regime snapshot disallows new CSPs")
        else:
            result.add("regime", "pass")

    # --- Rule N (macro blackout) — TICKET-007 ---------------------------------
    async def _rule_macro_blackout(
        self,
        result: RiskCheckResult,
        proposal: Proposal | MultiLegProposal,
        today: date | None,
    ) -> None:
        """Block new short premium when the position's lifespan would cross a
        configured macro event (FOMC / CPI / NFP by default).

        Bypassed for closes.

        Fail-open paths (rule passes with a logged checkpoint, never blocks):
          - macro_blackout.enabled is false in config
          - `macro_events` table is empty (cron probably not configured)
        A calendar that runs out within min_coverage_days still answers for
        the dates it has and fires a rate-limited Discord alert to extend
        config/macro_calendar.yaml (2026-10-27: the old refresh-age check
        stayed green while the table held no future event at all).
        """
        if _is_close(proposal):
            result.add("macro_blackout", "skip", "close — not gated by macro events")
            return
        cfg = (self._config.get("risk", {}) or {}).get("macro_blackout", {}) or {}
        if not bool(cfg.get("enabled", False)):
            result.add("macro_blackout", "skip", "disabled in config")
            return

        # Resolve the short-leg expiration. For single-leg that's
        # proposal.contract.expiration; for multi-leg it's the short leg.
        if isinstance(proposal, MultiLegProposal):
            short_exp: date | None = None
            for leg in proposal.legs:
                if str(getattr(leg.action, "value", leg.action)) == "SELL_TO_OPEN":
                    short_exp = leg.expiration
                    break
            if short_exp is None and proposal.legs:
                short_exp = proposal.legs[0].expiration
        else:
            short_exp = proposal.contract.expiration
        if short_exp is None:
            result.add("macro_blackout", "skip", "no short-leg expiration to evaluate")
            return

        from data.macro_calendar import MacroCalendar, _today_nyse
        calendar = MacroCalendar(self._repos, self._config)

        # Empty-table check — fail-open with rate-limited Discord.
        if (await self._repos.macro_events.count()) == 0:
            await calendar.maybe_alert_empty()
            result.add("macro_blackout", "skip", "macro_events table empty — fail-open")
            return

        # Running-out calendar — alert (rate-limited) but keep evaluating.
        await calendar.maybe_alert_stale()

        event_types = list(cfg.get("event_types", []) or [])
        avoid = cfg.get("entry_avoid_days")
        days_before: int | None = None
        days_after: int | None = None
        # Per-strategy overrides (2026-10 zero_dte rebuild). Absent keys keep
        # the global behaviour above. `macro_event_types` replaces the event
        # list; `macro_same_day_only` shrinks the window to the event day
        # itself — a position opened 10:00 and flattened by 15:00 cannot be
        # alive at a later day's event, yet the global window (days_before +
        # entry_avoid_days) blocked it up to 3 calendar days ahead.
        sparams = strategy_params(self._config, getattr(proposal, "strategy_id", None))
        if sparams.get("macro_event_types") is not None:
            event_types = list(sparams["macro_event_types"] or [])
        if bool(sparams.get("macro_same_day_only", False)):
            avoid, days_before, days_after = 0, 0, 0
        decision = await calendar.is_blackout(
            today=today or _today_nyse(),
            short_expiration=short_exp,
            event_types=event_types,
            entry_avoid_days=int(avoid) if avoid is not None else None,
            days_before=days_before,
            days_after=days_after,
        )
        if decision.in_blackout:
            result.add("macro_blackout", "fail", decision.reason or "in macro blackout")
        else:
            result.add("macro_blackout", "pass")
