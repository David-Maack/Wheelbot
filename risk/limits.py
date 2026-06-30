"""Pre-trade risk gates from spec §8.

Every order goes through `RiskGate.evaluate(proposal, ...)` before submission.
On any rule's failure the gate raises `RiskCheckFailed`; the router catches it,
logs loudly, and the order is not placed.

Rules implemented (1-7 of §8):

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
7. Regime gate             — if regime.enabled, refuse new entries when the
                              direction-appropriate flag is false:
                                CSPs / bull put spreads → csps_allowed
                                bear call spreads        → bear_calls_allowed
                              Closes (MULTI_LEG_CLOSE) bypass this rule.
                              Fail-open when no row exists yet.

Rules 8-10 (kill switch / consecutive losses / stop file) live in
`execution/kill_switch.py` — they halt the *runner*, not individual orders.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Any

from core.broker import Broker
from core.checkpoint import log_checkpoint
from core.config import effective_wheel_params
from core.models import OptionType, OrderStatus, OrderType
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
    cap, earnings blackout, regime, tier-2 screen, concurrent caps). A buyback
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
    # A long-option OPEN (BUY_TO_OPEN — PMCC LEAP, swing long) risks only the
    # DEBIT paid (the max loss), NOT the underlying notional. Without this a 1-
    # contract deep-ITM SPY long looks like ~$73k of exposure and trips the
    # per-position cap, even though we only put ~$4.5k at risk.
    if proposal.order_type == OrderType.BUY_TO_OPEN:
        mid = contract.mid
        if mid is None and contract.bid is not None and contract.ask is not None:
            mid = (contract.bid + contract.ask) / 2
        if mid is not None and mid > 0:
            return mid * 100 * proposal.quantity
        # No quote — fall through to the conservative underlying bound.
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
        # Ids of "swing"-type strategies — exempt from the CSP-oriented regime
        # gate (they carry their own 200-SMA direction gate in the signal).
        self._swing_ids = {
            s.get("id") for s in (config.get("strategies", []) or [])
            if s.get("type") == "swing"
        }

    async def evaluate(
        self,
        proposal: Proposal | MultiLegProposal,
        *,
        today: date | None = None,
        raise_on_fail: bool = True,
    ) -> RiskCheckResult:
        result = RiskCheckResult(proposal=proposal)
        params = effective_wheel_params(proposal.symbol, self._config, self._universe)
        account = await self._broker.get_account()

        if isinstance(proposal, MultiLegProposal):
            await self._rule_buying_power(result, proposal, account, params)
            await self._rule_position_cap(result, proposal, account, params)
            await self._rule_concurrent_cap(result, proposal, params)
            await self._rule_concurrent_total(result, proposal)
            await self._rule_earnings_multi_leg(result, proposal, params, today)
            await self._rule_regime_multi_leg(result, proposal, params)
            await self._rule_tier2_screen(result, proposal)
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
            await self._rule_tier2_screen(result, proposal)
            await self._rule_macro_blackout(result, proposal, today)
            # TICKET-015: PMCC-specific gates (no-op for non-pmcc proposals).
            self._rule_pmcc_position_cap(result, proposal, params)
            await self._rule_pmcc_short_above_breakeven(result, proposal)

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
        cap = int(params.get("max_concurrent_positions", 4))
        account_id = self._config.get("account", {}).get("id", "primary")
        # Per-strategy concurrent cap: count only positions belonging to the
        # same strategy. Different strategies share the account but have
        # independent slot accounting.
        strategy_id = proposal.strategy_id
        active = await self._repos.positions.list_active(
            account_id, strategy_id=strategy_id
        )
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
        """Dispatch the regime gate by spread direction.

        - bull_put: gated by csps_allowed (same gate as CSPs — bullish premium).
        - bear_call: gated by bear_calls_allowed (bearish premium, opposite gate).

        Closes always pass — reducing exposure should not be blocked by an
        unfavorable regime, only opens.
        """
        if proposal.order_type == OrderType.MULTI_LEG_CLOSE:
            result.add("regime", "skip", "closes bypass the regime gate")
            return
        if not self._config.get("regime", {}).get("enabled", False):
            result.add("regime", "skip", "regime gating disabled in config")
            return

        # Which column(s) to read depends on direction. For iron_condor we
        # AND-combine csps_allowed + bear_calls_allowed (both wings must be
        # independently allowed by the regime). This fallback is semantically
        # correct for the current 4-bucket regime: BULL_TREND fails the call
        # wing, BEAR_TREND fails the put wing, NEUTRAL allows both, HIGH_VOL
        # fails both. When TICKET-013's 6-bucket regime adds an explicit
        # `iron_condor_allowed` column (matrix can diverge from AND in
        # NEUTRAL_HIGH_IV etc.), swap this branch to read that column.
        # TICKET-016: a calendar wants a NEUTRAL, range-bound regime — which in
        # the 4-bucket scheme is exactly "both directional wings allowed", the
        # same AND-combine as iron_condor. (The LOW-IV requirement is enforced
        # separately by the selector's inverted IVR gate, not here.) When
        # TICKET-013's 6-bucket regime lands, swap to BULL_TREND_LOW_IV /
        # NEUTRAL_LOW_IV.
        if proposal.direction in ("iron_condor", "calendar"):
            c = await self._repos.db.connect()
            async with c.execute(
                "SELECT csps_allowed, bear_calls_allowed FROM regime_snapshots "
                "ORDER BY snapshot_date DESC LIMIT 1"
            ) as cur:
                row = await cur.fetchone()
            if row is None:
                result.add("regime", "skip", "no regime snapshots yet")
                return
            csps_ok = bool(row["csps_allowed"]) if row["csps_allowed"] is not None else True
            bear_ok = bool(row["bear_calls_allowed"]) if row["bear_calls_allowed"] is not None else True
            if csps_ok and bear_ok:
                result.add("regime", "pass")
            else:
                missing = []
                if not csps_ok:
                    missing.append("put side (csps_allowed=False)")
                if not bear_ok:
                    missing.append("call side (bear_calls_allowed=False)")
                result.add(
                    "regime", "fail",
                    f"{proposal.direction} requires a range-bound regime; blocked: {', '.join(missing)}",
                )
            return

        if proposal.direction == "bear_call":
            flag_column = "bear_calls_allowed"
            fail_detail = "current regime snapshot disallows new bearish-premium trades"
        else:
            flag_column = "csps_allowed"
            fail_detail = "current regime snapshot disallows new bullish-premium trades"

        c = await self._repos.db.connect()
        async with c.execute(
            f"SELECT {flag_column} FROM regime_snapshots ORDER BY snapshot_date DESC LIMIT 1"
        ) as cur:
            row = await cur.fetchone()
        if row is None:
            result.add("regime", "skip", "no regime snapshots yet")
            return
        flag = bool(row[flag_column]) if row[flag_column] is not None else True
        if not flag:
            result.add("regime", "fail", fail_detail)
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
        # Sub-sprint 2.4: swing strategies carry their OWN 200-SMA regime gate
        # inside the signal (direction = sign(close − 200SMA)). The CSP-oriented
        # regime rule must not second-guess them — it would wrongly block a
        # (bearish) swing PUT in a down regime, exactly when the signal wants it.
        if getattr(proposal, "strategy_id", None) in self._swing_ids:
            result.add("regime", "skip", "swing: regime handled by the strategy's own 200-SMA gate")
            return
        # CCs are about closing existing exposure — regime gate applies only to
        # new CSPs (spec §8 rule 7).
        # TICKET-015: a PMCC long (BUY_TO_OPEN call) IS a new bullish directional
        # bet — gate it on csps_allowed (the same bullish-premium flag a CSP
        # uses). A PMCC short (SELL_TO_OPEN call) is covered premium against the
        # long, so it falls through to the non-PUT skip below like a wheel CC.
        is_pmcc_long = (
            getattr(proposal, "strategy_id", None) == "pmcc"
            and proposal.order_type == OrderType.BUY_TO_OPEN
        )
        if proposal.contract.option_type != OptionType.PUT and not is_pmcc_long:
            result.add("regime", "skip", "not a CSP / PMCC long")
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

    # --- TICKET-015: PMCC rules ------------------------------------------------
    def _pmcc_param(self, key: str, default: Any = None) -> Any:
        """Read a value from the pmcc strategy's params block. The gate's
        `params` arg is effective_wheel_params (wheel-centric) and does NOT
        carry per-strategy params, so PMCC rules read straight from the
        strategies registry in config."""
        for s in self._config.get("strategies", []) or []:
            if s.get("id") == "pmcc":
                return (s.get("params", {}) or {}).get(key, default)
        return default

    def _rule_pmcc_position_cap(
        self,
        result: RiskCheckResult,
        proposal: Proposal,
        params: dict[str, Any],
    ) -> None:
        """The long-LEAP debit must fit max_capital_per_position_usd. Only
        applies to the PMCC long entry (BUY_TO_OPEN). Belt-and-suspenders with
        the selector's own cap check. Reads the cap from the strategies
        registry (see _pmcc_param)."""
        if getattr(proposal, "strategy_id", None) != "pmcc":
            result.add("pmcc_position_cap", "skip", "not a PMCC proposal")
            return
        if proposal.order_type != OrderType.BUY_TO_OPEN:
            result.add("pmcc_position_cap", "skip", "not a PMCC long entry")
            return
        cap = float(self._pmcc_param("max_capital_per_position_usd", 0) or 0)
        if cap <= 0:
            result.add("pmcc_position_cap", "skip", "no cap configured")
            return
        c = proposal.contract
        if c.bid is None or c.ask is None:
            result.add("pmcc_position_cap", "skip", "no quote for debit estimate")
            return
        debit = ((c.bid + c.ask) / 2) * 100 * proposal.quantity
        if debit > cap:
            result.add(
                "pmcc_position_cap", "fail",
                f"long debit ${debit:.0f} exceeds cap ${cap:.0f}",
            )
        else:
            result.add("pmcc_position_cap", "pass")

    async def _rule_pmcc_short_above_breakeven(
        self,
        result: RiskCheckResult,
        proposal: Proposal,
    ) -> None:
        """A PMCC short call must sit STRICTLY above the long's breakeven so an
        assigned short is covered at a profit. Only applies to the PMCC short
        entry (SELL_TO_OPEN call). Belt-and-suspenders with the selector."""
        if getattr(proposal, "strategy_id", None) != "pmcc":
            result.add("pmcc_short_above_breakeven", "skip", "not a PMCC proposal")
            return
        if proposal.order_type != OrderType.SELL_TO_OPEN or proposal.contract.option_type != OptionType.CALL:
            result.add("pmcc_short_above_breakeven", "skip", "not a PMCC short entry")
            return
        account_id = self._config.get("account", {}).get("id", "primary")
        position = await self._repos.positions.get_by_symbol(
            account_id, proposal.symbol, strategy_id="pmcc"
        )
        if position is None or position.current_cycle_id is None:
            # No open long to cover — selector shouldn't have produced this;
            # fail closed.
            result.add(
                "pmcc_short_above_breakeven", "fail",
                "no open PMCC long position to cover the short",
            )
            return
        c = await self._repos.db.connect()
        async with c.execute(
            "SELECT strike, fill_price FROM orders WHERE cycle_id = ? "
            "AND order_type = ? AND option_type = ? AND status = ? "
            "ORDER BY placed_at DESC LIMIT 1",
            (
                position.current_cycle_id,
                OrderType.BUY_TO_OPEN.value,
                OptionType.CALL.value,
                OrderStatus.FILLED.value,
            ),
        ) as cur:
            row = await cur.fetchone()
        if row is None or row["strike"] is None or row["fill_price"] is None:
            result.add(
                "pmcc_short_above_breakeven", "fail",
                "cannot recover long breakeven (no filled long order)",
            )
            return
        breakeven = float(row["strike"]) + abs(float(row["fill_price"]))
        if proposal.contract.strike > breakeven:
            result.add("pmcc_short_above_breakeven", "pass")
        else:
            result.add(
                "pmcc_short_above_breakeven", "fail",
                f"short strike {proposal.contract.strike} not above long "
                f"breakeven {breakeven:.2f}",
            )

    # --- Rule 8 (Sprint 14) — Tier-2 LLM screen --------------------------------
    async def _rule_tier2_screen(
        self,
        result: RiskCheckResult,
        proposal: Proposal | MultiLegProposal,
    ) -> None:
        """Tier-2 tickers require an LLM-screener candidate row from today
        with a score at or above the configured threshold. Spec §6 intent:
        tier-2 names carry catalyst / news risk that the rule-based selector
        doesn't see; the LLM has 7 days of headlines + earnings calendar.

        Skips when:
          - LLM screener disabled in config
          - Order is a close (this is an entry gate, not a close trigger)
          - Bearish-direction proposals (bear_call_spread) — current screener
            prompt is bullish-biased; using it to gate bear-calls would be
            wrong. Separate bearish screener is a future feature.
          - Ticker is tier-1 (the LLM doesn't add edge on mega-caps / ETFs
            we already trust)
          - Ticker isn't in the universe at all (shouldn't happen — but skip)
        """
        intel = self._config.get("intelligence", {}) or {}
        if not bool(intel.get("llm_screener_enabled", False)):
            result.add("tier2_screen", "skip", "screener disabled")
            return

        # Closes always pass — entry gate only.
        if proposal.order_type in (
            OrderType.BUY_TO_CLOSE, OrderType.SELL_TO_CLOSE,
            OrderType.MULTI_LEG_CLOSE,
        ):
            result.add("tier2_screen", "skip", "close — gate is entry-only")
            return

        # Bear-call direction: current screener prompt is bullish-biased.
        if isinstance(proposal, MultiLegProposal) and proposal.direction == "bear_call":
            result.add(
                "tier2_screen", "skip",
                "bear_call direction — screener is bullish-biased",
            )
            return

        # Look up universe entry; tier-1 bypasses the rule.
        symbol = proposal.symbol.upper()
        entry = None
        for t in self._universe.get("tickers", []):
            if t.symbol.upper() == symbol:
                entry = t
                break
        if entry is None:
            result.add("tier2_screen", "skip", f"{symbol} not in universe")
            return
        if entry.tier != 2:
            result.add("tier2_screen", "skip", f"{symbol} is tier-{entry.tier}")
            return

        # Tier-2 entry — require today's candidate row with score >= threshold.
        threshold = float(intel.get("tier2_min_score", 50.0))
        c = await self._repos.db.connect()
        async with c.execute(
            "SELECT score FROM candidates "
            "WHERE symbol = ? AND run_date = date('now') "
            "ORDER BY score DESC LIMIT 1",
            (symbol,),
        ) as cur:
            row = await cur.fetchone()
        if row is None:
            result.add(
                "tier2_screen", "fail",
                f"no LLM screener row for tier-2 {symbol} today — screener may not have run",
            )
            return
        score = float(row["score"]) if row["score"] is not None else 0.0
        if score < threshold:
            result.add(
                "tier2_screen", "fail",
                f"LLM score {score:.1f} < threshold {threshold:.1f} for {symbol}",
            )
        else:
            result.add(
                "tier2_screen", "pass",
                f"LLM score {score:.1f} ≥ {threshold:.1f} for {symbol}",
            )

    # --- Rule N (macro blackout) — TICKET-007 ---------------------------------
    async def _rule_macro_blackout(
        self,
        result: RiskCheckResult,
        proposal: Proposal | MultiLegProposal,
        today: date | None,
    ) -> None:
        """Block new short premium when the position's lifespan would cross a
        configured macro event (FOMC / CPI / NFP by default).

        Bypassed for closes. **NOT** bypassed for bear_call_spread — unlike
        `_rule_tier2_screen` (whose bypass exists because the screener's
        prompt is bullish-biased), macro events gap BOTH directions, so the
        bear_call rule needs the same protection as bull_put.

        Fail-open paths (rule passes with a logged checkpoint, never blocks):
          - macro_blackout.enabled is false in config
          - `macro_events` table is empty (cron probably not configured)
          - last refresh > stale_threshold_hours ago (cron probably broken)
        Each fail-open also fires a rate-limited Discord alert so the
        operator notices instead of silently flying blind.
        """
        if _is_close(proposal):
            result.add("macro_blackout", "skip", "close — not gated by macro events")
            return
        # TICKET-015: a long-option OPEN (the PMCC LEAP, BUY_TO_OPEN) is not
        # short premium — a macro gap caps a buyer's risk at the debit, unlike a
        # seller's. And a 90-180 DTE long's lifespan ALWAYS spans a monthly
        # macro event, which would block the PMCC long perpetually. Skip it
        # (the directional risk is covered by the bullish_long news_check).
        if proposal.order_type == OrderType.BUY_TO_OPEN:
            result.add("macro_blackout", "skip", "long-option open — not short premium")
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

        # Stale-calendar check — fail-open with rate-limited Discord.
        if await calendar.is_stale():
            await calendar.maybe_alert_stale()
            result.add("macro_blackout", "skip", "macro calendar stale — fail-open")
            return

        event_types = list(cfg.get("event_types", []) or [])
        decision = await calendar.is_blackout(
            today=today or _today_nyse(),
            short_expiration=short_exp,
            event_types=event_types,
        )
        if decision.in_blackout:
            result.add("macro_blackout", "fail", decision.reason or "in macro blackout")
        else:
            result.add("macro_blackout", "pass")
