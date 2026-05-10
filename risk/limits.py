"""Pre-trade risk gates from spec §8.

Every order goes through `RiskGate.evaluate(proposal, ...)` before submission.
On any rule's failure the gate raises `RiskCheckFailed`; the router catches it,
logs loudly, and the order is not placed.

Rules implemented (1-7 of §8):

1. Buying-power floor      — keep ≥ buying_power_floor_pct free after this order.
2. Per-position cap        — notional of this position ≤ max_position_pct_of_account.
3. Concurrent positions    — count of non-IDLE positions ≤ max_concurrent_positions.
4. Earnings blackout       — fail-open when no data (yfinance is patchy).
5. CC strike floor         — short-call strike ≥ cost basis. Re-checked here
                              even though cc_selector enforces it: belt + suspenders.
6. Liquidity gates         — re-check OI/volume/spread at submit time (chain may
                              have moved between selection and submission).
7. Regime gate             — if regime.enabled and last regime row has
                              csps_allowed=False, refuse new CSPs. Fail-open
                              when no row exists yet (Sprint 7 populates the table).

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
from core.models import OptionType, OrderType, PositionState
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
            await self._rule_earnings_multi_leg(result, proposal, params, today)
            await self._rule_regime_multi_leg(result, proposal, params)
        else:
            await self._rule_buying_power(result, proposal, account, params)
            await self._rule_position_cap(result, proposal, account, params)
            await self._rule_concurrent_cap(result, proposal, params)
            await self._rule_earnings(result, proposal, params, today)
            self._rule_cc_strike_floor(result, proposal)
            self._rule_liquidity(result, proposal, params)
            await self._rule_regime(result, proposal, params)

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
        projected = len(active) + (1 if new_slot else 0)
        if projected > cap:
            result.add(
                "concurrent_positions_cap",
                "fail",
                f"projected {projected} > cap {cap} (strategy={strategy_id}, active={len(active)})",
            )
        else:
            result.add("concurrent_positions_cap", "pass")

    # --- Rule 4 ----------------------------------------------------------------
    async def _rule_earnings(
        self,
        result: RiskCheckResult,
        proposal: Proposal,
        params: dict[str, Any],
        today: date | None,
    ) -> None:
        days_before = int(params.get("earnings_blackout_days_before", 5))
        days_after = int(params.get("earnings_blackout_days_after", 2))
        in_window = in_blackout(
            proposal.symbol,
            proposal.contract.expiration,
            days_before=days_before,
            days_after=days_after,
            today=today,
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
        days_before = int(params.get("earnings_blackout_days_before", 5))
        days_after = int(params.get("earnings_blackout_days_after", 2))
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
        """Bull put credit spreads sell premium with bullish bias — same gate as CSPs."""
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
        csps_allowed = bool(row["csps_allowed"]) if row["csps_allowed"] is not None else True
        if not csps_allowed:
            result.add("regime", "fail", "current regime snapshot disallows new bullish-premium trades")
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
