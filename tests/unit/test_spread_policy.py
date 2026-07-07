"""Exit-policy A/B engine for the put_spread stop-loss backtest.

Synthetic price paths crafted to force each scenario:
  - rally        -> every policy profit-closes identically
  - dip-recover  -> the 2x stop realizes a loss the no-stop policy recovers from
  - crash-stay   -> the 2x stop cuts the loss vs riding to DTE-21 / expiry
"""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from backtest.spread_policy import (
    PairedTrade,
    PolicyOutcome,
    PolicySpec,
    aggregate,
    model_entry,
    paired_stop_analysis,
    run_symbol,
    spread_debit,
    walk_policy,
)

IV = 0.30
LIVE = PolicySpec(name="live_2x", profit_close_pct=50, time_close_dte=21, stop_loss_mult=2.0)
NO_STOP = PolicySpec(name="no_stop", profit_close_pct=50, time_close_dte=21, stop_loss_mult=None)
HOLD = PolicySpec(name="hold_to_exp", profit_close_pct=50, time_close_dte=None, stop_loss_mult=None)


def _bars(closes: list[float], start: str = "2026-01-05") -> pd.DataFrame:
    idx = pd.bdate_range(start, periods=len(closes))
    return pd.DataFrame({"close": closes}, index=idx)


def _iv(bars: pd.DataFrame, iv: float = IV) -> pd.Series:
    return pd.Series(iv, index=bars.index)


def _entry(bars: pd.DataFrame, i: int = 0, spot: float | None = None):
    e = model_entry(
        "TEST", bars.index[i].date(), spot or float(bars["close"].iloc[i]), IV,
        dte_days=38, short_delta=0.25, width=5.0, strike_step=1.0,
        slippage_per_share=0.05,
    )
    assert e is not None
    return e


# -- entry modeling -----------------------------------------------------------


def test_model_entry_structure():
    bars = _bars([100.0] * 5)
    e = _entry(bars)
    assert e.long_strike == e.short_strike - 5.0
    assert e.short_strike < 100.0  # OTM put
    assert e.short_strike == round(e.short_strike)  # on the $1 grid
    assert 0 < e.credit < 5.0  # credit bounded by width
    assert e.credit_fill == pytest.approx(e.credit - 0.05)
    assert e.expiration == date(2026, 2, 12)  # entry + 38 calendar days


def test_model_entry_degenerate_returns_none():
    # Spot so low the long strike would go non-positive.
    assert model_entry(
        "TEST", date(2026, 1, 5), 3.0, IV,
        dte_days=38, short_delta=0.25, width=5.0, strike_step=1.0,
        slippage_per_share=0.05,
    ) is None


def test_spread_debit_settles_intrinsic_at_expiry():
    # Spot between strikes at dte=0: intrinsic = short_k - spot.
    assert spread_debit(90.0, IV, 92.0, 87.0, 0.0) == pytest.approx(2.0)
    # Below the long strike: full width.
    assert spread_debit(80.0, IV, 92.0, 87.0, 0.0) == pytest.approx(5.0)
    # Above the short strike: worthless.
    assert spread_debit(100.0, IV, 92.0, 87.0, 0.0) == pytest.approx(0.0)


# -- policy walks ---------------------------------------------------------------


def test_rally_profit_closes_all_policies_identically():
    closes = [100.0 + 0.4 * i for i in range(45)]
    bars = _bars(closes)
    e = _entry(bars, i=0)
    outs = {
        p.name: walk_policy(e, bars, _iv(bars), p, slippage_per_share=0.05)
        for p in (LIVE, NO_STOP, HOLD)
    }
    for out in outs.values():
        assert out is not None
        assert out.exit_reason == "profit"
        assert out.pnl > 0
    assert outs["live_2x"].pnl == pytest.approx(outs["no_stop"].pnl)
    assert outs["live_2x"].exit_date == outs["no_stop"].exit_date


def test_dip_recover_stop_realizes_recoverable_loss():
    # Fast drop through the short strike, then a full recovery well before
    # the DTE-21 close. The stop locks the loss; no_stop rides it back.
    closes = [100.0, 96.0, 91.0, 86.0, 85.0, 88.0, 93.0, 99.0, 104.0] + [105.0] * 36
    bars = _bars(closes)
    e = _entry(bars, i=0)
    live = walk_policy(e, bars, _iv(bars), LIVE, slippage_per_share=0.05)
    no_stop = walk_policy(e, bars, _iv(bars), NO_STOP, slippage_per_share=0.05)
    assert live is not None and no_stop is not None
    assert live.exit_reason == "stop"
    assert live.pnl < 0
    assert no_stop.exit_reason in ("profit", "time")
    assert no_stop.pnl > live.pnl


def test_crash_and_stay_stop_cuts_the_loss():
    # Grind down through both strikes and stay there: the early stop beats
    # the DTE-21 close, which beats settling at (near) max loss.
    closes = [100.0, 97.0, 94.0, 91.0, 88.0, 85.0, 82.0, 80.0] + [80.0] * 40
    bars = _bars(closes)
    e = _entry(bars, i=0)
    live = walk_policy(e, bars, _iv(bars), LIVE, slippage_per_share=0.05)
    no_stop = walk_policy(e, bars, _iv(bars), NO_STOP, slippage_per_share=0.05)
    hold = walk_policy(e, bars, _iv(bars), HOLD, slippage_per_share=0.05)
    assert live is not None and no_stop is not None and hold is not None
    assert live.exit_reason == "stop"
    assert no_stop.exit_reason == "time"
    assert hold.exit_reason == "expiry"
    assert live.pnl > no_stop.pnl >= hold.pnl
    # Defined risk: no outcome can lose more than (width - credit) x 100.
    max_loss = -(e.width - e.credit_fill) * 100 - 5.0  # + close slippage
    assert hold.pnl >= max_loss


def test_incomplete_tail_trade_returns_none():
    bars = _bars([100.0] * 10)  # data ends long before the 38-day expiry
    e = _entry(bars, i=0)
    assert walk_policy(e, bars, _iv(bars), HOLD, slippage_per_share=0.05) is None


# -- run_symbol + aggregation ----------------------------------------------------


def test_run_symbol_pairs_every_policy_on_every_trade():
    closes = [100.0 + 0.2 * i for i in range(160)]
    bars = _bars(closes)
    trades = run_symbol("TEST", bars, [LIVE, NO_STOP, HOLD], entry_every=10)
    assert trades  # rally path -> plenty of resolvable entries
    for t in trades:
        assert set(t.outcomes) == {"live_2x", "no_stop", "hold_to_exp"}
        # Expiration always inside the data window (tail entries dropped).
        assert t.entry.expiration <= bars.index[-1].date()


def test_aggregate_and_paired_math():
    e = _entry(_bars([100.0] * 5))

    def _t(live_pnl: float, live_reason: str, alt_pnl: float) -> PairedTrade:
        return PairedTrade(entry=e, outcomes={
            "live_2x": PolicyOutcome("live_2x", date(2026, 2, 1), live_reason,
                                     0.0, live_pnl, 10),
            "no_stop": PolicyOutcome("no_stop", date(2026, 2, 5), "time",
                                     0.0, alt_pnl, 14),
        })

    trades = [
        _t(40.0, "profit", 40.0),
        _t(-150.0, "stop", 30.0),   # stop hurt: alt recovered
        _t(-150.0, "stop", -300.0),  # stop helped: alt got worse
    ]
    stats = aggregate(trades, "live_2x")
    assert stats["n"] == 3
    assert stats["win_rate"] == pytest.approx(1 / 3)
    assert stats["expectancy"] == pytest.approx((40 - 150 - 150) / 3)
    assert stats["reasons"] == {"profit": 1, "stop": 2}

    pa = paired_stop_analysis(trades, "live_2x", "no_stop")
    assert pa["n_stopped"] == 2
    assert pa["stop_hurt"] == 1
    assert pa["stop_helped"] == 1
    assert pa["alt_minus_stop_total"] == pytest.approx((30 - -150) + (-300 - -150))
