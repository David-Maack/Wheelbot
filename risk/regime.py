"""Daily macro snapshot — spec §13 #33.

Run once per trading day (cron after market close). Pulls SPY and ^VIX
daily history from yfinance and writes one `regime_snapshots` row.

Fields populated:
  spy_close, spy_sma_200, spy_above_sma
  vix_close, vix_change_pct
  choppiness                  — 14-day Choppiness Index on SPY
  regime ∈ {BULL_TREND, NEUTRAL, BEAR_TREND, HIGH_VOL}
  csps_allowed                — False when the §8 #7 gate should block

Decision rule for `csps_allowed`:
  False  if  vix_close > vix_max
         OR  vix_change_pct >= pause_on_vix_spike_pct
         OR  regime == BEAR_TREND  (per spec — if SPY < 200d SMA, paused)
  True otherwise.

Choppiness Index 14d formula (Bollinger):
    100 * log10( SUM(TR, 14) / (max(High, 14) - min(Low, 14)) ) / log10(14)
    where TR = max(High-Low, |High-prevClose|, |prevClose-Low|).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import UTC, date, datetime
from typing import Any

from core.checkpoint import checkpoint, log_checkpoint
from core.models import Regime, RegimeSnapshot
from db.repo import Repos


def _today() -> date:
    return datetime.now(UTC).date()


@dataclass(frozen=True, slots=True)
class RegimeInputs:
    spy_close: float
    spy_sma_200: float
    vix_close: float
    vix_change_pct: float
    choppiness: float


def classify_regime(inputs: RegimeInputs, *, vix_max: float, vix_spike_pct: float) -> tuple[Regime, bool]:
    """Pure-function classification — used by run_regime() and tested directly."""
    if inputs.vix_close > vix_max or inputs.vix_change_pct >= vix_spike_pct:
        return Regime.HIGH_VOL, False
    if inputs.spy_close < inputs.spy_sma_200:
        return Regime.BEAR_TREND, False
    if inputs.choppiness >= 61.8:  # standard "consolidating" threshold
        return Regime.NEUTRAL, True
    return Regime.BULL_TREND, True


def _fetch_yfinance_inputs() -> RegimeInputs | None:
    """Pull SPY + VIX history and compute the snapshot inputs. None on data gap."""
    try:
        import yfinance as yf  # noqa: PLC0415
    except ImportError:
        log_checkpoint("regime_yfinance_missing", status="fail")
        return None

    try:
        spy = yf.download("SPY", period="1y", interval="1d", progress=False, auto_adjust=False)
        vix = yf.download("^VIX", period="1mo", interval="1d", progress=False, auto_adjust=False)
    except Exception as exc:  # network etc.
        log_checkpoint("regime_yfinance_fail", status="fail", error=str(exc))
        return None
    if spy is None or spy.empty or vix is None or vix.empty:
        return None

    spy_close = float(spy["Close"].iloc[-1])
    spy_sma_200 = float(spy["Close"].tail(200).mean()) if len(spy) >= 200 else float("nan")

    vix_close = float(vix["Close"].iloc[-1])
    vix_prev = float(vix["Close"].iloc[-2]) if len(vix) >= 2 else vix_close
    vix_change_pct = ((vix_close - vix_prev) / vix_prev * 100.0) if vix_prev else 0.0

    chop = _choppiness_14d(spy)
    if math.isnan(spy_sma_200) or chop is None:
        return None
    return RegimeInputs(
        spy_close=spy_close,
        spy_sma_200=spy_sma_200,
        vix_close=vix_close,
        vix_change_pct=vix_change_pct,
        choppiness=chop,
    )


def _choppiness_14d(spy_df: Any) -> float | None:
    """14-day Bollinger Choppiness Index on SPY."""
    if len(spy_df) < 15:
        return None
    last = spy_df.tail(15)
    high = last["High"]
    low = last["Low"]
    close = last["Close"]
    prev_close = close.shift(1)
    tr = (high - low).combine((high - prev_close).abs(), max).combine((prev_close - low).abs(), max)
    tr_sum = float(tr.tail(14).sum())
    high_max = float(high.tail(14).max())
    low_min = float(low.tail(14).min())
    rng = high_max - low_min
    if rng <= 0 or tr_sum <= 0:
        return None
    return 100.0 * math.log10(tr_sum / rng) / math.log10(14)


async def run_regime(repos: Repos, config: dict[str, Any]) -> RegimeSnapshot | None:
    """Fetch inputs, classify, persist. Returns the row written (or None on gap)."""
    today = _today()
    with checkpoint("regime_snapshot", date=str(today)) as ctx:
        inputs = _fetch_yfinance_inputs()
        if inputs is None:
            ctx["skipped"] = "no_data"
            return None
        section = config.get("regime", {}) or {}
        vix_max = float(section.get("vix_max", 35))
        vix_spike = float(section.get("pause_on_vix_spike_pct", 30))
        regime, csps_allowed = classify_regime(
            inputs, vix_max=vix_max, vix_spike_pct=vix_spike
        )
        snapshot = RegimeSnapshot(
            snapshot_date=today,
            spy_close=inputs.spy_close,
            spy_sma_200=inputs.spy_sma_200,
            spy_above_sma=inputs.spy_close >= inputs.spy_sma_200,
            vix_close=inputs.vix_close,
            vix_change_pct=inputs.vix_change_pct,
            choppiness=inputs.choppiness,
            regime=regime,
            csps_allowed=csps_allowed,
            notes=None,
        )
        await _upsert(repos, snapshot)
        ctx["regime"] = regime.value
        ctx["csps_allowed"] = csps_allowed
        return snapshot


async def _upsert(repos: Repos, snapshot: RegimeSnapshot) -> None:
    c = await repos.db.connect()
    await c.execute(
        "INSERT INTO regime_snapshots "
        "(snapshot_date, spy_close, spy_sma_200, spy_above_sma, vix_close, "
        " vix_change_pct, choppiness, regime, csps_allowed, notes) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?) "
        "ON CONFLICT(snapshot_date) DO UPDATE SET "
        " spy_close = excluded.spy_close, "
        " spy_sma_200 = excluded.spy_sma_200, "
        " spy_above_sma = excluded.spy_above_sma, "
        " vix_close = excluded.vix_close, "
        " vix_change_pct = excluded.vix_change_pct, "
        " choppiness = excluded.choppiness, "
        " regime = excluded.regime, "
        " csps_allowed = excluded.csps_allowed",
        (
            snapshot.snapshot_date.isoformat(),
            snapshot.spy_close,
            snapshot.spy_sma_200,
            int(snapshot.spy_above_sma) if snapshot.spy_above_sma is not None else None,
            snapshot.vix_close,
            snapshot.vix_change_pct,
            snapshot.choppiness,
            snapshot.regime.value if hasattr(snapshot.regime, "value") else snapshot.regime,
            int(snapshot.csps_allowed) if snapshot.csps_allowed is not None else None,
            snapshot.notes,
        ),
    )
    await c.commit()
