"""Diagnostic — what does the broker's option chain actually look like for a symbol?

Use this when csp_no_candidates is firing and you want to know which filter
is rejecting things.

    python -m scripts.probe_chain               # default symbol F
    python -m scripts.probe_chain --symbol BAC

Prints:
  - Total contracts returned by the broker.
  - How many fall in the configured DTE band.
  - For in-band contracts, the actual delta / bid/ask / OI / volume so you
    can see which filter is failing them.
  - If none in-band, the available DTE distribution.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from collections import Counter
from datetime import date
from typing import Any

from core.broker_factory import make_broker
from core.config import load_config
from core.models import OptionType


async def run(symbol: str, side: str) -> int:
    config = load_config()
    broker = make_broker(config)
    opt_type = OptionType.PUT if side == "put" else OptionType.CALL
    chain = await broker.get_option_chain(symbol, option_type=opt_type)
    today = date.today()

    print(f"symbol={symbol}  side={side}  total_contracts={len(chain)}")
    if not chain:
        print("(broker returned empty chain — check market-data subscription / market hours)")
        return 1

    dte_min = int(config.get("wheel", {}).get("dte_min", 30))
    dte_max = int(config.get("wheel", {}).get("dte_max", 45))
    delta_min = float(config.get("wheel", {}).get(
        "csp_delta_min" if side == "put" else "cc_delta_min", 0.20
    ))
    delta_max = float(config.get("wheel", {}).get(
        "csp_delta_max" if side == "put" else "cc_delta_max", 0.30
    ))
    oi_min = int(config.get("wheel", {}).get("open_interest_min", 0))
    vol_min = int(config.get("wheel", {}).get("volume_min", 0))
    spread_max = float(config.get("wheel", {}).get("bid_ask_spread_max_pct", 100.0))
    print(
        f"filters: dte={dte_min}-{dte_max} |delta|={delta_min}-{delta_max} "
        f"oi>={oi_min} vol>={vol_min} spread<={spread_max}%"
    )

    in_band = [c for c in chain if dte_min <= (c.expiration - today).days <= dte_max]
    print(f"in DTE band: {len(in_band)}")

    if not in_band:
        cnt = Counter((c.expiration - today).days for c in chain)
        print("available DTEs (count):")
        for dte, n in sorted(cnt.items())[:20]:
            print(f"  dte={dte:>3}  n={n}")
        return 0

    print("\nin-band contracts (first 20):")
    print(f"  {'occ':<25} {'strike':>7} {'dte':>4} {'delta':>7} {'bid':>6} {'ask':>6} {'oi':>7} {'vol':>7}  {'reason':<30}")
    rejected: dict[str, int] = Counter()
    for c in in_band[:20]:
        dte = (c.expiration - today).days
        delta_abs = abs(c.delta) if c.delta is not None else None
        spread_pct: Any = None
        if c.bid is not None and c.ask is not None and c.ask > 0:
            mid = (c.bid + c.ask) / 2
            if mid > 0:
                spread_pct = (c.ask - c.bid) / mid * 100

        reason = "PASS"
        if delta_abs is None:
            reason = "no_delta"
        elif delta_abs < delta_min or delta_abs > delta_max:
            reason = f"delta_outside_band ({delta_abs:.2f})"
        elif (c.open_interest or 0) < oi_min:
            reason = f"oi_low ({c.open_interest})"
        elif (c.volume or 0) < vol_min:
            reason = f"vol_low ({c.volume})"
        elif spread_pct is None:
            reason = "no_spread"
        elif spread_pct > spread_max:
            reason = f"spread_wide ({spread_pct:.1f}%)"

        rejected[reason] += 1
        print(
            f"  {c.occ_symbol:<25} {c.strike:>7} {dte:>4} "
            f"{(c.delta or 0):>7.2f} {(c.bid or 0):>6.2f} {(c.ask or 0):>6.2f} "
            f"{(c.open_interest or 0):>7} {(c.volume or 0):>7}  {reason}"
        )

    print("\nfilter outcomes across in-band contracts:")
    for reason, n in rejected.most_common():
        print(f"  {reason:<30} {n}")

    if hasattr(broker, "aclose"):
        await broker.aclose()
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbol", default="F")
    parser.add_argument("--side", choices=["put", "call"], default="put")
    args = parser.parse_args(argv)
    return asyncio.run(run(args.symbol, args.side))


if __name__ == "__main__":
    sys.exit(main())
