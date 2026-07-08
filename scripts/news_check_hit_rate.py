"""News-check hit-rate report — CLI wrapper around intelligence/news_hit_rate.

Answers: does the Haiku catalyst sniff predict anything? In advisory mode
proceed AND caution entries both trade at full size, so realized cycle P&L
per bucket is a clean natural experiment. This report is the evidence gate
for flipping `intelligence.news_check_advisory` to false before live.

Read-only. Run on the box with the live DB:
    docker exec wheelbot python -m scripts.news_check_hit_rate
"""

from __future__ import annotations

import asyncio

from core.config import load_config
from db.repo import Database, Repos
from intelligence.news_hit_rate import compute_hit_rate


def _fmt_bucket(name: str, s) -> str:
    win = f"{s.win_rate * 100:5.1f}%" if s.win_rate is not None else "    —"
    avg = f"${s.avg_pnl:8.2f}" if s.avg_pnl is not None else "       —"
    return (
        f"{name:<8} decisions={s.n_decisions:<4} matched={s.n_matched:<4} "
        f"closed={s.n_closed:<4} win={win}  avg={avg}  total=${s.total_pnl:9.2f}"
    )


async def main() -> None:
    config = load_config()
    db = Database(config["database"]["path"])
    repos = Repos(db)
    try:
        report = await compute_hit_rate(repos)
    finally:
        await db.close()

    print(f"news_check hit-rate — {report['n_total']} NEWS_CHECK decisions "
          f"({report['n_unparsed']} unparsed pre-fix rows excluded)")
    for name in ("proceed", "caution"):
        print(_fmt_bucket(name, report["buckets"][name]))
    blocks = report["blocks"]
    print(f"block    decisions={len(blocks):<4} (no trade — no realized outcome)")
    for b in blocks:
        print(f"    {b['created_at']}  {b['symbol']}  conf={b['confidence']}")
    print(f"\nverdict: {report['verdict']}")


if __name__ == "__main__":
    asyncio.run(main())
