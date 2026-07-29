# Automated 0DTE Options Trading — Research Report (July 2026)

Prepared for the WheelBot 0DTE module design. Audience: developer running a Python options bot on an Alpaca **paper** account, 5-minute tick loop, existing defined-risk credit-spread infrastructure (iron condors, verticals) on SPY/QQQ with deterministic risk gates.

---

## TL;DR

- **The dominant documented systematic 0DTE approach is the morning defined-risk premium sale**: iron condor or credit spread entered ~9:32–10:15 ET, shorts at 10–20 delta, profit target 25–35% of credit, stop 1–2x credit (or structural stop via narrow wings), flat well before the close. Multiple independent sources converge on these parameters ([Option Alpha 230k-trade study](https://optionalpha.com/blog/0dte-options-strategy-performance), [Theta Profits 9,100 live trades](https://www.thetaprofits.com/my-most-profitable-options-trading-strategy-0dte-breakeven-iron-condor/), [OptionsDecay](https://www.optionsdecay.com/0dte-iron-condor/), [Options Trading IQ / Option Omega](https://optionstradingiq.com/option-omega/)).
- **The academic verdict is sobering**: the intraday variance risk premium is tiny (~0.0011% of underlying from 10:00 ET to close), and iron condors/butterflies that look good gross flip to **negative** Sharpe after half-spread + slippage ([Vilkov et al.](https://github.com/vilkovgr/0dte-strategies/blob/main/docs/paper/paper-annotated.md)). Retail 0DTE traders lose ~$350k/day in aggregate ([Beckmeyer, Branger & Gayda](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4404704)) — but the *losing* subset is single-leg, debit-paid, high-IV buying; **multi-leg premium-capture trades are the relatively profitable subset**.
- **Fill assumptions are the whole ballgame.** One 7-year, 1-minute-resolution SPY credit-spread backtest found that switching from mid-fills to a realistic maker-fill model **cut CAGR 30–60% across the entire strategy grid**, and that at short DTE "the stop loss isn't a fine-tuning parameter, it's the entire strategy" ([FlashAlpha](https://flashalpha.com/articles/spy-put-credit-spread-active-backtest-mm-fills-vrp-signal-drawdown-breaker)).
- **A 5-minute loop is fine for entries but too slow for 0DTE stop management.** 0DTE spread deltas move ~10x faster than 45DTE equivalents; a position can go from +$150 to −$500 in five minutes ([Cboe/Henry Schwartz](https://www.cboe.com/insights/posts/henry-schwartzs-zero-day-spx-iron-condor-strategy-a-deep-dive/)). Stops must be **resting broker-side orders** (OCO stop-limit + stop-market) or a sub-30-second monitor — or eliminated entirely by using narrow-width spreads as structural stops ([zerodte.com](https://zerodte.com/outsmarting-slippage-why-narrow-vertical-spreads-beat-traditional-stops/)).
- **Alpaca specifics**: index options (SPX/SPXW/XSP/VIX) landed in Alpaca **paper trading on July 23, 2026, but market data for them is NOT yet available** — so XSP is not yet usable for a data-driven bot; SPY/QQQ remain the practical underlyings ([Alpaca announcement](https://alpaca.markets/blog/alpaca-introduces-index-options-paper-trading/)). On expiration day Alpaca **rejects option orders after ~3:15 PM ET (3:30 for broad-based ETFs) and auto-liquidates expiring positions at 3:30/3:45 PM ET** — the bot's day effectively ends at 3:15 ([Alpaca 0DTE guide](https://alpaca.markets/learn/how-to-trade-0dte-options-on-alpaca)). Alpaca's paper fill simulator fills marketable orders against NBBO with **no liquidity/size constraint and no slippage** (10% random partial fills only) — optimistic by construction; run a parallel penalized P&L ([Alpaca paper docs](https://docs.alpaca.markets/us/docs/paper-trading)).
- **Recommended first paper tests**: (1) morning iron condor, 10:00–10:15 entry, ~15Δ shorts, PT 30%, structural or 1x-credit stop, hard flatten 14:45–15:00 ET; (2) trend-filtered directional credit spread (SMA5-style filter) with **narrow width as the stop** (no stop orders — kills the biggest sim-to-live gap); (3) staggered multi-entry condors (Theta Profits/METF style) only after the slippage model is validated, because it leans hardest on stop-fill quality.

---

## 1. Documented systematic 0DTE approaches

### 1.1 Option Alpha's aggregate autotrading data (largest retail sample)

Option Alpha analyzed **230,000 live-automated 0DTE trades** over 483 trading days from Sept 2021 ([source](https://optionalpha.com/blog/0dte-options-strategy-performance)):

- **Strategy mix**: iron butterflies + iron condors = **78% of positions**; SPY = 81% of trades, QQQ second (98% of volume in SPY/QQQ/IWM).
- **Win rates**: iron condors **70.2%**, iron butterflies **66.8%** — yet only **49.6% of traders were profitable**, i.e., high win rate ≠ positive expectancy.
- **Profitable-trader pattern**: enter **~10:15 ET** (after opening volatility), exit **~12:00 ET**, average hold ~2 hours; profit target ~**15%**, stop ~**−25%**, time-based exit at noon if neither hits.
- **Calendar effects**: Monday and Wednesday most profitable; **Thursday consistently negative**.
- **Best filtered setups**: short call spreads on an SMA5 sell signal / short put spreads on an SMA5 buy signal exceeded **75% win rate with profit factor > 2.0** (~$1.8M cumulative across 37,894 trades in the studied cohort).
- Assignment overrides hit <0.005% of trades (bots that close before expiry make American-style SPY assignment a non-issue in practice).
- Noted profitability slowdown entering 2023 after daily expirations launched.

Option Alpha also publishes 0DTE bot templates, including a "peg research"-based SPX condor and a 1DTE condor entered after 11:45 ET ([templates](https://optionalpha.com/templates), [peg bot](https://optionalpha.com/videos/peg-research-based-0dte-spx-bot)), and a [0DTE backtester](https://optionalpha.com/backtester).

### 1.2 Theta Profits "Breakeven Iron Condor" (longest documented live track record)

**9,100 live SPX trades, April 2021 – February 2026** ([source](https://www.thetaprofits.com/my-most-profitable-options-trading-strategy-0dte-breakeven-iron-condor/)):

- **Entry**: staggered entries at regular intervals through the day (≥30 min apart, typically hourly); shorts at **10–15 delta**, longs **30 SPX points** further out; collect $100–$200 per side, equal credit both sides.
- **Stops**: set immediately on entry, **per side**, on **short legs only** (reduces slippage; longs closed manually after a stop fires); loss per side = total premium of the whole condor ("breakeven" construction); implemented as **OCO stop-limit + backup stop-market**; stops adjusted intraday for theta.
- **Exits**: 5-cent take-profit on each short.
- **Results**: **40% win rate**, average win ≈ **2.2x** average loss, avg net profit/trade 0.28%, premium capture 5.65% (declined to 3.53% in 2024), **49 of 57 months profitable**, double-stop (both sides stopped) on **8.6%** of trades.
- **Risk rules**: never risk >1–2% of the account per day; ≤50% of buying power.

His companion article on stops documents the mechanics and the **August 2023 Cboe liquidity change** that eliminated the SPX quote-spike problem that used to blow through stop-limits ([source](https://www.thetaprofits.com/stop-loss-on-credit-spreads-in-0dte-options-trading/)).

### 1.3 OptionsDecay flagship rules (representative of the retail "mechanical condor" school)

([source](https://www.optionsdecay.com/0dte-iron-condor/)) — entry **9:32 ET**; shorts ~**20 delta**; wings **$100** SPX (a $25 variant exists); **PT 35%** of credit; **SL 50%** of credit; flatten **by ~3:30 ET** ("respect gamma in the final 30 minutes"); skip or downsize FOMC/CPI/NFP days; size 2–5% of equity per trade; "honor the stop, every time."

### 1.4 Option Omega published backtest (and its cautionary tale)

Options Trading IQ's Option Omega backtest of an SPX 0DTE condor — entry **9:35**, **14Δ** shorts, **35-pt wings**, **PT 30%**, timed exit **10:59** — over Jan–Aug 2022: **82.7% win rate**, but average loss (−$2,181) was **~2.1x** average win ($1,052); largest single loss **−$11,128**; max drawdown −27.8% *with 100% capital allocation and compounding* — the headline 5,497% CAGR is a sizing artifact, not an edge measurement ([source](https://optionstradingiq.com/option-omega/)). It did model $0.05 slippage/leg and commissions. Lesson: 0DTE condor backtests are exquisitely sensitive to sizing and to a handful of tail days.

### 1.5 Prop/desk and platform ecosystem

- **Cboe's Henry Schwartz** walked through a zero-day SPX condor (10-pt wings, ~$1.00 credit per side, ~9:1 risk:reward per side) and stressed sizing over management: a position "up 150 bucks five minutes before close" can be "down 400 or 500" on one small move ([Cboe insights](https://www.cboe.com/insights/posts/henry-schwartzs-zero-day-spx-iron-condor-strategy-a-deep-dive/)).
- **Trade Automation Toolbox (TAT)** is the most-used retail 0DTE automation platform (hundreds of daily traders, Discord community); it supports 0DTE credit spreads/condors/flies with broker-side stop monitoring and management as its flagship feature, and popularized **METF** — six fixed-time entries per day of EMA-trend-following SPX credit spreads ([TAT](https://tradeautomationtoolbox.com/), [METF write-up](https://www.thetaprofits.com/how-to-trade-the-metf-0dte-options-strategy/), [options1k strategies](https://options1k.com/0dte-strategy/)).
- **Academic/quant systematic study** (Vilkov et al., SPXW 30-min bars, Sept 2016–Jan 2026): seven strategy families at 10:00 ET entry held to close. Best **conditional** performer out-of-sample: **put ratio spreads (net SR 0.93)**; a top-3 basket 0.82 net. Unconditional condors/flies: **+0.77 gross SR → −0.20 net SR** after half-spread + 0.5bp slippage. Time-of-day entries (10:00/13:00/15:00) gave broadly similar core results ([annotated paper](https://github.com/vilkovgr/0dte-strategies/blob/main/docs/paper/paper-annotated.md)).

---

## 2. Empirical evidence: what works vs. what doesn't

### 2.1 Academic and exchange research

- **Retail loses in aggregate — but composition matters.** Beckmeyer, Branger & Gayda: >75% of retail S&P 500 option trades are 0DTE; retail lost ~$241k/day (Feb 2021–Sep 2023), rising to **~$350k/day after daily expirations (May 2022)**. Losses are concentrated in **single-leg, debit-paid, high-IV** trades; **multi-leg trades and trades harvesting volatility/jump risk premia are significantly more profitable** ([SSRN](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4404704), [PDF](https://wp.lancs.ac.uk/fofi2024/files/2024/04/FoFI-2024-146-Leander-Gayda.pdf)). This is directly favorable to WheelBot's defined-risk credit-selling design.
- **The intraday premium is thin.** Vilkov et al.: median realized variance risk premium from 10:00 ET to expiry ≈ **0.0011% of the underlying**; strategy distributions are "wide, tail-heavy, and unstable across regimes"; expected shortfall (0.58–1.58% of underlying) dwarfs mean carry; appropriate use is a "small tactical sleeve" sized against tail risk ([paper](https://github.com/vilkovgr/0dte-strategies/blob/main/docs/paper/paper-annotated.md)).
- **Volume and market impact.** 0DTE = ~**57% of SPX options ADV** (2.15M contracts) in Q3 2025 ([Cboe](https://www.cboe.com/insights/posts/the-state-of-the-options-industry-quarter-three-2025/)). Cboe finds 0DTE flow **balanced buy-vs-sell**, leaving market makers near net-zero gamma — no systemic squeeze mechanics to exploit or fear ([Cboe research](https://cdn.cboe.com/resources/education/research_publications/gammasqueezes.pdf), [Cboe insights](https://www.cboe.com/insights/posts/volatility-insights-evaluating-the-market-impact-of-spx-0-dte-options/)). Brogaard, Han & Won find 0DTE activity *does* raise intraday volatility (~+9.1% of mean per 1σ of 0DTE volume), driven by speculative retail flow ([SSRN](https://papers.ssrn.com/sol3/Delivery.cfm/4426358.pdf?abstractid=4426358&mirid=1&type=2)).

### 2.2 Practitioner backtests and live records

- **FlashAlpha 96-configuration SPY put-credit-spread study** (7.16 years, 1-minute chains, 2019–2026): baseline win rate 74%; **switching from mid-fills to a realistic maker-fill model (post at ask+$0.04, ~20–25% fill rate, ~12-min waits, fills 4–7¢ worse than mid) cut CAGR 30–60% across the grid**. At 7 DTE/10Δ the same config went from **+CAGR with a 100% stop to a −100% account wipeout with no stop**. Counterintuitively, SL=200% *underperformed both* SL=100% and no-stop on identical data (stop-clustering artifact — a warning about stop-parameter overfitting). Days with the richest premium (top-quintile VRP) had **worse** win rates (66% vs 74%) — fat premium is compensation, not free money ([FlashAlpha](https://flashalpha.com/articles/spy-put-credit-spread-active-backtest-mm-fills-vrp-signal-drawdown-breaker)).
- **Win-rate asymmetry is universal.** Every credible source shows avg loss ≥ 2x avg win for stopped condors (Option Omega backtest: 2.1x; Theta Profits inverts it only by using breakeven-construction stops at 40% WR; Option Alpha: 70% WR with only half of traders profitable). Expectancy after costs, not win rate, is the metric.
- **Documented failure modes**: (a) tail days — one 2022 backtest loss of −$11,128 vs +$1,052 avg win ([Options Trading IQ](https://optionstradingiq.com/option-omega/)); (b) stop slippage "easily double or triple your expected loss" in fast tape ([zerodte.com](https://zerodte.com/outsmarting-slippage-why-narrow-vertical-spreads-beat-traditional-stops/)); (c) pre-Aug-2023 SPX quote spikes blowing through stop-limits without filling ([Theta Profits](https://www.thetaprofits.com/stop-loss-on-credit-spreads-in-0dte-options-trading/)); (d) 2024's higher intraday vol raising double-stop frequency and compressing premium capture (Theta Profits' own PCR fell 5.65%→3.53%).
- **Elite Trader practitioners** running 5–15Δ condors with 30-pt wings and per-side stops report ~25% losing trades, ~35% breakeven, rest winners — consistent with the "many small scratches, few real winners, occasional 2x losses" profile ([Elite Trader thread](https://www.elitetrader.com/et/threads/day-trading-0dte-condors.378216/page-6)).

**Bottom line**: what "works" (marginally, after costs) is *disciplined, defined-risk, morning-entry premium selling with mechanical exits and tail-aware sizing*, plus simple regime/trend filters. What demonstrably doesn't: buying single-leg 0DTE lottery tickets, selling without stops or structural caps, trading the last 30 minutes, oversizing, and trusting mid-fill backtests.

---

## 3. Execution realism

### 3.1 Gamma speed

A 0DTE short put spread's delta changes roughly **10x faster** than the same 45DTE spread ([Alpaca 0DTE guide](https://alpaca.markets/learn/how-to-trade-0dte-options-on-alpaca)). Near the short strike late in the day, a 10-point SPX move can flip a winner into a stop-out in minutes ([Elite Trader](https://www.elitetrader.com/et/threads/day-trading-0dte-condors.378216/page-6)); Schwartz's "+$150 to −$500 in five minutes" near the close is the canonical illustration ([Cboe](https://www.cboe.com/insights/posts/henry-schwartzs-zero-day-spx-iron-condor-strategy-a-deep-dive/)).

### 3.2 What cadence real 0DTE bots run at

- Real 0DTE automation (TAT, Option Alpha) does **continuous stop monitoring** — TAT's core selling point is broker-connected stop monitoring/management, not periodic polling ([TAT features](https://support.tradeautomationtoolbox.com/hc/en-us/articles/43583650491155-Trade-Automation-Toolbox-Feature-List)).
- Manual/semi-auto practitioners avoid the cadence problem entirely with **resting broker-side OCO orders** (stop-limit + backup stop-market on the short legs) placed at entry ([Theta Profits](https://www.thetaprofits.com/stop-loss-on-credit-spreads-in-0dte-options-trading/)).
- Implication for WheelBot: the **5-minute loop is acceptable for entry decisions and profit-target checks, but not for stop-loss protection**. Either (a) park resting stop orders at Alpaca on entry, (b) run a dedicated 10–30s monitor thread for open 0DTE positions, or (c) choose structures whose max loss is acceptable without any stop (narrow wings).

### 3.3 Stop-fill slippage reality

- Once triggered, a stop becomes a market order and "slippage can easily double or triple your expected loss"; liquidity vanishes exactly when stops fire; stop-limits can be jumped entirely ([zerodte.com](https://zerodte.com/outsmarting-slippage-why-narrow-vertical-spreads-beat-traditional-stops/)).
- Mitigations used in the wild: stops on **short legs only** (single-leg SPX shorts are more liquid than the spread complex); OCO stop-limit + wider stop-market backstop; post-Aug-2023 SPX liquidity improvements let many traders go back to plain market stops ([Theta Profits](https://www.thetaprofits.com/stop-loss-on-credit-spreads-in-0dte-options-trading/)).
- The structural alternative: **make the wing the stop** — a narrow (e.g., $1–$5 wide) vertical has its max loss "built into the position… no reliance on fill quality" ([zerodte.com](https://zerodte.com/outsmarting-slippage-why-narrow-vertical-spreads-beat-traditional-stops/)).

### 3.4 Settlement: American SPY/QQQ vs cash-settled SPX/XSP — and what Alpaca actually supports

- **SPY/QQQ**: physical delivery, American-style. Pin risk peaks in the final 30 minutes; the OCC exercise window runs to **5:30 PM ET**, so a short that looked OTM at 4:00 can be assigned on an after-hours move; a **partial-ITM spread** (short ITM, long OTM) leaves naked assigned shares over the weekend gap ([Option Alpha on partial-ITM pin risk](https://optionalpha.com/learn/0dte-partial-in-the-money-assignment), [Elite Trader](https://www.elitetrader.com/et/threads/confused-about-spy-0dte-cut-off-time-for-assignment.381388/)).
- **In practice Alpaca removes most of this**: on expiration day it **rejects new option orders after ~3:15 PM ET (3:30 PM for broad-based ETFs like SPY/QQQ) and auto-liquidates expiring positions at 3:30/3:45 PM ET** ([Alpaca 0DTE guide](https://alpaca.markets/learn/how-to-trade-0dte-options-on-alpaca)). Design consequence: the bot's own hard flatten must complete by ~15:00–15:10 ET, or it forfeits exit control to Alpaca's liquidation engine (which will exit at whatever the market gives). "Hold to expiry" condor variants are **not implementable** on Alpaca ETF options — the last ~45 minutes of theta is off the table.
- **XSP/SPX on Alpaca — verified**: Alpaca announced **index options (SPX, SPXW, VIX, VIXW, DJX, XSP) in paper trading on July 23, 2026** — cash-settled, European, SPXW PM-settled and explicitly pitched for same-day-expiration strategies. **But index-option market data is not yet on Alpaca's data API** ("coming months"), and live trading isn't announced ([Alpaca blog](https://alpaca.markets/blog/alpaca-introduces-index-options-paper-trading/)). So today: build on SPY/QQQ, architect the symbol layer so XSP (same 1/10 SPX notional as SPY, but cash-settled/European, no pin/assignment risk) can be swapped in once data ships. Note SPX/XSP have exchange fees and wider spreads than penny-quoted SPY.

---

## 4. Paper-test realism on Alpaca

### 4.1 Known gaps in Alpaca's paper fill simulation

From Alpaca's own docs ([paper trading](https://docs.alpaca.markets/us/docs/paper-trading), [GitHub docs mirror](https://github.com/alpacahq/alpaca-docs/blob/master/content/trading/paper-trading.md)):

- Orders fill via simulation against **real-time NBBO**; limit orders fill only when marketable (buy limit ≥ ask, sell limit ≤ bid) — i.e., you effectively **pay the touch, never improve inside the spread** for marketable orders, and resting orders fill on touch with no queue modeling.
- **Order size is not checked against NBBO size** — you can fill 100 contracts against a 5-lot quote. No market impact, ever.
- 10% of eligible fills are randomly partial; remainder re-evaluated.
- No slippage model on stops/market orders: a triggered paper stop fills at the quote, while a live 0DTE stop routinely fills 2–3x worse ([zerodte.com](https://zerodte.com/outsmarting-slippage-why-narrow-vertical-spreads-beat-traditional-stops/)).
- Alpaca itself flags that "paper trading is a simulated environment" with execution-quality differences ([index options post](https://alpaca.markets/blog/alpaca-introduces-index-options-paper-trading/)); multi-leg (MLEG) orders are supported in paper ([Alpaca 0DTE guide](https://alpaca.markets/learn/how-to-trade-0dte-options-on-alpaca)).

Net effect: **Alpaca paper is systematically optimistic for a 0DTE seller** — especially on stop exits, the exact place where the strategy's expectancy lives.

### 4.2 How others make simulated fills honest

- **FlashAlpha's maker-fill model** (the best-documented public example): post limit at ask+$0.04, accept ~20–25% fill rate and ~12-minute waits, realize fills 4–7¢/contract worse than mid — this alone cut CAGR 30–60% vs mid-fills ([FlashAlpha](https://flashalpha.com/articles/spy-put-credit-spread-active-backtest-mm-fills-vrp-signal-drawdown-breaker)).
- **Vilkov et al.**: half-spread + 0.5bp slippage per trade as the cost model; it flipped condor Sharpe from +0.77 to −0.20 ([paper](https://github.com/vilkovgr/0dte-strategies/blob/main/docs/paper/paper-annotated.md)).
- **Option Omega convention**: fixed per-leg slippage (e.g., $0.05) + per-contract commissions ($1.70 entry / $0.70 exit SPX in the OTIQ backtest) ([Options Trading IQ](https://optionstradingiq.com/option-omega/), [OO docs](https://docs.optionomega.com/backtesting/backtest-setup)).
- **Practitioner convention** for stops: assume the stop fills at trigger **plus** an adder (a half-spread or a fixed $0.05–$0.15/leg; more on event days).

### 4.3 What makes a 0DTE paper test credible vs worthless

Credible requires: (1) a **parallel penalized P&L** that re-prices every fill with a slippage model (entry: no better than mid − 25–50% of half-spread; stop exits: trigger + adder), because raw Alpaca paper P&L will overstate results; (2) **NBBO capture at decision time and at fill** for every order, so the penalty model can be calibrated later against live micro-lots; (3) inclusion of **event days** (FOMC/CPI/NFP) in the sample even if the strategy skips them — the skip rule is part of the strategy; (4) **no compounding at aggressive allocation** during evaluation; (5) a sample long enough to contain at least one vol shock (the Aug-2024-type day); (6) tracking **MAE/MFE per position at ≤1-minute granularity** so stop behavior can be studied offline; (7) honest treatment of Thursday/regime splits found by Option Alpha rather than cherry-picking. A paper test that just reads Alpaca fills at 5-minute polls with no slippage overlay is worthless for go/no-go.

### 4.4 Cheap/free historical 0DTE chain data for validation

- **ThetaData** — cheapest per-GB raw intraday option quotes/chains; standard choice for DIY frameworks ([comparison](https://flashalpha.com/articles/best-options-data-apis-2026)).
- **Cboe DataShop** — 1-minute (or N-minute) option quote intervals with NBBO and size; authoritative for SPX/SPXW ([DataShop](https://datashop.cboe.com/), [intervals product](https://datashop.cboe.com/options-intervals-subscription)).
- **Polygon.io** — OPRA aggregates/quotes API, mid-priced tier.
- **Alpaca's own historical option data** (OPRA, from Feb 2024) — free with the account; short history but zero-cost for recent-regime validation and for calibrating the fill-penalty model against the very quotes the bot trades on.
- **Hosted backtesters with 0DTE data**: [Option Omega](https://docs.optionomega.com/backtesting-faq) (minute-level SPX/SPY, slippage/commission settings) and [Option Alpha's 0DTE backtester](https://optionalpha.com/backtester) — fastest way to sanity-check a parameter grid before writing code. [IVolatility](https://www.ivolatility.com/historical-options-data/) sells 0DTE-oriented datasets (IVX0DTE index).

---

## 5. Synthesis

### 5(a) Requirements for a credible WheelBot 0DTE paper module

**Loop cadence**
- Keep the 5-minute loop for entry evaluation and profit-target/time-exit checks.
- Add either (preferred, matches live practitioners) **resting OCO stop orders placed at entry** (stop-limit + wider stop-market backstop, on short legs or the spread), or a **dedicated 10–30s monitor** for open 0DTE positions. Do not stop-manage 0DTE gamma on a 5-minute poll.
- **Hard flatten by 15:00 ET** (buffer ahead of Alpaca's 3:15/3:30 order cutoff and 3:30/3:45 auto-liquidation). Treat a position still open at 15:05 as an incident.

**Entry mechanics**
- Entry window 9:45–10:30 ET (evidence: profitable-cohort ~10:15 entry; avoid the opening auction chaos). Optional second window ~12:30–13:00 for staggered variants.
- Delta-targeted strike selection from the live chain (10–20Δ shorts); reject entry if quote age > a few seconds, if bid-ask width exceeds a per-strike threshold, or if credit < a floor relative to width (min credit/width ratio, e.g., ≥ 8–10% for condor sides).
- Order placement: limit at mid, reprice toward marketable in 2–3 steps with timeouts (ladder), abandon past a worst-acceptable price. Log every reprice.
- **Event-day gate**: skip or half-size FOMC/CPI/NFP days (reuse the existing macro-calendar infra).

**Fill/slippage model (the credibility core)**
- Record NBBO (bid/ask/size) at decision, at order placement, and at fill for every leg.
- Maintain **dual P&L**: raw Alpaca paper P&L and **penalized P&L** — entries assumed no better than mid − 25–50% of half-spread; profit-target exits at mid − half-spread fraction; **stop exits at trigger price + adder** (start: max(half-spread, $0.05)/leg normal days; 3x on high-vol days, VIX1D-scaled). Go/no-go decisions read penalized P&L only.
- Configurable so the penalty parameters can be recalibrated (later, against 1-lot live fills).

**Hard risk caps (deterministic gates, consistent with existing WheelBot design)**
- Defined-risk structures only; per-trade max loss ≤ 1% of equity; **daily aggregate max loss** (sum of worst-case if every open stop slips 2x) ≤ 2% of equity; max N concurrent 0DTE positions; kill-switch latch for the rest of the day after a double-stop or daily-cap hit; no new entries after 14:00 ET; no entries within 30 min of the close ever.
- Sizing fixed (no compounding) during the entire paper evaluation.

**Logging (per position)**
- Full chain snapshot at entry (strikes ± several, NBBO + sizes + greeks + IV), underlying and VIX/VIX1D, entry-time bucket, credit/width ratio, every order lifecycle event with NBBO stamps, 1-minute MAE/MFE path, stop trigger path (trigger quote vs fill), exit reason (PT/SL/time/flatten/auto-liq), and both P&L columns. This is what turns 3 months of paper into a dataset instead of an anecdote.

**Duration/sample**: minimum ~60 trading days and ≥100 positions per variant before judging; pre-register the evaluation metric (penalized expectancy per trade and tail ratio, not win rate).

### 5(b) Top 3 strategy variants to paper-test first

**1. Morning iron condor (the consensus baseline).** SPY (QQQ optional), enter 10:00–10:15 ET, shorts ~15Δ, wings sized so max loss ≈ 3–5x credit, PT 25–35% of credit, stop at 1x credit (per-side or whole-position), time exit 14:00–15:00 ET.
*For*: the most-documented approach — Option Alpha's profitable cohort (70% WR, 10:15/12:00 pattern), Theta Profits' 57-month live record, OptionsDecay's rules, Cboe's own walkthrough; multi-leg premium capture is the retail subset Beckmeyer et al. found relatively profitable.
*Against*: Vilkov et al. show the unconditional condor is **net-negative after costs** (SR −0.20); avg loss ≈ 2x avg win; premium capture compressing since 2023–24. This is the null-hypothesis test: if it can't beat zero on penalized P&L, the regime answer matters more than the strategy answer.

**2. Trend-filtered directional credit spread with structural stop (best fit for a 5-minute bot).** Short put spread on an SMA5/VWAP-style buy signal (short call spread on sell signal), shorts 10–15Δ, **narrow width ($1–2 SPY)** so the wing *is* the stop — no stop orders at all; PT 25–50% of credit or time exit 14:30.
*For*: Option Alpha's best-performing filtered setups (>75% WR, PF > 2.0); eliminates stop-fill slippage — the single largest paper-vs-live divergence ([zerodte.com](https://zerodte.com/outsmarting-slippage-why-narrow-vertical-spreads-beat-traditional-stops/)) — so the paper test is intrinsically more credible; simple regime filters carried most of the edge in FlashAlpha's 16k-trade comparison; WheelBot already has vertical infrastructure and MTF signal work (SPY swing project).
*Against*: narrow wings mean worse credit-to-fee ratio and near-certain max loss when wrong (win rate must stay high); directional filters are exactly where overfitting lives; Thursday-negative and chop-day whipsaw risk.

**3. Staggered multi-entry condors with per-side stops (Theta Profits / METF style).** 3–6 entries at fixed times, 10–15Δ shorts, per-side stop = total credit, OCO stop orders resting at broker.
*For*: the longest live track record in the space (9,100 trades, 49/57 profitable months); time-diversification across intraday vol regimes; TAT community runs this at scale.
*Against*: expectancy depends almost entirely on stop-fill quality (40% win rate design) — the hardest thing to simulate honestly on Alpaca paper; more trades = more fee/slippage drag; requires resting-order management the current loop doesn't have. **Test third, only after the fill-penalty model is calibrated by variants 1–2.**

---

## Sources

Academic / exchange:
- Beckmeyer, Branger & Gayda, "Retail Traders Love 0DTE Options... But Should They?" — https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4404704 (PDF: https://wp.lancs.ac.uk/fofi2024/files/2024/04/FoFI-2024-146-Leander-Gayda.pdf)
- Brogaard, Han & Won, "Does 0DTE Options Trading Increase Volatility?" — https://papers.ssrn.com/sol3/Delivery.cfm/4426358.pdf?abstractid=4426358&mirid=1&type=2
- Vilkov et al., 0DTE strategies paper (annotated) — https://github.com/vilkovgr/0dte-strategies/blob/main/docs/paper/paper-annotated.md
- Cboe, "0DTE Index Options and Market Volatility" — https://cdn.cboe.com/resources/education/research_publications/gammasqueezes.pdf
- Cboe, "Evaluating the Market Impact of SPX 0DTE Options" — https://www.cboe.com/insights/posts/volatility-insights-evaluating-the-market-impact-of-spx-0-dte-options/
- Cboe, "State of the Options Industry Q3 2025" — https://www.cboe.com/insights/posts/the-state-of-the-options-industry-quarter-three-2025/
- Cboe, Henry Schwartz zero-day SPX iron condor deep dive — https://www.cboe.com/insights/posts/henry-schwartzs-zero-day-spx-iron-condor-strategy-a-deep-dive/

Practitioner strategies / backtests:
- Option Alpha, 0DTE strategy performance (230k trades) — https://optionalpha.com/blog/0dte-options-strategy-performance
- Option Alpha, 0DTE backtester — https://optionalpha.com/backtester ; bot templates — https://optionalpha.com/templates ; partial-ITM pin risk — https://optionalpha.com/learn/0dte-partial-in-the-money-assignment
- Theta Profits, Breakeven Iron Condor (9,100 trades) — https://www.thetaprofits.com/my-most-profitable-options-trading-strategy-0dte-breakeven-iron-condor/
- Theta Profits, stop-loss mechanics on 0DTE credit spreads — https://www.thetaprofits.com/stop-loss-on-credit-spreads-in-0dte-options-trading/
- Theta Profits, METF strategy — https://www.thetaprofits.com/how-to-trade-the-metf-0dte-options-strategy/
- OptionsDecay, 0DTE iron condor rules — https://www.optionsdecay.com/0dte-iron-condor/
- Options Trading IQ, Option Omega SPX 0DTE IC backtest — https://optionstradingiq.com/option-omega/
- FlashAlpha, 96 SPY put-credit-spread strategies with realistic fills — https://flashalpha.com/articles/spy-put-credit-spread-active-backtest-mm-fills-vrp-signal-drawdown-breaker
- zerodte.com, "Outsmarting Slippage" (structural stops) — https://zerodte.com/outsmarting-slippage-why-narrow-vertical-spreads-beat-traditional-stops/
- Trade Automation Toolbox — https://tradeautomationtoolbox.com/ (features: https://support.tradeautomationtoolbox.com/hc/en-us/articles/43583650491155-Trade-Automation-Toolbox-Feature-List)
- Elite Trader, "Day trading 0DTE Condors" — https://www.elitetrader.com/et/threads/day-trading-0dte-condors.378216/page-6 ; SPY 0DTE assignment cutoff — https://www.elitetrader.com/et/threads/confused-about-spy-0dte-cut-off-time-for-assignment.381388/

Alpaca / data:
- Alpaca, index options in paper trading (July 23, 2026) — https://alpaca.markets/blog/alpaca-introduces-index-options-paper-trading/
- Alpaca, How to trade 0DTE options (cutoffs, auto-liquidation) — https://alpaca.markets/learn/how-to-trade-0dte-options-on-alpaca
- Alpaca, paper trading fill simulation — https://docs.alpaca.markets/us/docs/paper-trading (mirror: https://github.com/alpacahq/alpaca-docs/blob/master/content/trading/paper-trading.md)
- ThetaData / provider comparison — https://flashalpha.com/articles/best-options-data-apis-2026
- Cboe DataShop — https://datashop.cboe.com/ (option quote intervals: https://datashop.cboe.com/options-intervals-subscription)
- IVolatility historical options data — https://www.ivolatility.com/historical-options-data/
- Option Omega docs — https://docs.optionomega.com/backtesting/backtest-setup
