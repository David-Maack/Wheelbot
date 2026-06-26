"""Backtest harness for the SPY swing strategy (sub-sprint 1).

Read-only / offline. Validates the multi-timeframe VWAP-EMA crossover signal
and compares ITM vs OTM option structures using a Black-Scholes-modeled option
leg (no paid historical option-chain data required). Nothing here trades or
mutates state; it is a go/no-go gate before any live wiring.
"""
