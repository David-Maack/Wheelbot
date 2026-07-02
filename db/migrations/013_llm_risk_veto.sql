-- 2026-07-01 AI audit item #2: LLM regime exception veto.
--
-- Once daily (after the numeric regime snapshot is written), an LLM reads the
-- numeric dossier + general-market headlines and answers ONE question: "do the
-- headlines contain a material risk the numbers can't see yet (war escalation,
-- bank failure, surprise policy move, credit event)?" A YES sets llm_risk_veto
-- on today's snapshot and the bot halves entry sizes for the day. The veto is
-- REDUCE-ONLY by construction — it can never increase risk, and every failure
-- mode (no API key, budget exceeded, parse failure, network) fails open to
-- no-veto. See intelligence/regime_veto.py.
ALTER TABLE regime_snapshots ADD COLUMN llm_risk_veto BOOLEAN;
ALTER TABLE regime_snapshots ADD COLUMN llm_veto_reason TEXT;
