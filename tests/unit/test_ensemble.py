"""intelligence/ensemble — voting helper."""

from __future__ import annotations

from typing import Any

import pytest

from core.models import LlmDecisionType
from intelligence.ensemble import EnsembleResult, ensemble_vote


class _StubClient:
    """Returns a pre-set decision per model."""

    def __init__(self, mapping: dict[str, str | None]):
        self._mapping = mapping

    async def call(self, *, decision_type, model, system, user_payload, max_output_tokens=512, context=None):
        decision = self._mapping[model]
        return {
            "parsed": {"decision": decision} if decision else {},
            "raw_text": "",
            "tokens_in": 1,
            "tokens_out": 1,
            "cost_usd": 0.0001,
            "decision_id": 1,
        }


@pytest.mark.asyncio
async def test_unanimous_when_all_agree():
    client = _StubClient({"a": "roll", "b": "roll", "c": "roll"})
    result = await ensemble_vote(
        client,  # type: ignore[arg-type]
        decision_type=LlmDecisionType.ROLL_ADVISE,
        system="s",
        user_payload="p",
        models=["a", "b", "c"],
        quorum=2,
    )
    assert result.agreement == "unanimous"
    assert result.decision == "roll"


@pytest.mark.asyncio
async def test_majority_when_quorum_reached():
    client = _StubClient({"a": "roll", "b": "roll", "c": "close"})
    result = await ensemble_vote(
        client,  # type: ignore[arg-type]
        decision_type=LlmDecisionType.ROLL_ADVISE,
        system="s",
        user_payload="p",
        models=["a", "b", "c"],
        quorum=2,
    )
    assert result.agreement == "majority"
    assert result.decision == "roll"


@pytest.mark.asyncio
async def test_no_consensus_when_models_split():
    client = _StubClient({"a": "roll", "b": "close", "c": "assign"})
    result = await ensemble_vote(
        client,  # type: ignore[arg-type]
        decision_type=LlmDecisionType.ROLL_ADVISE,
        system="s",
        user_payload="p",
        models=["a", "b", "c"],
        quorum=2,
    )
    assert result.agreement == "no_consensus"
    assert result.decision is None


@pytest.mark.asyncio
async def test_no_consensus_when_all_failed():
    client = _StubClient({"a": None, "b": None})
    result = await ensemble_vote(
        client,  # type: ignore[arg-type]
        decision_type=LlmDecisionType.ROLL_ADVISE,
        system="s",
        user_payload="p",
        models=["a", "b"],
        quorum=2,
    )
    assert result.agreement == "no_consensus"
