"""scripts/bootstrap_tastytrade — secrets.env upsert + safety prompts."""

from __future__ import annotations

from pathlib import Path

import pytest

from scripts.bootstrap_tastytrade import _detect_existing_mode, upsert_secrets


def test_upsert_creates_file_if_missing(tmp_path: Path):
    target = tmp_path / "secrets.env"
    upsert_secrets(target, {"TASTYTRADE_REMEMBER_TOKEN": "abc", "TASTYTRADE_USE_SANDBOX": "true"})
    text = target.read_text(encoding="utf-8")
    assert "TASTYTRADE_REMEMBER_TOKEN=abc" in text
    assert "TASTYTRADE_USE_SANDBOX=true" in text


def test_upsert_replaces_only_specified_keys(tmp_path: Path):
    target = tmp_path / "secrets.env"
    target.write_text(
        "# managed by hand\n"
        "ALPACA_API_KEY=keep_me\n"
        "TASTYTRADE_REMEMBER_TOKEN=old\n"
        "ANTHROPIC_API_KEY=also_keep\n",
        encoding="utf-8",
    )
    upsert_secrets(target, {"TASTYTRADE_REMEMBER_TOKEN": "new"})
    text = target.read_text(encoding="utf-8")
    assert "ALPACA_API_KEY=keep_me" in text
    assert "ANTHROPIC_API_KEY=also_keep" in text
    assert "TASTYTRADE_REMEMBER_TOKEN=new" in text
    assert "TASTYTRADE_REMEMBER_TOKEN=old" not in text


def test_upsert_preserves_comments_and_blank_lines(tmp_path: Path):
    target = tmp_path / "secrets.env"
    target.write_text(
        "# top comment\n\n# section\nALPACA_API_KEY=xyz\n\n",
        encoding="utf-8",
    )
    upsert_secrets(target, {"TASTYTRADE_REMEMBER_TOKEN": "tok"})
    text = target.read_text(encoding="utf-8")
    assert "# top comment" in text
    assert "# section" in text


def test_detect_existing_mode_reports_sandbox(tmp_path: Path):
    target = tmp_path / "secrets.env"
    target.write_text("TASTYTRADE_USE_SANDBOX=true\n", encoding="utf-8")
    assert _detect_existing_mode(target) == "sandbox"


def test_detect_existing_mode_reports_prod(tmp_path: Path):
    target = tmp_path / "secrets.env"
    target.write_text("TASTYTRADE_USE_SANDBOX=false\n", encoding="utf-8")
    assert _detect_existing_mode(target) == "prod"


def test_detect_existing_mode_none_when_absent(tmp_path: Path):
    target = tmp_path / "secrets.env"
    assert _detect_existing_mode(target) is None
