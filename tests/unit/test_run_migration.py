"""Regression tests for scripts.run_migration statement splitting.

Migration 015 shipped with trailing `-- packages; ...` comments; the old
splitter stripped only full-line comments before splitting on ';', so the
CREATE TABLE was truncated mid-column (sqlite "incomplete input") and the
bot crash-looped on deploy (2026-07-30).
"""

import sqlite3

from scripts.run_migration import _discover, _split_statements


def test_inline_comment_semicolon_does_not_split():
    sql = (
        "CREATE TABLE t (\n"
        "    a INTEGER,  -- packages; pnl_* are dollars\n"
        "    b REAL      -- penalized; feeds the loss cap\n"
        ");\n"
    )
    stmts = _split_statements(sql)
    assert len(stmts) == 1
    sqlite3.connect(":memory:").execute(stmts[0])


def test_full_line_comments_still_stripped():
    sql = "-- header comment\nCREATE TABLE t (a INTEGER);\n-- trailer\n"
    assert _split_statements(sql) == ["CREATE TABLE t (a INTEGER)"]


def test_double_dash_inside_string_literal_is_preserved():
    sql = "INSERT INTO t (note) VALUES ('a -- not a comment');"
    assert _split_statements(sql) == [
        "INSERT INTO t (note) VALUES ('a -- not a comment')"
    ]


def test_every_real_migration_splits_into_complete_statements():
    migrations = _discover()
    assert migrations, "no migration files discovered"
    for m in migrations:
        for stmt in _split_statements(m.path.read_text(encoding="utf-8")):
            assert sqlite3.complete_statement(stmt + ";"), (
                f"migration {m.version} produced an incomplete statement: "
                f"{stmt[:120]!r}"
            )
