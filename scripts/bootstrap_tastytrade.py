"""One-time Tastytrade OAuth bootstrap.

Captures a long-lived refresh token from a username/password login and writes
just the relevant keys into `config/secrets.env` — leaves all other lines
untouched.

Two modes:

    Interactive (default):
        python -m scripts.bootstrap_tastytrade --sandbox
        # prompts for username + password (getpass)

    Non-interactive (CI / headless):
        python -m scripts.bootstrap_tastytrade --sandbox \
            --username you@example.com --password-stdin

Sandbox vs prod is a hard choice: the script refuses to overwrite an existing
prod token with a sandbox one (and vice versa) without `--force`. The current
mode is determined by `--sandbox` / `--prod` (mutually exclusive, --sandbox
default for safety).

Note: as of Tastytrade SDK 12.x the developer flow is OAuth2. You also need a
`provider_secret` (your app's client secret from developer.tastytrade.com)
written to secrets.env once — this script doesn't manage that key. After
running this script the env will contain:

    TASTYTRADE_PROVIDER_SECRET=...     (you set this manually, once)
    TASTYTRADE_REMEMBER_TOKEN=...      (this script writes it)
    TASTYTRADE_USE_SANDBOX=true|false  (this script writes it)
    TASTYTRADE_ACCOUNT_NUMBER=...      (this script optionally writes it)
"""

from __future__ import annotations

import argparse
import asyncio
import getpass
import os
import sys
from pathlib import Path

from core.checkpoint import configure_logging, log_checkpoint
from core.config import _config_dir


SECRETS_KEYS = {
    "TASTYTRADE_REMEMBER_TOKEN",
    "TASTYTRADE_USE_SANDBOX",
    "TASTYTRADE_ACCOUNT_NUMBER",
}


def upsert_secrets(path: Path, updates: dict[str, str]) -> None:
    """Set/replace `KEY=value` lines in `path`. Untouched lines stay untouched."""
    path.parent.mkdir(parents=True, exist_ok=True)
    existing_lines: list[str] = []
    if path.exists():
        existing_lines = path.read_text(encoding="utf-8").splitlines()

    seen = set()
    new_lines: list[str] = []
    for line in existing_lines:
        stripped = line.strip()
        if "=" in line and not stripped.startswith("#"):
            key = line.split("=", 1)[0].strip()
            if key in updates:
                new_lines.append(f"{key}={updates[key]}")
                seen.add(key)
                continue
        new_lines.append(line)
    for key, value in updates.items():
        if key not in seen:
            new_lines.append(f"{key}={value}")
    path.write_text("\n".join(new_lines).rstrip() + "\n", encoding="utf-8")


def _read_password(stdin: bool) -> str:
    if stdin:
        # Read once, strip trailing newline.
        return sys.stdin.readline().rstrip("\n")
    return getpass.getpass("Tastytrade password: ")


def _detect_existing_mode(secrets_path: Path) -> str | None:
    """Returns 'sandbox', 'prod', or None based on what's currently in secrets.env."""
    if not secrets_path.exists():
        return None
    for line in secrets_path.read_text(encoding="utf-8").splitlines():
        if line.strip().startswith("TASTYTRADE_USE_SANDBOX="):
            value = line.split("=", 1)[1].strip().lower()
            return "sandbox" if value in ("true", "1", "yes") else "prod"
    return None


async def fetch_remember_token(
    username: str,
    password: str,
    *,
    is_test: bool,
) -> tuple[str, str | None]:
    """Authenticate with username/password → return (remember_token, account_number_hint).

    Uses the Tastytrade legacy session-token endpoint via `tastytrade.Session`'s
    older constructor surface. As of SDK 12.x the *primary* flow is OAuth refresh
    tokens; this helper exists so users without a developer-portal app can still
    bootstrap. If the SDK doesn't expose a legacy login on the local version we
    surface a clear error.
    """
    try:
        from tastytrade import Session  # noqa: PLC0415
    except ImportError as exc:
        raise SystemExit(
            "tastytrade SDK not installed. Run: pip install -e '.[broker]'"
        ) from exc

    # SDK 12.x dropped the username/password constructor. Try a couple of legacy
    # entry points; if none work, instruct the user to use the OAuth refresh-
    # token flow at developer.tastytrade.com.
    if hasattr(Session, "from_credentials"):
        async with Session.from_credentials(  # type: ignore[attr-defined]
            login=username, password=password, is_test=is_test, remember_me=True
        ) as session:
            token = getattr(session, "remember_token", None) or getattr(
                session, "session_token", None
            )
            return str(token), getattr(session, "default_account_number", None)
    raise SystemExit(
        "This Tastytrade SDK version doesn't expose a username/password login. "
        "Generate a refresh token via your developer-portal app and set "
        "TASTYTRADE_REMEMBER_TOKEN directly. See README §Operations."
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--sandbox", action="store_true", help="(default) sandbox creds")
    mode.add_argument("--prod", action="store_true", help="production creds")
    parser.add_argument("--username")
    parser.add_argument(
        "--password-stdin",
        action="store_true",
        help="Read password from stdin instead of an interactive prompt",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Skip the sandbox/prod safety prompt",
    )
    parser.add_argument("--account-number", help="Optional default account number")
    args = parser.parse_args(argv)

    configure_logging()
    is_test = not args.prod  # default sandbox

    secrets_path = _config_dir() / "secrets.env"
    existing = _detect_existing_mode(secrets_path)
    target = "sandbox" if is_test else "prod"
    if existing and existing != target and not args.force:
        print(
            f"secrets.env already configured for {existing!r}; refusing to overwrite "
            f"with {target!r} credentials. Re-run with --force to proceed.",
            file=sys.stderr,
        )
        return 2

    username = args.username or input("Tastytrade username: ").strip()
    password = _read_password(args.password_stdin)
    if not username or not password:
        print("Username and password are required.", file=sys.stderr)
        return 2

    token, account_hint = asyncio.run(
        fetch_remember_token(username, password, is_test=is_test)
    )
    if not token:
        print("Login succeeded but no remember token returned.", file=sys.stderr)
        return 1

    updates = {
        "TASTYTRADE_REMEMBER_TOKEN": token,
        "TASTYTRADE_USE_SANDBOX": "true" if is_test else "false",
    }
    chosen_account = args.account_number or account_hint
    if chosen_account:
        updates["TASTYTRADE_ACCOUNT_NUMBER"] = chosen_account

    upsert_secrets(secrets_path, updates)
    log_checkpoint(
        "tastytrade_bootstrap",
        status="ok",
        mode=target,
        account=chosen_account,
        path=str(secrets_path),
    )
    print(f"Wrote remember token + mode to {secrets_path} ({target}).")
    if not chosen_account:
        print("(no default account captured; set TASTYTRADE_ACCOUNT_NUMBER manually)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
