"""
Anonymous usage telemetry via PostHog.

Telemetry is opt-out (default on) and can be disabled two ways:
  - SemTax(telemetry=False) per-instance
  - SEMTAX_DISABLE_TELEMETRY=1 environment variable (process-wide)

What is captured:
  - classifier_initialized: taxonomy, model_id, python_version, semtax_version
  - classify_called: batch_size, taxonomy, model_id
  - cache_miss: taxonomy, level, model_id
  - cache_hit: taxonomy, level

What is NEVER captured:
  - Description text or any user data
  - Spend amounts or financial information
  - User identity or machine identifiers beyond the anonymous device UUID

The device UUID is generated once and stored at ~/.semtax/device_id.
It is a random UUID4 — not tied to any user account or machine fingerprint.

All PostHog calls are wrapped in try/except — telemetry never crashes the
library regardless of network state or PostHog SDK errors.
"""

from __future__ import annotations

import os
import sys
import uuid
from pathlib import Path
from typing import Optional

_POSTHOG_API_KEY = os.environ.get(
    "POSTHOG_PROJECT_API_KEY",
    "phc_X1ExZn5I16Dl8fdOx1FMNHtsK6Kdvn8PeoZrzh18O7g",
)
_POSTHOG_HOST = os.environ.get("POSTHOG_HOST", "https://us.i.posthog.com")

_OPT_OUT_ENV_VAR = "SEMTAX_DISABLE_TELEMETRY"
_DEVICE_ID_PATH = Path.home() / ".semtax" / "device_id"

# Keys allowed to be sent in event properties (allowlist approach)
_ALLOWED_KEYS = frozenset(
    {
        "batch_size",
        "taxonomy",
        "model_id",
        "python_version",
        "semtax_version",
        "level",
        "match_level",
    }
)


def _is_opted_out() -> bool:
    return os.environ.get(_OPT_OUT_ENV_VAR, "").strip().lower() in ("1", "true", "yes")


def _get_device_id() -> str:
    """
    Return or create a persistent anonymous UUID for this installation.

    Stored at ~/.semtax/device_id as a plain text file.  Never tied to
    user identity — it is a random UUID4 generated on first run.
    """
    try:
        _DEVICE_ID_PATH.parent.mkdir(parents=True, exist_ok=True)
        if _DEVICE_ID_PATH.exists():
            stored = _DEVICE_ID_PATH.read_text().strip()
            if stored:
                return stored
        device_id = str(uuid.uuid4())
        _DEVICE_ID_PATH.write_text(device_id)
        return device_id
    except Exception:
        return "unknown"


def _safe_properties(props: dict) -> dict:
    """Strip any keys not on the allowlist to prevent accidental data leakage."""
    return {k: v for k, v in props.items() if k in _ALLOWED_KEYS}


def _get_semtax_version() -> str:
    try:
        from importlib.metadata import version
        return version("semtax")
    except Exception:
        return "unknown"


def capture(
    event: str,
    properties: Optional[dict] = None,
    opt_out: bool = False,
) -> None:
    """
    Fire a PostHog event.

    Args:
        event:      Event name string.
        properties: Dict of event properties.  Only allowlisted keys are sent.
        opt_out:    If True (forwarded from SemTax(telemetry=False)), suppress.
    """
    if opt_out or _is_opted_out():
        return

    try:
        import posthog  # type: ignore[import]

        posthog.api_key = _POSTHOG_API_KEY
        posthog.host = _POSTHOG_HOST

        base_props = {
            "python_version": sys.version.split()[0],
            "semtax_version": _get_semtax_version(),
        }
        merged = {**base_props, **(properties or {})}

        posthog.capture(
            distinct_id=_get_device_id(),
            event=event,
            properties=_safe_properties(merged),
        )
    except Exception:
        # Telemetry errors are always silently swallowed
        pass
