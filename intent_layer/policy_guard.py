"""
PolicyGuard — conflict resolution and TTL management

Rules (hardcoded for now, make data-driven later):
  - URLLC > eMBB > mMTC  (slice priority ordering)
  - Last-write-wins within the same slice
  - Default TTL: 300s
  - Write intents block conflicting write intents on the same slice
  - Read intents (get_latency, get_utilization) always pass
"""

import time
from dataclasses import dataclass, field
from typing import Optional
from .intent_parser import ParsedIntent

READ_ACTIONS = {"get_latency", "get_utilization"}

SLICE_PRIORITY = {"URLLC": 3, "eMBB": 2, "mMTC": 1, None: 0}

DEFAULT_TTL = 300  # seconds


class PolicyConflictError(Exception):
    pass


@dataclass
class ActiveIntent:
    intent: ParsedIntent
    expires_at: float


class PolicyGuard:
    def __init__(self, default_ttl: int = DEFAULT_TTL):
        self.default_ttl = default_ttl
        self._active: dict[str, ActiveIntent] = {}  # slice → active intent

    def check_and_register(self, intent: ParsedIntent) -> float:
        """
        Validate intent against active policy state.
        Returns expires_at timestamp on success.
        Raises PolicyConflictError if blocked by a higher-priority active intent.
        """
        self._expire_stale()

        if intent.action in READ_ACTIONS:
            return 0  # reads always pass, no TTL needed

        key = intent.slice or "__global__"
        existing = self._active.get(key)

        if existing:
            existing_priority = SLICE_PRIORITY.get(existing.intent.slice, 0)
            new_priority      = SLICE_PRIORITY.get(intent.slice, 0)

            if existing_priority > new_priority:
                raise PolicyConflictError(
                    f"Blocked: active {existing.intent.slice!r} intent "
                    f"has higher priority than {intent.slice!r}"
                )
            # same or lower priority → last-write-wins (overwrite)

        expires_at = time.time() + self.default_ttl
        self._active[key] = ActiveIntent(intent=intent, expires_at=expires_at)
        return expires_at

    def _expire_stale(self) -> None:
        now = time.time()
        self._active = {k: v for k, v in self._active.items() if v.expires_at > now}
