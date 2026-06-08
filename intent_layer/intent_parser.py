"""
IntentParser — NL string → structured intent dict

Responsibilities:
- Extract action, slice, parameters, and target DPID from free-form text
- Validate extracted fields against known vocabulary
- Raise IntentParseError for unsupported or ambiguous inputs

This module is intentionally stateless. Feed it a string, get a dict back.
Conflict resolution and TTL assignment happen downstream in PolicyGuard.
"""

import re
from dataclasses import dataclass, field
from typing import Optional


# ── Vocabulary ────────────────────────────────────────────────────────────────

KNOWN_ACTIONS = {
    "cap_traffic",
    "prioritize_slice",
    "reroute",
    "failover",
    "get_latency",
    "get_utilization",
}

KNOWN_SLICES = {"mMTC", "eMBB", "URLLC"}

# keyword → canonical action
ACTION_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\b(cap|limit|restrict)\b", re.I),       "cap_traffic"),
    (re.compile(r"\b(prioritize|prefer|boost)\b", re.I),  "prioritize_slice"),
    (re.compile(r"\b(reroute|redirect|route)\b", re.I),   "reroute"),
    (re.compile(r"\b(failover|fail.?over)\b", re.I),      "failover"),
    (re.compile(r"\b(latency|ping|delay)\b", re.I),       "get_latency"),
    (re.compile(r"\b(utilization|utilisation|throughput|bandwidth)\b", re.I), "get_utilization"),
]

# e.g. "5 Mbps", "10mbps", "100 kbps"
RATE_PATTERN = re.compile(r"(\d+(?:\.\d+)?)\s*(mbps|kbps|gbps)", re.I)

# e.g. "0x30", "switch 48", "dpid 0x1a"
DPID_PATTERN = re.compile(r"(?:switch|dpid|sw)?\s*(0x[0-9a-fA-F]+|\d+)", re.I)

SLICE_PATTERN = re.compile(r"\b(mMTC|eMBB|URLLC)\b")


# ── Errors ────────────────────────────────────────────────────────────────────

class IntentParseError(ValueError):
    pass


# ── Result ────────────────────────────────────────────────────────────────────

@dataclass
class ParsedIntent:
    action: str
    slice: Optional[str] = None
    limit_mbps: Optional[float] = None
    dpid: Optional[str] = None           # raw, pre-normalization
    dst_dpid: Optional[str] = None       # second DPID, e.g. latency dst
    raw: str = ""
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if v is not None and v != {}}


# ── Parser ────────────────────────────────────────────────────────────────────

class IntentParser:
    """
    Lightweight regex-based NL parser.

    For production you'd swap the internals for an LLM call or a
    fine-tuned classifier — the interface stays identical.
    """

    def parse(self, text: str) -> ParsedIntent:
        action = self._extract_action(text)
        if action is None:
            raise IntentParseError(
                f"Cannot identify a supported action in: {text!r}\n"
                f"Supported actions: {sorted(KNOWN_ACTIONS)}"
            )

        intent = ParsedIntent(action=action, raw=text)
        intent.slice     = self._extract_slice(text)
        intent.limit_mbps = self._extract_rate_mbps(text)
        intent.dpid      = self._extract_dpid(text)
        intent.dst_dpid  = self._extract_dst_dpid(text)

        self._validate(intent)
        return intent

    # ── private ───────────────────────────────────────────────────────────────

    def _extract_action(self, text: str) -> Optional[str]:
        for pattern, action in ACTION_PATTERNS:
            if pattern.search(text):
                return action
        return None

    def _extract_slice(self, text: str) -> Optional[str]:
        m = SLICE_PATTERN.search(text)
        return m.group(1) if m else None

    def _extract_rate_mbps(self, text: str) -> Optional[float]:
        m = RATE_PATTERN.search(text)
        if not m:
            return None
        value, unit = float(m.group(1)), m.group(2).lower()
        if unit == "kbps":
            return value / 1000
        if unit == "gbps":
            return value * 1000
        return value  # mbps

    def _extract_dpid(self, text: str) -> Optional[str]:
        # Prefer an explicit 0x-prefixed hex DPID (e.g. "0x30").
        hex_match = re.search(r"0x[0-9a-fA-F]+", text)
        if hex_match:
            return hex_match.group(0)
        # Fall back to a bare integer, but skip any number that is a rate
        # value (e.g. the "5" in "5 Mbps") — i.e. preceded or followed by a unit.
        for m in re.finditer(r"\d+", text):
            before = text[max(0, m.start() - 6):m.start()]
            after  = text[m.end():m.end() + 6]
            if re.search(r"(kbps|mbps|gbps)\s*$", before, re.I):
                continue
            if re.match(r"\s*(kbps|mbps|gbps)", after, re.I):
                continue
            return m.group(0)
        return None

    def _extract_dst_dpid(self, text: str) -> Optional[str]:
        # Second DPID in the string, if the intent names a pair.
        matches = DPID_PATTERN.findall(text)
        return matches[1] if len(matches) > 1 else None

    def _validate(self, intent: ParsedIntent) -> None:
        if intent.slice and intent.slice not in KNOWN_SLICES:
            raise IntentParseError(f"Unknown slice: {intent.slice!r}")

        if intent.action == "cap_traffic":
            if intent.limit_mbps is None:
                raise IntentParseError("cap_traffic requires a rate (e.g. '5 Mbps')")
            if intent.slice is None:
                raise IntentParseError("cap_traffic requires a slice target (mMTC / eMBB / URLLC)")

        if intent.action == "reroute" and intent.dpid is None:
            raise IntentParseError("reroute requires a switch/DPID target")
