"""
IntentTranslator — structured intent dict → RyuClient method calls

Lives in community 0 alongside RyuClient.
Never calls Ryu REST directly — always goes through RyuClient so
_normalize_dpid() and _request() retry logic are inherited for free.

Slice → queue mapping is the one piece of config that MUST stay
in sync with your Ryu QoS setup. Centralised here so there's
one place to update it.
"""

import time
from typing import Optional
from ai_layer.network_interface.ryu_client import RyuClient
from .intent_parser import ParsedIntent, IntentParseError
from .policy_guard import PolicyGuard, PolicyConflictError

# ── Slice config ──────────────────────────────────────────────────────────────
# Must match your Ryu QoS queue assignments
SLICE_QUEUE_MAP: dict[str, int] = {
    "URLLC": 1,   # highest priority queue
    "eMBB":  2,
    "mMTC":  3,   # lowest priority queue
}

# DSCP marks used to classify each slice's traffic onto its queue.
# Mirrors the prod.json qos_baseline rules (EF=46, AF41=34, AF11=10).
SLICE_DSCP_MAP: dict[str, str] = {
    "URLLC": "46",
    "eMBB":  "34",
    "mMTC":  "10",
}

# Default DPID if none extracted from intent
DEFAULT_DPID = "0x30"

# Default latency destination when an intent names only one endpoint.
# get_latency() requires a (src, dst) pair; this is the fallback dst.
# Must be a real switch (prod.json: 0x30, 0x40, 0x41).
DEFAULT_DST = "0x40"

# Link budget (bps) split across slices when prioritizing. Per-slice shares
# are derived from SLICE_QUEUE_MAP priority, not hardcoded individually.
PRIORITY_BASE_RATE_BPS = 20_000_000


# ── Result ────────────────────────────────────────────────────────────────────

class IntentResult:
    def __init__(self, status: str, intent: ParsedIntent,
                 details: dict = None, error: str = None,
                 expires_at: Optional[float] = None):
        self.status     = status
        self.intent     = intent
        self.details    = details or {}
        self.error      = error
        self.expires_at = expires_at

    def to_dict(self) -> dict:
        d = {
            "status": self.status,
            "action": self.intent.action,
        }
        if self.details:    d.update(self.details)
        if self.error:      d["error"] = self.error
        if self.expires_at: d["expires_at"] = int(self.expires_at)
        return d


# ── Translator ────────────────────────────────────────────────────────────────

class IntentTranslator:
    """
    Entry point for the intent layer.
    Owns the parse → policy → translate pipeline.
    """

    def __init__(self, ryu_client: RyuClient, policy_guard: Optional[PolicyGuard] = None):
        from .intent_parser import IntentParser
        self.ryu     = ryu_client
        self.parser  = IntentParser()
        self.policy  = policy_guard or PolicyGuard()

        self._dispatch = {
            "cap_traffic":      self._handle_cap_traffic,
            "prioritize_slice": self._handle_prioritize_slice,
            "reroute":          self._handle_reroute,
            "failover":         self._handle_failover,
            "get_latency":      self._handle_get_latency,
            "get_utilization":  self._handle_get_utilization,
        }

    def resolve(self, text: str) -> IntentResult:
        # 1. Parse
        try:
            intent = self.parser.parse(text)
        except IntentParseError as e:
            return IntentResult("unsupported", _stub_intent(text), error=str(e))

        # 2. Policy check
        try:
            expires_at = self.policy.check_and_register(intent)
        except PolicyConflictError as e:
            return IntentResult("blocked", intent, error=str(e))

        # 3. Translate → RyuClient
        handler = self._dispatch.get(intent.action)
        if not handler:
            return IntentResult("unsupported", intent,
                                error=f"No handler for action: {intent.action!r}")
        try:
            details = handler(intent)
            return IntentResult("success", intent, details=details,
                                expires_at=expires_at or None)
        except Exception as e:
            return IntentResult("error", intent, error=str(e))

    # ── Handlers ──────────────────────────────────────────────────────────────

    def _handle_cap_traffic(self, intent: ParsedIntent) -> dict:
        dpid     = intent.dpid or DEFAULT_DPID
        queue_id = SLICE_QUEUE_MAP[intent.slice]
        rate_bps = int(intent.limit_mbps * 1_000_000)

        self.ryu.apply_qos(dpid, {"queue_id": queue_id, "max_rate": str(rate_bps)})
        self.ryu.post_qos_rule(dpid, {
            "match": {"eth_type": 2048, "ip_dscp": SLICE_DSCP_MAP[intent.slice]},
            "actions": {"queue": str(queue_id)},
        })

        return {"dpid": dpid, "queue_id": queue_id, "max_rate_bps": rate_bps}

    def _handle_prioritize_slice(self, intent: ParsedIntent) -> dict:
        # Boost the target slice with a min_rate floor; cap every other slice
        # with a proportional max_rate. Shares are derived from each slice's
        # queue priority in SLICE_QUEUE_MAP (lower queue_id = higher priority).
        dpid   = intent.dpid or DEFAULT_DPID
        target = intent.slice
        if target not in SLICE_QUEUE_MAP:
            raise ValueError(
                f"prioritize_slice requires a known target slice, got {target!r}"
            )

        n = len(SLICE_QUEUE_MAP)
        weight = {s: n - q + 1 for s, q in SLICE_QUEUE_MAP.items()}
        total  = sum(weight.values())

        applied = []
        for slice_name, queue_id in SLICE_QUEUE_MAP.items():
            share = str(int(PRIORITY_BASE_RATE_BPS * weight[slice_name] / total))
            qos = {"queue_id": queue_id}
            if slice_name == target:
                qos["min_rate"] = share   # guarantee the boosted slice a floor
            else:
                qos["max_rate"] = share   # cap the others, proportional to priority
            self.ryu.apply_qos(dpid, qos)
            applied.append({"slice": slice_name, **qos})

        return {"dpid": dpid, "prioritized": target, "applied": applied}

    def _handle_reroute(self, intent: ParsedIntent) -> dict:
        dpid = intent.dpid or DEFAULT_DPID
        self.ryu.post_router_entry(dpid, {})
        return {"dpid": dpid}

    def _handle_failover(self, intent: ParsedIntent) -> dict:
        dpid = intent.dpid or DEFAULT_DPID
        self.ryu.post_router_entry(dpid, {})
        return {"dpid": dpid, "mode": "failover"}

    def _handle_get_latency(self, intent: ParsedIntent) -> dict:
        result = self.ryu.get_latency(intent.dpid or DEFAULT_DPID,
                                      intent.dst_dpid or DEFAULT_DST)
        return {"latency": result}

    def _handle_get_utilization(self, intent: ParsedIntent) -> dict:
        result = self.ryu.get_link_utilization()
        return {"utilization": result}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _stub_intent(text: str) -> ParsedIntent:
    from .intent_parser import ParsedIntent
    return ParsedIntent(action="unknown", raw=text)
