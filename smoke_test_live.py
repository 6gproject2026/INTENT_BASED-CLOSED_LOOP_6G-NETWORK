#!/usr/bin/env python3
"""Live smoke test for the intent layer.

Drives IntentTranslator against the REAL RyuClient (no mocks). Mininet and the
Ryu controller must already be running before this script is executed.

Order of operations:
  1. "Show latency between 0x30 and 0x40"        — read-only
  2. "Get utilization for switch 0x30"           — read-only
  3. "Cap mMTC traffic to 5 Mbps on switch 0x30" — write, runs ONLY if 1 & 2 pass

Exit code: 0 if every step passes, 1 on any failure (including an unreachable
controller). The write intent is skipped entirely if either read fails, so we
never apply QoS to a network we couldn't even read.

Run from the AI_Layer/ directory:  python smoke_test_live.py
"""

import json
import sys

from ai_layer.network_interface.ryu_client import RyuClient, wait_for_controller
from intent_layer import IntentTranslator

CONFIG_PATH = "prod.json"
PREFLIGHT_WAIT_SECONDS = 10  # do not retry the controller indefinitely


def load_translator():
    """Build IntentTranslator on a real RyuClient using prod.json config."""
    with open(CONFIG_PATH, encoding="utf-8") as fh:
        cfg = json.load(fh)
    ryu_cfg = cfg["environment"]["ryu_controller"]
    ryu = RyuClient(ryu_cfg)               # base_url comes from config, not hardcoded
    return IntentTranslator(ryu), ryu, ryu_cfg["base_url"]


def _value_is_present(val) -> bool:
    """Reject None and empty containers; allow numbers (including 0)."""
    if val is None:
        return False
    if isinstance(val, (list, dict, str)) and len(val) == 0:
        return False
    return True


def run_intent(translator, text, *, detail_key=None, expect=None) -> bool:
    """Resolve one intent, print the result dict, and report PASS/FAIL."""
    print(f"\n--- intent: {text!r}")
    result = translator.resolve(text)
    d = result.to_dict()
    print(f"    result: {json.dumps(d)}")

    ok, reason = True, ""

    if result.status != "success":
        ok, reason = False, f"status={result.status!r} error={result.error}"

    # Read intents must return a non-null value under their detail key.
    if ok and detail_key is not None:
        val = result.details.get(detail_key)
        if not _value_is_present(val):
            ok, reason = False, f"{detail_key!r} is null/empty"

    # Write intent must match expected fields (queue_id, dpid, ...).
    if ok and expect:
        for key, want in expect.items():
            got = d.get(key)
            if got != want:
                ok, reason = False, f"expected {key}={want!r}, got {got!r}"
                break

    print(f"    {'PASS' if ok else 'FAIL'}" + (f" — {reason}" if reason else ""))
    return ok


def main() -> int:
    translator, ryu, base_url = load_translator()
    print(f"Live intent-layer smoke test against Ryu at {base_url}")

    if not wait_for_controller(ryu, max_wait_seconds=PREFLIGHT_WAIT_SECONDS):
        print(f"\nABORT: Ryu controller unreachable after {PREFLIGHT_WAIT_SECONDS}s. "
              f"Ensure Mininet and the Ryu controller are running.")
        return 1

    # Steps 1 & 2 — read-only intents.
    read1 = run_intent(translator, "Show latency between 0x30 and 0x40", detail_key="latency")
    read2 = run_intent(translator, "Get utilization for switch 0x30", detail_key="utilization")

    if not (read1 and read2):
        print("\nABORT: a read intent failed — skipping the write intent. "
              "Refusing to apply QoS to a network we cannot read.")
        return 1

    # Step 3 — write intent, only reached after both reads pass.
    write = run_intent(
        translator,
        "Cap mMTC traffic to 5 Mbps on switch 0x30",
        expect={"status": "success", "queue_id": 3, "dpid": "0x30"},
    )
    if not write:
        print("\nFAIL: write intent did not pass.")
        return 1

    print("\nALL PASS — reads returned data and the QoS write succeeded.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
