"""Resolve a natural-language intent against the LIVE Ryu controller.

Unlike test_intent_cli.py (which uses a recording mock and never touches the
network), this runner builds a real RyuClient from prod.json and actually
issues the HTTP calls to the controller.

Usage:
    python run_intent.py "Cap mMTC traffic to 5 Mbps on switch 0x30"
    python run_intent.py "Show utilization" --config prod.json
    python run_intent.py "Cap mMTC traffic to 5 Mbps on switch 0x30" --dry-run

--dry-run swaps in a recording client: it prints the RyuClient calls that
*would* be made and returns a synthetic success, without contacting Ryu.

Verifying a live run (the controller returning 200 is only step 1):
    1. Ryu HTTP log shows e.g. `POST /qos/queue/0000000000000030 ... 200`
    2. In Mininet/OVS: `ovs-vsctl list qos` / `list queue` shows the new rate
    3. Behavioural: iperf to the slice host caps at the requested rate
"""

import argparse
import json
import sys

from ai_layer.network_interface.ryu_client import RyuClient
from intent_layer import IntentTranslator


class _RecordingRyuClient:
    """Dry-run stand-in: records calls instead of issuing HTTP requests."""

    def __init__(self):
        self.calls = []

    def __getattr__(self, name):
        def recorder(*args, **kwargs):
            self.calls.append((name, args, kwargs))
            return {"_dry_run": name, "args": list(args)}
        return recorder

    def dump_calls(self):
        if not self.calls:
            print("ryu calls: <none>")
            return
        print("ryu calls (dry-run, not sent):")
        for name, args, kwargs in self.calls:
            parts = [repr(a) for a in args] + [f"{k}={v!r}" for k, v in kwargs.items()]
            print(f"  {name}({', '.join(parts)})")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Resolve a natural-language intent against the live Ryu controller"
    )
    parser.add_argument("intent", help="Intent text, e.g. 'Cap mMTC traffic to 5 Mbps on switch 0x30'")
    parser.add_argument("--config", default="prod.json", help="Path to config JSON")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the RyuClient calls that would be made without contacting Ryu",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = json.load(f)

    ryu_cfg = config["environment"]["ryu_controller"]

    if args.dry_run:
        ryu = _RecordingRyuClient()
    else:
        ryu = RyuClient(ryu_cfg)

    # resolve() catches handler-level errors (incl. unreachable Ryu) and returns
    # an IntentResult with status "error" — no need to wrap it here.
    result = IntentTranslator(ryu).resolve(args.intent)

    if args.dry_run:
        ryu.dump_calls()

    print(json.dumps(result.to_dict(), indent=2))
    return 0 if result.status in ("success", "unsupported") else 1


if __name__ == "__main__":
    raise SystemExit(main())
