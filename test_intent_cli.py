"""CLI test harness for the intent layer — no Ryu/Mininet required.

Usage:
    python test_intent_cli.py "Cap mMTC traffic to 5 Mbps on switch 0x30"

Resolves the intent string against a recording mock RyuClient (no network
calls), prints every RyuClient method invocation with its arguments, then
prints the IntentResult as a JSON dict.
"""

import json
import sys


class MockRyuClient:
    """Stand-in for RyuClient: records every method call instead of doing HTTP.

    Any attribute access returns a recorder that logs (name, args, kwargs)
    and returns a fake response dict, so handlers can build their results.
    """

    def __init__(self):
        self.calls = []

    def __getattr__(self, name):
        def recorder(*args, **kwargs):
            self.calls.append((name, args, kwargs))
            return {"_mock": name, "args": list(args)}
        return recorder

    def dump_calls(self):
        if not self.calls:
            print("ryu calls: <none>")
            return
        print("ryu calls:")
        for name, args, kwargs in self.calls:
            parts = [repr(a) for a in args] + [f"{k}={v!r}" for k, v in kwargs.items()]
            print(f"  {name}({', '.join(parts)})")


def main(argv):
    if len(argv) < 2:
        print(f'usage: python {argv[0]} "<intent text>"')
        return 2

    text = argv[1]

    # Imported here so a missing/broken import surfaces as a clean error.
    from intent_layer import IntentTranslator

    ryu = MockRyuClient()
    result = IntentTranslator(ryu).resolve(text)

    ryu.dump_calls()
    print(json.dumps(result.to_dict(), indent=2))

    return 0 if result.status in ("success", "unsupported") else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
