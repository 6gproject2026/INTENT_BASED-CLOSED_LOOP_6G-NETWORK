"""Record per-gateway L2 rewrites used by route-override actions.

Run AFTER traffic is flowing and a cross-fabric ping succeeds.
qos_rest_router installs routes as packet-in stubs and only writes the
eth_src/eth_dst/output flow once ARP resolves, so on a quiet network there is
genuinely nothing to capture.

Deliberately NOT part of setup_network.py: setup must run on a quiet network,
capture must not. Bundling them made the ordering invisible and produced an
empty map that only surfaced later, as a per-episode failure in train.py.
"""

import argparse
import json
import sys

from ai_layer.network_setup.next_hop_capture import (
    capture_next_hops,
    NextHopCaptureError,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Capture next-hop L2 rewrites (run while traffic is flowing)"
    )
    parser.add_argument("--config", default="prod.json", help="Path to config JSON")
    parser.add_argument(
        "--output",
        default=None,
        help="Override environment.network.route_override.next_hop_map_path",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = json.load(f)

    try:
        result = capture_next_hops(config, output_path=args.output)
    except NextHopCaptureError as exc:
        print(exc)
        print("\nNothing written; any existing map is unchanged.")
        return 1

    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
