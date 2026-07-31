import argparse
import json

from ai_layer.network_setup import NetworkInitializer
from ai_layer.network_setup.next_hop_capture import capture_next_hops
from ai_layer.network_interface.ryu_client import RyuClient
from ai_layer.network_interface.action_translator import ActionTranslator


def parse_args():
    parser = argparse.ArgumentParser(description="Initialize routing and baseline QoS once")
    parser.add_argument("--config", default="prod.json", help="Path to config JSON")
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue setup when a step fails",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned setup steps without calling APIs",
    )
    parser.add_argument(
        "--capture-next-hops",
        action="store_true",
        help="Record per-gateway L2 rewrites for route-override actions",
    )
    parser.add_argument(
        "--next-hop-map",
        default=None,
        help="Where to write the next-hop map (default: config value)",
    )
    return parser.parse_args()


def clear_route_overrides(config: dict) -> dict:
    """Remove any leftover route-override flow so runs start from a known state.

    ActionTranslator tracks override state in memory only, so a run that died
    mid-failover leaves the override installed while the next process believes
    none exists — the first reroute would then be treated as a no-op. Deleting
    the override cookie unconditionally makes that unrepresentable.

    delete_strict on a flow that does not exist is a no-op on OVS, so this is
    safe on a clean network.

    Delegates to ActionTranslator so the override's table/priority/cookie are
    defined in exactly one place — a mismatch here would silently fail to
    delete the flow the translator installs.
    """
    client = RyuClient(config["environment"]["ryu_controller"])
    return ActionTranslator(client, config).reset_overrides()


def main():
    args = parse_args()
    with open(args.config, "r", encoding="utf-8") as f:
        config = json.load(f)

    if args.continue_on_error:
        config.setdefault("environment", {}).setdefault("startup_setup", {})[
            "continue_on_error"
        ] = True
    if args.dry_run:
        config.setdefault("debugging", {})["dry_run_setup"] = True

    initializer = NetworkInitializer(config)
    summary = initializer.initialize()

    print("Network setup summary:")
    print(json.dumps(summary.as_dict(), indent=2))

    if not args.dry_run:
        print("\nRoute override reset:")
        print(json.dumps(clear_route_overrides(config), indent=2))

    if args.capture_next_hops and not args.dry_run:
        # Must run after routing setup: the rewrites are read back out of the
        # flows rest_router installs, which is also what makes them correct —
        # they are the MACs the controller itself resolved via ARP.
        result = capture_next_hops(config, output_path=args.next_hop_map)
        print("\nNext-hop capture:")
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
