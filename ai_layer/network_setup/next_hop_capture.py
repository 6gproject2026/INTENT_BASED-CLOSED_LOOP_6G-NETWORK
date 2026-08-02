"""Capture per-gateway L2 rewrites for route-override actions.

Route overrides are installed as raw flow-mods, so they must carry the same
eth_src/eth_dst/out_port that rest_router would have used for that next hop.
Those values cannot be read back over REST on this controller: ofctl_rest's
flow-stats replies never arrive (1.0s timeout, empty body). They are instead
read directly from OVS and correlated with the route table:

    GET /router/{dpid}   ->  route_id -> gateway
    ovs-ofctl dump-flows ->  cookie   -> eth_src / eth_dst / out_port

qos_rest_router encodes route_id in the upper 16 bits of the flow cookie, so
cookie >> 16 == route_id joins the two.

ORDERING: this must run while traffic is flowing. qos_rest_router installs
routes as packet-in stubs (actions=CONTROLLER:65535) and only writes the
eth_src/eth_dst/output flow once ARP resolves, which requires live traffic.
On a quiet network there is genuinely nothing to read.
"""

import json
import re
import subprocess

from ai_layer.network_interface.ryu_client import RyuClient


class NextHopCaptureError(RuntimeError):
    """Capture produced no usable map."""


_FLOW_RE = re.compile(
    r"cookie=(?P<cookie>0x[0-9a-f]+).*?"
    r"set_field:(?P<eth_src>[0-9a-f:]{17})->eth_src.*?"
    r"set_field:(?P<eth_dst>[0-9a-f:]{17})->eth_dst.*?"
    r"output:(?P<out_port>\d+)",
    re.IGNORECASE,
)

DEFAULT_DUMP_CMD = [
    "./scripts/mn.sh", "sw", "ovs-ofctl", "-O", "OpenFlow13",
    "dump-flows", "{bridge}",
]


def _dump_flows(bridge: str, cmd_template: list) -> str:
    cmd = [part.format(bridge=bridge) for part in cmd_template]
    return subprocess.run(
        cmd, capture_output=True, text=True, timeout=30, check=True
    ).stdout


def _rewrites_by_route_id(flow_text: str) -> dict:
    rewrites = {}
    for match in _FLOW_RE.finditer(flow_text):
        route_id = int(match.group("cookie"), 16) >> 16
        if route_id == 0:
            continue  # table-0 QoS / LLDP flows carry no route id
        rewrites[route_id] = {
            "eth_src": match.group("eth_src").lower(),
            "eth_dst": match.group("eth_dst").lower(),
            "out_port": int(match.group("out_port")),
        }
    return rewrites


def _required_gateways(config: dict) -> set:
    """Gateways the configured actions will actually ask for at runtime.

    A non-empty map is not sufficient: train.py fails on the specific key it
    needs ("No next-hop rewrite captured for 48:14.0.0.2"), so capture only
    succeeded if every gateway the action space references was resolved.
    """
    network = config["environment"]["network"]
    default_switch = str(network.get("switch_dpid", ""))
    required = set()
    for action in config["environment"]["action_space"]["actions"].values():
        target = action.get("target", {})
        gateway = target.get("route", {}).get("gateway")
        if gateway:
            switch_id = str(target.get("switch_dpid", default_switch))
            required.add(f"{switch_id}:{gateway}")
    return required


def _diagnose(next_hops: dict, missing: set, dumped: list, skipped: list) -> str:
    """Explain which of the two distinct failure modes actually occurred."""
    lines = [
        "Next-hop capture failed: %d gateway(s) resolved, missing %s."
        % (len(next_hops), sorted(missing) if missing else "(none, but map is empty)")
    ]
    if skipped:
        lines.append(
            "  %d bridge dump(s) FAILED - check scripts/mn.sh CONTAINER (it is "
            "hardcoded) and that this ran from the repo root:" % len(skipped)
        )
        lines += [
            "    %s (%s): %s" % (s["bridge"], s["dpid"], s["error"]) for s in skipped
        ]
    if dumped and all(count == 0 for _, count in dumped):
        lines.append(
            "  All %d bridge dumps succeeded but contained no MAC-rewrite flows.\n"
            "  qos_rest_router installs routes as packet-in stubs "
            "(actions=CONTROLLER:65535) and only writes the\n"
            "  eth_src/eth_dst/output flow after ARP resolves, which requires live "
            "traffic. Start traffic,\n"
            "  confirm a host-to-host ping across the fabric succeeds, then re-run."
            % len(dumped)
        )
    return "\n".join(lines)


def capture_next_hops(config: dict, output_path=None) -> dict:
    network = config["environment"]["network"]
    override_cfg = network.get("route_override", {})
    output_path = output_path or override_cfg.get(
        "next_hop_map_path", "models/next_hops.json"
    )
    cmd_template = override_cfg.get("dump_flows_command", DEFAULT_DUMP_CMD)

    client = RyuClient(config["environment"]["ryu_controller"])
    # alias -> dpid  becomes  dpid -> alias, since dump-flows takes bridge names
    bridges = {str(v): k for k, v in network.get("switch_dpids", {}).items()}

    if not bridges:
        raise NextHopCaptureError(
            "environment.network.switch_dpids is empty - nothing to dump."
        )

    next_hops, skipped, dumped = {}, [], []
    for dpid, bridge in sorted(bridges.items()):
        try:
            flow_text = _dump_flows(bridge, cmd_template)
        except (subprocess.SubprocessError, OSError) as exc:
            skipped.append({"dpid": dpid, "bridge": bridge, "error": str(exc)})
            continue

        rewrites = _rewrites_by_route_id(flow_text)
        dumped.append((bridge, len(rewrites)))

        for entry in client.get_router_config(dpid) or []:
            for net in entry.get("internal_network", []) or []:
                for route in net.get("route", []) or []:
                    hop = rewrites.get(route.get("route_id"))
                    if hop:
                        next_hops[f"{dpid}:{route['gateway']}"] = hop

    # Validate BEFORE writing. An empty or incomplete map must never overwrite a
    # previously-good one, and must never be reported as a successful capture:
    # the point is that train.py should not be the thing that discovers this,
    # one failed episode at a time.
    missing = _required_gateways(config) - set(next_hops)
    if not next_hops or missing:
        raise NextHopCaptureError(_diagnose(next_hops, missing, dumped, skipped))

    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(next_hops, fh, indent=2, sort_keys=True)

    return {
        "path": output_path,
        "captured": len(next_hops),
        "gateways": sorted(next_hops),
        "skipped": skipped,
    }
