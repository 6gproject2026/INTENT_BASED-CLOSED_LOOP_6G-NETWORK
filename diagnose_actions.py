#!/usr/bin/env python3
"""Diagnose whether RL actions physically move traffic on the live network.

Read-only intent: this script issues the SAME action calls the agent issues
(through ActionTranslator.execute, the real code path) and measures the network
before/after each one. It changes network state exactly as training does — it
does NOT add any new kind of write. Run it against a live Ryu controller with
Mininet up and traffic flowing, otherwise every link reads ~0 and the verdict
is meaningless.

    python diagnose_actions.py --config prod.json

Everything is read from prod.json. Nothing is hardcoded.
"""

import argparse
import json
import sys
import time

from ai_layer.network_interface.ryu_client import (
    RyuClient,
    RyuClientError,
    RyuConnectionError,
)
from ai_layer.network_interface.action_translator import ActionTranslator

# Exact link names the telemetry parser sums over (spaces around ->).
MAIN_LINKS = ["core -> sp1", "sp1 -> lf1"]
BACKUP_LINKS = ["core -> sp2", "sp2 -> lf1"]
ALL_LINKS = MAIN_LINKS + BACKUP_LINKS

# Action id -> human label, matching prod.json environment.action_space.
ACTION_LABELS = {0: "do_nothing", 1: "update_queue", 2: "failover", 3: "reroute"}

# The route every failover/reroute action targets, per prod.json action_space.
ROUTE_DEST = "20.0.0.0/24"

UTIL_CHANGE_THRESHOLD_MBPS = 0.5


def parse_args():
    p = argparse.ArgumentParser(description="Diagnose physical effect of RL actions")
    p.add_argument("--config", default="prod.json", help="Path to config JSON")
    return p.parse_args()


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def parse_links(links_response):
    """{link_name: tx_mbps} from GET /links/utilization."""
    out = {}
    for item in (links_response or {}).get("links", []):
        name = item.get("link")
        if name is not None:
            out[name] = float(item.get("tx_mbps", 0.0))
    return out


def count_dest_routes(router_response, dest):
    """Count route entries matching `dest` anywhere in the /router/{dpid} body.

    Ryu rest_router returns a nested structure; parse defensively so a format
    quirk does not crash the diagnostic. Returns (count, sample_entries).
    """
    found = []

    def walk(obj):
        if isinstance(obj, dict):
            d = obj.get("destination")
            if isinstance(d, str) and d.replace(" ", "") == dest.replace(" ", ""):
                found.append(obj)
            for v in obj.values():
                walk(v)
        elif isinstance(obj, list):
            for v in obj:
                walk(v)

    walk(router_response)
    return len(found), found[:5]


def snapshot(client, core_dpid, lat_src, lat_dst, stage):
    """Read links + latency + route count. Never raises for latency nulls."""
    links_raw = client.get_link_utilization()
    links = parse_links(links_raw)

    latency_ms = loss_pct = None
    latency_null = loss_null = False
    try:
        lat = client.get_latency(lat_src, lat_dst)
        latency_ms = lat.get("latency_ms")
        loss_pct = lat.get("packet_loss_percent")
        latency_null = latency_ms is None
        loss_null = loss_pct is None
    except RyuClientError as exc:
        latency_null = loss_null = True
        print(f"    [warn] latency read failed: {exc}")

    # GET /router/{dpid}; normalize dpid exactly as the write path does.
    hexid = client._normalize_dpid(core_dpid)
    try:
        router_raw = client._request("GET", f"/router/{hexid}")
    except RyuClientError as exc:
        router_raw = {"_error": str(exc)}
    route_count, route_sample = count_dest_routes(router_raw, ROUTE_DEST)

    snap = {
        "stage": stage,
        "links": links,
        "main_tx": sum(links.get(n, 0.0) for n in MAIN_LINKS),
        "backup_tx": sum(links.get(n, 0.0) for n in BACKUP_LINKS),
        "latency_ms": latency_ms,
        "loss_pct": loss_pct,
        "latency_null": latency_null,
        "loss_null": loss_null,
        "route_count": route_count,
        "route_sample": route_sample,
    }
    return snap


def print_snapshot(s):
    print(f"  [{s['stage']}]")
    for n in ALL_LINKS:
        present = "" if n in s["links"] else "  (MISSING from response)"
        print(f"      {n:14s} = {s['links'].get(n, 0.0):8.3f} Mbps{present}")
    print(f"      main_tx (sum) = {s['main_tx']:.3f} Mbps | backup_tx (sum) = {s['backup_tx']:.3f} Mbps")
    lat = "NULL" if s["latency_null"] else f"{s['latency_ms']}"
    loss = "NULL" if s["loss_null"] else f"{s['loss_pct']}"
    print(f"      latency_ms = {lat} | packet_loss_percent = {loss}")
    print(f"      {ROUTE_DEST} route entries on core = {s['route_count']}")


def max_link_delta(before, after):
    """Largest absolute per-link tx change (Mbps) across all four links."""
    worst = 0.0
    worst_link = None
    for n in ALL_LINKS:
        d = abs(after["links"].get(n, 0.0) - before["links"].get(n, 0.0))
        if d > worst:
            worst, worst_link = d, n
    return worst, worst_link


def main():
    args = parse_args()
    config = load_config(args.config)

    env_cfg = config["environment"]
    ryu_cfg = env_cfg["ryu_controller"]
    net_cfg = env_cfg["network"]
    mon = env_cfg.get("monitoring", {}).get("main_pair", {})

    core_dpid = str(net_cfg.get("switch_dpids", {}).get("core", net_cfg.get("switch_dpid", "48")))
    lat_src = mon.get("src", "G6_D1")
    lat_dst = mon.get("dst", "URLLC")
    stab = float(net_cfg.get("stabilization_delay_seconds", 1.0))

    client = RyuClient(ryu_cfg)
    translator = ActionTranslator(client, config)

    print(f"Live action diagnostic against {client.base_url}")
    print(f"core dpid={core_dpid} (hex {client._normalize_dpid(core_dpid)}) | "
          f"latency pair {lat_src}->{lat_dst} | stabilization={stab}s\n")

    # --- Preflight -------------------------------------------------------
    try:
        switches = client._request("GET", "/stats/switches")
    except RyuConnectionError as exc:
        print(f"ABORT: controller unreachable — {exc}\n"
              "Start Mininet, then Ryu, then retry. See docs/RUNNING_GUIDE.md §3.")
        return 1
    print(f"Controller reachable. /stats/switches -> {switches}\n")

    # --- Baseline --------------------------------------------------------
    baseline = snapshot(client, core_dpid, lat_src, lat_dst, "baseline")
    print_snapshot(baseline)

    # --- Drive each action through the REAL code path --------------------
    # Order chosen to expose route effects: failover (to backup), then reroute
    # (back to main), then a queue change, then a no-op control.
    sequence = [2, 3, 1, 0]
    results = {}   # action_id -> (ActionResult, before_snap, after_snap)
    prev = baseline

    for action_id in sequence:
        label = ACTION_LABELS[action_id]
        print(f"\n--- executing action {action_id} ({label}) ---")
        try:
            result = translator.execute(action_id)
        except RyuConnectionError as exc:
            print(f"ABORT: controller lost mid-diagnostic — {exc}")
            return 1

        # ActionResult carries no HTTP status/body (see report finding a);
        # print every field it does have.
        print("  ActionResult:")
        print(f"      success      = {result.success}")
        print(f"      action_name  = {result.action_name}")
        print(f"      message      = {result.message}")
        print(f"      metadata     = {result.metadata}")

        time.sleep(stab)
        after = snapshot(client, core_dpid, lat_src, lat_dst, f"after action {action_id} ({label})")
        print_snapshot(after)
        results[action_id] = (result, prev, after)
        prev = after

    # --- Verdict table ---------------------------------------------------
    print("\n" + "=" * 68)
    print("VERDICT: did any link utilization move > "
          f"{UTIL_CHANGE_THRESHOLD_MBPS} Mbps because of the action?")
    print("=" * 68)
    print(f"{'action':22s} {'reported':10s} {'max Δ link':22s} {'moved?'}")
    print("-" * 68)
    for action_id in sequence:
        result, before, after = results[action_id]
        delta, link = max_link_delta(before, after)
        moved = "YES" if delta > UTIL_CHANGE_THRESHOLD_MBPS else "NO"
        where = f"{delta:.3f} @ {link}" if link else f"{delta:.3f}"
        print(f"{action_id} ({ACTION_LABELS[action_id]}):".ljust(22),
              f"{'ok' if result.success else 'FAIL':10s} {where:22s} {moved}")

    # --- Route accumulation check ---------------------------------------
    print("\nDuplicate-route check on core switch:")
    print(f"  {ROUTE_DEST} entries: baseline={baseline['route_count']} -> "
          f"final={prev['route_count']}")
    if prev["route_count"] > baseline["route_count"]:
        print("  -> POSTs ADDED routes (rest_router does not replace). "
              "Over a full run this stacks hundreds of duplicates.")
    elif prev["route_count"] == baseline["route_count"] and baseline["route_count"] > 1:
        print("  -> multiple entries already present before this run.")

    print("\nDone. This measured the real network; interpret per the report questions.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
