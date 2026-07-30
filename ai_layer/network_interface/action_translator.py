"""
Action Translator for runtime optimization actions.

Action mapping (from prod.json):
    0 -> do_nothing   : no API call
    1 -> update_queue : POST /qos/queue/{switch_id}
    2 -> failover     : POST /stats/flowentry/add           (install route override)
    3 -> reroute      : POST /stats/flowentry/delete_strict (remove route override)

Route changes deliberately bypass rest_router's route table. Two measured
behaviours on this controller make the original POST /router approach a no-op:

  * POSTing a destination that already exists is rejected by RoutingTable.add's
    overlap check ("Destination overlaps [route_id=2]"), and both failover and
    reroute target 20.0.0.0/24, which is already installed.
  * DELETE /router cannot free the destination first: it returns
    "Data is nothing." after a 1.0s flow-stats timeout.

A higher-priority flow in the routing table overrides the installed route
without removing it; delete_strict reverses it. Flow-mods are unaffected by the
stats-reply defect because they expect no reply.
"""

import json
import logging
from dataclasses import dataclass, field

from ai_layer.network_interface.ryu_client import RyuClient, raise_on_command_failure

logger = logging.getLogger(__name__)


@dataclass
class ActionResult:
    success: bool
    action_id: int
    action_name: str
    message: str
    metadata: dict = field(default_factory=dict)


class ActionTranslator:
    """Converts runtime action IDs into concrete Ryu API operations."""

    def __init__(self, client: RyuClient, config: dict):
        self.client = client
        self.config = config
        self.default_switch = config["environment"]["network"]["switch_dpid"]
        self._actions_cfg = config["environment"]["action_space"]["actions"]
        self._override_cfg = config["environment"]["network"].get("route_override", {})
        self._next_hops = self._load_next_hops()
        # (switch_id, destination) -> bool. The override flow is the only thing
        # that can diverge from the route table, so tracking it is enough to
        # answer "is the desired state already in effect?". Assumed absent at
        # construction; setup_network.py clears stale overrides so that holds.
        self._override_active = {}

        self._handlers = {
            0: self._do_nothing,
            1: self._update_queue,
            2: self._failover,
            3: self._reroute,
        }

    def execute(self, action_id: int) -> ActionResult:
        handler = self._handlers.get(action_id)
        if handler is None:
            return ActionResult(
                success=False,
                action_id=action_id,
                action_name="unknown",
                message=f"Invalid action_id: {action_id}",
            )

        action_key = str(action_id)
        action_name = self._actions_cfg.get(action_key, {}).get("name", "unknown")

        try:
            metadata = handler(action_key) or {}
            logger.info("Action %d (%s) executed", action_id, action_name)
            return ActionResult(
                success=True,
                action_id=action_id,
                action_name=action_name,
                message=f"Executed {action_name}",
                metadata=metadata,
            )
        except Exception as exc:
            logger.error("Action %d (%s) failed: %s", action_id, action_name, exc)
            return ActionResult(
                success=False,
                action_id=action_id,
                action_name=action_name,
                message=str(exc),
            )

    def _do_nothing(self, _action_key: str) -> dict:
        return {}

    def _update_queue(self, action_key: str) -> dict:
        target = self._actions_cfg[action_key].get("target", {})
        switch_id = str(target.get("switch_dpid", self.default_switch))
        qos_config = target.get("qos_config", {})
        if not qos_config:
            raise ValueError("Missing qos_config for update_queue action")
        resp = self.client.apply_qos(switch_id, qos_config)
        raise_on_command_failure(resp, f"update_queue on {switch_id}")
        return {"operation": "update_queue", "switch_id": switch_id}

    def _failover(self, action_key: str) -> dict:
        return self._set_route_override(action_key, activate=True)

    def _reroute(self, action_key: str) -> dict:
        return self._set_route_override(action_key, activate=False)

    # ------------------------------------------------------------------ #
    #  Route override
    # ------------------------------------------------------------------ #

    def _set_route_override(self, action_key: str, activate: bool) -> dict:
        cfg = self._actions_cfg[action_key]
        operation = cfg.get("name", "failover" if activate else "reroute")
        target = cfg.get("target", {})
        switch_id = str(target.get("switch_dpid", self.default_switch))
        route = target.get("route", {})
        if not route:
            raise ValueError(f"Missing route payload for {operation} action")
        destination = route["destination"]
        gateway = route["gateway"]

        state_key = (switch_id, destination)
        if self._override_active.get(state_key, False) == activate:
            logger.info("%s: already in desired state, no API call", operation)
            return {
                "operation": operation,
                "switch_id": switch_id,
                "failover_active": activate,
                "no_op": True,
                "reason": "already in desired state",
            }

        if activate:
            next_hop = self._next_hop(switch_id, gateway)
            self.client.install_flow(
                switch_id, self._override_flow(destination, next_hop)
            )
            logger.info(
                "%s: override installed for %s via %s (port %s)",
                operation, destination, gateway, next_hop["out_port"],
            )
        else:
            # Falling back to the route table only restores the intended path if
            # the base entry still points where reroute expects. Verify before
            # removing the override, otherwise this silently lands elsewhere.
            base_gw = self._base_route_gateway(switch_id, destination)
            if base_gw != gateway:
                raise RuntimeError(
                    f"{operation}: base route for {destination} on {switch_id} is "
                    f"via {base_gw}, expected {gateway}; refusing to remove override"
                )
            self.client.delete_flow_strict(
                switch_id, self._override_flow_key(destination)
            )
            logger.info(
                "%s: override removed, %s falls back to %s",
                operation, destination, gateway,
            )

        self._override_active[state_key] = activate
        return {
            "operation": operation,
            "switch_id": switch_id,
            "failover_active": activate,
            "no_op": False,
        }

    def _override_flow_key(self, destination: str) -> dict:
        """Flow identity shared by add and delete_strict.

        table_id must be the table rest_router installs routes into (1 under
        qos_rest_router, which reserves table 0 for queue classification), and
        priority must exceed the route's own 2 + netmask.
        """
        return {
            "table_id": int(self._override_cfg.get("table_id", 1)),
            "priority": int(self._override_cfg.get("priority", 100)),
            "cookie": int(self._override_cfg.get("cookie", 0xA17E)),
            "match": {"eth_type": 0x0800, "ipv4_dst": destination},
        }

    def _override_flow(self, destination: str, next_hop: dict) -> dict:
        flow = self._override_flow_key(destination)
        flow["actions"] = [
            {"type": "DEC_NW_TTL"},
            {"type": "SET_FIELD", "field": "eth_src", "value": next_hop["eth_src"]},
            {"type": "SET_FIELD", "field": "eth_dst", "value": next_hop["eth_dst"]},
            {"type": "OUTPUT", "port": int(next_hop["out_port"])},
        ]
        return flow

    def _base_route_gateway(self, switch_id: str, destination: str):
        """Gateway of rest_router's own route entry for `destination`, or None."""
        resp = self.client.get_router_config(switch_id)
        for entry in resp or []:
            for network in entry.get("internal_network", []) or []:
                for route in network.get("route", []) or []:
                    if route.get("destination") == destination:
                        return route.get("gateway")
        return None

    def _next_hop(self, switch_id: str, gateway: str) -> dict:
        key = f"{switch_id}:{gateway}"
        hop = self._next_hops.get(key)
        if not hop:
            raise RuntimeError(
                f"No next-hop rewrite captured for {key}. Run "
                f"'python setup_network.py --config <cfg> --capture-next-hops'. "
                f"Map path: {self._next_hop_path}"
            )
        return hop

    def _load_next_hops(self) -> dict:
        self._next_hop_path = self._override_cfg.get(
            "next_hop_map_path", "models/next_hops.json"
        )
        try:
            with open(self._next_hop_path, "r", encoding="utf-8") as fh:
                return json.load(fh)
        except FileNotFoundError:
            logger.warning(
                "Next-hop map %s not found; failover will fail until captured",
                self._next_hop_path,
            )
            return {}
