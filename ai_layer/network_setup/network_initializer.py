import logging
from dataclasses import dataclass, field
from typing import List

from ai_layer.network_interface.ryu_client import (
    RyuClient,
    RyuResponseError,
    command_failures,
)

logger = logging.getLogger(__name__)

# Re-applying setup against a configured controller is routine: a plain re-run
# and train.py's reconnect path both replay every step. rest_router reports
# those replays as body-level failures — "Address overlaps [address_id=1]",
# "Destination overlaps [route_id=2]" — which mean the desired state is already
# present, not that the step failed. They are tracked separately so that real
# failures stop being counted as successes without making setup non-idempotent.
IDEMPOTENT_CONFLICT_MARKERS = ("overlaps",)


def _is_idempotent_conflict(details: List[str]) -> bool:
    return bool(details) and all(
        any(marker in detail.lower() for marker in IDEMPOTENT_CONFLICT_MARKERS)
        for detail in details
    )


@dataclass
class SetupSummary:
    enabled: bool
    dry_run: bool
    routing_steps: int = 0
    qos_steps: int = 0
    successes: int = 0
    failures: int = 0
    already_configured: int = 0
    errors: List[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "enabled": self.enabled,
            "dry_run": self.dry_run,
            "routing_steps": self.routing_steps,
            "qos_steps": self.qos_steps,
            "successes": self.successes,
            "failures": self.failures,
            "already_configured": self.already_configured,
            "errors": self.errors,
        }


class NetworkInitializer:
    """Runs one-time routing and QoS baseline setup against Ryu APIs."""

    def __init__(self, config: dict):
        env_cfg = config.get("environment", {})
        self.startup_cfg = env_cfg.get("startup_setup", {})
        self.debug_cfg = config.get("debugging", {})
        self.continue_on_error = bool(self.startup_cfg.get("continue_on_error", False))
        # When True, an idempotent conflict is a hard failure instead of a no-op.
        self.strict_conflicts = bool(self.startup_cfg.get("strict_conflicts", False))
        self.dry_run = bool(self.debug_cfg.get("dry_run_setup", False))
        self.enabled = bool(self.startup_cfg.get("enabled", False))
        self.client = RyuClient(env_cfg.get("ryu_controller", {}))
        self.summary = SetupSummary(enabled=self.enabled, dry_run=self.dry_run)

    def initialize(self) -> SetupSummary:
        if not self.enabled:
            logger.info("Startup setup disabled by config")
            return self.summary

        logger.info("Starting network initialization (dry_run=%s)", self.dry_run)
        self._initialize_routing()
        self._initialize_qos_baseline()

        if self.summary.failures > 0 and not self.continue_on_error:
            raise RuntimeError(
                "Network setup failed with "
                f"{self.summary.failures} error(s): {self.summary.errors}"
            )

        logger.info(
            "Network initialization done: successes=%d failures=%d",
            self.summary.successes,
            self.summary.failures,
        )
        return self.summary

    def _run_step(self, phase: str, label: str, fn):
        if phase == "routing":
            self.summary.routing_steps += 1
        elif phase == "qos":
            self.summary.qos_steps += 1

        if self.dry_run:
            logger.info("[dry-run] %s: %s", phase, label)
            self.summary.successes += 1
            return

        try:
            response = fn()
            # rest_qos and qos_rest_router answer 200 OK with the real verdict in
            # the body, so an unchecked call cannot distinguish applied from
            # rejected. This is what let failover/reroute no-op silently.
            details = command_failures(response)

            if details and not self.strict_conflicts and _is_idempotent_conflict(details):
                logger.info(
                    "%s: %s (already configured: %s)", phase, label, "; ".join(details)
                )
                self.summary.already_configured += 1
                self.summary.successes += 1
                return

            if details:
                raise RyuResponseError("; ".join(details))

            logger.info("%s: %s", phase, label)
            self.summary.successes += 1
        except Exception as exc:
            msg = f"{phase} | {label} | {exc}"
            logger.error(msg)
            self.summary.failures += 1
            self.summary.errors.append(msg)
            if not self.continue_on_error:
                raise

    def _initialize_routing(self):
        routing_cfg = self.startup_cfg.get("routing", {})
        node_order = routing_cfg.get("node_order", [])
        nodes = routing_cfg.get("nodes", {})
        ovsdb_addr = str(self.startup_cfg.get("ovsdb_addr", "tcp:127.0.0.1:6632"))

        for switch_id in node_order:
            node_cfg = nodes.get(switch_id, {})

            self._run_step(
                "routing",
                f"set ovsdb addr for {switch_id}",
                lambda sid=switch_id: self.client.set_switch_ovsdb_addr(sid, ovsdb_addr),
            )

            for address in node_cfg.get("addresses", []):
                self._run_step(
                    "routing",
                    f"add address {address} on {switch_id}",
                    lambda sid=switch_id, addr=address: self.client.add_router_address(sid, addr),
                )

            for route in node_cfg.get("routes", []):
                destination = str(route.get("destination", ""))
                gateway = str(route.get("gateway", ""))
                if destination and gateway:
                    self._run_step(
                        "routing",
                        f"add route {destination} via {gateway} on {switch_id}",
                        lambda sid=switch_id, dst=destination, gw=gateway: self.client.add_router_route(
                            sid, dst, gw
                        ),
                    )

            default_gateway = node_cfg.get("default_gateway")
            if default_gateway:
                self._run_step(
                    "routing",
                    f"set default gateway {default_gateway} on {switch_id}",
                    lambda sid=switch_id, gw=default_gateway: self.client.set_router_default_gateway(
                        sid, str(gw)
                    ),
                )

    def _initialize_qos_baseline(self):
        qos_cfg = self.startup_cfg.get("qos_baseline", {})

        for rule_entry in qos_cfg.get("rules", []):
            switch_id = str(rule_entry.get("switch_dpid", ""))
            rule = rule_entry.get("rule", {})
            if not switch_id or not rule:
                continue
            self._run_step(
                "qos",
                f"install qos rule on {switch_id}",
                lambda sid=switch_id, payload=rule: self.client.post_qos_rule(sid, payload),
            )

        for queue_entry in qos_cfg.get("queues", []):
            switch_id = str(queue_entry.get("switch_dpid", ""))
            qos_config = queue_entry.get("qos_config", {})
            port_name = str(qos_config.get("port_name", "unknown"))
            if not switch_id or not qos_config:
                continue
            self._run_step(
                "qos",
                f"configure queue on {switch_id} {port_name}",
                lambda sid=switch_id, payload=qos_config: self.client.apply_qos(sid, payload),
            )
