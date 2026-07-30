import time
import logging
import requests

logger = logging.getLogger(__name__)


class RyuClientError(Exception):
    """Base exception for Ryu REST client errors."""


class RyuConnectionError(RyuClientError):
    """Failed to connect to Ryu controller."""


class RyuResponseError(RyuClientError):
    """Invalid or error response from Ryu controller."""


class RyuClient:
    """REST client for communicating with a Ryu SDN controller."""

    def __init__(self, config: dict):
        """
        Args:
            config: The 'environment.ryu_controller' section from prod.json.
        """
        self.base_url = config["base_url"].rstrip("/")
        self.timeout = config.get("timeout_seconds", 5)
        self.retries = config.get("retry_attempts", 3)
        self.retry_delay = config.get("retry_delay_seconds", 1)
        # Retries use capped exponential backoff. Kept deliberately short: this
        # rides out a brief blip, while a full controller restart is handled by
        # wait_for_controller() at the training-loop level.
        self.backoff_max = config.get("retry_backoff_max_seconds", 8)
        self.reconnect_max_wait = config.get("reconnect_max_wait_seconds", 180)
        self.reconnect_poll = config.get("reconnect_poll_seconds", 5)

    # ------------------------------------------------------------------ #
    #  Internal helpers
    # ------------------------------------------------------------------ #

    def _request(self, method: str, path: str, json_body=None) -> dict:
        """Send an HTTP request with retry logic.

        Returns the parsed JSON response body, or an empty dict on 204/no-content.
        """
        url = f"{self.base_url}{path}"

        for attempt in range(1, self.retries + 1):
            try:
                resp = requests.request(
                    method, url, json=json_body, timeout=self.timeout
                )
                resp.raise_for_status()

                if resp.status_code == 204 or not resp.content:
                    return {}
                try:
                    return resp.json()
                except ValueError:
                    # Some controller apps return plain text for successful writes.
                    return {"raw": resp.text}

            except requests.ConnectionError as exc:
                logger.warning("Connection failed (attempt %d/%d): %s", attempt, self.retries, exc)
                if attempt == self.retries:
                    raise RyuConnectionError(f"Cannot reach Ryu at {url}") from exc
                time.sleep(self._backoff_delay(attempt))

            except requests.Timeout as exc:
                logger.warning("Request timed out (attempt %d/%d): %s", attempt, self.retries, exc)
                if attempt == self.retries:
                    raise RyuConnectionError(f"Timeout reaching Ryu at {url}") from exc
                time.sleep(self._backoff_delay(attempt))

            except requests.HTTPError as exc:
                raise RyuResponseError(
                    f"{method} {url} returned {resp.status_code}: {resp.text}"
                ) from exc

        # Only reachable when retry_attempts < 1, which would otherwise return None
        # and surface as a confusing TypeError far from the real cause.
        raise RyuConnectionError(
            f"No request attempted for {url}: retry_attempts={self.retries}"
        )

    def _backoff_delay(self, attempt: int) -> float:
        """Capped exponential backoff: retry_delay * 2^(attempt-1), clamped to backoff_max."""
        return min(self.retry_delay * (2 ** (attempt - 1)), self.backoff_max)

    @staticmethod
    def _normalize_dpid(dpid: str) -> str:
        if dpid is None:
            return ""
        text = str(dpid).strip().lower()
        if text.startswith("0x"):
            value = int(text, 16)
        elif len(text) == 16:
            value = int(text, 16)
        elif any(ch in text for ch in "abcdef"):
            value = int(text, 16)
        else:
            value = int(text, 10)
        return f"{value:016x}"

    @staticmethod
    def _decimal_dpid(dpid: str) -> int:
        """/stats/* endpoints take a decimal dpid, unlike conf/router/qos."""
        return int(RyuClient._normalize_dpid(dpid), 16)

    # ------------------------------------------------------------------ #
    #  GET endpoints – telemetry
    # ------------------------------------------------------------------ #

    def get_port_stats(self, dpid: str) -> dict:
        """GET /stats/port/{dpid}  →  port statistics."""
        return self._request("GET", f"/stats/port/{dpid}")

    def get_flow_stats(self, dpid: str) -> dict:
        """GET /stats/flow/{dpid}  →  flow statistics."""
        return self._request("GET", f"/stats/flow/{dpid}")

    def get_queue_stats(self, dpid: str) -> dict:
        """GET /qos/queue/{dpid}  →  queue statistics."""
        return self._request("GET", f"/qos/queue/{dpid}")

    def get_link_utilization(self) -> dict:
        """GET /links/utilization  →  per-link tx_mbps statistics."""
        return self._request("GET", "/links/utilization")

    def ping(self) -> bool:
        """Single-shot reachability check. Returns True/False, never raises.

        Deliberately bypasses _request()'s retry loop: callers poll this on their
        own schedule, so an internal retry budget would stack delays on top of theirs.
        """
        try:
            resp = requests.get(
                f"{self.base_url}/links/utilization", timeout=self.timeout
            )
            return resp.status_code < 500
        except requests.RequestException:
            return False

    def get_latency(self, src: str, dst: str) -> dict:
        """GET /latency/{src}/{dst}  →  latency/loss for a monitored pair."""
        return self._request("GET", f"/latency/{src}/{dst}")

    def get_router_config(self, switch_id: str) -> list:
        """GET /router/{switch_id}  →  addresses + static routes for one switch."""
        dpid = self._normalize_dpid(switch_id)
        return self._request("GET", f"/router/{dpid}")

    # ------------------------------------------------------------------ #
    #  POST endpoints – actions and setup
    # ------------------------------------------------------------------ #

    def install_flow(self, dpid: str, flow_rule: dict) -> dict:
        """POST /stats/flowentry/add  →  install a flow rule.

        Args:
            dpid:      Switch datapath ID.
            flow_rule: Dict with match/actions fields.
                       dpid is injected automatically.
        """
        body = {"dpid": self._decimal_dpid(dpid), **flow_rule}
        return self._request("POST", "/stats/flowentry/add", json_body=body)

    def delete_flow(self, dpid: str, flow_rule: dict) -> dict:
        """POST /stats/flowentry/delete  →  remove a flow rule."""
        body = {"dpid": self._decimal_dpid(dpid), **flow_rule}
        return self._request("POST", "/stats/flowentry/delete", json_body=body)

    def delete_flow_strict(self, dpid: str, flow_rule: dict) -> dict:
        """POST /stats/flowentry/delete_strict  →  remove exactly one flow.

        Strict matching is required for route overrides: a non-strict delete on
        ipv4_dst=<prefix> would also remove rest_router's own route flow for the
        same prefix, which is the entry the override is supposed to fall back to.
        """
        body = {"dpid": self._decimal_dpid(dpid), **flow_rule}
        return self._request("POST", "/stats/flowentry/delete_strict", json_body=body)

    def apply_qos(self, switch_id: str, qos_config: dict) -> dict:
        """POST /qos/queue/{switch_id}  →  configure a QoS queue.

        Args:
            switch_id:  Switch identifier (e.g. '0000000000000001').
            qos_config: Dict with port_name, max_rate, etc.
        """
        dpid = self._normalize_dpid(switch_id)
        return self._request("POST", f"/qos/queue/{dpid}", json_body=qos_config)

    def post_qos_rule(self, switch_id: str, rule: dict) -> dict:
        """POST /qos/rules/{switch_id}  →  add a QoS routing rule."""
        dpid = self._normalize_dpid(switch_id)
        return self._request("POST", f"/qos/rules/{dpid}", json_body=rule)

    def set_switch_ovsdb_addr(self, switch_id: str, ovsdb_addr: str) -> dict:
        """PUT /v1.0/conf/switches/{switch_id}/ovsdb_addr."""
        dpid = self._normalize_dpid(switch_id)
        return self._request(
            "PUT",
            f"/v1.0/conf/switches/{dpid}/ovsdb_addr",
            json_body=ovsdb_addr,
        )

    def post_router_entry(self, switch_id: str, payload: dict) -> dict:
        """POST /router/{switch_id} with address/route/default-gateway payload."""
        dpid = self._normalize_dpid(switch_id)
        return self._request("POST", f"/router/{dpid}", json_body=payload)

    def add_router_address(self, switch_id: str, address_cidr: str) -> dict:
        """POST router address assignment."""
        return self.post_router_entry(switch_id, {"address": address_cidr})

    def add_router_route(self, switch_id: str, destination_cidr: str, gateway_ip: str) -> dict:
        """POST static route assignment."""
        return self.post_router_entry(
            switch_id,
            {
                "destination": destination_cidr,
                "gateway": gateway_ip,
            },
        )

    def set_router_default_gateway(self, switch_id: str, gateway_ip: str) -> dict:
        """POST default gateway assignment."""
        return self.post_router_entry(switch_id, {"gateway": gateway_ip})

    def reset_network(self) -> dict:
        """POST /network/reset  →  reset network state."""
        return self._request("POST", "/network/reset")


# ---------------------------------------------------------------------- #
#  Response-body inspection
# ---------------------------------------------------------------------- #

def command_failures(response) -> list:
    """Details strings for every failed command in a Ryu REST response body.

    rest_qos and qos_rest_router answer 200 OK even when the operation failed,
    putting the verdict in the body:

        [{"switch_id": "...", "command_result": [
            {"result": "failure", "details": "Destination overlaps [route_id=2]"}]}]

    Without inspecting this, a rejected write is indistinguishable from a
    successful one, which is how failover/reroute silently no-op'd.

    The shape of `command_result` varies by app and endpoint: qos_rest_router
    returns a list of result dicts, while rest_qos returns a single dict (and
    its `details` may itself be a nested dict, not a string). Anything that is
    not a result dict is skipped rather than assumed.
    """
    entries = response if isinstance(response, list) else [response]
    failures = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        results = entry.get("command_result")
        if isinstance(results, dict):
            results = [results]
        elif not isinstance(results, list):
            continue
        for result in results:
            if not isinstance(result, dict):
                continue
            if str(result.get("result", "")).lower() == "failure":
                failures.append(str(result.get("details", "unspecified failure")))
    return failures


def raise_on_command_failure(response, context: str = "") -> None:
    """Raise RyuResponseError when a 200 response reports a failed command."""
    failures = command_failures(response)
    if failures:
        prefix = f"{context}: " if context else ""
        raise RyuResponseError(prefix + "; ".join(failures))


# ---------------------------------------------------------------------- #
#  Controller availability
# ---------------------------------------------------------------------- #

def wait_for_controller(client: RyuClient, max_wait_seconds=None, poll_seconds=None) -> bool:
    """Poll the controller until it answers or the budget expires.

    Used both as a pre-run preflight (fail fast instead of dying mid-training) and
    to ride out a controller restart without losing the run. Returns True as soon
    as the controller responds, False if max_wait_seconds elapses.
    """
    if max_wait_seconds is None:
        max_wait_seconds = client.reconnect_max_wait
    if poll_seconds is None:
        poll_seconds = client.reconnect_poll

    deadline = time.monotonic() + max_wait_seconds
    attempt = 0
    while True:
        attempt += 1
        if client.ping():
            if attempt > 1:
                logger.info("Controller reachable again after %d attempt(s)", attempt)
            return True

        remaining = deadline - time.monotonic()
        if remaining <= 0:
            logger.error(
                "Controller still unreachable at %s after %ss",
                client.base_url, max_wait_seconds,
            )
            return False

        logger.warning(
            "Controller unreachable (attempt %d), retrying in %ss (%.0fs budget left)",
            attempt, poll_seconds, remaining,
        )
        time.sleep(min(poll_seconds, remaining))
