"""
SDN Gymnasium Environment for the routing-first optimization pipeline.

Observation: np.array of 6 floats in [0, 1]
Action:      Discrete(4)

step() cycle:
    1. Execute runtime optimization action
    2. Wait stabilization_delay seconds
    3. Collect /links/utilization + /latency/{src}/{dst}
    4. Parse telemetry into 6D state
    5. Compute operational reward
"""

import logging
import time

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from ai_layer.network_interface.action_translator import ActionTranslator
from ai_layer.network_interface.ryu_client import RyuClient
from ai_layer.network_interface.telemetry_parser import TelemetryParser
from ai_layer.utils.reward import compute_reward_details

logger = logging.getLogger(__name__)


class SDNEnv(gym.Env):
    """Gymnasium environment wrapping the live SDN network."""

    metadata = {"render_modes": ["human"]}

    def __init__(self, config: dict):
        super().__init__()

        self.config = config
        env_cfg = config["environment"]
        net_cfg = env_cfg["network"]
        mon_cfg = env_cfg.get("monitoring", {})
        main_pair = mon_cfg.get("main_pair", {})
        episode_cfg = env_cfg.get("episode", {})

        self.stabilization_delay = float(net_cfg.get("stabilization_delay_seconds", 2.0))
        self.max_steps = int(episode_cfg.get("max_steps", 50))
        self.call_network_reset_on_reset = bool(episode_cfg.get("call_network_reset_on_reset", False))

        self.latency_src = main_pair.get("src", "G6_D1")
        self.latency_dst = main_pair.get("dst", "URLLC")
        self.failover_active = False

        self.client = RyuClient(env_cfg["ryu_controller"])
        self.parser = TelemetryParser(
            main_link_capacity_mbps=net_cfg.get("main_link_capacity_mbps", 20.0),
            backup_link_capacity_mbps=net_cfg.get("backup_link_capacity_mbps", 10.0),
            latency_min_ms=mon_cfg.get("latency_min_ms", 10.0),
            latency_max_ms=mon_cfg.get("latency_max_ms", 80.0),
            packet_loss_max_percent=mon_cfg.get("packet_loss_max_percent", 5.0),
        )
        self.translator = ActionTranslator(self.client, config)
        self.reward_cfg = config.get("reward_function", None)

        repeat_cfg = self.reward_cfg.get("components", {}).get("action_repeat_penalty", {}) if self.reward_cfg else {}
        self.repeat_penalty_weight = (
            float(repeat_cfg.get("weight", 0.0)) if bool(repeat_cfg.get("enabled", False)) else 0.0
        )
        outcome_cfg = self.reward_cfg.get("components", {}).get("outcome_improvement_bonus", {}) if self.reward_cfg else {}
        self.outcome_bonus_weight = (
            float(outcome_cfg.get("weight", 0.0)) if bool(outcome_cfg.get("enabled", False)) else 0.0
        )
        self.outcome_bonus_max = float(outcome_cfg.get("max_bonus", 0.2)) if outcome_cfg else 0.2

        state_dim = int(env_cfg.get("state_space", {}).get("dimension", 6))
        action_dim = int(env_cfg.get("action_space", {}).get("dimension", 4))

        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(state_dim,),
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(action_dim)

        self._current_step = 0
        self._state = np.zeros(state_dim, dtype=np.float32)
        self._last_action = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._current_step = 0
        self._last_action = None

        # Episodes must start from a known path. Without this, a failover left
        # active by the previous episode silently becomes the next episode's
        # initial condition, and with a shared env one policy's final state
        # becomes the next policy's starting state.
        #
        # Clearing the flag alone is not enough: the override flow lives on the
        # switch, owned by ActionTranslator, so the fabric is reset too --
        # otherwise the env would report failover_active=False while the
        # override is still forwarding traffic down the backup path.
        if options and "failover_active" in options:
            # Deliberate mid-failover start. The caller owns fabric/belief
            # agreement here; nothing is issued to the switch.
            self.failover_active = bool(options["failover_active"])
        else:
            result = self.translator.reset_overrides()
            if result["errors"]:
                logger.warning("Route override reset incomplete: %s", result["errors"])
            self.failover_active = False

        if self.call_network_reset_on_reset:
            try:
                self.client.reset_network()
            except Exception:
                logger.debug("Network reset endpoint unavailable, continuing without reset")

        time.sleep(self.stabilization_delay)
        self._state = self._observe()

        return self._state, {"failover_active": self.failover_active}

    def step(self, action: int):
        prev_congestion = float(max(self._state[3], self._state[4]))

        result = self.translator.execute(action)
        if not result.success:
            logger.warning("Action %d failed: %s", action, result.message)

        if result.success and "failover_active" in result.metadata:
            self.failover_active = bool(result.metadata["failover_active"])

        time.sleep(self.stabilization_delay)
        self._state = self._observe()

        reward_details = compute_reward_details(self._state, self.reward_cfg)

        repeat_penalty = 0.0
        if self.repeat_penalty_weight > 0.0 and self._last_action is not None and action == self._last_action:
            repeat_penalty = -self.repeat_penalty_weight

        outcome_bonus = 0.0
        post_congestion = float(max(self._state[3], self._state[4]))
        if self.outcome_bonus_weight > 0.0 and action in (1, 2, 3):
            improvement = max(0.0, prev_congestion - post_congestion)
            outcome_bonus = min(self.outcome_bonus_max, improvement * self.outcome_bonus_weight)

        reward = reward_details["total"] + repeat_penalty + outcome_bonus

        clip_lo, clip_hi = -10.0, 10.0
        if self.reward_cfg is not None:
            clip_lo, clip_hi = self.reward_cfg.get("normalization", {}).get("clip_range", [clip_lo, clip_hi])
        reward = float(np.clip(reward, clip_lo, clip_hi))

        reward_details["action_repeat_penalty"] = repeat_penalty
        reward_details["outcome_improvement_bonus"] = outcome_bonus
        reward_details["total"] = reward
        self._last_action = action

        self._current_step += 1
        terminated = False
        truncated = self._current_step >= self.max_steps

        info = {
            "step": self._current_step,
            "failover_active": self.failover_active,
            "action_name": result.action_name,
            "action_success": result.success,
            "action_metadata": result.metadata,
            "reward": reward,
            "reward_components": reward_details,
        }
        return self._state, reward, terminated, truncated, info

    def render(self):
        labels = [
            "latency",
            "loss",
            "throughput",
            "u_main",
            "u_backup",
            "failover",
        ]
        parts = [f"{label}={value:.3f}" for label, value in zip(labels, self._state)]
        print(f"[Step {self._current_step:3d}] {' | '.join(parts)}")

    def _observe(self) -> np.ndarray:
        links = self.client.get_link_utilization()
        latency = self.client.get_latency(self.latency_src, self.latency_dst)
        return self.parser.build_state(
            link_util_response=links,
            latency_response=latency,
            failover_active=self.failover_active,
        )
