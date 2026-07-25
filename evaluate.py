import argparse
import json
import os
import random
from datetime import datetime
from typing import Callable

import numpy as np
import torch

from ai_layer.agent.dqn_agent import DQNAgent
from ai_layer.environments.sdn_env import SDNEnv
from ai_layer.network_interface.ryu_client import RyuConnectionError, wait_for_controller
from ai_layer.network_setup import NetworkInitializer

# Evaluation runs are short, so a dead controller should fail fast rather than
# waiting it out the way training does.
PREFLIGHT_WAIT_SECONDS = 10


def build_environment(config: dict):
    """Build the live SDN environment backed by Ryu telemetry/actions."""
    return SDNEnv(config)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate trained DQN and baselines")
    parser.add_argument("--config", default="prod.json", help="Path to config JSON")
    parser.add_argument("--seed", type=int, default=None, help="Random seed override")
    parser.add_argument(
        "--skip-setup",
        action="store_true",
        help="Skip startup setup before evaluation",
    )
    parser.add_argument(
        "--model-path",
        default=os.path.join("models", "dqn_model.pth"),
        help="Path to trained model",
    )
    parser.add_argument(
        "--metrics-path",
        default=os.path.join("logs", "evaluation_metrics.json"),
        help="Path for evaluation metrics JSON",
    )
    return parser.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def run_policy(
    env,
    policy_name: str,
    num_episodes: int,
    action_fn: Callable[[object], int],
    base_seed: int,
    congestion_threshold: float,
    render: bool = False,
) -> dict:
    env_action_dim = int(env.action_space.n)
    episode_rewards = []
    episode_steps = []
    action_counts = {str(i): 0 for i in range(env_action_dim)}
    success_steps = 0
    total_steps = 0
    switch_count = 0
    switch_opportunities = 0
    prev_action = None

    latency_sum = 0.0
    loss_sum = 0.0
    throughput_sum = 0.0
    failover_sum = 0.0
    congestion_hits = 0

    comp_sums = {}
    comp_steps = 0

    for episode in range(num_episodes):
        reset_out = env.reset(seed=base_seed + episode)
        state = reset_out[0] if isinstance(reset_out, tuple) else reset_out

        done = False
        reward_sum = 0.0
        steps = 0

        while not done:
            action = int(action_fn(state))
            action_counts[str(action)] += 1

            if prev_action is not None:
                switch_opportunities += 1
                if action != prev_action:
                    switch_count += 1
            prev_action = action

            step_out = env.step(action)
            if len(step_out) == 5:
                next_state, reward, terminated, truncated, info = step_out
                done = bool(terminated or truncated)
                if info.get("action_success", True):
                    success_steps += 1

                components = info.get("reward_components", {}) if isinstance(info, dict) else {}
                for k, v in components.items():
                    if isinstance(v, (int, float)):
                        comp_sums[k] = comp_sums.get(k, 0.0) + float(v)
                if components:
                    comp_steps += 1
            else:
                next_state, reward, done = step_out
                done = bool(done)
                success_steps += 1

            state = next_state
            reward_sum += float(reward)
            steps += 1
            total_steps += 1

            latency_sum += float(next_state[0])
            loss_sum += float(next_state[1])
            throughput_sum += float(next_state[2])
            failover_sum += float(next_state[5])

            if float(max(next_state[3], next_state[4])) > congestion_threshold:
                congestion_hits += 1

            if render:
                env.render()

        episode_rewards.append(reward_sum)
        episode_steps.append(steps)
        print(
            f"{policy_name} Episode {episode + 1}/{num_episodes} | "
            f"Reward: {reward_sum:.4f} | Steps: {steps}"
        )

    avg_components = {}
    if comp_steps:
        for k, v in comp_sums.items():
            avg_components[k] = v / comp_steps

    avg_reward = (sum(episode_rewards) / len(episode_rewards)) if episode_rewards else 0.0
    avg_steps = (sum(episode_steps) / len(episode_steps)) if episode_steps else 0.0
    min_reward = min(episode_rewards) if episode_rewards else 0.0
    max_reward = max(episode_rewards) if episode_rewards else 0.0

    action_success_rate = (success_steps / total_steps) if total_steps > 0 else 0.0
    action_switch_rate = (switch_count / switch_opportunities) if switch_opportunities > 0 else 0.0
    dominant_action_ratio = (max(action_counts.values()) / total_steps) if total_steps > 0 else 0.0

    avg_latency_proxy = (latency_sum / total_steps) if total_steps > 0 else 0.0
    avg_packet_loss_proxy = (loss_sum / total_steps) if total_steps > 0 else 0.0
    avg_throughput_proxy = (throughput_sum / total_steps) if total_steps > 0 else 0.0
    failover_active_rate = (failover_sum / total_steps) if total_steps > 0 else 0.0
    congestion_hit_rate = (congestion_hits / total_steps) if total_steps > 0 else 0.0

    contrib_keys = [
        "latency_penalty",
        "packet_loss_penalty",
        "utilization_penalty",
        "throughput_bonus",
        "congestion_penalty",
        "failover_penalty",
        "outcome_improvement_bonus",
        "action_repeat_penalty",
    ]
    abs_total = 0.0
    for key in contrib_keys:
        abs_total += abs(float(avg_components.get(key, 0.0)))

    reward_component_shares = {}
    for key in contrib_keys:
        val = abs(float(avg_components.get(key, 0.0)))
        reward_component_shares[key] = (val / abs_total) if abs_total > 0 else 0.0

    return {
        "policy": policy_name,
        "num_episodes": num_episodes,
        "average_reward": avg_reward,
        "min_reward": min_reward,
        "max_reward": max_reward,
        "average_steps": avg_steps,
        "action_success_rate": action_success_rate,
        "action_switch_rate": action_switch_rate,
        "dominant_action_ratio": dominant_action_ratio,
        "avg_latency_proxy": avg_latency_proxy,
        "avg_packet_loss_proxy": avg_packet_loss_proxy,
        "avg_throughput_proxy": avg_throughput_proxy,
        "failover_active_rate": failover_active_rate,
        "congestion_hit_rate": congestion_hit_rate,
        "action_counts": action_counts,
        "avg_reward_components": avg_components,
        "reward_component_shares": reward_component_shares,
        "episode_rewards": episode_rewards,
        "episode_steps": episode_steps,
    }


def main():
    args = parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = json.load(f)

    seed = int(args.seed) if args.seed is not None else int(config.get("system", {}).get("random_seed", 42))
    set_seed(seed)

    env = build_environment(config)

    if not wait_for_controller(env.client, max_wait_seconds=PREFLIGHT_WAIT_SECONDS):
        print(
            f"ABORT: Ryu controller unreachable at {env.client.base_url}.\n"
            "Start Mininet and ryu-manager, then retry. If a stale controller is\n"
            "still holding the port, clear it first: fuser -k 8080/tcp"
        )
        return 1

    run_setup = bool(config.get("evaluation", {}).get("run_startup_setup", False)) and not bool(args.skip_setup)
    if run_setup:
        summary = NetworkInitializer(config).initialize()
        print(f"Startup setup completed: {summary.as_dict()}")
    eval_cfg = config.get("evaluation", {})
    agent_cfg = config["agent"]
    nn_cfg = agent_cfg["neural_network"]
    hp = agent_cfg["hyperparameters"]
    rb_cfg = agent_cfg["replay_buffer"]

    env_state_dim = int(env.observation_space.shape[0])
    env_action_dim = int(env.action_space.n)
    if nn_cfg["input_dim"] != env_state_dim:
        raise ValueError(f"State dimension mismatch: config={nn_cfg['input_dim']} env={env_state_dim}")
    if nn_cfg["output_dim"] != env_action_dim:
        raise ValueError(f"Action dimension mismatch: config={nn_cfg['output_dim']} env={env_action_dim}")

    agent = DQNAgent(
        state_dim=nn_cfg["input_dim"],
        action_dim=nn_cfg["output_dim"],
        gamma=hp["gamma"],
        epsilon=0.0,
        epsilon_decay=1.0,
        epsilon_min=0.0,
        batch_size=hp["batch_size"],
        learning_rate=hp["learning_rate"],
        replay_buffer_capacity=rb_cfg["capacity"],
        device=config.get("system", {}).get("device", None),
    )

    model_path = args.model_path
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Trained model not found at {model_path}")

    state_dict = torch.load(model_path, map_location=agent.device)
    agent.q_network.load_state_dict(state_dict)
    agent.update_target_network()
    agent.q_network.eval()

    num_episodes = int(eval_cfg.get("num_episodes", 10))
    render = bool(eval_cfg.get("render", False))
    congestion_threshold = float(
        config.get("reward_function", {})
        .get("components", {})
        .get("congestion_threshold", {})
        .get("threshold", 0.9)
    )

    baseline_results = []
    try:
        dqn_metrics = run_policy(
            env=env,
            policy_name="dqn",
            num_episodes=num_episodes,
            action_fn=lambda s: agent.select_action(s),
            base_seed=seed,
            congestion_threshold=congestion_threshold,
            render=render,
        )

        baseline_cfg = eval_cfg.get("baseline_policies", [])
        for baseline in baseline_cfg:
            name = str(baseline.get("name", "")).lower()
            if name == "random":
                result = run_policy(
                    env=env,
                    policy_name="random",
                    num_episodes=num_episodes,
                    action_fn=lambda _s: env.action_space.sample(),
                    base_seed=seed,
                    congestion_threshold=congestion_threshold,
                    render=False,
                )
                baseline_results.append(result)
            elif name == "do_nothing":
                result = run_policy(
                    env=env,
                    policy_name="do_nothing",
                    num_episodes=num_episodes,
                    action_fn=lambda _s: 0,
                    base_seed=seed,
                    congestion_threshold=congestion_threshold,
                    render=False,
                )
                baseline_results.append(result)
    except RyuConnectionError as exc:
        # Evaluation is short and its numbers are only meaningful over a full run,
        # so bail out rather than reporting metrics from a half-dead network.
        print(f"\nABORT: controller lost during evaluation: {exc}")
        print(
            "Restart it, clearing any stale listener first:\n"
            "    fuser -k 8080/tcp && ss -lntp | grep 8080\n"
            "then re-run this evaluation."
        )
        return 1

    metrics = {
        "num_episodes": num_episodes,
        "model_path": model_path,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "dqn": dqn_metrics,
        "baselines": baseline_results,
    }

    metrics_path = args.metrics_path
    metrics_dir = os.path.dirname(metrics_path)
    if metrics_dir:
        os.makedirs(metrics_dir, exist_ok=True)
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("\nEvaluation Summary")
    print(
        f"DQN -> AvgReward: {dqn_metrics['average_reward']:.4f} | "
        f"Range: [{dqn_metrics['min_reward']:.4f}, {dqn_metrics['max_reward']:.4f}] | "
        f"AvgSteps: {dqn_metrics['average_steps']:.2f}"
    )
    print(f"DQN Action counts: {dqn_metrics['action_counts']}")
    print(
        f"DQN switch_rate={dqn_metrics['action_switch_rate']:.3f} | "
        f"dominant_action_ratio={dqn_metrics['dominant_action_ratio']:.3f} | "
        f"latency_proxy={dqn_metrics['avg_latency_proxy']:.4f} | "
        f"loss_proxy={dqn_metrics['avg_packet_loss_proxy']:.4f} | "
        f"throughput_proxy={dqn_metrics['avg_throughput_proxy']:.4f} | "
        f"failover_active_rate={dqn_metrics['failover_active_rate']:.3f} | "
        f"congestion_hit_rate={dqn_metrics['congestion_hit_rate']:.3f}"
    )
    if dqn_metrics["avg_reward_components"]:
        print(f"DQN Avg reward components: {dqn_metrics['avg_reward_components']}")

    for baseline in baseline_results:
        print(
            f"{baseline['policy']} -> AvgReward: {baseline['average_reward']:.4f} | "
            f"Range: [{baseline['min_reward']:.4f}, {baseline['max_reward']:.4f}] | "
            f"AvgSteps: {baseline['average_steps']:.2f}"
        )
        print(f"{baseline['policy']} Action counts: {baseline['action_counts']}")

    print(f"Saved metrics to {metrics_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
