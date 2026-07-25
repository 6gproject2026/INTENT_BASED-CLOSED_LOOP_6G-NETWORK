import argparse
import json
import os
import random
import sys

import numpy as np
import torch

from ai_layer.agent.dqn_agent import DQNAgent
from ai_layer.environments.sdn_env import SDNEnv
from ai_layer.network_interface.ryu_client import RyuConnectionError, wait_for_controller
from ai_layer.network_setup import NetworkInitializer

# Rich resume state, written every episode. Kept separate from the plain
# state_dict checkpoints (dqn_ep*.pth / --model-path) that evaluate.py loads.
RESUME_FILENAME = "train_state.pth"

# Startup reachability budget. Short on purpose: a controller that is down should
# fail here in seconds, not 60 episodes into a multi-hour run.
PREFLIGHT_WAIT_SECONDS = 10


def build_environment(config: dict):
    """Build the live SDN environment backed by Ryu telemetry/actions."""
    return SDNEnv(config)


def parse_args():
    parser = argparse.ArgumentParser(description="Train DQN agent for SDN control")
    parser.add_argument("--config", default="prod.json", help="Path to config JSON")
    parser.add_argument("--seed", type=int, default=None, help="Random seed override")
    parser.add_argument(
        "--skip-setup",
        action="store_true",
        help="Skip one-time startup setup before training",
    )
    parser.add_argument(
        "--model-path",
        default=os.path.join("models", "dqn_model.pth"),
        help="Output path for final trained model",
    )
    parser.add_argument(
        "--resume",
        default=None,
        help=f"Resume training from a saved state file (models/{RESUME_FILENAME})",
    )
    return parser.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def save_resume_state(path, agent, episode, global_step, reward_history, seed):
    """Persist everything needed to continue training from the next episode.

    Written every episode so an outage costs at most one episode of work.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(
        {
            "q_network": agent.q_network.state_dict(),
            "target_network": agent.target_network.state_dict(),
            "optimizer": agent.optimizer.state_dict(),
            "epsilon": agent.epsilon,
            "episode": episode,
            "global_step": global_step,
            "reward_history": reward_history,
            "seed": seed,
        },
        path,
    )


def load_resume_state(path, agent):
    """Restore agent + loop counters saved by save_resume_state().

    Returns (start_episode, global_step, reward_history, seed).
    The replay buffer is intentionally not persisted: it is large, and refilling it
    costs only the warmup steps.
    """
    state = torch.load(path, map_location=agent.device, weights_only=False)
    agent.q_network.load_state_dict(state["q_network"])
    agent.target_network.load_state_dict(state["target_network"])
    agent.optimizer.load_state_dict(state["optimizer"])
    agent.epsilon = float(state["epsilon"])

    start_episode = int(state["episode"]) + 1
    print(
        f"Resumed from {path}: continuing at episode {start_episode + 1} "
        f"with epsilon={agent.epsilon:.4f}"
    )
    return start_episode, int(state["global_step"]), list(state["reward_history"]), state["seed"]


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

    run_setup = bool(config.get("training", {}).get("run_startup_setup", False)) and not bool(args.skip_setup)
    if run_setup:
        summary = NetworkInitializer(config).initialize()
        print(f"Startup setup completed: {summary.as_dict()}")
    env_state_dim = int(env.observation_space.shape[0])
    env_action_dim = int(env.action_space.n)

    agent_cfg = config["agent"]
    hp = agent_cfg["hyperparameters"]
    nn_cfg = agent_cfg["neural_network"]
    rb_cfg = agent_cfg["replay_buffer"]

    if nn_cfg["input_dim"] != env_state_dim:
        raise ValueError(
            f"State dimension mismatch: config={nn_cfg['input_dim']} env={env_state_dim}"
        )
    if nn_cfg["output_dim"] != env_action_dim:
        raise ValueError(
            f"Action dimension mismatch: config={nn_cfg['output_dim']} env={env_action_dim}"
        )

    agent = DQNAgent(
        state_dim=nn_cfg["input_dim"],
        action_dim=nn_cfg["output_dim"],
        gamma=hp["gamma"],
        epsilon=hp["epsilon_start"],
        epsilon_decay=hp["epsilon_decay"],
        epsilon_min=hp.get("epsilon_end", 0.01),
        batch_size=hp["batch_size"],
        learning_rate=hp["learning_rate"],
        replay_buffer_capacity=rb_cfg["capacity"],
        device=config.get("system", {}).get("device", None),
    )

    num_episodes = config["training"]["num_episodes"]
    warmup_steps = int(config["training"].get("warmup_steps", 0))
    min_buffer_size = int(rb_cfg.get("min_size_for_training", 0))
    train_start_size = max(warmup_steps, min_buffer_size)
    save_frequency = int(config["training"].get("save_frequency", 50))
    target_update_frequency = hp["target_update_frequency"]

    global_step = 0
    reward_history = []
    start_episode = 0

    os.makedirs("models", exist_ok=True)
    resume_path = os.path.join("models", RESUME_FILENAME)

    if args.resume:
        start_episode, global_step, reward_history, seed = load_resume_state(args.resume, agent)
        seed = int(seed)

    for episode in range(start_episode, num_episodes):
        try:
            reset_out = env.reset(seed=seed + episode)
            state = reset_out[0] if isinstance(reset_out, tuple) else reset_out

            done = False
            episode_reward = 0.0
            episode_losses = []
            comp_sums = {}
            comp_steps = 0

            while not done:
                action = agent.select_action(state)

                step_out = env.step(action)
                if len(step_out) == 5:
                    next_state, reward, terminated, truncated, info = step_out
                    done = bool(terminated or truncated)
                    components = info.get("reward_components", {}) if isinstance(info, dict) else {}
                    for k, v in components.items():
                        if isinstance(v, (int, float)):
                            comp_sums[k] = comp_sums.get(k, 0.0) + float(v)
                    if components:
                        comp_steps += 1
                else:
                    next_state, reward, done = step_out
                    done = bool(done)

                agent.store_transition(state, action, reward, next_state, done)
                if len(agent.replay_buffer) >= train_start_size:
                    loss = agent.train_step()
                    if loss is not None:
                        episode_losses.append(loss)

                state = next_state
                episode_reward += float(reward)
                global_step += 1

                if global_step % target_update_frequency == 0:
                    agent.update_target_network()

                # Decay exploration per environment step so epsilon anneals smoothly
                # across the whole training budget rather than once per episode.
                agent.decay_epsilon()

        except RyuConnectionError as exc:
            # The controller died mid-episode. Save first, then try to ride it out:
            # episode-1 means a resume restarts this episode from the top.
            save_resume_state(resume_path, agent, episode - 1, global_step, reward_history, seed)
            print(f"\nController lost during episode {episode + 1}: {exc}")
            print(f"State saved to {resume_path}. Waiting for the controller to come back...")

            if not wait_for_controller(env.client):
                print(
                    f"\nABORT: controller still down at {env.client.base_url}.\n"
                    "Restart it, clearing any stale listener first:\n"
                    "    fuser -k 8080/tcp && ss -lntp | grep 8080\n"
                    "then resume with:\n"
                    f"    python train.py --config {args.config} --resume {resume_path}"
                )
                return 1

            # A Ryu restart wipes the routing/QoS baseline, so reinstall it before
            # continuing or training would run against an unconfigured network.
            if run_setup:
                summary = NetworkInitializer(config).initialize()
                print(f"Controller back. Startup setup re-applied: {summary.as_dict()}")
            else:
                print("Controller back. Continuing (startup setup disabled).")

            # The interrupted episode spans a network-state discontinuity, so it is
            # discarded rather than stitched together across the outage.
            print(f"Discarding interrupted episode {episode + 1}.")
            continue

        reward_history.append(episode_reward)

        avg_reward = sum(reward_history[-20:]) / min(len(reward_history), 20)
        avg_loss = (sum(episode_losses) / len(episode_losses)) if episode_losses else 0.0
        avg_lat_pen = (comp_sums.get("latency_penalty", 0.0) / comp_steps) if comp_steps else 0.0
        avg_loss_pen = (comp_sums.get("packet_loss_penalty", 0.0) / comp_steps) if comp_steps else 0.0
        avg_cong_pen = (comp_sums.get("congestion_penalty", 0.0) / comp_steps) if comp_steps else 0.0
        avg_repeat_pen = (comp_sums.get("action_repeat_penalty", 0.0) / comp_steps) if comp_steps else 0.0
        avg_outcome_bonus = (comp_sums.get("outcome_improvement_bonus", 0.0) / comp_steps) if comp_steps else 0.0

        print(
            f"Episode {episode + 1}/{num_episodes} | "
            f"Reward: {episode_reward:.4f} | "
            f"AvgReward(20): {avg_reward:.4f} | "
            f"AvgLoss: {avg_loss:.6f} | "
            f"LatPen: {avg_lat_pen:.4f} | "
            f"LossPen: {avg_loss_pen:.4f} | "
            f"CongPen: {avg_cong_pen:.4f} | "
            f"RepeatPen: {avg_repeat_pen:.4f} | "
            f"OutcomeBonus: {avg_outcome_bonus:.4f} | "
            f"Epsilon: {agent.epsilon:.4f}"
        )

        # Written every episode so an outage costs at most one episode of work.
        save_resume_state(resume_path, agent, episode, global_step, reward_history, seed)

        if (episode + 1) % save_frequency == 0:
            ckpt_path = os.path.join("models", f"dqn_ep{episode + 1}.pth")
            torch.save(agent.q_network.state_dict(), ckpt_path)
            print(f"Checkpoint saved to {ckpt_path}")

    model_path = args.model_path
    model_dir = os.path.dirname(model_path)
    if model_dir:
        os.makedirs(model_dir, exist_ok=True)
    torch.save(agent.q_network.state_dict(), model_path)
    print(f"Saved trained model to {model_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
