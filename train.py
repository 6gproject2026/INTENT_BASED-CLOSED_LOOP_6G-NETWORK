import argparse
import json
import os
import random

import numpy as np
import torch

from ai_layer.agent.dqn_agent import DQNAgent
from ai_layer.environments.sdn_env import SDNEnv
from ai_layer.network_setup import NetworkInitializer


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
    return parser.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def main():
    args = parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = json.load(f)

    run_setup = bool(config.get("training", {}).get("run_startup_setup", False)) and not bool(args.skip_setup)
    if run_setup:
        summary = NetworkInitializer(config).initialize()
        print(f"Startup setup completed: {summary.as_dict()}")

    seed = int(args.seed) if args.seed is not None else int(config.get("system", {}).get("random_seed", 42))
    set_seed(seed)

    env = build_environment(config)
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

    os.makedirs("models", exist_ok=True)

    for episode in range(num_episodes):
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

        if (episode + 1) % save_frequency == 0:
            ckpt_path = os.path.join("models", f"dqn_ep{episode + 1}.pth")
            torch.save(agent.q_network.state_dict(), ckpt_path)
            print(f"Checkpoint saved to {ckpt_path}")

    model_path = os.path.join("models", "dqn_model.pth")
    model_path = args.model_path
    model_dir = os.path.dirname(model_path)
    if model_dir:
        os.makedirs(model_dir, exist_ok=True)
    torch.save(agent.q_network.state_dict(), model_path)
    print(f"Saved trained model to {model_path}")


if __name__ == "__main__":
    main()
