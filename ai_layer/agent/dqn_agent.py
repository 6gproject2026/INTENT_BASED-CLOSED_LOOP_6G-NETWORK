import random
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from ai_layer.agent.replay_buffer import ReplayBuffer
from ai_layer.models.q_network import QNetwork


class DQNAgent:
	"""DQN agent with target network and experience replay."""

	def __init__(
		self,
		state_dim: int,
		action_dim: int,
		gamma: float = 0.99,
		epsilon: float = 1.0,
		epsilon_decay: float = 0.995,
		epsilon_min: float = 0.01,
		batch_size: int = 64,
		learning_rate: float = 1e-3,
		replay_buffer_capacity: int = 10000,
		device: Optional[str] = None,
	):
		self.state_dim = int(state_dim)
		self.action_dim = int(action_dim)

		self.gamma = float(gamma)
		self.epsilon = float(epsilon)
		self.epsilon_decay = float(epsilon_decay)
		self.epsilon_min = float(epsilon_min)
		self.batch_size = int(batch_size)
		self.learning_rate = float(learning_rate)

		self.device = torch.device(
			device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
		)

		self.q_network = QNetwork(self.state_dim, self.action_dim).to(self.device)
		self.target_network = QNetwork(self.state_dim, self.action_dim).to(self.device)
		self.update_target_network()

		self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.learning_rate)
		self.loss_fn = nn.MSELoss()
		self.replay_buffer = ReplayBuffer(replay_buffer_capacity)

	def select_action(self, state) -> int:
		"""Select action using epsilon-greedy policy."""
		if random.random() < self.epsilon:
			return random.randrange(self.action_dim)

		state_tensor = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
		with torch.no_grad():
			q_values = self.q_network(state_tensor)
		return int(torch.argmax(q_values, dim=1).item())

	def store_transition(self, state, action, reward, next_state, done) -> None:
		"""Store one transition into replay memory."""
		self.replay_buffer.add(state, action, reward, next_state, done)

	def train_step(self) -> Optional[float]:
		"""Run one DQN update step from a sampled minibatch.

		Returns:
			Loss value when an update is performed, else None.
		"""
		if len(self.replay_buffer) < self.batch_size:
			return None

		states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)

		states_t = torch.as_tensor(states, dtype=torch.float32, device=self.device)
		actions_t = torch.as_tensor(actions, dtype=torch.int64, device=self.device).unsqueeze(1)
		rewards_t = torch.as_tensor(rewards, dtype=torch.float32, device=self.device)
		next_states_t = torch.as_tensor(next_states, dtype=torch.float32, device=self.device)
		dones_t = torch.as_tensor(dones, dtype=torch.float32, device=self.device)

		current_q = self.q_network(states_t).gather(1, actions_t).squeeze(1)

		with torch.no_grad():
			next_q = self.target_network(next_states_t).max(dim=1).values
			target_q = rewards_t + (1.0 - dones_t) * self.gamma * next_q

		loss = self.loss_fn(current_q, target_q)

		self.optimizer.zero_grad()
		loss.backward()
		torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
		self.optimizer.step()

		return float(loss.item())

	def update_target_network(self) -> None:
		"""Hard-update target network parameters from the online network."""
		self.target_network.load_state_dict(self.q_network.state_dict())

	def decay_epsilon(self) -> None:
		"""Decay epsilon one step toward epsilon_min.

		train.py calls this once per environment *step*, not per episode, so
		epsilon_decay is coupled to num_episodes * max_steps: reaching epsilon_min
		takes log(epsilon_min) / log(epsilon_decay) steps. Changing the episode
		budget without retuning epsilon_decay leaves the agent stuck exploring.
		"""
		self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
