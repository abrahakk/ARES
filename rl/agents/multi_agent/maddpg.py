from __future__ import annotations

import copy
from typing import Dict, List

import torch
import torch.nn as nn
import torch.optim as optim

from rl.utils.normalization import RunningNormalizer
from rl.utils.replay_buffer import ReplayBuffer

ACTION_DIM = 3


class _DeterministicActor(nn.Module):
    def __init__(self, observation_dim: int, hidden_dim: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(observation_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, ACTION_DIM), nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _JointCritic(nn.Module):
    def __init__(self, observation_dim: int, num_agents: int, hidden_dim: int = 256) -> None:
        super().__init__()
        input_dim = num_agents * (observation_dim + ACTION_DIM)
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, joint_obs: torch.Tensor, joint_actions: torch.Tensor) -> torch.Tensor:

        flat = torch.cat(
            [joint_obs.flatten(start_dim=1), joint_actions.flatten(start_dim=1)], dim=-1
        )
        return self.net(flat).squeeze(-1)


class MADDPG:
    def __init__(
        self,
        robot_ids: List[str],
        observation_dim: int,
        hidden_dim: int = 256,
        actor_lr: float = 1e-3,
        critic_lr: float = 1e-3,
        gamma: float = 0.99,
        tau: float = 0.01,
        exploration_noise_std: float = 0.1,
        batch_size: int = 256,
        device: str = "cpu",
    ) -> None:
        self.robot_ids = robot_ids
        self.num_agents = len(robot_ids)
        self.observation_dim = observation_dim
        self.device = device
        self.gamma = gamma
        self.tau = tau
        self.exploration_noise_std = exploration_noise_std
        self.batch_size = batch_size

        self.actors = {rid: _DeterministicActor(observation_dim, hidden_dim).to(device) for rid in robot_ids}
        self.target_actors = {rid: copy.deepcopy(actor) for rid, actor in self.actors.items()}
        self.actor_optimizers = {
            rid: optim.Adam(actor.parameters(), lr=actor_lr) for rid, actor in self.actors.items()
        }


        self.critics = {
            rid: _JointCritic(observation_dim, self.num_agents, hidden_dim).to(device) for rid in robot_ids
        }
        self.target_critics = {rid: copy.deepcopy(critic) for rid, critic in self.critics.items()}
        self.critic_optimizers = {
            rid: optim.Adam(critic.parameters(), lr=critic_lr) for rid, critic in self.critics.items()
        }

        self.normalizer = RunningNormalizer(shape=(observation_dim,))
        self.buffer = ReplayBuffer()

    def act(self, observations: Dict[str, "np.ndarray"], deterministic: bool = False) -> Dict[str, "np.ndarray"]:
        actions = {}
        for rid, obs in observations.items():
            normalized = self.normalizer.normalize(obs.reshape(1, -1))
            obs_tensor = torch.as_tensor(normalized, dtype=torch.float32, device=self.device)
            with torch.no_grad():
                action = self.actors[rid](obs_tensor).squeeze(0)
            if not deterministic:
                action = action + torch.randn_like(action) * self.exploration_noise_std
                action = torch.clamp(action, -1.0, 1.0)
            actions[rid] = action.cpu().numpy()
        return actions

    def store_transition(self, joint_obs, joint_actions, joint_rewards, joint_next_obs, joint_done) -> None:

        self.buffer.add(joint_obs, joint_actions, joint_rewards, joint_next_obs, joint_done)

    def update(self) -> Dict[str, float]:
        if len(self.buffer) < self.batch_size:
            return {}

        batch = self.buffer.sample(self.batch_size, device=self.device)
        joint_obs = batch["observations"]     
        joint_actions = batch["actions"]      
        joint_next_obs = batch["next_observations"]
        rewards = batch["rewards"]            
        dones = batch["dones"]                

        losses = {}

        with torch.no_grad():
            next_actions = torch.stack(
                [self.target_actors[rid](joint_next_obs[:, i, :]) for i, rid in enumerate(self.robot_ids)],
                dim=1,
            )

        for i, rid in enumerate(self.robot_ids):
            with torch.no_grad():
                target_q = self.target_critics[rid](joint_next_obs, next_actions)
                y = rewards[:, i] + self.gamma * (1.0 - dones[:, i]) * target_q

            current_q = self.critics[rid](joint_obs, joint_actions)
            critic_loss = nn.functional.mse_loss(current_q, y)

            self.critic_optimizers[rid].zero_grad()
            critic_loss.backward()
            self.critic_optimizers[rid].step()

            actions_for_actor_loss = joint_actions.clone()
            actions_for_actor_loss[:, i, :] = self.actors[rid](joint_obs[:, i, :])
            actor_loss = -self.critics[rid](joint_obs, actions_for_actor_loss).mean()

            self.actor_optimizers[rid].zero_grad()
            actor_loss.backward()
            self.actor_optimizers[rid].step()

            self._soft_update(self.actors[rid], self.target_actors[rid])
            self._soft_update(self.critics[rid], self.target_critics[rid])

            losses[f"{rid}_critic_loss"] = critic_loss.item()
            losses[f"{rid}_actor_loss"] = actor_loss.item()

        return losses

    def _soft_update(self, source: nn.Module, target: nn.Module) -> None:
        for src_param, tgt_param in zip(source.parameters(), target.parameters()):
            tgt_param.data.copy_(self.tau * src_param.data + (1.0 - self.tau) * tgt_param.data)
