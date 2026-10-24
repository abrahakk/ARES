from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn as nn
import torch.optim as optim

from rl.networks.actor import ActorNetwork
from rl.networks.centralized_critic import CentralizedCritic
from rl.utils.normalization import RunningNormalizer
from rl.environments.action_space import ACTION_DIM


class MAPPO:
    def __init__(
        self,
        robot_ids: List[str],
        observation_dim: int,
        hidden_dim: int = 256,
        actor_lr: float = 3e-4,
        critic_lr: float = 1e-3,
        clip_epsilon: float = 0.2,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        num_epochs: int = 4,
        device: str = "cpu",
    ) -> None:
        self.robot_ids = robot_ids
        self.device = device
        self.clip_epsilon = clip_epsilon
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.num_epochs = num_epochs

        self.actor = ActorNetwork(observation_dim, hidden_dim).to(device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)

        self.critic = CentralizedCritic(observation_dim, ACTION_DIM, hidden_dim).to(device)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)

        self.normalizer = RunningNormalizer(shape=(observation_dim,))

        from rl.utils.replay_buffer import RolloutBuffer
        self.buffers: Dict[str, RolloutBuffer] = {rid: RolloutBuffer() for rid in robot_ids}

    def act(self, observations: Dict[str, "np.ndarray"], deterministic: bool = False) -> Dict[str, dict]:
        
       import numpy as np

        results = {}
        for rid, obs in observations.items():
            normalized = self.normalizer.normalize(obs.reshape(1, -1))
            obs_tensor = torch.as_tensor(normalized, dtype=torch.float32, device=self.device)
            with torch.no_grad():
                action, log_prob, _local_value = self.actor.act(obs_tensor, deterministic=deterministic)
            results[rid] = {
                "action": action.squeeze(0).cpu().numpy(),
                "log_prob": log_prob.item(),
            }
        return results

    def compute_centralized_value(self, joint_observations, joint_actions) -> float:

        obs_tensor = torch.as_tensor(joint_observations, dtype=torch.float32, device=self.device).unsqueeze(0)
        act_tensor = torch.as_tensor(joint_actions, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            value = self.critic(obs_tensor, act_tensor)
        return value.item()

    def store_transition(self, robot_id: str, observation, action, log_prob, value, reward, done) -> None:
        self.normalizer.update(observation.reshape(1, -1))
        self.buffers[robot_id].add(observation, action, log_prob, value, reward, done)



    def update(self, last_values: Dict[str, float]) -> Dict[str, float]:
        all_obs, all_actions, all_old_log_probs, all_returns, all_advantages = [], [], [], [], []

        for rid in self.robot_ids:
            buffer = self.buffers[rid]
            if len(buffer) == 0:
                continue
            batch = buffer.as_tensors(last_values.get(rid, 0.0), device=self.device)
            all_obs.append(batch["observations"])
            all_actions.append(batch["actions"])
            all_old_log_probs.append(batch["old_log_probs"])
            all_returns.append(batch["returns"])
            all_advantages.append(batch["advantages"])

        observations = torch.cat(all_obs, dim=0)
        actions = torch.cat(all_actions, dim=0)
        old_log_probs = torch.cat(all_old_log_probs, dim=0)
        returns = torch.cat(all_returns, dim=0)
        advantages = torch.cat(all_advantages, dim=0)

        n = observations.shape[0]
        total_policy_loss = total_entropy = 0.0
        num_updates = 0

        for _epoch in range(self.num_epochs):
            indices = torch.randperm(n)
            for start in range(0, n, 64):
                mb_idx = indices[start:start + 64]

                log_probs, entropy, _values = self.actor.evaluate_actions(
                    observations[mb_idx], actions[mb_idx]
                )
                ratio = torch.exp(log_probs - old_log_probs[mb_idx])
                surr1 = ratio * advantages[mb_idx]
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages[mb_idx]
                policy_loss = -torch.min(surr1, surr2).mean() - self.entropy_coef * entropy.mean()

                self.actor_optimizer.zero_grad()
                policy_loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                self.actor_optimizer.step()

                total_policy_loss += policy_loss.item()
                total_entropy += entropy.mean().item()
                num_updates += 1

        for rid in self.robot_ids:
            self.buffers[rid].clear()

        return {
            "policy_loss": total_policy_loss / max(num_updates, 1),
            "entropy": total_entropy / max(num_updates, 1),
        }

    def state_dict(self) -> dict:
        return {
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "normalizer": self.normalizer.state_dict(),
        }

    def load_state_dict(self, state: dict) -> None:
        self.actor.load_state_dict(state["actor"])
        self.critic.load_state_dict(state["critic"])
        self.normalizer.load_state_dict(state["normalizer"])
