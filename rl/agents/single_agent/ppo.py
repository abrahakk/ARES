from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
import torch.optim as optim

from rl.networks.actor import ActorNetwork
from rl.utils.replay_buffer import RolloutBuffer
from rl.utils.normalization import RunningNormalizer


class SingleAgentPPO:
    def __init__(
        self,
        observation_dim: int,
        hidden_dim: int = 256,
        learning_rate: float = 3e-4,
        clip_epsilon: float = 0.2,
        value_loss_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        num_epochs: int = 4,
        minibatch_size: int = 64,
        device: str = "cpu",
    ) -> None:
        self.device = device
        self.clip_epsilon = clip_epsilon
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.num_epochs = num_epochs
        self.minibatch_size = minibatch_size

        self.actor = ActorNetwork(observation_dim, hidden_dim).to(device)
        self.optimizer = optim.Adam(self.actor.parameters(), lr=learning_rate)
        self.normalizer = RunningNormalizer(shape=(observation_dim,))
        self.buffer = RolloutBuffer()

    def act(self, observation, deterministic: bool = False):
        normalized = self.normalizer.normalize(observation.reshape(1, -1))
        obs_tensor = torch.as_tensor(normalized, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            action, log_prob, value = self.actor.act(obs_tensor, deterministic=deterministic)
        return action.squeeze(0).cpu().numpy(), log_prob.item(), value.item()

    def store_transition(self, observation, action, log_prob, value, reward, done) -> None:
        self.normalizer.update(observation.reshape(1, -1))
        self.buffer.add(observation, action, log_prob, value, reward, done)

    def update(self, last_value: float) -> Dict[str, float]:
        batch = self.buffer.as_tensors(last_value, device=self.device)
        n = batch["observations"].shape[0]

        total_policy_loss = total_value_loss = total_entropy = 0.0
        num_updates = 0

        for _epoch in range(self.num_epochs):
            indices = torch.randperm(n)
            for start in range(0, n, self.minibatch_size):
                mb_idx = indices[start:start + self.minibatch_size]

                log_probs, entropy, values = self.actor.evaluate_actions(
                    batch["observations"][mb_idx], batch["actions"][mb_idx]
                )
                ratio = torch.exp(log_probs - batch["old_log_probs"][mb_idx])
                advantages = batch["advantages"][mb_idx]

                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = nn.functional.mse_loss(values, batch["returns"][mb_idx])
                entropy_bonus = entropy.mean()

                loss = policy_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy_bonus

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                self.optimizer.step()

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy_bonus.item()
                num_updates += 1

        self.buffer.clear()
        return {
            "policy_loss": total_policy_loss / num_updates,
            "value_loss": total_value_loss / num_updates,
            "entropy": total_entropy / num_updates,
        }

    def state_dict(self) -> dict:
        return {
            "actor": self.actor.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "normalizer": self.normalizer.state_dict(),
        }

    def load_state_dict(self, state: dict) -> None:
        self.actor.load_state_dict(state["actor"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.normalizer.load_state_dict(state["normalizer"])
