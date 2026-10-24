
from __future__ import annotations

import copy
from typing import Dict, List

import torch
import torch.nn as nn
import torch.optim as optim

from rl.agents.single_agent.dqn import ACTION_TABLE, NUM_DISCRETE_ACTIONS
from rl.utils.normalization import RunningNormalizer
from rl.utils.replay_buffer import ReplayBuffer


class _AgentQNetwork(nn.Module):
    def __init__(self, observation_dim: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(observation_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, NUM_DISCRETE_ACTIONS),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _MixingNetwork(nn.Module):


    def __init__(self, num_agents: int, state_dim: int, mixing_hidden_dim: int = 32) -> None:
        super().__init__()
        self.num_agents = num_agents
        self.mixing_hidden_dim = mixing_hidden_dim

        self.hyper_w1 = nn.Linear(state_dim, num_agents * mixing_hidden_dim)
        self.hyper_w2 = nn.Linear(state_dim, mixing_hidden_dim)
        self.hyper_b1 = nn.Linear(state_dim, mixing_hidden_dim)
        self.hyper_b2 = nn.Sequential(nn.Linear(state_dim, mixing_hidden_dim), nn.ReLU(), nn.Linear(mixing_hidden_dim, 1))

    def forward(self, agent_qs: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        batch_size = agent_qs.shape[0]

        w1 = torch.abs(self.hyper_w1(state)).view(batch_size, self.num_agents, self.mixing_hidden_dim)
        b1 = self.hyper_b1(state).view(batch_size, 1, self.mixing_hidden_dim)
        hidden = torch.relu(torch.bmm(agent_qs.unsqueeze(1), w1) + b1) 

        w2 = torch.abs(self.hyper_w2(state)).view(batch_size, self.mixing_hidden_dim, 1)
        b2 = self.hyper_b2(state).view(batch_size, 1, 1)
        team_q = torch.bmm(hidden, w2) + b2  

        return team_q.view(batch_size)


class QMIX:
    def __init__(
        self,
        robot_ids: List[str],
        observation_dim: int,
        learning_rate: float = 5e-4,
        gamma: float = 0.99,
        target_update_interval: int = 200,
        batch_size: int = 128,
        device: str = "cpu",
    ) -> None:
        self.robot_ids = robot_ids
        self.num_agents = len(robot_ids)
        self.device = device
        self.gamma = gamma
        self.target_update_interval = target_update_interval
        self.batch_size = batch_size

        
       self.agent_q = _AgentQNetwork(observation_dim).to(device)
        self.target_agent_q = copy.deepcopy(self.agent_q)

        joint_state_dim = self.num_agents * observation_dim
        self.mixer = _MixingNetwork(self.num_agents, joint_state_dim).to(device)
        self.target_mixer = copy.deepcopy(self.mixer)

        self.optimizer = optim.Adam(
            list(self.agent_q.parameters()) + list(self.mixer.parameters()), lr=learning_rate
        )

        self.normalizer = RunningNormalizer(shape=(observation_dim,))
        self.buffer = ReplayBuffer()
        self._train_steps = 0

    def act(self, observations: Dict[str, "np.ndarray"], epsilon: float = 0.05) -> Dict[str, "np.ndarray"]:
        import random
        actions = {}
        for rid, obs in observations.items():
            normalized = self.normalizer.normalize(obs.reshape(1, -1))
            if random.random() < epsilon:
                idx = random.randrange(NUM_DISCRETE_ACTIONS)
            else:
                obs_tensor = torch.as_tensor(normalized, dtype=torch.float32, device=self.device)
                with torch.no_grad():
                    q_values = self.agent_q(obs_tensor)
                idx = int(torch.argmax(q_values, dim=-1).item())
            actions[rid] = ACTION_TABLE[idx]
        return actions

    def store_transition(self, joint_obs, joint_action_indices, team_reward, joint_next_obs, done) -> None:
        self.buffer.add(joint_obs, joint_action_indices, team_reward, joint_next_obs, done)

    def update(self) -> float:
        if len(self.buffer) < self.batch_size:
            return 0.0

        batch = self.buffer.sample(self.batch_size, device=self.device)
        joint_obs = batch["observations"]        
        action_indices = batch["actions"].long()
        team_reward = batch["rewards"]          
        joint_next_obs = batch["next_observations"]
        done = batch["dones"]                      

        batch_size = joint_obs.shape[0]

        agent_qs = self.agent_q(joint_obs.view(-1, joint_obs.shape[-1])).view(
            batch_size, self.num_agents, NUM_DISCRETE_ACTIONS
        )
        chosen_qs = agent_qs.gather(-1, action_indices.unsqueeze(-1)).squeeze(-1)  

        state = joint_obs.view(batch_size, -1)
        team_q = self.mixer(chosen_qs, state)

        with torch.no_grad():
            next_agent_qs = self.target_agent_q(joint_next_obs.view(-1, joint_next_obs.shape[-1])).view(
                batch_size, self.num_agents, NUM_DISCRETE_ACTIONS
            )
            next_best_qs = next_agent_qs.max(dim=-1).values
            next_state = joint_next_obs.view(batch_size, -1)
            next_team_q = self.target_mixer(next_best_qs, next_state)
            target = team_reward + self.gamma * (1.0 - done) * next_team_q

        loss = nn.functional.mse_loss(team_q, target)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(
            list(self.agent_q.parameters()) + list(self.mixer.parameters()), 10.0
        )
        self.optimizer.step()

        self._train_steps += 1
        if self._train_steps % self.target_update_interval == 0:
            self.target_agent_q.load_state_dict(self.agent_q.state_dict())
            self.target_mixer.load_state_dict(self.mixer.state_dict())

        return loss.item()
