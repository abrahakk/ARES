from __future__ import annotations

import logging
import random
import uuid
from typing import Callable, List, Optional

import numpy as np

from data.evaluation_writer import EvaluationWriter
from data.schemas import EpisodeStep, EvaluationRun, TrajectoryRecord
from data.trajectory_writer import TrajectoryWriter
from evaluation.metrics import (
    coverage_fraction,
    collision_rate,
    communication_overhead,
    energy_consumption,
    mean_episode_reward,
    mission_completion,
    path_efficiency,
    reward_breakdown_totals,
)
from rl.environments.ares_env import AresEnv
from simulation.scenarios.exploration import ExplorationScenario

logger = logging.getLogger(__name__)

ActFunction = Callable[[np.ndarray], np.ndarray]


class PolicyEvaluator:
    def __init__(
        self,
        scenario: ExplorationScenario,
        experiment_name: str,
        condition: str,
    ) -> None:
        self.scenario = scenario
        self.experiment_name = experiment_name
        self.condition = condition

        self.trajectory_writer = TrajectoryWriter(experiment_name, condition)
        self.evaluation_writer = EvaluationWriter(experiment_name)

    def run(self, act_fns: dict, seeds: List[int], deterministic: bool = True) -> List[EvaluationRun]:

        results: List[EvaluationRun] = []
        for seed in seeds:
            record = self._run_single_episode(act_fns, seed, deterministic)
            self.trajectory_writer.write(record)

            evaluation_run = EvaluationRun(
                experiment_name=self.experiment_name,
                condition=self.condition,
                seed=seed,
                environment_name=self.scenario.environment.name,
                coverage=coverage_fraction(record),
                mission_completed=mission_completion(record, self.scenario.coverage_success_threshold),
                completion_time_steps=len(record.steps) if record.outcome == "success" else None,
                collision_count=int(collision_rate(record) * len({s.robot_id for s in record.steps} or {1})),
                path_efficiency=path_efficiency(record),
                energy_consumption=energy_consumption(record),
                communication_messages=communication_overhead(record),
                mean_episode_reward=mean_episode_reward(record),
                reward_breakdown=reward_breakdown_totals(record),
            )
            self.evaluation_writer.write(evaluation_run)
            results.append(evaluation_run)

            logger.info(
                "seed=%d condition=%s coverage=%.3f completed=%s",
                seed, self.condition, evaluation_run.coverage, evaluation_run.mission_completed,
            )

        return results

    def _run_single_episode(self, act_fns: dict, seed: int, deterministic: bool) -> TrajectoryRecord:
        random.seed(seed)
        np.random.seed(seed)

        env = AresEnv(**self.scenario.build_env_kwargs())
        observations = env.reset()

        steps: List[EpisodeStep] = []
        outcome = "timeout"
        step_index = 0

        while observations:
            actions = {}
            for robot_id, obs in observations.items():
                act_fn = act_fns.get(robot_id)
                if act_fn is None:
                    logger.warning("no act_fn for robot_id=%s, using zero action", robot_id)
                    actions[robot_id] = np.zeros(3, dtype=np.float32)
                else:
                    actions[robot_id] = act_fn(obs)

            observations, rewards, dones, infos = env.step(actions)

            for robot_id in actions:
                info = infos.get(robot_id, {})
                steps.append(
                    EpisodeStep(
                        step=step_index,
                        robot_id=robot_id,
                        pose=list(env.robot_manager.robots[robot_id].state.pose.as_tuple()),
                        action=list(actions[robot_id]),
                        reward=float(rewards.get(robot_id, 0.0)),
                        reward_breakdown={k: v for k, v in info.items() if k.startswith("r_") or k == "total"},
                        coverage_fraction=info.get("coverage_fraction", 0.0),
                        messages_sent=int(actions[robot_id][2] > 0.5) if len(actions[robot_id]) > 2 else 0,
                        messages_received=0,
                        collided=env.robot_manager.robots[robot_id].state.collision_count > 0,
                    )
                )

            step_index += 1
            if dones and all(dones.values()):
                if env.coverage_map.coverage_fraction() >= self.scenario.coverage_success_threshold:
                    outcome = "success"
                elif not observations:
                    outcome = "robot_failure"
                break

        return TrajectoryRecord(
            episode_id=uuid.uuid4().hex[:12],
            experiment_name=self.experiment_name,
            condition=self.condition,
            seed=seed,
            environment_name=self.scenario.environment.name,
            steps=steps,
            outcome=outcome,
        )
