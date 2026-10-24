from __future__ import annotations

import tempfile
import textwrap

from rl.environments.reward import RewardConfig, aggregate_episode_breakdown, compute_reward


def test_default_reward_config_values():
    config = RewardConfig()
    assert config.w_coverage == 1.0
    assert config.w_collision == 2.0


def test_reward_config_from_yaml_matches_repo_defaults():
    config = RewardConfig.from_yaml("config/rewards.yaml")
    assert config.w_coverage == 1.0
    assert config.w_information == 0.5
    assert config.w_cooperation == 0.3
    assert config.w_collision == 2.0
    assert config.w_energy == 0.05
    assert config.w_communication == 0.02


def test_reward_config_ablated_zeroes_only_the_named_term():
    config = RewardConfig()
    ablated = config.ablated("cooperation")
    assert ablated.w_cooperation == 0.0
    assert ablated.w_coverage == config.w_coverage
    assert ablated.w_collision == config.w_collision
    assert config.w_cooperation == 0.3


def test_reward_config_ablated_rejects_unknown_term():
    config = RewardConfig()
    try:
        config.ablated("not_a_real_term")
        assert False, "expected ValueError for unknown reward term"
    except ValueError:
        pass


def test_compute_reward_combines_all_six_terms():
    config = RewardConfig(
        w_coverage=1.0, w_information=1.0, w_cooperation=1.0,
        w_collision=1.0, w_energy=1.0, w_communication=1.0,
    )
    total, breakdown = compute_reward(
        newly_covered_area_m2=2.0,
        joint_entropy_reduction=1.5,
        cooperation_event=True,
        collided=True,
        distance_traveled_m=0.5,
        message_sent=True,
        config=config,
    )
    assert abs(total - 2.0) < 1e-9
    assert breakdown["r_coverage"] == 2.0
    assert breakdown["r_collision"] == 1.0
    assert breakdown["total"] == total


def test_compute_reward_ablated_term_has_no_effect():

    config = RewardConfig(
        w_coverage=1.0, w_information=1.0, w_cooperation=1.0,
        w_collision=1.0, w_energy=1.0, w_communication=1.0,
    )
    ablated_config = config.ablated("communication")

    kwargs = dict(
        newly_covered_area_m2=1.0, joint_entropy_reduction=0.0,
        cooperation_event=False, collided=False,
        distance_traveled_m=0.0,
    )
    total_with_comm, _ = compute_reward(**kwargs, message_sent=True, config=config)
    total_without_comm_penalty, _ = compute_reward(**kwargs, message_sent=True, config=ablated_config)

    assert total_without_comm_penalty > total_with_comm


def test_aggregate_episode_breakdown_sums_components():
    steps = [
        {"total": 1.0, "r_coverage": 1.0, "r_collision": 0.0},
        {"total": 0.5, "r_coverage": 0.5, "r_collision": 1.0},
    ]
    totals = aggregate_episode_breakdown(steps)
    assert totals["total"] == 1.5
    assert totals["r_coverage"] == 1.5
    assert totals["r_collision"] == 1.0


def test_aggregate_episode_breakdown_handles_empty_list():
    assert aggregate_episode_breakdown([]) == {}
