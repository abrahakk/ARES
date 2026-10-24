from __future__ import annotations

from simulation.environments.dynamic import DYNAMIC_ENV_CONFIG, communication_sweep_configs
from simulation.environments.randomized import RandomizationConfig, apply_domain_randomization
from simulation.environments.static import STATIC_ENV_CONFIG
from simulation.scenarios.exploration import ExplorationScenario
from simulation.scenarios.robot_failure import RobotFailureScenario
from simulation.scenarios.search_rescue import SearchRescueScenario
from simulation.scenarios.surveillance import SurveillanceScenario




def test_exploration_scenario_default_spawn_positions_are_evenly_spaced():
    scenario = ExplorationScenario(environment=STATIC_ENV_CONFIG, robot_ids=["r1", "r2", "r3"])
    positions = scenario.default_spawn_positions()
    assert len(positions) == 3

    xs = [p[0] for p in positions]
    assert xs == sorted(xs)  

def test_exploration_scenario_respects_explicit_spawn_positions():
    explicit = [(1.0, 1.0, 0.0), (2.0, 2.0, 0.0)]
    scenario = ExplorationScenario(robot_ids=["r1", "r2"], spawn_positions=explicit)
    assert scenario.default_spawn_positions() == explicit


def test_exploration_scenario_build_env_kwargs_has_required_keys():
    scenario = ExplorationScenario()
    kwargs = scenario.build_env_kwargs()
    for key in ("robot_ids", "world_bounds_m", "reward_config", "max_steps", "coverage_success_threshold"):
        assert key in kwargs



def test_robot_failure_scenario_validates_failure_robot_in_roster():
    try:
        RobotFailureScenario(robot_ids=["r1", "r2"], failure_robot_id="not_in_roster")
        assert False, "expected ValueError when failure_robot_id is not in robot_ids"
    except ValueError:
        pass


def test_robot_failure_scenario_triggers_only_at_failure_step():
    scenario = RobotFailureScenario(failure_step=100)
    assert scenario.should_trigger_failure(100) is True
    assert scenario.should_trigger_failure(99) is False
    assert scenario.should_trigger_failure(101) is False



def test_search_rescue_scenario_detects_target_within_radius():
    scenario = SearchRescueScenario(target_positions=[(5.0, 5.0)], detection_radius_m=0.5)
    found = scenario.check_targets_found({"r1": (5.1, 5.05)}, already_found=set())
    assert found == {0}


def test_search_rescue_scenario_does_not_detect_out_of_radius():
    scenario = SearchRescueScenario(target_positions=[(5.0, 5.0)], detection_radius_m=0.5)
    found = scenario.check_targets_found({"r1": (10.0, 10.0)}, already_found=set())
    assert found == set()


def test_search_rescue_scenario_preserves_already_found():
    scenario = SearchRescueScenario(target_positions=[(1.0, 1.0), (2.0, 2.0)], detection_radius_m=0.3)
    found = scenario.check_targets_found({"r1": (2.05, 2.0)}, already_found={0})
    assert found == {0, 1}




def test_surveillance_scenario_no_bonus_for_never_seen_cell():
    scenario = SurveillanceScenario()
    bonus = scenario.compute_staleness_bonus((0, 0), last_observed_step={}, current_step=1000)
    assert bonus == 0.0


def test_surveillance_scenario_no_bonus_within_revisit_interval():
    scenario = SurveillanceScenario(revisit_interval_steps=300)
    bonus = scenario.compute_staleness_bonus((0, 0), last_observed_step={(0, 0): 950}, current_step=1000)
    assert bonus == 0.0


def test_surveillance_scenario_scales_bonus_with_overdue_time():
    scenario = SurveillanceScenario(revisit_interval_steps=300, staleness_bonus_scale=1.0)
    slightly_overdue = scenario.compute_staleness_bonus((0, 0), {(0, 0): 0}, current_step=400)
    very_overdue = scenario.compute_staleness_bonus((0, 0), {(0, 0): 0}, current_step=1000)
    assert 0.0 < slightly_overdue < very_overdue <= 1.0




def test_static_and_dynamic_configs_have_distinct_world_files():
    assert STATIC_ENV_CONFIG.world_file != DYNAMIC_ENV_CONFIG.world_file


def test_dynamic_config_has_imperfect_communication_by_default():
    assert DYNAMIC_ENV_CONFIG.protocol_config.packet_loss_probability > 0.0


def test_communication_sweep_spans_from_perfect_to_degraded():
    sweep = communication_sweep_configs()
    loss_levels = [c.packet_loss_probability for c in sweep]
    assert loss_levels[0] == 0.0
    assert loss_levels[-1] > 0.5
    assert loss_levels == sorted(loss_levels)  


def test_domain_randomization_jitters_within_bound():
    config = RandomizationConfig(obstacle_position_jitter_m=0.1)
    original = [(1.0, 1.0)]
    jittered = apply_domain_randomization(original, config)
    dx = abs(jittered[0][0] - original[0][0])
    dy = abs(jittered[0][1] - original[0][1])
    assert dx <= 0.1
    assert dy <= 0.1
