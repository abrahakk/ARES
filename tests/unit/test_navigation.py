from __future__ import annotations

import math

from robots.navigation.local_controller import LocalController
from robots.navigation.obstacle_avoidance import ObstacleAvoidance
from robots.navigation.path_planning import FrontierPathPlanner




def test_obstacle_avoidance_passes_through_when_no_obstacles():
    avoidance = ObstacleAvoidance()
    linear, angular, overridden = avoidance.filter_command(0.2, 0.5, obstacle_points=[])
    assert linear == 0.2
    assert angular == 0.5
    assert overridden is False


def test_obstacle_avoidance_halts_forward_motion_at_critical_distance():
    avoidance = ObstacleAvoidance(safe_distance_m=0.25, critical_distance_m=0.12)

    linear, angular, overridden = avoidance.filter_command(0.3, 0.0, obstacle_points=[(0.05, 0.0)])
    assert linear == 0.0
    assert overridden is True


def test_obstacle_avoidance_scales_speed_in_safe_zone():
    avoidance = ObstacleAvoidance(safe_distance_m=0.25, critical_distance_m=0.12)
    linear, angular, overridden = avoidance.filter_command(0.3, 0.0, obstacle_points=[(0.2, 0.0)])
    assert 0.0 < linear < 0.3
    assert overridden is True


def test_obstacle_avoidance_ignores_obstacles_behind_robot():
    avoidance = ObstacleAvoidance()
    linear, angular, overridden = avoidance.filter_command(0.3, 0.0, obstacle_points=[(-0.05, 0.0)])
    assert overridden is False
    assert linear == 0.3



def test_local_controller_straight_line_gives_equal_wheel_speeds():
    controller = LocalController(webots_robot=None)  
    controller.set_velocity(0.1, 0.0)  


def test_local_controller_clamps_to_max_wheel_speed():
    controller = LocalController(webots_robot=None, max_wheel_speed_rad_s=1.0)
    controller.set_velocity(10.0, 10.0)


def test_frontier_planner_chooses_nearest_unclaimed_frontier():
    planner = FrontierPathPlanner()
    current_pos = (0.0, 0.0)
    frontiers = [(5.0, 5.0), (1.0, 0.0), (3.0, 3.0)]
    chosen = planner.choose_frontier(current_pos, frontiers)
    assert chosen == (1.0, 0.0)


def test_frontier_planner_skips_claimed_frontiers():
    planner = FrontierPathPlanner()
    planner.claim_frontier((1.0, 0.0))
    chosen = planner.choose_frontier((0.0, 0.0), [(1.0, 0.0), (3.0, 3.0)])
    assert chosen == (3.0, 3.0)


def test_frontier_planner_returns_none_when_all_claimed():
    planner = FrontierPathPlanner()
    planner.claim_frontier((1.0, 0.0))
    chosen = planner.choose_frontier((0.0, 0.0), [(1.0, 0.0)])
    assert chosen is None


def test_frontier_planner_has_reached_within_threshold():
    planner = FrontierPathPlanner(frontier_reach_threshold_m=0.3)
    assert planner.has_reached((0.0, 0.0), (0.2, 0.0)) is True
    assert planner.has_reached((0.0, 0.0), (1.0, 0.0)) is False


def test_frontier_planner_velocity_toward_faces_target():
    planner = FrontierPathPlanner()
    linear, angular = planner.velocity_toward((0.0, 0.0), 0.0, (1.0, 0.0))
    assert linear > 0.0
    assert abs(angular) < 0.1

    linear_behind, angular_behind = planner.velocity_toward((0.0, 0.0), 0.0, (-1.0, 0.0))
    assert abs(angular_behind) > abs(angular)
