from __future__ import annotations

import time

from robots.agents.robot_manager import RobotManager
from robots.agents.robot_state import Pose, RobotState, RobotStatus




def test_robot_state_defaults():
    state = RobotState(robot_id="robot_01")
    assert state.status == RobotStatus.IDLE
    assert state.battery_level == 1.0
    assert state.pose.as_tuple() == (0.0, 0.0, 0.0)


def test_robot_state_touch_updates_timestamp():
    state = RobotState(robot_id="robot_01")
    original = state.last_updated
    time.sleep(0.01)
    state.touch()
    assert state.last_updated > original


def test_robot_state_is_stale_respects_max_age():
    state = RobotState(robot_id="robot_01")
    state.last_updated = time.time() - 10.0
    assert state.is_stale(max_age_seconds=5.0) is True
    assert state.is_stale(max_age_seconds=20.0) is False


def test_robot_state_round_trips_through_dict():
    state = RobotState(robot_id="robot_01", pose=Pose(x=1.0, y=2.0, theta=0.5), status=RobotStatus.EXPLORING)
    restored = RobotState.from_dict(state.to_dict())
    assert restored.robot_id == "robot_01"
    assert restored.pose.as_tuple() == (1.0, 2.0, 0.5)
    assert restored.status == RobotStatus.EXPLORING



def test_robot_manager_spawns_all_robots():
    manager = RobotManager(["robot_01", "robot_02"])
    manager.spawn_all()
    assert set(manager.robots.keys()) == {"robot_01", "robot_02"}


def test_robot_manager_get_observations_excludes_failed_robots():
    manager = RobotManager(["robot_01", "robot_02"])
    manager.spawn_all()
    manager.fail_robot("robot_02")

    observations = manager.get_observations()
    assert "robot_01" in observations
    assert "robot_02" not in observations


def test_robot_manager_active_robot_ids_excludes_failed():
    manager = RobotManager(["robot_01", "robot_02"])
    manager.spawn_all()
    manager.fail_robot("robot_01")
    assert manager.active_robot_ids() == ["robot_02"]


def test_robot_manager_fail_robot_raises_for_unknown_id():
    manager = RobotManager(["robot_01"])
    manager.spawn_all()
    try:
        manager.fail_robot("robot_99")
        assert False, "expected KeyError for unknown robot_id"
    except KeyError:
        pass


def test_robot_manager_deliver_message_skips_failed_robots():
    manager = RobotManager(["robot_01"])
    manager.spawn_all()
    manager.fail_robot("robot_01")


    manager.deliver_message("robot_01", message=object())
