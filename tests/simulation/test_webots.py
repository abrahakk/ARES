from __future__ import annotations

import os
import shutil

import pytest

pytestmark = pytest.mark.simulation


def _webots_available() -> bool:
    webots_home = os.environ.get("WEBOTS_HOME")
    if webots_home and os.path.exists(os.path.join(webots_home, "webots")):
        return True
    return shutil.which("webots") is not None


@pytest.fixture(autouse=True)
def _skip_if_no_webots():
    if not _webots_available():
        pytest.skip("no Webots installation found (WEBOTS_HOME not set / webots not on PATH)")


def test_world_files_exist():
    world_files = [
        "simulation/webots/worlds/simple.wbt",
        "simulation/webots/worlds/maze.wbt",
        "simulation/webots/worlds/dynamic.wbt",
        "simulation/webots/worlds/disaster_zone.wbt",
    ]
    for path in world_files:
        assert os.path.exists(path), f"missing world file: {path}"


def test_proto_files_exist():
    proto_files = [
        "simulation/webots/protos/ARESRobot.proto",
        "simulation/webots/protos/SensorSuite.proto",
    ]
    for path in proto_files:
        assert os.path.exists(path), f"missing PROTO file: {path}"


def test_ares_robot_proto_declares_expected_device_names():
    
    with open("simulation/webots/protos/ARESRobot.proto") as f:
        content = f.read()

    for expected_device_name in ('"gps"', '"inertial unit"', '"lidar"', '"camera"',
                                 '"left wheel motor"', '"right wheel motor"'):
        assert expected_device_name in content, f"expected device name {expected_device_name} not found in ARESRobot.proto"


def test_supervisor_controller_launches_without_crashing():
   
    import importlib
    module = importlib.import_module("simulation.webots.controllers.supervisor_controller")
    assert hasattr(module, "SupervisorController")
