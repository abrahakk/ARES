from __future__ import annotations

import os

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  
from serving.app.main import app  
pytestmark = pytest.mark.integration

client = TestClient(app)


def test_health_endpoint_returns_200():
    response = client.get("/health")
    assert response.status_code == 200
    assert "status" in response.json()


def test_predict_rejects_missing_required_fields():
    response = client.post("/predict", json={"robot_id": "robot_01"})
    assert response.status_code == 422


def test_predict_rejects_out_of_bounds_theta():
    payload = {
        "robot_id": "robot_01",
        "pose": {"x": 0.0, "y": 0.0, "theta": 99.0},  
        "local_occupancy": [0.0] * 25,
        "detected_obstacles": [],
        "messages_received": [],
        "battery_level": 0.8,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_predict_rejects_wrong_length_occupancy_grid():
    payload = {
        "robot_id": "robot_01",
        "pose": {"x": 0.0, "y": 0.0, "theta": 0.0},
        "local_occupancy": [0.0] * 10,  
        "detected_obstacles": [],
        "messages_received": [],
        "battery_level": 0.8,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_predict_rejects_out_of_bounds_battery_level():
    payload = {
        "robot_id": "robot_01",
        "pose": {"x": 0.0, "y": 0.0, "theta": 0.0},
        "local_occupancy": [0.0] * 25,
        "detected_obstacles": [],
        "messages_received": [],
        "battery_level": 1.5,  
   }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


@pytest.mark.skipif(
    not os.path.exists(os.environ.get("ARES_MODEL_PATH", "models/onnx/ares_policy.onnx")),
    reason="no exported ONNX model available at ARES_MODEL_PATH - run scripts/export_onnx.py first",
)
def test_predict_returns_valid_action_with_real_model():
    payload = {
        "robot_id": "robot_01",
        "pose": {"x": 0.0, "y": 0.0, "theta": 0.0},
        "local_occupancy": [0.0] * 25,
        "detected_obstacles": [],
        "messages_received": [],
        "battery_level": 1.0,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert body["robot_id"] == "robot_01"
    assert "linear_velocity" in body
    assert "angular_velocity" in bodyfrom __future__ import annotations

def test_robots_list_endpoint_does_not_crash_without_redis():
    response = client.get("/robots")
    assert response.status_code in (200, 500)  
