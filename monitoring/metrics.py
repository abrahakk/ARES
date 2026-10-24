from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram


COLLISIONS_TOTAL = Counter(
    "ares_collisions_total",
    "Total collision events reported, labeled by robot.",
    ["robot_id"],
)

COVERAGE_FRACTION = Gauge(
    "ares_coverage_fraction",
    "Current team-wide exploration coverage fraction for the active mission.",
    ["mission_id"],
)

MISSION_COMPLETIONS_TOTAL = Counter(
    "ares_mission_completions_total",
    "Total missions that reached the coverage success threshold, labeled by outcome.",
    ["outcome"],  
)

ROBOT_FAILURES_TOTAL = Counter(
    "ares_robot_failures_total",
    "Total robot failures detected, labeled by robot.",
    ["robot_id"],
)



EPISODE_REWARD = Histogram(
    "ares_episode_reward",
    "Per-episode mean reward, labeled by reward component.",
    ["component"], 
    buckets=(-5.0, -2.0, -1.0, 0.0, 0.5, 1.0, 2.0, 5.0, 10.0),
)


MESSAGES_SENT_TOTAL = Counter(
    "ares_messages_sent_total",
    "Total messages sent, labeled by sender robot and message type.",
    ["robot_id", "message_type"],
)

MESSAGE_PAYLOAD_BYTES = Histogram(
    "ares_message_payload_bytes",
    "Size distribution of sent message payloads.",
    buckets=(64, 128, 256, 512, 1024, 2048, 4096),
)

COMMUNICATION_LATENCY_SECONDS = Histogram(
    "ares_communication_latency_seconds",
    "Round-trip time from a robot sending a message to a recipient processing it.",
)



INFERENCE_LATENCY_SECONDS = Histogram(
    "ares_inference_latency_seconds",
    "ONNX Runtime forward-pass latency, excluding network and Redis time - "
    "the specific breakdown the proposal's Experiment 8 (Section 11.8) needs "
    "to isolate where end-to-end latency actually accumulates.",
    buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0),
)


def record_collision(robot_id: str) -> None:
    COLLISIONS_TOTAL.labels(robot_id=robot_id).inc()


def record_coverage(mission_id: str, fraction: float) -> None:
    COVERAGE_FRACTION.labels(mission_id=mission_id).set(fraction)


def record_mission_completion(outcome: str) -> None:
    MISSION_COMPLETIONS_TOTAL.labels(outcome=outcome).inc()


def record_robot_failure(robot_id: str) -> None:
    ROBOT_FAILURES_TOTAL.labels(robot_id=robot_id).inc()


def record_episode_reward(component: str, value: float) -> None:
    EPISODE_REWARD.labels(component=component).observe(value)


def record_message_sent(robot_id: str, message_type: str, payload_bytes: int) -> None:
    MESSAGES_SENT_TOTAL.labels(robot_id=robot_id, message_type=message_type).inc()
    MESSAGE_PAYLOAD_BYTES.observe(payload_bytes)


def record_inference_latency(seconds: float) -> None:
    INFERENCE_LATENCY_SECONDS.observe(seconds)
