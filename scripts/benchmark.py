from __future__ import annotations

import argparse
import logging
import statistics
import time

import numpy as np

from rl.environments.observation_space import OBSERVATION_DIM

logger = logging.getLogger(__name__)


def _sample_observation() -> dict:

    rng = np.random.default_rng()
    return {
        "robot_id": "benchmark_robot",
        "pose": {"x": float(rng.uniform(-2, 2)), "y": float(rng.uniform(-2, 2)), "theta": float(rng.uniform(-3.1, 3.1))},
        "local_occupancy": [float(v) for v in rng.uniform(0, 1, size=25)],
        "detected_obstacles": [],
        "messages_received": [],
        "battery_level": 1.0,
    }


def benchmark_api_latency(api_url: str, num_requests: int) -> list:

    import requests

    latencies = []
    payload = _sample_observation()

    for _ in range(num_requests):
        start = time.perf_counter()
        response = requests.post(f"{api_url}/predict", json=payload, timeout=5.0)
        elapsed = time.perf_counter() - start
        if response.status_code != 200:
            logger.warning("request failed with status=%d: %s", response.status_code, response.text)
            continue
        latencies.append(elapsed)

    return latencies


def benchmark_inference_only_latency(checkpoint_path: str, num_requests: int, device: str = "cpu") -> list:

    import onnxruntime as ort
    import torch

    from rl.networks.actor import ActorNetwork

    actor = ActorNetwork(OBSERVATION_DIM).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    actor.load_state_dict(checkpoint["actor"] if "actor" in checkpoint else checkpoint)
    actor.eval()

    rng = np.random.default_rng()
    latencies = []

    with torch.no_grad():
        for _ in range(num_requests):
            obs = rng.standard_normal(OBSERVATION_DIM).astype(np.float32)
            obs_tensor = torch.as_tensor(obs, device=device).unsqueeze(0)
            start = time.perf_counter()
            actor.export_forward(obs_tensor)
            latencies.append(time.perf_counter() - start)

    return latencies


def _summarize(name: str, latencies: list) -> None:
    if not latencies:
        logger.warning("%s: no successful measurements", name)
        return
    sorted_latencies = sorted(latencies)
    p50 = sorted_latencies[len(sorted_latencies) // 2]
    p95 = sorted_latencies[int(len(sorted_latencies) * 0.95)]
    logger.info(
        "%s: n=%d mean=%.4fs p50=%.4fs p95=%.4fs max=%.4fs",
        name, len(latencies), statistics.mean(latencies), p50, p95, max(latencies),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api-url", type=str, default="http://localhost:8000")
    parser.add_argument("--checkpoint", type=str, default=None, help="If given, also benchmarks inference-only latency.")
    parser.add_argument("--num-requests", type=int, default=100)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--skip-api", action="store_true", help="Skip the live-API benchmark (inference-only benchmark still runs if --checkpoint is given).")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    if not args.skip_api:
        logger.info("benchmarking live API at %s ...", args.api_url)
        api_latencies = benchmark_api_latency(args.api_url, args.num_requests)
        _summarize("end-to-end API latency", api_latencies)

    if args.checkpoint:
        logger.info("benchmarking inference-only latency ...")
        inference_latencies = benchmark_inference_only_latency(args.checkpoint, args.num_requests, args.device)
        _summarize("inference-only latency (ONNX Runtime, no network/Redis)", inference_latencies)

        if not args.skip_api and api_latencies:
            overhead = statistics.mean(api_latencies) - statistics.mean(inference_latencies)
            logger.info(
                "estimated non-inference overhead (network + FastAPI + Redis): %.4fs "
                "(%.1f%% of end-to-end latency)",
                overhead, 100 * overhead / statistics.mean(api_latencies) if statistics.mean(api_latencies) > 0 else 0,
            )


if __name__ == "__main__":
    main()
