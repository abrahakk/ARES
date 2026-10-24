from __future__ import annotations

import logging
import time
from typing import Tuple

import numpy as np
import onnxruntime as ort

from monitoring.metrics import record_inference_latency

logger = logging.getLogger(__name__)


class ONNXEngine:
    def __init__(self, model_path: str, providers: list | None = None) -> None:
        self.model_path = model_path
        self.session = ort.InferenceSession(
            model_path, providers=providers or ["CPUExecutionProvider"]
        )
        self._input_name = self.session.get_inputs()[0].name
        logger.info("loaded ONNX model from %s (input=%s)", model_path, self._input_name)

    def predict(self, observation_vector: np.ndarray) -> Tuple[np.ndarray, float]:
        if observation_vector.ndim == 1:
            observation_vector = observation_vector.reshape(1, -1)
        observation_vector = observation_vector.astype(np.float32)

        start = time.perf_counter()
        action_out, value_out = self.session.run(
            None, {self._input_name: observation_vector}
        )
        record_inference_latency(time.perf_counter() - start)

        return action_out[0], float(value_out[0])
