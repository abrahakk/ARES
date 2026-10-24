from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from rl.environments.action_space import action_vector_to_dict
from rl.environments.observation_space import flatten_observation
from serving.inference.onnx_engine import ONNXEngine

logger = logging.getLogger(__name__)


class ModelManager:
    def __init__(self, model_path: str) -> None:
        self.model_path = model_path
        self.engine: Optional[ONNXEngine] = None

    def load(self) -> None:
        self.engine = ONNXEngine(self.model_path)

    @property
    def is_loaded(self) -> bool:
        return self.engine is not None

    def predict(self, observation: dict) -> dict:

        if self.engine is None:
            raise RuntimeError("ModelManager.predict() called before load() - model not loaded")

        vector = flatten_observation(observation) 
        action, value = self.engine.predict(vector)

        action_dict = action_vector_to_dict(action)
        action_dict["value_estimate"] = value
        return action_dict
