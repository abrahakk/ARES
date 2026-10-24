from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class RunningNormalizer:
    shape: tuple
    epsilon: float = 1e-8
    mean: np.ndarray = field(init=False)
    var: np.ndarray = field(init=False)
    count: float = field(default=1e-4, init=False)

    def __post_init__(self) -> None:
        self.mean = np.zeros(self.shape, dtype=np.float64)
        self.var = np.ones(self.shape, dtype=np.float64)

    def update(self, batch: np.ndarray) -> None:
        """batch: (N, *shape)"""
        batch_mean = np.mean(batch, axis=0)
        batch_var = np.var(batch, axis=0)
        batch_count = batch.shape[0]
        self._update_from_moments(batch_mean, batch_var, batch_count)

    def _update_from_moments(self, batch_mean, batch_var, batch_count) -> None:
        delta = batch_mean - self.mean
        total_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / total_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + np.square(delta) * self.count * batch_count / total_count
        new_var = m2 / total_count

        self.mean, self.var, self.count = new_mean, new_var, total_count

    def normalize(self, x: np.ndarray, clip: float = 10.0) -> np.ndarray:
        normalized = (x - self.mean) / np.sqrt(self.var + self.epsilon)
        return np.clip(normalized, -clip, clip)

    def state_dict(self) -> dict:
        return {"mean": self.mean.tolist(), "var": self.var.tolist(), "count": self.count}

    def load_state_dict(self, state: dict) -> None:
        self.mean = np.array(state["mean"], dtype=np.float64)
        self.var = np.array(state["var"], dtype=np.float64)
        self.count = state["count"]
