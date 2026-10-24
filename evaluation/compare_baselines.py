from __future__ import annotations

import logging
import statistics
from dataclasses import dataclass
from typing import List, Optional

from data.evaluation_writer import EvaluationWriter
from data.schemas import EvaluationRun

logger = logging.getLogger(__name__)

SIGNIFICANCE_THRESHOLD = 0.05


@dataclass
class StatisticalComparison:
    experiment_name: str
    metric: str
    condition_a: str
    condition_b: str
    mean_a: float
    mean_b: float
    stdev_a: float
    stdev_b: float
    n_a: int
    n_b: int
    u_statistic: Optional[float]
    p_value: Optional[float]
    significant: bool
    inconclusive_reason: Optional[str] = None

    def summary_line(self) -> str:
        if self.inconclusive_reason is not None:
            return (
                f"{self.metric}: {self.condition_a} (n={self.n_a}, mean={self.mean_a:.3f}) vs "
                f"{self.condition_b} (n={self.n_b}, mean={self.mean_b:.3f}) - INCONCLUSIVE "
                f"({self.inconclusive_reason})"
            )
        verdict = "SIGNIFICANT" if self.significant else "not significant"
        return (
            f"{self.metric}: {self.condition_a} (mean={self.mean_a:.3f}) vs "
            f"{self.condition_b} (mean={self.mean_b:.3f}) - p={self.p_value:.4f} ({verdict})"
        )


def _mann_whitney_u(sample_a: List[float], sample_b: List[float]):

    try:
        from scipy.stats import mannwhitneyu
        result = mannwhitneyu(sample_a, sample_b, alternative="two-sided")
        return float(result.statistic), float(result.pvalue)
    except ImportError:
        logger.warning("scipy not available - falling back to normal-approximation Mann-Whitney U")
        return _mann_whitney_u_normal_approx(sample_a, sample_b)


def _mann_whitney_u_normal_approx(sample_a: List[float], sample_b: List[float]):
    n1, n2 = len(sample_a), len(sample_b)
    combined = [(v, "a") for v in sample_a] + [(v, "b") for v in sample_b]
    combined.sort(key=lambda pair: pair[0])

    ranks = {}
    i = 0
    rank = 1
    while i < len(combined):
        j = i
        while j < len(combined) and combined[j][0] == combined[i][0]:
            j += 1
        avg_rank = (rank + (rank + (j - i) - 1)) / 2.0
        for k in range(i, j):
            ranks[k] = avg_rank
        rank += (j - i)
        i = j

    rank_sum_a = sum(ranks[idx] for idx, (_, label) in enumerate(combined) if label == "a")
    u_a = rank_sum_a - n1 * (n1 + 1) / 2.0
    u = min(u_a, n1 * n2 - u_a)

    mean_u = n1 * n2 / 2.0
    std_u = ((n1 * n2 * (n1 + n2 + 1)) / 12.0) ** 0.5
    if std_u == 0:
        return u, 1.0

    z = (u - mean_u) / std_u
    import math
    p_value = 2 * (1 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2))))
    return u, min(1.0, p_value)


def compare_conditions(
    experiment_name: str,
    condition_a: str,
    condition_b: str,
    metric: str = "coverage",
    min_seeds: int = 5,
) -> StatisticalComparison:

    writer = EvaluationWriter(experiment_name)
    runs_a = writer.read_by_condition(condition_a)
    runs_b = writer.read_by_condition(condition_b)

    values_a = [getattr(r, metric) for r in runs_a]
    values_b = [getattr(r, metric) for r in runs_b]

    mean_a = statistics.mean(values_a) if values_a else float("nan")
    mean_b = statistics.mean(values_b) if values_b else float("nan")
    stdev_a = statistics.stdev(values_a) if len(values_a) > 1 else 0.0
    stdev_b = statistics.stdev(values_b) if len(values_b) > 1 else 0.0

    if len(values_a) < min_seeds or len(values_b) < min_seeds:
        return StatisticalComparison(
            experiment_name=experiment_name, metric=metric,
            condition_a=condition_a, condition_b=condition_b,
            mean_a=mean_a, mean_b=mean_b, stdev_a=stdev_a, stdev_b=stdev_b,
            n_a=len(values_a), n_b=len(values_b),
            u_statistic=None, p_value=None, significant=False,
            inconclusive_reason=(
                f"fewer than {min_seeds} seeds recorded "
                f"(condition_a={len(values_a)}, condition_b={len(values_b)})"
            ),
        )

    u_statistic, p_value = _mann_whitney_u(values_a, values_b)

    return StatisticalComparison(
        experiment_name=experiment_name, metric=metric,
        condition_a=condition_a, condition_b=condition_b,
        mean_a=mean_a, mean_b=mean_b, stdev_a=stdev_a, stdev_b=stdev_b,
        n_a=len(values_a), n_b=len(values_b),
        u_statistic=u_statistic, p_value=p_value,
        significant=p_value < SIGNIFICANCE_THRESHOLD,
    )


def compare_all_metrics(
    experiment_name: str, condition_a: str, condition_b: str, min_seeds: int = 5
) -> List[StatisticalComparison]:

    metrics = [
        "coverage", "path_efficiency", "energy_consumption",
        "communication_messages", "mean_episode_reward", "collision_count",
    ]
    return [
        compare_conditions(experiment_name, condition_a, condition_b, metric=m, min_seeds=min_seeds)
        for m in metrics
    ]
