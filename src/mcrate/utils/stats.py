from __future__ import annotations

import math
import random
from typing import Callable, Sequence

import numpy as np


def safe_mean(values: Sequence[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def safe_div(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def bootstrap_ci(
    items: Sequence[dict],
    metric_fn: Callable[[Sequence[dict]], float],
    *,
    iters: int = 1000,
    seed: int = 1,
) -> tuple[float, float]:
    if not items:
        return (0.0, 0.0)
    rng = random.Random(seed)
    values = []
    for _ in range(iters):
        sample = [items[rng.randrange(len(items))] for _ in range(len(items))]
        values.append(metric_fn(sample))
    values.sort()
    low = values[int(0.025 * (len(values) - 1))]
    high = values[int(0.975 * (len(values) - 1))]
    return (float(low), float(high))


def wilson_ci(successes: int, total: int, *, z: float = 1.959963984540054) -> tuple[float, float]:
    if total <= 0:
        return (0.0, 0.0)
    p_hat = successes / total
    denominator = 1.0 + (z * z) / total
    center = (p_hat + (z * z) / (2.0 * total)) / denominator
    margin = (
        z
        * math.sqrt((p_hat * (1.0 - p_hat) / total) + ((z * z) / (4.0 * total * total)))
        / denominator
    )
    return (max(0.0, center - margin), min(1.0, center + margin))


def agresti_caffo_diff_ci(
    successes_a: int,
    total_a: int,
    successes_b: int,
    total_b: int,
    *,
    z: float = 1.959963984540054,
) -> tuple[float, float]:
    if total_a <= 0 or total_b <= 0:
        return (0.0, 0.0)
    adj_success_a = successes_a + 1
    adj_total_a = total_a + 2
    adj_success_b = successes_b + 1
    adj_total_b = total_b + 2
    rate_a = adj_success_a / adj_total_a
    rate_b = adj_success_b / adj_total_b
    diff = rate_a - rate_b
    variance = (rate_a * (1.0 - rate_a) / adj_total_a) + (rate_b * (1.0 - rate_b) / adj_total_b)
    margin = z * math.sqrt(max(variance, 0.0))
    return (max(-1.0, diff - margin), min(1.0, diff + margin))


def roc_auc_score(y_true: Sequence[int], y_score: Sequence[float]) -> float:
    positives = [(s, y) for s, y in zip(y_score, y_true) if y == 1]
    negatives = [(s, y) for s, y in zip(y_score, y_true) if y == 0]
    if not positives or not negatives:
        return 0.5
    better = 0.0
    ties = 0.0
    for p_score, _ in positives:
        for n_score, _ in negatives:
            if p_score > n_score:
                better += 1.0
            elif p_score == n_score:
                ties += 1.0
    total = len(positives) * len(negatives)
    return (better + 0.5 * ties) / total


def sigmoid(value: float) -> float:
    if value >= 0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


def softplus(value: float) -> float:
    if value > 30:
        return value
    return math.log1p(math.exp(value))


def to_numpy(rows: Sequence[Sequence[float]]) -> np.ndarray:
    return np.asarray(rows, dtype=float)
