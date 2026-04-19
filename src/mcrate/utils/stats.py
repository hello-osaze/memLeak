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
