from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import numpy as np
from mcrate.models.hf import load_activation_array
from mcrate.utils.io import load_yaml, write_json
from mcrate.utils.logging import get_logger
from mcrate.utils.stats import roc_auc_score, sigmoid


LOGGER = get_logger(__name__)


GROUP_LABELS = {
    "success_low_cue_member": 1,
    "fail_low_cue_member": 0,
    "success_high_cue_member": 2,
    "low_cue_nonmember": 3,
}


def _fit_logistic(X: np.ndarray, y: np.ndarray, *, steps: int = 300, lr: float = 0.2) -> tuple[np.ndarray, float]:
    if len(X) == 0:
        return np.zeros((X.shape[1],), dtype=float), 0.0
    weights = np.zeros((X.shape[1],), dtype=float)
    bias = 0.0
    for _ in range(steps):
        logits = X @ weights + bias
        preds = np.vectorize(sigmoid)(logits)
        diff = preds - y
        grad_w = (X.T @ diff) / len(X) + 0.01 * weights
        grad_b = diff.mean()
        weights -= lr * grad_w
        bias -= lr * grad_b
    return weights, bias


def _comparison_pairs() -> list[tuple[str, str, str]]:
    return [
        ("success_low_cue_member", "fail_low_cue_member", "success_low_vs_fail_low"),
        ("success_low_cue_member", "low_cue_nonmember", "success_low_vs_nonmember_low"),
        ("success_low_cue_member", "success_high_cue_member", "success_low_vs_success_high"),
    ]


def _standardize_with_train_stats(train_X: np.ndarray, eval_X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = train_X.mean(axis=0, keepdims=True)
    std = np.maximum(train_X.std(axis=0, keepdims=True), 1e-6)
    return (train_X - mean) / std, (eval_X - mean) / std


def _balanced_train_eval_split(
    pos: np.ndarray,
    neg: np.ndarray,
    *,
    eval_fraction: float,
    seed: int,
    min_examples_per_class: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    size = min(len(pos), len(neg))
    if size < max(2, min_examples_per_class):
        return None
    pos = pos[:size]
    neg = neg[:size]
    rng = np.random.default_rng(seed)
    pos_order = rng.permutation(size)
    neg_order = rng.permutation(size)
    eval_per_class = max(1, int(round(size * eval_fraction)))
    eval_per_class = min(eval_per_class, size - 1)
    if size - eval_per_class < 1:
        return None
    pos_eval = pos[pos_order[:eval_per_class]]
    pos_train = pos[pos_order[eval_per_class:]]
    neg_eval = neg[neg_order[:eval_per_class]]
    neg_train = neg[neg_order[eval_per_class:]]
    train_X = np.concatenate([pos_train, neg_train], axis=0)
    train_y = np.asarray([1] * len(pos_train) + [0] * len(neg_train), dtype=float)
    eval_X = np.concatenate([pos_eval, neg_eval], axis=0)
    eval_y = np.asarray([1] * len(pos_eval) + [0] * len(neg_eval), dtype=float)
    train_X, eval_X = _standardize_with_train_stats(train_X, eval_X)
    train_order = rng.permutation(len(train_X))
    eval_order = rng.permutation(len(eval_X))
    return train_X[train_order], train_y[train_order], eval_X[eval_order], eval_y[eval_order]


def _available_layers(activations_path: Path, resid_site: str) -> list[int]:
    layers = set()
    for group_dir in activations_path.iterdir():
        if not group_dir.is_dir():
            continue
        for path in group_dir.glob(f"layer_*_site_{resid_site}.pt"):
            try:
                layer_value = int(path.name.split("_")[1])
            except (IndexError, ValueError):
                continue
            layers.add(layer_value)
    return sorted(layers)


def train_probes(*, activations_root: str, config_path: str, out_dir: str) -> dict[str, Any]:
    config = load_yaml(config_path)
    activations_path = Path(activations_root)
    if activations_path.is_file():
        activations_path = activations_path.parent
    resid_site = "resid_post"
    layers = _available_layers(activations_path, resid_site)
    rows = []
    avg_auc_by_layer: dict[int, list[float]] = {layer: [] for layer in layers}
    eval_fraction = float(config.get("probe", {}).get("eval_fraction", 0.2))
    split_seed = int(config.get("probe", {}).get("split_seed", 1))
    min_examples_per_class = int(config.get("probe", {}).get("min_examples_per_class", 2))

    for positive_group, negative_group, label in _comparison_pairs():
        pos_dir = activations_path / positive_group
        neg_dir = activations_path / negative_group
        if not pos_dir.exists() or not neg_dir.exists():
            continue
        for layer in layers:
            pos_path = pos_dir / f"layer_{layer:02d}_site_{resid_site}.pt"
            neg_path = neg_dir / f"layer_{layer:02d}_site_{resid_site}.pt"
            if not pos_path.exists() or not neg_path.exists():
                continue
            pos = load_activation_array(pos_path)
            neg = load_activation_array(neg_path)
            size = min(len(pos), len(neg))
            if size == 0:
                continue
            pos = pos[:size]
            neg = neg[:size]
            split = _balanced_train_eval_split(
                pos,
                neg,
                eval_fraction=eval_fraction,
                seed=split_seed + layer,
                min_examples_per_class=min_examples_per_class,
            )
            if split is None:
                continue
            train_X, train_y, eval_X, eval_y = split
            weights, bias = _fit_logistic(train_X, train_y)
            scores = eval_X @ weights + bias
            auc = roc_auc_score(y_true=eval_y.astype(int).tolist(), y_score=scores.tolist())
            avg_auc_by_layer[layer].append(auc)
            rows.append(
                {
                    "comparison": label,
                    "layer": layer,
                    "auc": round(float(auc), 4),
                    "site": resid_site,
                    "train_examples": int(len(train_y)),
                    "eval_examples": int(len(eval_y)),
                    "positive_examples": int(size),
                    "negative_examples": int(size),
                    "status": "ok",
                }
            )

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    csv_path = out_path / "probe_results.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "comparison",
                "layer",
                "auc",
                "site",
                "train_examples",
                "eval_examples",
                "positive_examples",
                "negative_examples",
                "status",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    ranked_layers = sorted(
        ((layer, scores) for layer, scores in avg_auc_by_layer.items() if scores),
        key=lambda item: sum(item[1]) / max(1, len(item[1])),
        reverse=True,
    )
    top_layers = ranked_layers[: int(config.get("top_k_layers_from_probe", 5))]
    candidates = {
        "all_candidate_layers": [
            {"layer": layer, "mean_auc": round(sum(scores) / max(1, len(scores)), 4)}
            for layer, scores in ranked_layers
            if scores
        ],
        "candidate_layers": [
            {"layer": layer, "mean_auc": round(sum(scores) / max(1, len(scores)), 4)}
            for layer, scores in top_layers
        ]
    }
    write_json(out_path / "candidate_layers.json", candidates)
    LOGGER.info("Wrote probe results to %s", csv_path)
    return {"probe_results": str(csv_path.resolve()), **candidates}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train simple layerwise probes from cached activations.")
    parser.add_argument("--activations", required=True, help="Activation root directory.")
    parser.add_argument("--config", required=True, help="Probe config YAML.")
    parser.add_argument("--out", required=True, help="Output directory.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_probes(activations_root=args.activations, config_path=args.config, out_dir=args.out)


if __name__ == "__main__":
    main()
