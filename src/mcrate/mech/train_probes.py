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
            X = np.concatenate([pos, neg], axis=0)
            y = np.asarray([1] * len(pos) + [0] * len(neg), dtype=float)
            order = np.random.default_rng(1 + layer).permutation(len(X))
            X = X[order]
            y = y[order]
            X = (X - X.mean(axis=0, keepdims=True)) / np.maximum(X.std(axis=0, keepdims=True), 1e-6)
            split = max(1, int(0.8 * len(X)))
            weights, bias = _fit_logistic(X[:split], y[:split])
            eval_X = X[split:] if len(set(y[split:].astype(int).tolist())) == 2 else X
            eval_y = y[split:] if len(set(y[split:].astype(int).tolist())) == 2 else y
            scores = eval_X @ weights + bias
            auc = roc_auc_score(y_true=eval_y.astype(int).tolist(), y_score=scores.tolist())
            avg_auc_by_layer[layer].append(auc)
            rows.append({"comparison": label, "layer": layer, "auc": round(float(auc), 4), "site": resid_site})

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    csv_path = out_path / "probe_results.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["comparison", "layer", "auc", "site"])
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
