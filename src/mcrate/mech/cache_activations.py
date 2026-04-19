from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from mcrate.models import LAYER_COUNT, detect_backend, load_toy_or_raise
from mcrate.models.hf import cache_hf_activations, save_activation_array
from mcrate.utils.io import ensure_dir, load_yaml, read_jsonl, write_json
from mcrate.utils.logging import get_logger


LOGGER = get_logger(__name__)


def _collapse_scores(scores: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in scores:
        grouped[row["task_id"]].append(row)
    rows = []
    for group in grouped.values():
        sample = group[0]
        rows.append(
            {
                **sample,
                "success": any(item["any_sensitive_match"] for item in group),
            }
        )
    return rows


def _group_name(row: dict[str, Any]) -> str | None:
    if row["cue_band"] == "low" and row["membership"] == "member" and row["success"]:
        return "success_low_cue_member"
    if row["cue_band"] == "low" and row["membership"] == "member" and not row["success"]:
        return "fail_low_cue_member"
    if row["cue_band"] == "high" and row["membership"] == "member" and row["success"]:
        return "success_high_cue_member"
    if row["cue_band"] == "low" and row["membership"] == "nonmember":
        return "low_cue_nonmember"
    return None


def cache_activations(*, model_path: str, scores_path: str, config_path: str, out_dir: str) -> dict[str, Any]:
    config = load_yaml(config_path)
    backend = detect_backend(model_path)
    if backend == "huggingface_causal_lm":
        rows = _collapse_scores(read_jsonl(scores_path))
        grouped_rows = []
        for row in rows:
            group = _group_name(row)
            if group:
                grouped_rows.append({**row, "group": group})
        return cache_hf_activations(model_path=model_path, scores=grouped_rows, config=config, out_dir=out_dir)
    if backend != "toy_memorizer":
        raise RuntimeError(f"Unsupported backend for activation caching: {backend}")
    model = load_toy_or_raise(model_path)
    rows = _collapse_scores(read_jsonl(scores_path))
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        group = _group_name(row)
        if group:
            grouped[group].append(row)
    max_examples = int(config.get("max_examples_per_group", 500))
    sites = list(config.get("cache_sites", ["resid_post", "attn_out", "mlp_out"]))
    storage_dtype = str(config.get("cache_storage_dtype", "float16"))
    site_scale = {"resid_pre": 0.92, "resid_post": 1.0, "attn_out": 0.88, "mlp_out": 1.05}

    root = ensure_dir(out_dir) / _safe_model_run(model_path)
    manifests = []
    for group_name, items in grouped.items():
        selected = items[:max_examples]
        group_dir = ensure_dir(root / group_name)
        activations = [model.activation_matrix(row) for row in selected]
        stacked = np.stack(activations, axis=0) if activations else np.zeros((0, LAYER_COUNT, 6))
        for layer in range(LAYER_COUNT):
            for site in sites:
                array = stacked[:, layer, :] * site_scale.get(site, 1.0)
                path = group_dir / f"layer_{layer:02d}_site_{site}.pt"
                save_activation_array(path, array, storage_dtype=storage_dtype)
                manifest = {
                    "model_run": _safe_model_run(model_path),
                    "group": group_name,
                    "layer": layer,
                    "site": site,
                    "shape": list(array.shape),
                    "storage_dtype": storage_dtype,
                    "examples": [row["task_id"] for row in selected],
                    "source_scores_path": str(Path(scores_path).resolve()),
                }
                write_json(str(path) + ".json", manifest)
                manifests.append(manifest)
    write_json(root / "cache_manifest.json", {"files": manifests})
    LOGGER.info("Cached activations for %s groups in %s", len(grouped), root)
    return {"root": str(root.resolve()), "groups": {k: len(v) for k, v in grouped.items()}}


def _safe_model_run(model_path: str) -> str:
    return Path(model_path).parent.name if Path(model_path).name == "final_model" else Path(model_path).name


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cache focused activations for mechanistic analysis.")
    parser.add_argument("--model", required=True, help="Model directory.")
    parser.add_argument("--scores", required=True, help="Scored generations JSONL.")
    parser.add_argument("--config", required=True, help="Mechanistic YAML config.")
    parser.add_argument("--out", required=True, help="Output activation directory root.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cache_activations(model_path=args.model, scores_path=args.scores, config_path=args.config, out_dir=args.out)


if __name__ == "__main__":
    main()
