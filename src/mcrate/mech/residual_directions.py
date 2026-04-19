from __future__ import annotations

import argparse
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from mcrate.models import detect_backend, load_toy_or_raise
from mcrate.models.hf import (
    TextBlockDataset,
    _get_transformer_layers,
    _load_model_and_tokenizer,
    capture_residual_post,
    evaluate_dataset_with_residual_edits,
    load_corpus_manifest,
    load_record_map,
    load_training_args,
    model_run_name,
    target_logprob_with_optional_edits,
    target_text_from_record,
)
from mcrate.utils.io import load_yaml, read_jsonl, write_jsonl
from mcrate.utils.logging import get_logger


LOGGER = get_logger(__name__)


def _collapse(scores: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped = defaultdict(list)
    for row in scores:
        grouped[row["task_id"]].append(row)
    rows = []
    for group in grouped.values():
        sample = dict(group[0])
        sample["success"] = any(item["any_sensitive_match"] for item in group)
        rows.append(sample)
    return rows


def residual_directions(*, model_path: str, scores_path: str, config_path: str, out_path: str) -> list[dict[str, Any]]:
    backend = detect_backend(model_path)
    config = load_yaml(config_path)
    rows = _collapse(read_jsonl(scores_path))
    success = [row for row in rows if row["cue_band"] == "low" and row["membership"] == "member" and row["success"]]
    fail = [row for row in rows if row["cue_band"] == "low" and row["membership"] == "member" and not row["success"]]
    eval_rows = [row for row in rows if row["cue_band"] == "low" and row["membership"] == "member"]
    if not success or not fail or not eval_rows:
        raise RuntimeError("Need successful and failed low-cue member examples for residual direction analysis.")
    alphas = list(config.get("alpha_grid", [0.0, 0.25, 0.5, 1.0, 2.0]))

    if backend == "toy_memorizer":
        model = load_toy_or_raise(model_path)
        success_mean = np.mean([model.activation_matrix(row) for row in success], axis=0)
        fail_mean = np.mean([model.activation_matrix(row) for row in fail], axis=0)
        direction = success_mean - fail_mean
        results = []
        for alpha in alphas:
            probabilities = []
            logprobs = []
            for row in eval_rows:
                modified = model.activation_matrix(row) - float(alpha) * direction
                probabilities.append(model.score_components(row, activations=modified)["probability"])
                logprobs.append(model.target_logprob(row, activations=modified))
            results.append(
                {
                    "model_run": Path(model_path).parent.name,
                    "alpha": float(alpha),
                    "low_cue_extraction_rate": round(float(np.mean(probabilities)), 4),
                    "mean_target_logprob": round(float(np.mean(logprobs)), 4),
                    "background_perplexity": round(float(model.utility_perplexity() * (1 + 0.01 * alpha)), 4),
                }
            )
        write_jsonl(out_path, results)
        LOGGER.info("Wrote residual direction results to %s", out_path)
        return results

    if backend != "huggingface_causal_lm":
        raise RuntimeError(f"Unsupported backend for residual direction analysis: {backend}")

    model, tokenizer, device = _load_model_and_tokenizer(model_path)
    record_map = load_record_map(model_path)
    capture_cache: dict[str, dict[str, Any]] = {}

    def cached_capture(row: dict[str, Any]) -> dict[str, Any]:
        task_id = row["task_id"]
        if task_id not in capture_cache:
            capture_cache[task_id] = capture_residual_post(model, tokenizer, device, row["prompt"])
        return capture_cache[task_id]

    layer_count = len(_get_transformer_layers(model))
    directions: dict[int, np.ndarray] = {}
    for layer in range(layer_count):
        success_vectors = [cached_capture(row)["vectors"][layer] for row in success if layer in cached_capture(row)["vectors"]]
        fail_vectors = [cached_capture(row)["vectors"][layer] for row in fail if layer in cached_capture(row)["vectors"]]
        if success_vectors and fail_vectors:
            directions[layer] = np.mean(np.asarray(success_vectors, dtype=float), axis=0) - np.mean(np.asarray(fail_vectors, dtype=float), axis=0)
    ranked_layers = sorted(directions, key=lambda layer: float(np.linalg.norm(directions[layer])), reverse=True)
    selected_layers = ranked_layers[: int(config.get("direction_top_k_layers", 5))]

    manifest = load_corpus_manifest(model_path)
    training_args = load_training_args(model_path)
    validation_file = manifest.get("validation_file")
    utility_dataset = None
    if validation_file:
        utility_dataset = TextBlockDataset(
            validation_file,
            tokenizer,
            int(training_args.get("sequence_length", manifest.get("sequence_length", 1024))),
        )

    results = []
    for alpha in alphas:
        probabilities = []
        logprobs = []
        for row in eval_rows:
            record = record_map.get(row["record_id"])
            if not record:
                continue
            target_text = target_text_from_record(record)
            if not target_text:
                continue
            capture = cached_capture(row)
            modified = target_logprob_with_optional_edits(
                model,
                tokenizer,
                device,
                prompt=row["prompt"],
                target_text=target_text,
                edits=[
                    {
                        "layer": layer,
                        "mode": "subtract",
                        "vector": directions[layer],
                        "alpha": float(alpha),
                        "token_index": capture["prompt_index"],
                    }
                    for layer in selected_layers
                ],
            )
            probabilities.append(math.exp(modified["target_logprob"]))
            logprobs.append(modified["target_logprob"])

        background_perplexity = 0.0
        if utility_dataset is not None:
            utility_metrics = evaluate_dataset_with_residual_edits(
                model,
                utility_dataset,
                batch_size=1,
                device=device,
                edits=[
                    {
                        "layer": layer,
                        "mode": "subtract",
                        "vector": directions[layer],
                        "alpha": float(alpha),
                        "token_index": None,
                    }
                    for layer in selected_layers
                ],
                max_examples=int(config.get("utility_eval_max_examples", 8)),
            )
            background_perplexity = utility_metrics["perplexity"]
        results.append(
            {
                "model_run": model_run_name(model_path),
                "alpha": float(alpha),
                "layers": selected_layers,
                "low_cue_extraction_rate": round(float(np.mean(probabilities) if probabilities else 0.0), 4),
                "mean_target_logprob": round(float(np.mean(logprobs) if logprobs else 0.0), 4),
                "background_perplexity": round(float(background_perplexity), 4),
            }
        )
    write_jsonl(out_path, results)
    LOGGER.info("Wrote residual direction results to %s", out_path)
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate residual direction interventions.")
    parser.add_argument("--model", required=True, help="Model directory.")
    parser.add_argument("--scores", required=True, help="Scored generations JSONL.")
    parser.add_argument("--config", required=True, help="Mechanistic YAML config.")
    parser.add_argument("--out", required=True, help="Output JSONL.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    residual_directions(model_path=args.model, scores_path=args.scores, config_path=args.config, out_path=args.out)


if __name__ == "__main__":
    main()
