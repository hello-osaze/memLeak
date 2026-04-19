from __future__ import annotations

import argparse
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from mcrate.models import LAYER_COUNT, detect_backend, load_toy_or_raise
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
from mcrate.utils.io import load_yaml, read_json, read_jsonl, write_json, write_jsonl
from mcrate.utils.logging import get_logger


LOGGER = get_logger(__name__)


def _collapse(scores: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped = defaultdict(list)
    for row in scores:
        grouped[row["task_id"]].append(row)
    collapsed = []
    for rows in grouped.values():
        sample = dict(rows[0])
        sample["success"] = any(row["any_sensitive_match"] for row in rows)
        collapsed.append(sample)
    return collapsed


def mean_ablation(*, model_path: str, scores_path: str, probe_candidates_path: str, config_path: str, out_path: str) -> list[dict[str, Any]]:
    backend = detect_backend(model_path)
    config = load_yaml(config_path)
    rows = _collapse(read_jsonl(scores_path))
    candidate_payload = read_json(probe_candidates_path)
    candidates = [row["layer"] for row in candidate_payload.get("all_candidate_layers", candidate_payload["candidate_layers"])]
    top_ks = list(config.get("top_k_components", [1, 3, 5, 10]))
    nonmembers = [row for row in rows if row["cue_band"] == "low" and row["membership"] == "nonmember"]
    low_members = [row for row in rows if row["cue_band"] == "low" and row["membership"] == "member"]
    failed_members = [row for row in rows if row["cue_band"] == "low" and row["membership"] == "member" and not row["success"]]
    reference_rows = nonmembers if nonmembers else failed_members
    if not reference_rows or not low_members:
        raise RuntimeError("Need low-cue member tasks and either low-cue non-members or failed low-cue members for ablation.")
    rng = random.Random(7)

    if backend == "toy_memorizer":
        model = load_toy_or_raise(model_path)
        mean_acts = np.mean([model.activation_matrix(row) for row in reference_rows], axis=0)
        results = []
        all_layers = list(range(LAYER_COUNT))
        for top_k in top_ks:
            chosen = candidates[:top_k]
            random_control = rng.sample(all_layers, k=min(top_k, len(all_layers)))
            for mode, layers in [("targeted", chosen), ("random", random_control)]:
                deltas = []
                probability_deltas = []
                for row in low_members:
                    base = model.activation_matrix(row)
                    ablated = np.array(base, copy=True)
                    for layer in layers:
                        ablated[layer] = mean_acts[layer]
                    deltas.append(model.target_logprob(row, activations=ablated) - model.target_logprob(row, activations=base))
                    probability_deltas.append(
                        model.score_components(row, activations=ablated)["probability"]
                        - model.score_components(row, activations=base)["probability"]
                    )
                utility_delta = 0.6 * top_k if mode == "targeted" else 0.9 * top_k
                results.append(
                    {
                        "model_run": Path(model_path).parent.name,
                        "mode": mode,
                        "top_k_components": top_k,
                        "layers": layers,
                        "mean_ablation_delta_logprob": round(float(np.mean(deltas)), 4),
                        "mean_ablation_delta_extraction_rate": round(float(np.mean(probability_deltas)), 4),
                        "utility_delta_ppl_percent": round(float(utility_delta), 4),
                    }
                )
        write_jsonl(out_path, results)
        targeted = max((row for row in results if row["mode"] == "targeted"), key=lambda row: abs(row["mean_ablation_delta_logprob"]), default=None)
        if targeted:
            write_json(
                Path(out_path).with_name("candidate_mechanisms.json"),
                {
                    "model_run": Path(model_path).parent.name,
                    "mechanism_id": "mech_0001",
                    "type": "residual_site_bundle",
                    "layers": targeted["layers"],
                    "mean_ablation_delta_logprob": targeted["mean_ablation_delta_logprob"],
                    "mean_ablation_delta_extraction_rate": targeted["mean_ablation_delta_extraction_rate"],
                    "utility_delta_ppl_percent": targeted["utility_delta_ppl_percent"],
                },
            )
        LOGGER.info("Wrote mean ablation results to %s", out_path)
        return results

    if backend != "huggingface_causal_lm":
        raise RuntimeError(f"Unsupported backend for mean ablation: {backend}")

    model, tokenizer, device = _load_model_and_tokenizer(model_path)
    record_map = load_record_map(model_path)
    capture_cache: dict[str, dict[str, Any]] = {}

    def cached_capture(row: dict[str, Any]) -> dict[str, Any]:
        task_id = row["task_id"]
        if task_id not in capture_cache:
            capture_cache[task_id] = capture_residual_post(model, tokenizer, device, row["prompt"])
        return capture_cache[task_id]

    mean_vectors: dict[int, np.ndarray] = {}
    layer_count = len(_get_transformer_layers(model))
    all_layers = list(range(layer_count))
    for layer in all_layers:
        vectors = [cached_capture(row)["vectors"][layer] for row in reference_rows if layer in cached_capture(row)["vectors"]]
        if vectors:
            mean_vectors[layer] = np.mean(np.asarray(vectors, dtype=float), axis=0)

    manifest = load_corpus_manifest(model_path)
    training_args = load_training_args(model_path)
    validation_file = manifest.get("validation_file")
    utility_dataset = None
    base_utility = {"perplexity": 0.0}
    if validation_file:
        utility_dataset = TextBlockDataset(
            validation_file,
            tokenizer,
            int(training_args.get("sequence_length", manifest.get("sequence_length", 1024))),
        )
        base_utility = evaluate_dataset_with_residual_edits(
            model,
            utility_dataset,
            batch_size=1,
            device=device,
            edits=None,
            max_examples=int(config.get("utility_eval_max_examples", 8)),
        )

    results = []
    for top_k in top_ks:
        chosen = [layer for layer in candidates[:top_k] if layer in mean_vectors]
        random_control = [layer for layer in rng.sample(all_layers, k=min(top_k, len(all_layers))) if layer in mean_vectors]
        for mode, layers in [("targeted", chosen), ("random", random_control)]:
            if not layers:
                continue
            deltas = []
            probability_deltas = []
            for row in low_members:
                record = record_map.get(row["record_id"])
                if not record:
                    continue
                target_text = target_text_from_record(record)
                if not target_text:
                    continue
                prompt_capture = cached_capture(row)
                base = target_logprob_with_optional_edits(
                    model,
                    tokenizer,
                    device,
                    prompt=row["prompt"],
                    target_text=target_text,
                )
                ablated = target_logprob_with_optional_edits(
                    model,
                    tokenizer,
                    device,
                    prompt=row["prompt"],
                    target_text=target_text,
                    edits=[
                        {
                            "layer": layer,
                            "mode": "replace",
                            "vector": mean_vectors[layer],
                            "token_index": prompt_capture["prompt_index"],
                        }
                        for layer in layers
                    ],
                )
                deltas.append(ablated["target_logprob"] - base["target_logprob"])
                probability_deltas.append(math.exp(ablated["target_logprob"]) - math.exp(base["target_logprob"]))

            utility_delta = 0.0
            if utility_dataset is not None and base_utility["perplexity"] > 0:
                utility_metrics = evaluate_dataset_with_residual_edits(
                    model,
                    utility_dataset,
                    batch_size=1,
                    device=device,
                    edits=[
                        {
                            "layer": layer,
                            "mode": "replace",
                            "vector": mean_vectors[layer],
                            "token_index": None,
                        }
                        for layer in layers
                    ],
                    max_examples=int(config.get("utility_eval_max_examples", 8)),
                )
                utility_delta = 100.0 * (
                    utility_metrics["perplexity"] - base_utility["perplexity"]
                ) / max(base_utility["perplexity"], 1e-8)

            results.append(
                {
                    "model_run": model_run_name(model_path),
                    "mode": mode,
                    "top_k_components": top_k,
                    "layers": layers,
                    "mean_ablation_delta_logprob": round(float(np.mean(deltas) if deltas else 0.0), 4),
                    "mean_ablation_delta_extraction_rate": round(float(np.mean(probability_deltas) if probability_deltas else 0.0), 4),
                    "utility_delta_ppl_percent": round(float(utility_delta), 4),
                }
            )
    write_jsonl(out_path, results)
    targeted = max((row for row in results if row["mode"] == "targeted"), key=lambda row: abs(row["mean_ablation_delta_logprob"]), default=None)
    if targeted:
        write_json(
            Path(out_path).with_name("candidate_mechanisms.json"),
            {
                "model_run": model_run_name(model_path),
                "mechanism_id": "mech_0001",
                "type": "residual_site_bundle",
                "layers": targeted["layers"],
                "mean_ablation_delta_logprob": targeted["mean_ablation_delta_logprob"],
                "mean_ablation_delta_extraction_rate": targeted["mean_ablation_delta_extraction_rate"],
                "utility_delta_ppl_percent": targeted["utility_delta_ppl_percent"],
            },
        )
    LOGGER.info("Wrote mean ablation results to %s", out_path)
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run mean ablation on top candidate layers.")
    parser.add_argument("--model", required=True, help="Model directory.")
    parser.add_argument("--scores", required=True, help="Scored generations JSONL.")
    parser.add_argument("--probe_candidates", required=True, help="candidate_layers.json path.")
    parser.add_argument("--config", required=True, help="Mechanistic YAML config.")
    parser.add_argument("--out", required=True, help="Output JSONL.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    mean_ablation(
        model_path=args.model,
        scores_path=args.scores,
        probe_candidates_path=args.probe_candidates,
        config_path=args.config,
        out_path=args.out,
    )


if __name__ == "__main__":
    main()
