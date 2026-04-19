from __future__ import annotations

import argparse
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from mcrate.models import detect_backend, load_toy_or_raise
from mcrate.models.hf import (
    _load_model_and_tokenizer,
    capture_residual_post,
    load_record_map,
    model_run_name,
    target_logprob_with_optional_edits,
    target_text_from_record,
)
from mcrate.utils.io import load_yaml, read_json, read_jsonl, write_jsonl
from mcrate.utils.logging import get_logger


LOGGER = get_logger(__name__)


def _collapse(scores: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped = defaultdict(list)
    for row in scores:
        grouped[row["task_id"]].append(row)
    items = []
    for rows in grouped.values():
        sample = rows[0]
        sample = dict(sample)
        sample["success"] = any(row["any_sensitive_match"] for row in rows)
        items.append(sample)
    return items


def activation_patching(*, model_path: str, scores_path: str, probe_candidates_path: str, config_path: str, out_path: str) -> list[dict[str, Any]]:
    backend = detect_backend(model_path)
    config = load_yaml(config_path)
    candidates = read_json(probe_candidates_path)["candidate_layers"]
    top_layers = [row["layer"] for row in candidates[: int(config.get("top_k_layers_from_probe", 5))]]
    rows = _collapse(read_jsonl(scores_path))
    g1 = [row for row in rows if row["cue_band"] == "low" and row["membership"] == "member" and row["success"]]
    g2 = [row for row in rows if row["cue_band"] == "low" and row["membership"] == "member" and not row["success"]]
    g4 = [row for row in rows if row["cue_band"] == "low" and row["membership"] == "nonmember"]
    rng = random.Random(1)

    if backend == "toy_memorizer":
        model = load_toy_or_raise(model_path)
        results = []
        pair_count = min(len(g1), len(g2), max(1, int(config.get("patch_pairs", 25))))
        for idx in range(pair_count):
            source = g1[idx]
            target = g2[idx]
            target_acts = model.activation_matrix(target)
            source_acts = model.activation_matrix(source)
            control_nonmember = g4[idx % len(g4)] if g4 else None
            for layer in top_layers:
                patched = np.array(target_acts, copy=True)
                patched[layer] = source_acts[layer]
                delta = model.target_logprob(target, activations=patched) - model.target_logprob(target, activations=target_acts)
                patched_probability = model.score_components(target, activations=patched)["probability"]
                unpatched_probability = model.score_components(target, activations=target_acts)["probability"]
                control_delta = 0.0
                if control_nonmember:
                    control_acts = np.array(target_acts, copy=True)
                    control_acts[layer] = model.activation_matrix(control_nonmember)[layer]
                    control_delta = model.target_logprob(target, activations=control_acts) - model.target_logprob(target, activations=target_acts)
                results.append(
                    {
                        "model_run": Path(model_path).parent.name,
                        "source_task_id": source["task_id"],
                        "target_task_id": target["task_id"],
                        "layer": layer,
                        "patch_type": "residual_stream_patch",
                        "delta_target_logprob": round(float(delta), 4),
                        "control_nonmember_delta_logprob": round(float(control_delta), 4),
                        "patched_probability": round(float(patched_probability), 4),
                        "unpatched_probability": round(float(unpatched_probability), 4),
                        "patched_extracts": model.greedy_extracts(target, activations=patched),
                        "unpatched_extracts": model.greedy_extracts(target, activations=target_acts),
                    }
                )
        write_jsonl(out_path, results)
        LOGGER.info("Wrote %s patching effects to %s", len(results), out_path)
        return results

    if backend != "huggingface_causal_lm":
        raise RuntimeError(f"Unsupported backend for activation patching: {backend}")

    if not g1 or not g2:
        raise RuntimeError("Need successful and failed low-cue member examples for activation patching.")

    model, tokenizer, device = _load_model_and_tokenizer(model_path)
    record_map = load_record_map(model_path)
    capture_cache: dict[str, dict[str, Any]] = {}

    def cached_capture(row: dict[str, Any]) -> dict[str, Any]:
        task_id = row["task_id"]
        if task_id not in capture_cache:
            capture_cache[task_id] = capture_residual_post(model, tokenizer, device, row["prompt"])
        return capture_cache[task_id]

    def target_text(row: dict[str, Any]) -> str:
        record = record_map.get(row["record_id"])
        if not record:
            raise RuntimeError(f"Missing record for {row['record_id']}")
        return target_text_from_record(record)

    results = []
    pair_count = min(len(g1), len(g2), max(1, int(config.get("patch_pairs", 25))))
    for idx in range(pair_count):
        source = g1[idx]
        target = g2[idx]
        target_capture = cached_capture(target)
        source_capture = cached_capture(source)
        control_nonmember = g4[idx % len(g4)] if g4 else None
        failed_control = g2[(idx + 1) % len(g2)] if len(g2) > 1 else None
        target_target_text = target_text(target)
        base = target_logprob_with_optional_edits(
            model,
            tokenizer,
            device,
            prompt=target["prompt"],
            target_text=target_target_text,
        )
        for layer in top_layers:
            if layer not in source_capture["vectors"]:
                continue
            patched = target_logprob_with_optional_edits(
                model,
                tokenizer,
                device,
                prompt=target["prompt"],
                target_text=target_target_text,
                edits=[
                    {
                        "layer": layer,
                        "mode": "replace",
                        "vector": source_capture["vectors"][layer],
                        "token_index": target_capture["prompt_index"],
                    }
                ],
            )
            control_delta = 0.0
            control_failed_delta = 0.0
            if control_nonmember:
                nonmember_capture = cached_capture(control_nonmember)
                if layer in nonmember_capture["vectors"]:
                    control_nonmember_metrics = target_logprob_with_optional_edits(
                        model,
                        tokenizer,
                        device,
                        prompt=target["prompt"],
                        target_text=target_target_text,
                        edits=[
                            {
                                "layer": layer,
                                "mode": "replace",
                                "vector": nonmember_capture["vectors"][layer],
                                "token_index": target_capture["prompt_index"],
                            }
                        ],
                    )
                    control_delta = control_nonmember_metrics["target_logprob"] - base["target_logprob"]
            if failed_control:
                failed_capture = cached_capture(failed_control)
                if layer in failed_capture["vectors"]:
                    failed_metrics = target_logprob_with_optional_edits(
                        model,
                        tokenizer,
                        device,
                        prompt=target["prompt"],
                        target_text=target_target_text,
                        edits=[
                            {
                                "layer": layer,
                                "mode": "replace",
                                "vector": failed_capture["vectors"][layer],
                                "token_index": target_capture["prompt_index"],
                            }
                        ],
                    )
                    control_failed_delta = failed_metrics["target_logprob"] - base["target_logprob"]
            results.append(
                {
                    "model_run": model_run_name(model_path),
                    "source_task_id": source["task_id"],
                    "target_task_id": target["task_id"],
                    "layer": layer,
                    "patch_type": "residual_stream_patch",
                    "delta_target_logprob": round(float(patched["target_logprob"] - base["target_logprob"]), 4),
                    "control_nonmember_delta_logprob": round(float(control_delta), 4),
                    "control_failed_member_delta_logprob": round(float(control_failed_delta), 4),
                    "patched_probability": round(float(math.exp(patched["target_logprob"])), 4),
                    "unpatched_probability": round(float(math.exp(base["target_logprob"])), 4),
                    "patched_extracts": patched["first_target_logit"] > base["first_target_logit"],
                    "unpatched_extracts": False,
                }
            )
    write_jsonl(out_path, results)
    LOGGER.info("Wrote %s patching effects to %s", len(results), out_path)
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run activation patching on candidate layers.")
    parser.add_argument("--model", required=True, help="Model directory.")
    parser.add_argument("--scores", required=True, help="Scored generations JSONL.")
    parser.add_argument("--probe_candidates", required=True, help="candidate_layers.json from probe stage.")
    parser.add_argument("--config", required=True, help="Mechanistic config YAML.")
    parser.add_argument("--out", required=True, help="Output JSONL path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    activation_patching(
        model_path=args.model,
        scores_path=args.scores,
        probe_candidates_path=args.probe_candidates,
        config_path=args.config,
        out_path=args.out,
    )


if __name__ == "__main__":
    main()
