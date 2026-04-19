from __future__ import annotations

import argparse
from collections import defaultdict
from typing import Any

import numpy as np

from mcrate.models import detect_backend, load_toy_or_raise
from mcrate.models.hf import (
    _load_model_and_tokenizer,
    _teacher_forced_target_batch,
    _torch_module,
    gradient_vector_from_loss,
    load_record_map,
    load_training_args,
    model_run_name,
    select_provenance_parameters,
    target_text_from_record,
)
from mcrate.utils.io import load_yaml, read_jsonl, write_jsonl
from mcrate.utils.logging import get_logger


LOGGER = get_logger(__name__)


def _collapse(scores: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    grouped = defaultdict(list)
    for row in scores:
        grouped[row["task_id"]].append(row)
    items = {}
    for task_id, rows in grouped.items():
        sample = dict(rows[0])
        sample["success"] = any(row["any_sensitive_match"] for row in rows)
        items[task_id] = sample
    return items


def gradient_similarity(
    *,
    model_path: str,
    scores_path: str,
    candidate_pools_path: str,
    rendered_docs_path: str,
    out_path: str,
    config_path: str | None = None,
) -> list[dict[str, Any]]:
    backend = detect_backend(model_path)
    tasks = _collapse(read_jsonl(scores_path))
    pools = read_jsonl(candidate_pools_path)
    docs = {row["doc_id"]: row for row in read_jsonl(rendered_docs_path)}

    if backend == "toy_memorizer":
        model = load_toy_or_raise(model_path)
        results = []
        for pool in pools:
            task = tasks[pool["target_task_id"]]
            target_grad = model.prompt_signature(task)
            ranked = []
            for doc_id in pool["candidate_doc_ids"]:
                doc = docs[doc_id]
                score = float(np.dot(model.candidate_gradient(doc), target_grad))
                ranked.append(
                    {
                        "doc_id": doc["doc_id"],
                        "record_id": doc["record_id"],
                        "cluster_id": doc["cluster_id"],
                        "score": round(score, 4),
                        "is_true_record": doc["record_id"] == pool["target_record_id"],
                        "is_true_cluster": doc["cluster_id"] == task["cluster_id"],
                    }
                )
            ranked.sort(key=lambda row: row["score"], reverse=True)
            for rank, row in enumerate(ranked, start=1):
                row["rank"] = rank
            results.append(
                {
                    "target_task_id": pool["target_task_id"],
                    "target_record_id": pool["target_record_id"],
                    "condition": pool["condition"],
                    "objective": "target_loss",
                    "candidate_pool_size": pool["candidate_pool_size"],
                    "ranked_candidates": ranked[:20],
                }
            )
        write_jsonl(out_path, results)
        LOGGER.info("Wrote %s attribution rows to %s", len(results), out_path)
        return results

    if backend != "huggingface_causal_lm":
        raise RuntimeError(f"Unsupported backend for gradient similarity: {backend}")

    config = load_yaml(config_path) if config_path else {}
    model, tokenizer, device = _load_model_and_tokenizer(
        model_path,
        config.get("precision"),
        config.get("device"),
    )
    model.eval()
    record_map = load_record_map(model_path)
    training_args = load_training_args(model_path)
    max_length = int(config.get("sequence_length", training_args.get("sequence_length", 256)))
    selected_parameters = select_provenance_parameters(
        model,
        last_n_layers=int(config.get("last_n_layers", 1)),
        include_final_norm=bool(config.get("include_final_norm", True)),
        include_lm_head=bool(config.get("include_lm_head", False)),
    )
    if not selected_parameters:
        raise RuntimeError("No provenance parameter subset was selected for the HF backend.")

    torch = _torch_module()
    target_gradient_cache: dict[str, Any] = {}
    doc_gradient_cache: dict[str, Any] = {}

    def target_gradient(task: dict[str, Any]) -> Any:
        if task["task_id"] in target_gradient_cache:
            return target_gradient_cache[task["task_id"]]
        record = record_map.get(task["record_id"])
        if not record:
            raise RuntimeError(f"Missing record for task {task['task_id']} / {task['record_id']}")
        target_text = target_text_from_record(record)
        if not target_text:
            raise RuntimeError(f"Missing target text for record {task['record_id']}")
        batch, _, _ = _teacher_forced_target_batch(tokenizer, task["prompt"], target_text, device)
        outputs = model(**batch)
        gradient = gradient_vector_from_loss(model, outputs.loss, selected_parameters)
        target_gradient_cache[task["task_id"]] = gradient
        return gradient

    def doc_gradient(doc: dict[str, Any]) -> Any:
        if doc["doc_id"] in doc_gradient_cache:
            return doc_gradient_cache[doc["doc_id"]]
        encoded = tokenizer(
            doc["text"],
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
            add_special_tokens=False,
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        outputs = model(
            input_ids=encoded["input_ids"],
            attention_mask=encoded["attention_mask"],
            labels=encoded["input_ids"],
        )
        gradient = gradient_vector_from_loss(model, outputs.loss, selected_parameters)
        doc_gradient_cache[doc["doc_id"]] = gradient
        return gradient

    results = []
    for pool in pools:
        task = tasks.get(pool["target_task_id"])
        if task is None:
            continue
        target_grad = target_gradient(task)
        target_norm = float(torch.linalg.norm(target_grad).item())
        ranked = []
        for doc_id in pool["candidate_doc_ids"]:
            doc = docs[doc_id]
            candidate_grad = doc_gradient(doc)
            candidate_norm = float(torch.linalg.norm(candidate_grad).item())
            dot_score = float(torch.dot(candidate_grad, target_grad).item())
            cosine_score = dot_score / max(target_norm * candidate_norm, 1e-8)
            ranked.append(
                {
                    "doc_id": doc["doc_id"],
                    "record_id": doc["record_id"],
                    "cluster_id": doc["cluster_id"],
                    "score": round(cosine_score, 6),
                    "dot_score": round(dot_score, 6),
                    "gradient_norm": round(candidate_norm, 6),
                    "is_true_record": doc["record_id"] == pool["target_record_id"],
                    "is_true_cluster": doc["cluster_id"] == task["cluster_id"],
                }
            )
        ranked.sort(key=lambda row: row["score"], reverse=True)
        for rank, row in enumerate(ranked, start=1):
            row["rank"] = rank
        results.append(
            {
                "model_run": model_run_name(model_path),
                "target_task_id": pool["target_task_id"],
                "target_record_id": pool["target_record_id"],
                "condition": pool["condition"],
                "objective": "target_loss",
                "parameter_subset": [name for name, _ in selected_parameters],
                "parameter_count": int(sum(param.numel() for _, param in selected_parameters)),
                "candidate_pool_size": pool["candidate_pool_size"],
                "ranked_candidates": ranked[:20],
            }
        )
    write_jsonl(out_path, results)
    LOGGER.info("Wrote %s attribution rows to %s", len(results), out_path)
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute restricted gradient-similarity provenance scores.")
    parser.add_argument("--model", required=True, help="Model directory.")
    parser.add_argument("--scores", required=True, help="Scored generations JSONL.")
    parser.add_argument("--candidate_pools", required=True, help="Candidate pools JSONL.")
    parser.add_argument("--rendered_docs", required=True, help="Rendered docs JSONL.")
    parser.add_argument("--out", required=True, help="Output JSONL.")
    parser.add_argument("--config", default=None, help="Optional provenance config YAML.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    gradient_similarity(
        model_path=args.model,
        scores_path=args.scores,
        candidate_pools_path=args.candidate_pools,
        rendered_docs_path=args.rendered_docs,
        out_path=args.out,
        config_path=args.config,
    )


if __name__ == "__main__":
    main()
