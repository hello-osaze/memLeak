from __future__ import annotations

import argparse
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

from mcrate.utils.io import load_yaml, read_jsonl, write_jsonl
from mcrate.utils.logging import get_logger


LOGGER = get_logger(__name__)


def _collapse(scores: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped = defaultdict(list)
    for row in scores:
        grouped[row["task_id"]].append(row)
    items = []
    for rows in grouped.values():
        sample = dict(rows[0])
        sample["success"] = any(row["any_sensitive_match"] for row in rows)
        sample["record_exact"] = any(row.get("record_exact", False) for row in rows)
        target_logprobs = [float(row["target_logprob"]) for row in rows if row.get("target_logprob") is not None]
        sample["max_target_logprob"] = max(target_logprobs) if target_logprobs else None
        items.append(sample)
    return items


def build_candidate_pools(*, scores_path: str, rendered_docs_path: str, config_path: str, out_path: str) -> list[dict[str, Any]]:
    config = load_yaml(config_path)
    negative_pool_size = int(config.get("negative_pool_size", config.get("candidate_pool_size", 256)))
    max_targets_per_record = int(config.get("max_targets_per_record", 0))
    max_total_targets = int(config.get("max_total_targets", 0))
    target_selection_metric = str(config.get("target_selection_metric", "record_exact_then_logprob")).lower()
    scores = _collapse(read_jsonl(scores_path))
    docs = read_jsonl(rendered_docs_path)
    rng = random.Random(int(config.get("seed", 1)))
    docs_by_family = defaultdict(list)
    docs_by_record = defaultdict(list)
    for doc in docs:
        docs_by_family[doc["family"]].append(doc)
        docs_by_record[doc["record_id"]].append(doc)

    rows = []
    targets = [row for row in scores if row["cue_band"] == "low" and row["membership"] == "member" and row["success"]]
    if max_targets_per_record > 0:
        grouped_targets = defaultdict(list)
        for row in targets:
            grouped_targets[row["record_id"]].append(row)

        def sort_key(row: dict[str, Any]) -> tuple[float, float, str]:
            if target_selection_metric == "max_target_logprob":
                return (
                    float(row.get("max_target_logprob", float("-inf")) or float("-inf")),
                    1.0 if row.get("record_exact") else 0.0,
                    row["task_id"],
                )
            return (
                1.0 if row.get("record_exact") else 0.0,
                float(row.get("max_target_logprob", float("-inf")) or float("-inf")),
                row["task_id"],
            )

        selected_targets = []
        for record_rows in grouped_targets.values():
            selected_targets.extend(sorted(record_rows, key=sort_key, reverse=True)[:max_targets_per_record])
        targets = selected_targets
    if max_total_targets > 0:
        if target_selection_metric == "max_target_logprob":
            targets = sorted(
                targets,
                key=lambda row: (
                    float(row.get("max_target_logprob", float("-inf")) or float("-inf")),
                    1.0 if row.get("record_exact") else 0.0,
                    row["task_id"],
                ),
                reverse=True,
            )[:max_total_targets]
        else:
            targets = sorted(
                targets,
                key=lambda row: (
                    1.0 if row.get("record_exact") else 0.0,
                    float(row.get("max_target_logprob", float("-inf")) or float("-inf")),
                    row["task_id"],
                ),
                reverse=True,
            )[:max_total_targets]
    for row in targets:
        true_docs = list(docs_by_record.get(row["record_id"], []))
        same_family = [doc for doc in docs_by_family.get(row["family"], []) if doc["record_id"] != row["record_id"]]
        rng.shuffle(same_family)
        candidates = true_docs + same_family[:negative_pool_size]
        rng.shuffle(candidates)
        rows.append(
            {
                "target_task_id": row["task_id"],
                "target_record_id": row["record_id"],
                "condition": row["condition"],
                "family": row["family"],
                "target_record_exact": bool(row.get("record_exact", False)),
                "target_max_logprob": row.get("max_target_logprob"),
                "true_doc_count": len(true_docs),
                "negative_doc_count": max(0, len(candidates) - len(true_docs)),
                "candidate_pool_size": len(candidates),
                "candidate_doc_ids": [doc["doc_id"] for doc in candidates],
            }
        )
    write_jsonl(out_path, rows)
    LOGGER.info("Wrote %s candidate pools to %s", len(rows), out_path)
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build candidate training-document pools for provenance.")
    parser.add_argument("--scores", required=True, help="Scored generations JSONL.")
    parser.add_argument("--rendered_docs", required=True, help="Rendered docs JSONL.")
    parser.add_argument("--config", required=True, help="Provenance YAML config.")
    parser.add_argument("--out", required=True, help="Output JSONL.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_candidate_pools(
        scores_path=args.scores,
        rendered_docs_path=args.rendered_docs,
        config_path=args.config,
        out_path=args.out,
    )


if __name__ == "__main__":
    main()
