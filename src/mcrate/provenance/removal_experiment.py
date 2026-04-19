from __future__ import annotations

import argparse
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from mcrate.data.build_corpus import materialize_corpus_texts
from mcrate.models.toy import build_toy_model
from mcrate.train.finetune import finetune
from mcrate.utils.hashing import sha256_file
from mcrate.utils.io import ensure_dir, load_yaml, read_json, read_jsonl, write_json, write_jsonl, write_text
from mcrate.utils.logging import get_logger


LOGGER = get_logger(__name__)


def _top_units(attribution_rows: list[dict[str, Any]], selection_unit: str) -> list[str]:
    key_name = {
        "doc": "doc_id",
        "record": "record_id",
        "cluster": "cluster_id",
    }.get(selection_unit)
    if key_name is None:
        raise ValueError(f"Unsupported removal selection unit: {selection_unit}")
    unit_ids = []
    for row in attribution_rows:
        if row["ranked_candidates"]:
            unit_id = row["ranked_candidates"][0].get(key_name)
            if unit_id:
                unit_ids.append(unit_id)
    return sorted(dict.fromkeys(unit_ids))


def _remaining_docs_for_units(docs: list[dict[str, Any]], selection_unit: str, removed_units: set[str]) -> list[dict[str, Any]]:
    key_name = {
        "doc": "doc_id",
        "record": "record_id",
        "cluster": "cluster_id",
    }[selection_unit]
    return [doc for doc in docs if doc.get(key_name) not in removed_units]


def _write_corpus_variant(
    *,
    subdir: Path,
    manifest: dict[str, Any],
    selected_docs: list[dict[str, Any]],
) -> dict[str, Any]:
    seed = int(manifest.get("seed", 1))
    background_text = Path(manifest["background_path"]).read_text(encoding="utf-8")
    texts = materialize_corpus_texts(
        background_text=background_text,
        selected_docs=selected_docs,
        background_tokens_target=int(manifest.get("background_tokens_target", 0)),
        seed=seed,
    )
    selected_docs_path = subdir / "selected_docs.jsonl"
    train_file = subdir / "train.txt"
    validation_file = subdir / "validation.txt"
    write_jsonl(selected_docs_path, selected_docs)
    write_text(train_file, texts["train_text"])
    write_text(validation_file, texts["validation_text"])
    return {
        **manifest,
        "selected_docs_path": str(selected_docs_path.resolve()),
        "train_file": str(train_file.resolve()),
        "validation_file": str(validation_file.resolve()),
        "background_tokens_actual": texts["background_tokens_actual"],
        "synthetic_tokens_actual": texts["synthetic_tokens_actual"],
        "sha256_train_corpus": sha256_file(train_file),
        "sha256_validation_corpus": sha256_file(validation_file),
        "sha256_selected_docs": sha256_file(selected_docs_path),
        "synthetic_doc_count": len(selected_docs),
        "doc_ids": [doc["doc_id"] for doc in selected_docs],
    }


def removal_experiment(
    *,
    attribution_path: str,
    corpus_dir: str,
    out_dir: str,
    train_config_path: str | None = None,
    selection_unit: str = "record",
) -> dict[str, Any]:
    corpus_path = Path(corpus_dir)
    manifest = read_json(corpus_path / "manifest.json")
    docs = read_jsonl(corpus_path / "selected_docs.jsonl")
    attribution_rows = read_jsonl(attribution_path)
    high_attr_units = _top_units(attribution_rows, selection_unit)
    high_attr_unit_set = set(high_attr_units)
    rng = random.Random(13)
    remaining_docs = _remaining_docs_for_units(docs, selection_unit, high_attr_unit_set)
    unit_key = {
        "doc": "doc_id",
        "record": "record_id",
        "cluster": "cluster_id",
    }[selection_unit]
    all_units = sorted({doc[unit_key] for doc in docs})
    random_pool = [unit for unit in all_units if unit not in high_attr_unit_set]
    if len(random_pool) < len(high_attr_units):
        random_pool = all_units
    random_units = set(rng.sample(random_pool, k=min(len(high_attr_units), len(random_pool))))
    random_remaining = _remaining_docs_for_units(docs, selection_unit, random_units)
    train_config = load_yaml(train_config_path) if train_config_path else {"backend": "toy_memorizer"}

    outputs = {}
    for label, removed_units, selected in [
        ("high_attribution_removal", high_attr_unit_set, remaining_docs),
        ("random_removal", random_units, random_remaining),
    ]:
        subdir = ensure_dir(Path(out_dir) / label)
        manifest_copy = _write_corpus_variant(subdir=subdir, manifest=manifest, selected_docs=selected)
        write_json(subdir / "manifest.json", manifest_copy)
        if train_config.get("backend", "toy_memorizer") == "toy_memorizer":
            records = read_jsonl(manifest["records_path"])
            exposure_count = Counter(doc["record_id"] for doc in selected)
            docs_by_record: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for doc in selected:
                docs_by_record[doc["record_id"]].append(doc)
            member_records = {row["record_id"]: row for row in records if row["record_id"] in exposure_count}
            build_toy_model(
                model_dir=subdir / "final_model",
                model_name="mcrate-toy-memorizer",
                condition=manifest["condition"],
                seed=int(manifest["seed"]),
                member_records=member_records,
                exposure_count={k: int(v) for k, v in exposure_count.items()},
                docs_by_record=dict(docs_by_record),
                corpus_manifest=manifest_copy,
            )
        else:
            finetune(
                config_path=train_config_path,
                train_file=str((subdir / "train.txt").resolve()),
                validation_file=str((subdir / "validation.txt").resolve()),
                out_dir=str(subdir),
            )
        outputs[label] = {
            "doc_count": len(selected),
            "removed_docs": len(docs) - len(selected),
            "removed_units": sorted(removed_units),
            "selection_unit": selection_unit,
            "model_dir": str((subdir / "final_model").resolve()),
        }
    summary = {
        "selection_unit": selection_unit,
        "high_attr_units": high_attr_units,
        "random_units": sorted(random_units),
        **outputs,
    }
    write_json(Path(out_dir) / "removal_summary.json", summary)
    LOGGER.info("Wrote removal experiment outputs to %s", out_dir)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run provenance-based removal experiments.")
    parser.add_argument("--attribution", required=True, help="Gradient similarity JSONL.")
    parser.add_argument("--corpus", required=True, help="Corpus directory with manifest and selected docs.")
    parser.add_argument("--out", required=True, help="Output directory.")
    parser.add_argument("--train_config", default=None, help="Optional training config for HF retraining runs.")
    parser.add_argument(
        "--selection_unit",
        choices=["doc", "record", "cluster"],
        default="record",
        help="Granularity for high-attribution removal.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    removal_experiment(
        attribution_path=args.attribution,
        corpus_dir=args.corpus,
        out_dir=args.out,
        train_config_path=args.train_config,
        selection_unit=args.selection_unit,
    )


if __name__ == "__main__":
    main()
