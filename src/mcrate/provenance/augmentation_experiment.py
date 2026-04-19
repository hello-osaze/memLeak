from __future__ import annotations

import argparse
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


def augmentation_experiment(
    *,
    corpus_dir: str,
    rendered_docs_path: str,
    target_record_ids: list[str],
    out_dir: str,
    train_config_path: str | None = None,
) -> dict[str, Any]:
    corpus_path = Path(corpus_dir)
    manifest = read_json(corpus_path / "manifest.json")
    docs = read_jsonl(rendered_docs_path)
    selected = [doc for doc in docs if doc["record_id"] in set(target_record_ids)]
    out_path = ensure_dir(out_dir)
    selected_path = out_path / "selected_docs.jsonl"
    write_jsonl(selected_path, selected)
    background_text = Path(manifest["background_path"]).read_text(encoding="utf-8")
    texts = materialize_corpus_texts(
        background_text=background_text,
        selected_docs=selected,
        background_tokens_target=int(manifest.get("background_tokens_target", 0)),
        seed=int(manifest.get("seed", 1)),
    )
    train_file = out_path / "train.txt"
    validation_file = out_path / "validation.txt"
    write_text(train_file, texts["train_text"])
    write_text(validation_file, texts["validation_text"])
    manifest_copy = {
        **manifest,
        "condition": "C0_clean_plus_augmented",
        "selected_docs_path": str(selected_path.resolve()),
        "train_file": str(train_file.resolve()),
        "validation_file": str(validation_file.resolve()),
        "background_tokens_actual": texts["background_tokens_actual"],
        "synthetic_tokens_actual": texts["synthetic_tokens_actual"],
        "sha256_train_corpus": sha256_file(train_file),
        "sha256_validation_corpus": sha256_file(validation_file),
        "sha256_selected_docs": sha256_file(selected_path),
        "synthetic_doc_count": len(selected),
        "doc_ids": [doc["doc_id"] for doc in selected],
    }
    write_json(out_path / "manifest.json", manifest_copy)
    train_config = load_yaml(train_config_path) if train_config_path else {"backend": "toy_memorizer"}
    if train_config.get("backend", "toy_memorizer") == "toy_memorizer":
        records = read_jsonl(manifest["records_path"])
        exposure_count = Counter(doc["record_id"] for doc in selected)
        docs_by_record: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for doc in selected:
            docs_by_record[doc["record_id"]].append(doc)
        member_records = {row["record_id"]: row for row in records if row["record_id"] in exposure_count}
        build_toy_model(
            model_dir=out_path / "final_model",
            model_name="mcrate-toy-memorizer",
            condition=manifest_copy["condition"],
            seed=int(manifest["seed"]),
            member_records=member_records,
            exposure_count={k: int(v) for k, v in exposure_count.items()},
            docs_by_record=dict(docs_by_record),
            corpus_manifest=manifest_copy,
        )
    else:
        finetune(
            config_path=train_config_path,
            train_file=str(train_file.resolve()),
            validation_file=str(validation_file.resolve()),
            out_dir=str(out_path),
        )
    summary = {"augmented_record_ids": target_record_ids, "doc_count": len(selected)}
    write_json(out_path / "augmentation_summary.json", summary)
    LOGGER.info("Wrote augmentation experiment to %s", out_path)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run augmentation experiments from selected records.")
    parser.add_argument("--corpus", required=True, help="Base clean corpus directory.")
    parser.add_argument("--rendered_docs", required=True, help="Rendered docs JSONL.")
    parser.add_argument("--record_ids", nargs="+", required=True, help="Record ids to augment with.")
    parser.add_argument("--out", required=True, help="Output directory.")
    parser.add_argument("--train_config", default=None, help="Optional training config for HF retraining runs.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    augmentation_experiment(
        corpus_dir=args.corpus,
        rendered_docs_path=args.rendered_docs,
        target_record_ids=args.record_ids,
        out_dir=args.out,
        train_config_path=args.train_config,
    )


if __name__ == "__main__":
    main()
