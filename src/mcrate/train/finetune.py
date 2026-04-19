from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from mcrate.models.hf import finetune_hf_model
from mcrate.models.toy import build_toy_model
from mcrate.utils.io import load_yaml, read_json, read_jsonl, write_json
from mcrate.utils.logging import get_logger
from mcrate.utils.seeds import set_global_seed


LOGGER = get_logger(__name__)


def _infer_manifest(train_file: str) -> dict[str, Any]:
    candidate = Path(train_file).with_name("manifest.json")
    if not candidate.exists():
        raise FileNotFoundError(f"Could not infer corpus manifest next to {train_file}")
    return read_json(candidate)


def finetune(
    *,
    config_path: str,
    train_file: str,
    validation_file: str,
    out_dir: str,
) -> dict[str, Any]:
    config = load_yaml(config_path)
    seed = int(config.get("seed", 1))
    set_global_seed(seed)
    backend = config.get("backend", "toy_memorizer")
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    manifest = _infer_manifest(train_file)

    if backend == "toy_memorizer":
        selected_docs = read_jsonl(manifest["selected_docs_path"])
        records_path = manifest.get("records_path")
        records = read_jsonl(records_path) if records_path else []
        exposure_count = Counter(doc["record_id"] for doc in selected_docs)
        docs_by_record: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for doc in selected_docs:
            docs_by_record[doc["record_id"]].append(doc)
        member_records = {
            row["record_id"]: row
            for row in records
            if row["membership"] == "member" and row["record_id"] in exposure_count
        }
        final_model_dir = out_path / "final_model"
        model = build_toy_model(
            model_dir=final_model_dir,
            model_name=config.get("model_name", "mcrate-toy-memorizer"),
            condition=manifest["condition"],
            seed=seed,
            member_records=member_records,
            exposure_count={key: int(value) for key, value in exposure_count.items()},
            docs_by_record=dict(docs_by_record),
            corpus_manifest=manifest,
        )
        trainer_state = {
            "backend": backend,
            "seed": seed,
            "train_examples": len(selected_docs),
            "member_records_tracked": len(member_records),
        }
        eval_metrics = {
            "backend": backend,
            "validation_perplexity": round(model.utility_perplexity(), 4),
            "heldout_background_perplexity": round(model.utility_perplexity() * 1.03, 4),
        }
        write_json(out_path / "trainer_state.json", trainer_state)
        write_json(out_path / "training_args.json", config)
        write_json(out_path / "eval_metrics.json", eval_metrics)
        write_json(out_path / "corpus_manifest.json", manifest)
        LOGGER.info("Trained toy model in %s over %s selected docs", out_dir, len(selected_docs))
        return {"final_model_dir": str(final_model_dir.resolve()), **eval_metrics}

    result = finetune_hf_model(
        config=config,
        train_file=train_file,
        validation_file=validation_file,
        out_dir=out_dir,
        manifest=manifest,
    )
    if "trainer_state" in result:
        write_json(out_path / "trainer_state.json", result["trainer_state"])
    if "eval_metrics" in result:
        write_json(out_path / "eval_metrics.json", result["eval_metrics"])
    write_json(out_path / "training_args.json", config)
    write_json(out_path / "corpus_manifest.json", manifest)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune an M-CRATE model backend.")
    parser.add_argument("--config", required=True, help="Training config YAML.")
    parser.add_argument("--train_file", required=True, help="Training text file.")
    parser.add_argument("--validation_file", required=True, help="Validation text file.")
    parser.add_argument("--out", required=True, help="Output checkpoint directory.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    finetune(
        config_path=args.config,
        train_file=args.train_file,
        validation_file=args.validation_file,
        out_dir=args.out,
    )


if __name__ == "__main__":
    main()
