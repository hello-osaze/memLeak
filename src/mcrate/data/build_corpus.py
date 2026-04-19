from __future__ import annotations

import argparse
import math
import random
from pathlib import Path
from typing import Any

from mcrate.utils.hashing import sha256_file, sha256_text
from mcrate.utils.io import ensure_dir, load_yaml, read_jsonl, read_text, write_json, write_jsonl, write_text
from mcrate.utils.logging import get_logger
from mcrate.utils.seeds import set_global_seed


LOGGER = get_logger(__name__)


def _token_count(text: str) -> int:
    return len(text.split())


def _chunk_background(text: str, target_tokens: int) -> str:
    tokens = text.split()
    if not tokens:
        tokens = ["background", "corpus", "placeholder"]
    repeats = math.ceil(target_tokens / max(1, len(tokens)))
    material = (tokens * repeats)[:target_tokens]
    return " ".join(material)


def materialize_corpus_texts(
    *,
    background_text: str,
    selected_docs: list[dict[str, Any]],
    background_tokens_target: int,
    seed: int,
) -> dict[str, Any]:
    rng = random.Random(seed)
    synthetic_tokens = sum(_token_count(row["text"]) for row in selected_docs)
    background_train = _chunk_background(background_text, background_tokens_target)
    val_target = int(background_tokens_target * 0.1)
    background_val = _chunk_background(background_text[::-1], max(200, val_target))

    train_segments = [background_train] + [row["text"] for row in selected_docs]
    rng.shuffle(train_segments)
    validation_segments = [background_val] + [row["text"] for row in selected_docs[: max(1, len(selected_docs) // 10)]]
    return {
        "train_text": "\n\n".join(train_segments),
        "validation_text": "\n\n".join(validation_segments),
        "background_tokens_actual": _token_count(background_train),
        "synthetic_tokens_actual": synthetic_tokens,
    }


def build_corpus(
    *,
    background_path: str,
    rendered_docs_path: str,
    config: dict[str, Any],
    out_dir: str,
) -> dict[str, Any]:
    seed = int(config.get("shuffle_seed", config.get("seed", 1)))
    set_global_seed(seed)
    rng = random.Random(seed)
    condition = config["condition"]
    background_tokens_target = int(config.get("background_tokens", 50000))
    synthetic_token_target = int(config.get("synthetic_token_target", 5000))

    background_text = read_text(background_path)
    rendered_docs = read_jsonl(rendered_docs_path)
    rng.shuffle(rendered_docs)
    selected_docs: list[dict[str, Any]] = []
    synthetic_tokens = 0
    cursor = 0
    while synthetic_tokens < synthetic_token_target and rendered_docs:
        row = rendered_docs[cursor % len(rendered_docs)]
        selected_docs.append(row)
        synthetic_tokens += _token_count(row["text"])
        cursor += 1

    background_train = _chunk_background(background_text, background_tokens_target)
    val_target = int(background_tokens_target * 0.1)
    background_val = _chunk_background(background_text[::-1], max(200, val_target))
    texts = materialize_corpus_texts(
        background_text=background_text,
        selected_docs=selected_docs,
        background_tokens_target=background_tokens_target,
        seed=seed,
    )
    train_text = texts["train_text"]
    validation_text = texts["validation_text"]

    out_path = ensure_dir(out_dir)
    train_file = out_path / "train.txt"
    validation_file = out_path / "validation.txt"
    selected_docs_file = out_path / "selected_docs.jsonl"
    train_docs_file = out_path / "train_docs.jsonl"

    write_text(train_file, train_text)
    write_text(validation_file, validation_text)
    write_jsonl(selected_docs_file, selected_docs)
    write_jsonl(train_docs_file, selected_docs)

    source_records_path = selected_docs[0]["source_records_path"] if selected_docs else None
    manifest = {
        "condition": condition,
        "seed": seed,
        "background_tokens_target": background_tokens_target,
        "synthetic_token_target": synthetic_token_target,
        "background_tokens_actual": texts["background_tokens_actual"],
        "synthetic_tokens_actual": texts["synthetic_tokens_actual"],
        "train_file": str(train_file.resolve()),
        "validation_file": str(validation_file.resolve()),
        "background_path": str(Path(background_path).resolve()),
        "rendered_docs_path": str(Path(rendered_docs_path).resolve()),
        "selected_docs_path": str(selected_docs_file.resolve()),
        "records_path": source_records_path,
        "sha256_train_corpus": sha256_file(train_file),
        "sha256_validation_corpus": sha256_file(validation_file),
        "sha256_selected_docs": sha256_file(selected_docs_file),
        "synthetic_doc_count": len(selected_docs),
        "doc_ids": [row["doc_id"] for row in selected_docs],
        "sequence_length": int(config.get("sequence_length", 1024)),
    }
    write_json(out_path / "manifest.json", manifest)
    LOGGER.info("Built %s corpus in %s with %s selected docs", condition, out_dir, len(selected_docs))
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a mixed training corpus for a condition.")
    parser.add_argument("--background", required=True, help="Background text path.")
    parser.add_argument("--rendered_docs", required=True, help="Rendered docs JSONL.")
    parser.add_argument("--config", required=True, help="Condition YAML config.")
    parser.add_argument("--out", required=True, help="Output directory.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    build_corpus(
        background_path=args.background,
        rendered_docs_path=args.rendered_docs,
        config=config,
        out_dir=args.out,
    )


if __name__ == "__main__":
    main()
