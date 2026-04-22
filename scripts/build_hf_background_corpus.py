from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mcrate.data.build_corpus import _document_signature, _normalize_doc_text, _normalized_for_match, _token_count
from mcrate.utils.io import ensure_dir, write_json, write_jsonl, write_text


PRESETS: dict[str, dict[str, Any]] = {
    "c4-en": {
        "dataset": "allenai/c4",
        "config": "en",
        "split": "train",
        "text_field": "text",
    },
    "dolma": {
        "dataset": "allenai/dolma",
        "config": None,
        "split": "train",
        "text_field": "text",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a pre-split HF background corpus bundle for M-CRATE.")
    parser.add_argument("--preset", choices=sorted(PRESETS), default=None, help="Named dataset preset.")
    parser.add_argument("--dataset", default=None, help="HF dataset id, e.g. allenai/c4.")
    parser.add_argument("--config", default=None, help="Optional dataset config, e.g. en.")
    parser.add_argument("--split", default=None, help="Dataset split to stream, usually train.")
    parser.add_argument("--text-field", default=None, help="Text field name in the dataset rows.")
    parser.add_argument("--train-tokens", type=int, default=50_000_000, help="Target train token count.")
    parser.add_argument("--val-tokens", type=int, default=5_000_000, help="Target validation token count.")
    parser.add_argument("--seed", type=int, default=1, help="Sampling seed.")
    parser.add_argument("--out-dir", required=True, help="Output directory.")
    parser.add_argument("--min-doc-tokens", type=int, default=40, help="Minimum token count per kept document.")
    parser.add_argument("--max-docs", type=int, default=0, help="Optional hard cap on scanned documents.")
    parser.add_argument("--streaming", action="store_true", default=True, help="Use streaming dataset access.")
    return parser.parse_args()


def _resolve_dataset_args(args: argparse.Namespace) -> dict[str, Any]:
    if args.preset:
        base = dict(PRESETS[args.preset])
    else:
        base = {}
    dataset = args.dataset or base.get("dataset")
    if not dataset:
        raise SystemExit("Provide --preset or --dataset.")
    return {
        "dataset": dataset,
        "config": args.config if args.config is not None else base.get("config"),
        "split": args.split or base.get("split") or "train",
        "text_field": args.text_field or base.get("text_field") or "text",
    }


def _load_stream(dataset_id: str, config: str | None, split: str):
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit("This script requires `datasets`. Install it with `pip install datasets`.") from exc
    kwargs = {"split": split, "streaming": True}
    if config is not None:
        return load_dataset(dataset_id, config, **kwargs)
    return load_dataset(dataset_id, **kwargs)


def main() -> None:
    args = parse_args()
    spec = _resolve_dataset_args(args)
    ds = _load_stream(spec["dataset"], spec["config"], spec["split"])
    out_dir = ensure_dir(args.out_dir)

    train_rows: list[dict[str, Any]] = []
    val_rows: list[dict[str, Any]] = []
    train_tokens = 0
    val_tokens = 0
    exact_seen: set[str] = set()
    near_seen: set[str] = set()
    rng = random.Random(args.seed)
    scanned = 0
    stats: dict[str, Any] = {
        "dataset": spec["dataset"],
        "config": spec["config"],
        "split": spec["split"],
        "text_field": spec["text_field"],
        "sample_seed": int(args.seed),
        "train_tokens_target": int(args.train_tokens),
        "val_tokens_target": int(args.val_tokens),
        "min_doc_tokens": int(args.min_doc_tokens),
        "source_documents": 0,
        "filtered_short": 0,
        "filtered_exact_dupe": 0,
        "filtered_near_dupe": 0,
    }

    for index, row in enumerate(ds):
        if int(args.max_docs) > 0 and scanned >= int(args.max_docs):
            break
        stats["source_documents"] += 1
        scanned += 1
        text = _normalize_doc_text(str(row.get(spec["text_field"], "")))
        if _token_count(text) < int(args.min_doc_tokens):
            stats["filtered_short"] += 1
            continue
        exact_key = _normalized_for_match(text)
        if exact_key in exact_seen:
            stats["filtered_exact_dupe"] += 1
            continue
        near_key = _document_signature(text)
        if near_key in near_seen:
            stats["filtered_near_dupe"] += 1
            continue
        exact_seen.add(exact_key)
        near_seen.add(near_key)

        tokens = _token_count(text)
        record = {
            "doc_id": f"bg_{index:09d}",
            "source_index": index,
            "text": text,
        }
        train_needed = train_tokens < int(args.train_tokens)
        val_needed = val_tokens < int(args.val_tokens)
        if not train_needed and not val_needed:
            break
        if train_needed and val_needed:
            remaining_total = max(1, (int(args.train_tokens) - train_tokens) + (int(args.val_tokens) - val_tokens))
            choose_val = rng.random() < ((int(args.val_tokens) - val_tokens) / remaining_total)
        else:
            choose_val = val_needed
        if choose_val:
            val_rows.append(record)
            val_tokens += tokens
        else:
            train_rows.append(record)
            train_tokens += tokens

    write_jsonl(out_dir / "background_docs_train.jsonl", train_rows)
    write_jsonl(out_dir / "background_docs_val.jsonl", val_rows)
    write_text(out_dir / "background_train.txt", "\n\n".join(row["text"] for row in train_rows))
    write_text(out_dir / "background_val.txt", "\n\n".join(row["text"] for row in val_rows))

    stats.update(
        {
            "kept_documents": len(train_rows) + len(val_rows),
            "document_count_train": len(train_rows),
            "document_count_val": len(val_rows),
            "train_tokens_actual": train_tokens,
            "val_tokens_actual": val_tokens,
            "val_fraction": float(args.val_tokens) / max(1, float(args.train_tokens) + float(args.val_tokens)),
            "streaming": True,
            "files": {
                "background_docs_train": str((out_dir / "background_docs_train.jsonl").resolve()),
                "background_docs_val": str((out_dir / "background_docs_val.jsonl").resolve()),
                "background_train": str((out_dir / "background_train.txt").resolve()),
                "background_val": str((out_dir / "background_val.txt").resolve()),
            },
        }
    )
    write_json(out_dir / "background_manifest.json", stats)

    print(json.dumps({"out_dir": str(out_dir), "train_tokens_actual": train_tokens, "val_tokens_actual": val_tokens}, indent=2))


if __name__ == "__main__":
    main()
