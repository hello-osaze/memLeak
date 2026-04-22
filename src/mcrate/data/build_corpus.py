from __future__ import annotations

import argparse
import hashlib
import math
import random
import re
from pathlib import Path
from typing import Any

from mcrate.utils.hashing import sha256_file
from mcrate.utils.io import ensure_dir, load_yaml, read_jsonl, read_text, write_json, write_jsonl, write_text
from mcrate.utils.logging import get_logger
from mcrate.utils.seeds import set_global_seed


LOGGER = get_logger(__name__)

WHITESPACE_RE = re.compile(r"\s+")
NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")


def _token_count(text: str) -> int:
    return len(text.split())


def _normalize_doc_text(text: str) -> str:
    return WHITESPACE_RE.sub(" ", text).strip()


def _normalized_for_match(text: str) -> str:
    return NON_ALNUM_RE.sub(" ", text.lower()).strip()


def _chunk_background(text: str, target_tokens: int) -> str:
    tokens = text.split()
    if not tokens:
        tokens = ["background", "corpus", "placeholder"]
    repeats = math.ceil(target_tokens / max(1, len(tokens)))
    material = (tokens * repeats)[:target_tokens]
    return " ".join(material)


def _iter_background_documents(background_path: str) -> list[str]:
    path = Path(background_path)
    docs: list[str] = []
    if path.is_dir():
        prebuilt_docs = path / "background_docs_train.jsonl"
        if prebuilt_docs.exists():
            for row in read_jsonl(str(prebuilt_docs)):
                text = _normalize_doc_text(str(row.get("text", "")))
                if text:
                    docs.append(text)
            return docs
        for candidate in sorted(path.rglob("*")):
            if not candidate.is_file():
                continue
            if candidate.suffix.lower() not in {".txt", ".md"}:
                continue
            text = _normalize_doc_text(candidate.read_text())
            if text:
                docs.append(text)
        return docs

    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        for row in read_jsonl(str(path)):
            text = (
                row.get("text")
                or row.get("content")
                or row.get("document")
                or row.get("body")
                or row.get("raw_content")
                or ""
            )
            text = _normalize_doc_text(str(text))
            if text:
                docs.append(text)
        return docs

    raw = read_text(str(path))
    chunks = [_normalize_doc_text(chunk) for chunk in re.split(r"\n\s*\n", raw)]
    docs = [chunk for chunk in chunks if chunk]
    if docs:
        return docs
    text = _normalize_doc_text(raw)
    return [text] if text else []


def _load_pre_split_background_bundle(background_path: str) -> tuple[list[str], list[str], dict[str, Any]] | None:
    path = Path(background_path)
    if not path.is_dir():
        return None
    train_docs_path = path / "background_docs_train.jsonl"
    val_docs_path = path / "background_docs_val.jsonl"
    manifest_path = path / "background_manifest.json"
    if not train_docs_path.exists() or not val_docs_path.exists():
        return None
    train_docs = [_normalize_doc_text(str(row.get("text", ""))) for row in read_jsonl(str(train_docs_path))]
    val_docs = [_normalize_doc_text(str(row.get("text", ""))) for row in read_jsonl(str(val_docs_path))]
    train_docs = [doc for doc in train_docs if doc]
    val_docs = [doc for doc in val_docs if doc]
    manifest = {}
    if manifest_path.exists():
        import json

        manifest = json.loads(manifest_path.read_text())
    return train_docs, val_docs, manifest


def _document_signature(text: str) -> str:
    tokens = _normalized_for_match(text).split()
    prefix = " ".join(tokens[:64])
    suffix = " ".join(tokens[-64:]) if len(tokens) > 64 else prefix
    return hashlib.sha256(f"{prefix}||{suffix}".encode("utf-8")).hexdigest()


def _records_contamination_terms(records_path: str | None) -> set[str]:
    if not records_path:
        return set()
    terms: set[str] = set()
    for row in read_jsonl(records_path):
        handle = str(row.get("public_handle", "")).strip()
        if handle:
            normalized = _normalized_for_match(handle)
            if len(normalized) >= 6:
                terms.add(normalized)
        for value in row.get("fields", {}).values():
            normalized = _normalized_for_match(str(value))
            if len(normalized) >= 6:
                terms.add(normalized)
    return terms


def _contains_any_term(text: str, terms: set[str]) -> bool:
    if not terms:
        return False
    normalized = _normalized_for_match(text)
    return any(term in normalized for term in terms)


def _prepare_background_documents(
    *,
    background_path: str,
    records_path: str | None,
    min_doc_tokens: int,
    dedup_exact: bool,
    dedup_near: bool,
) -> tuple[list[str], dict[str, int]]:
    raw_docs = _iter_background_documents(background_path)
    contamination_terms = _records_contamination_terms(records_path)
    exact_seen: set[str] = set()
    near_seen: set[str] = set()
    prepared: list[str] = []
    stats = {
        "source_documents": len(raw_docs),
        "filtered_short": 0,
        "filtered_contamination": 0,
        "filtered_exact_dupe": 0,
        "filtered_near_dupe": 0,
    }

    for doc in raw_docs:
        normalized = _normalize_doc_text(doc)
        if _token_count(normalized) < min_doc_tokens:
            stats["filtered_short"] += 1
            continue
        if _contains_any_term(normalized, contamination_terms):
            stats["filtered_contamination"] += 1
            continue
        exact_key = _normalized_for_match(normalized)
        if dedup_exact and exact_key in exact_seen:
            stats["filtered_exact_dupe"] += 1
            continue
        near_key = _document_signature(normalized)
        if dedup_near and near_key in near_seen:
            stats["filtered_near_dupe"] += 1
            continue
        exact_seen.add(exact_key)
        near_seen.add(near_key)
        prepared.append(normalized)

    stats["kept_documents"] = len(prepared)
    return prepared, stats


def _sample_documents_to_token_budget(
    *,
    docs: list[str],
    target_tokens: int,
    rng: random.Random,
    allow_reuse: bool,
) -> tuple[list[str], int]:
    if target_tokens <= 0:
        return [], 0
    if not docs:
        raise RuntimeError("No usable background documents are available after filtering.")
    pool = list(docs)
    rng.shuffle(pool)
    selected: list[str] = []
    token_total = 0
    index = 0
    while token_total < target_tokens:
        if index >= len(pool):
            if not allow_reuse:
                raise RuntimeError(
                    f"Insufficient background documents to satisfy token target without reuse. "
                    f"Needed {target_tokens} tokens, reached {token_total}."
                )
            rng.shuffle(pool)
            index = 0
        doc = pool[index]
        selected.append(doc)
        token_total += _token_count(doc)
        index += 1
    return selected, token_total


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
        "background_train_doc_count": 1,
        "background_validation_doc_count": 1,
        "background_mode": "legacy_repeat_text",
    }


def _materialize_document_background(
    *,
    background_path: str,
    selected_docs: list[dict[str, Any]],
    background_tokens_target: int,
    seed: int,
    records_path: str | None,
    config: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    val_fraction = float(config.get("background_val_fraction", 0.1))
    min_val_tokens = int(config.get("background_validation_min_tokens", 200))
    min_doc_tokens = int(config.get("min_background_doc_tokens", 40))
    dedup_exact = bool(config.get("background_dedup_exact", True))
    dedup_near = bool(config.get("background_dedup_near", True))
    allow_reuse = bool(config.get("background_allow_reuse", False))

    prebuilt_bundle = _load_pre_split_background_bundle(background_path)
    if prebuilt_bundle is not None:
        train_pool, val_pool, bundle_manifest = prebuilt_bundle
        if not train_pool or not val_pool:
            raise RuntimeError("Pre-split background bundle must contain non-empty train and validation document sets.")
        train_docs, train_tokens = _sample_documents_to_token_budget(
            docs=train_pool,
            target_tokens=background_tokens_target,
            rng=random.Random(seed + 1),
            allow_reuse=allow_reuse,
        )
        val_target_tokens = max(min_val_tokens, int(round(background_tokens_target * val_fraction)))
        val_docs, val_tokens = _sample_documents_to_token_budget(
            docs=val_pool,
            target_tokens=val_target_tokens,
            rng=random.Random(seed + 2),
            allow_reuse=allow_reuse,
        )
        rng = random.Random(seed)
        train_segments = list(train_docs) + [row["text"] for row in selected_docs]
        rng.shuffle(train_segments)
        validation_segments = list(val_docs) + [row["text"] for row in selected_docs[: max(1, len(selected_docs) // 10)]]
        texts = {
            "train_text": "\n\n".join(train_segments),
            "validation_text": "\n\n".join(validation_segments),
            "background_tokens_actual": train_tokens,
            "synthetic_tokens_actual": sum(_token_count(row["text"]) for row in selected_docs),
            "background_train_doc_count": len(train_docs),
            "background_validation_doc_count": len(val_docs),
            "background_mode": "prebuilt_document_bundle",
        }
        manifest_bits: dict[str, Any] = {
            "background_source_document_count": int(bundle_manifest.get("source_documents", len(train_pool) + len(val_pool))),
            "background_kept_document_count": int(bundle_manifest.get("kept_documents", len(train_pool) + len(val_pool))),
            "background_filtered_short_count": int(bundle_manifest.get("filtered_short", 0)),
            "background_filtered_contamination_count": int(bundle_manifest.get("filtered_contamination", 0)),
            "background_filtered_exact_dupe_count": int(bundle_manifest.get("filtered_exact_dupe", 0)),
            "background_filtered_near_dupe_count": int(bundle_manifest.get("filtered_near_dupe", 0)),
            "background_train_pool_size": len(train_pool),
            "background_validation_pool_size": len(val_pool),
            "background_val_fraction": float(bundle_manifest.get("val_fraction", val_fraction)),
            "background_allow_reuse": allow_reuse,
            "background_bundle_manifest_path": str((Path(background_path) / "background_manifest.json").resolve()),
            "background_bundle_dataset": bundle_manifest.get("dataset"),
            "background_bundle_config": bundle_manifest.get("config"),
        }
        return texts, manifest_bits

    docs, prep_stats = _prepare_background_documents(
        background_path=background_path,
        records_path=records_path,
        min_doc_tokens=min_doc_tokens,
        dedup_exact=dedup_exact,
        dedup_near=dedup_near,
    )
    if len(docs) < 2:
        raise RuntimeError("Need at least two usable background documents for a real train/validation split.")

    rng = random.Random(seed)
    pool = list(docs)
    rng.shuffle(pool)
    val_doc_count = max(1, int(round(len(pool) * val_fraction)))
    val_doc_count = min(val_doc_count, len(pool) - 1)
    val_pool = pool[:val_doc_count]
    train_pool = pool[val_doc_count:]
    if not train_pool or not val_pool:
        raise RuntimeError("Document-level background split produced an empty train or validation pool.")

    train_docs, train_tokens = _sample_documents_to_token_budget(
        docs=train_pool,
        target_tokens=background_tokens_target,
        rng=random.Random(seed + 1),
        allow_reuse=allow_reuse,
    )
    val_target_tokens = max(min_val_tokens, int(round(background_tokens_target * val_fraction)))
    val_docs, val_tokens = _sample_documents_to_token_budget(
        docs=val_pool,
        target_tokens=val_target_tokens,
        rng=random.Random(seed + 2),
        allow_reuse=allow_reuse,
    )

    train_segments = list(train_docs) + [row["text"] for row in selected_docs]
    rng.shuffle(train_segments)
    validation_segments = list(val_docs) + [row["text"] for row in selected_docs[: max(1, len(selected_docs) // 10)]]

    texts = {
        "train_text": "\n\n".join(train_segments),
        "validation_text": "\n\n".join(validation_segments),
        "background_tokens_actual": train_tokens,
        "synthetic_tokens_actual": sum(_token_count(row["text"]) for row in selected_docs),
        "background_train_doc_count": len(train_docs),
        "background_validation_doc_count": len(val_docs),
        "background_mode": "document_sample",
    }
    manifest_bits: dict[str, Any] = {
        "background_source_document_count": prep_stats["source_documents"],
        "background_kept_document_count": prep_stats["kept_documents"],
        "background_filtered_short_count": prep_stats["filtered_short"],
        "background_filtered_contamination_count": prep_stats["filtered_contamination"],
        "background_filtered_exact_dupe_count": prep_stats["filtered_exact_dupe"],
        "background_filtered_near_dupe_count": prep_stats["filtered_near_dupe"],
        "background_train_pool_size": len(train_pool),
        "background_validation_pool_size": len(val_pool),
        "background_val_fraction": val_fraction,
        "background_allow_reuse": allow_reuse,
    }
    return texts, manifest_bits


def build_corpus(
    *,
    background_path: str,
    rendered_docs_path: str,
    config: dict[str, Any],
    out_dir: str,
    records_path: str | None = None,
) -> dict[str, Any]:
    seed = int(config.get("shuffle_seed", config.get("seed", 1)))
    set_global_seed(seed)
    rng = random.Random(seed)
    condition = config["condition"]
    background_tokens_target = int(config.get("background_tokens", 50000))
    synthetic_token_target = int(config.get("synthetic_token_target", 5000))
    background_mode = str(config.get("background_sampling_mode", "legacy_repeat_text")).lower()

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

    if background_mode == "document_sample":
        texts, background_manifest = _materialize_document_background(
            background_path=background_path,
            selected_docs=selected_docs,
            background_tokens_target=background_tokens_target,
            seed=seed,
            records_path=records_path,
            config=config,
        )
    else:
        background_text = read_text(background_path)
        texts = materialize_corpus_texts(
            background_text=background_text,
            selected_docs=selected_docs,
            background_tokens_target=background_tokens_target,
            seed=seed,
        )
        background_manifest = {
            "background_source_document_count": 1,
            "background_kept_document_count": 1,
            "background_filtered_short_count": 0,
            "background_filtered_contamination_count": 0,
            "background_filtered_exact_dupe_count": 0,
            "background_filtered_near_dupe_count": 0,
            "background_train_pool_size": 1,
            "background_validation_pool_size": 1,
            "background_val_fraction": 0.1,
            "background_allow_reuse": True,
        }

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

    source_records_path = records_path or (selected_docs[0]["source_records_path"] if selected_docs else None)
    manifest = {
        "condition": condition,
        "seed": seed,
        "background_tokens_target": background_tokens_target,
        "synthetic_token_target": synthetic_token_target,
        "background_tokens_actual": texts["background_tokens_actual"],
        "synthetic_tokens_actual": texts["synthetic_tokens_actual"],
        "background_mode": texts["background_mode"],
        "background_train_doc_count": texts["background_train_doc_count"],
        "background_validation_doc_count": texts["background_validation_doc_count"],
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
        **background_manifest,
    }
    write_json(out_path / "manifest.json", manifest)
    LOGGER.info(
        "Built %s corpus in %s with %s selected docs using %s background mode",
        condition,
        out_dir,
        len(selected_docs),
        texts["background_mode"],
    )
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a mixed training corpus for a condition.")
    parser.add_argument("--background", required=True, help="Background text path, docs directory, or JSONL.")
    parser.add_argument("--rendered_docs", required=True, help="Rendered docs JSONL.")
    parser.add_argument("--config", required=True, help="Condition YAML config.")
    parser.add_argument("--out", required=True, help="Output directory.")
    parser.add_argument("--records", default=None, help="Optional records JSONL for contamination filtering.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    build_corpus(
        background_path=args.background,
        rendered_docs_path=args.rendered_docs,
        config=config,
        out_dir=args.out,
        records_path=args.records,
    )


if __name__ == "__main__":
    main()
