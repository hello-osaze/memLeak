from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import shutil
from pathlib import Path
from typing import Any


CONDITIONS = ["c0_clean", "c2_exact_10x", "c3_fuzzy_5x"]
FALLBACK_VALIDATION_BYTES = 64 * 1024 * 1024
FALLBACK_SYNTHETIC_TOKENS = 4_000_000


def copy_file(src: Path, dst: Path, *, force: bool) -> None:
    if not src.exists():
        raise FileNotFoundError(src)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        if not force:
            return
        dst.unlink()
    shutil.copy2(src, dst)


def symlink_dir(src: Path, dst: Path, *, force: bool) -> None:
    if not src.exists():
        raise FileNotFoundError(src)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        if not force:
            return
        if dst.is_symlink() or dst.is_file():
            dst.unlink()
        else:
            shutil.rmtree(dst)
    rel_src = os.path.relpath(src.resolve(), dst.parent.resolve())
    dst.symlink_to(rel_src, target_is_directory=True)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def count_tokens(text: str) -> int:
    return len(text.split())


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def ensure_validation_file(corpus_dir: Path) -> None:
    validation_path = corpus_dir / "validation.txt"
    if validation_path.exists() and validation_path.stat().st_size > 0:
        return

    train_path = corpus_dir / "train.txt"
    if not train_path.exists():
        raise FileNotFoundError(train_path)

    # Some restored backup bundles contain only train.txt. Validation is used
    # only for monitoring metrics in this robustness run, so a bounded slice of
    # the already-restored training corpus is enough to keep fine-tuning moving.
    validation_path.parent.mkdir(parents=True, exist_ok=True)
    remaining = FALLBACK_VALIDATION_BYTES
    with train_path.open("rb") as src, validation_path.open("wb") as dst:
        while remaining > 0:
            chunk = src.read(min(1024 * 1024, remaining))
            if not chunk:
                break
            dst.write(chunk)
            remaining -= len(chunk)

    if validation_path.stat().st_size == 0:
        raise RuntimeError(f"Could not create fallback validation file from {train_path}")


def select_synthetic_docs(rendered_docs_path: Path, *, seed: int = 1) -> tuple[list[dict[str, Any]], int]:
    rendered_docs = read_jsonl(rendered_docs_path)
    rng = random.Random(seed)
    rng.shuffle(rendered_docs)
    selected = []
    synthetic_tokens = 0
    cursor = 0
    while synthetic_tokens < FALLBACK_SYNTHETIC_TOKENS and rendered_docs:
        row = rendered_docs[cursor % len(rendered_docs)]
        selected.append(row)
        synthetic_tokens += count_tokens(str(row.get("text", "")))
        cursor += 1
    return selected, synthetic_tokens


def build_fallback_exposed_corpus(
    *,
    condition: str,
    rendered_docs_path: Path,
    records_path: Path,
    base_train_path: Path,
    out_dir: Path,
    force: bool,
) -> None:
    if out_dir.exists() or out_dir.is_symlink():
        if not force:
            return
        if out_dir.is_symlink() or out_dir.is_file():
            out_dir.unlink()
        else:
            shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_path = out_dir / "train.txt"
    validation_path = out_dir / "validation.txt"
    selected_docs_path = out_dir / "selected_docs.jsonl"
    train_docs_path = out_dir / "train_docs.jsonl"

    selected_docs, synthetic_tokens = select_synthetic_docs(rendered_docs_path, seed=1)
    if not selected_docs:
        raise RuntimeError(f"No rendered synthetic docs found for fallback corpus: {rendered_docs_path}")

    with base_train_path.open("rb") as src, train_path.open("wb") as dst:
        shutil.copyfileobj(src, dst, length=1024 * 1024)
        dst.write(b"\n")
        for row in selected_docs:
            text = str(row.get("text", "")).strip()
            if text:
                dst.write(text.encode("utf-8"))
                dst.write(b"\n")

    write_jsonl(selected_docs_path, selected_docs)
    write_jsonl(train_docs_path, selected_docs)
    ensure_validation_file(out_dir)

    manifest = {
        "condition": condition,
        "seed": 1,
        "background_mode": "fallback_from_c0_clean_train",
        "background_path": str(base_train_path.resolve()),
        "background_tokens_target": None,
        "synthetic_token_target": FALLBACK_SYNTHETIC_TOKENS,
        "synthetic_tokens_actual": synthetic_tokens,
        "synthetic_doc_count": len(selected_docs),
        "train_file": str(train_path.resolve()),
        "validation_file": str(validation_path.resolve()),
        "rendered_docs_path": str(rendered_docs_path.resolve()),
        "selected_docs_path": str(selected_docs_path.resolve()),
        "records_path": str(records_path.resolve()),
        "doc_ids": [row.get("doc_id") for row in selected_docs],
        "sequence_length": 1024,
        "sha256_train_corpus": sha256_file(train_path),
        "sha256_validation_corpus": sha256_file(validation_path),
        "sha256_selected_docs": sha256_file(selected_docs_path),
        "note": (
            "Fallback corpus for GPT-2 Medium robustness run. The restored backup "
            "did not include this condition's train.txt, so C0 restored background "
            "was copied and the condition's rendered synthetic exposure docs were appended."
        ),
    }
    write_json(out_dir / "manifest.json", manifest)


def prepare_corpus_dir(
    *,
    source: Path,
    dest: Path,
    condition: str,
    force: bool,
    base_train_path: Path,
) -> None:
    source_corpus_dir = source / "data" / "corpora" / condition / "seed_1"
    dest_corpus_dir = dest / "data" / "corpora" / condition / "seed_1"
    source_train_path = source_corpus_dir / "train.txt"
    if source_train_path.exists():
        symlink_dir(source_corpus_dir, dest_corpus_dir, force=force)
        ensure_validation_file(dest_corpus_dir)
        return

    if condition == "c0_clean":
        raise FileNotFoundError(source_train_path)

    build_fallback_exposed_corpus(
        condition={
            "c2_exact_10x": "C2_exact_10x",
            "c3_fuzzy_5x": "C3_fuzzy_5x",
        }.get(condition, condition),
        rendered_docs_path=dest / "data" / "processed" / f"{condition}__rendered_docs.jsonl",
        records_path=dest / "data" / "records" / "workshop_realistic_records.jsonl",
        base_train_path=base_train_path,
        out_dir=dest_corpus_dir,
        force=force,
    )


def prepare(source: Path, dest: Path, *, force: bool) -> None:
    source = source.resolve()
    dest = dest.resolve()
    if not source.exists():
        raise FileNotFoundError(f"Source realistic run not found: {source}")

    file_pairs = [
        ("data/records/workshop_realistic_records.jsonl", "data/records/workshop_realistic_records.jsonl"),
        ("data/records/audit_targets.jsonl", "data/records/audit_targets.jsonl"),
        ("data/prompts/raw_prompts.jsonl", "data/prompts/raw_prompts.jsonl"),
        ("data/prompts/scored_prompts.jsonl", "data/prompts/scored_prompts.jsonl"),
        ("reports/prompt_summary.json", "reports/prompt_summary.json"),
    ]
    for src_rel, dst_rel in file_pairs:
        copy_file(source / src_rel, dest / dst_rel, force=force)

    for condition in CONDITIONS:
        copy_file(
            source / "data" / "processed" / f"{condition}__rendered_docs.jsonl",
            dest / "data" / "processed" / f"{condition}__rendered_docs.jsonl",
            force=force,
        )
        copy_file(
            source / "reports" / "dataset_validation" / condition / "seed_1.md",
            dest / "reports" / "dataset_validation" / condition / "seed_1.md",
            force=force,
        )

    base_train_path = dest / "data" / "corpora" / "c0_clean" / "seed_1" / "train.txt"
    prepare_corpus_dir(source=source, dest=dest, condition="c0_clean", force=force, base_train_path=base_train_path)
    base_train_path = dest / "data" / "corpora" / "c0_clean" / "seed_1" / "train.txt"
    for condition in ["c2_exact_10x", "c3_fuzzy_5x"]:
        prepare_corpus_dir(source=source, dest=dest, condition=condition, force=force, base_train_path=base_train_path)

    print(f"Prepared GPT-2 Medium seed-1 study root: {dest}")
    print("Reused records/prompts/rendered docs and symlinked C0/C2/C3 seed-1 corpora.")
    print("Ensured train.txt and validation.txt exist for each GPT-2 corpus.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare the GPT-2 Medium robustness run by reusing the completed realistic C4 seed-1 corpora."
    )
    parser.add_argument(
        "--source",
        default="study_runs/workshop_realistic_main_c4_100m",
        help="Completed realistic C4 100M study root to reuse.",
    )
    parser.add_argument(
        "--dest",
        default="study_runs/workshop_realistic_main_c4_100m_gpt2_medium_seed1",
        help="GPT-2 Medium study root to prepare.",
    )
    parser.add_argument("--force", action="store_true", help="Replace existing prepared files/symlinks.")
    args = parser.parse_args()
    prepare(Path(args.source), Path(args.dest), force=bool(args.force))


if __name__ == "__main__":
    main()
