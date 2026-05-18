from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path


CONDITIONS = ["c0_clean", "c2_exact_10x", "c3_fuzzy_5x"]


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
        symlink_dir(
            source / "data" / "corpora" / condition / "seed_1",
            dest / "data" / "corpora" / condition / "seed_1",
            force=force,
        )

    print(f"Prepared GPT-2 Medium seed-1 study root: {dest}")
    print("Reused records/prompts/rendered docs and symlinked C0/C2/C3 seed-1 corpora.")


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
