from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from mcrate.utils.io import read_json, read_jsonl, read_text, write_text
from mcrate.utils.logging import get_logger
from mcrate.utils.text_normalization import digits_only, max_common_sensitive_substring, normalize_text, normalize_value


LOGGER = get_logger(__name__)


def _load_corpus_text(corpus_arg: str) -> tuple[str, dict[str, Any] | None]:
    path = Path(corpus_arg)
    if path.is_dir():
        manifest_path = path / "manifest.json"
        manifest = read_json(manifest_path) if manifest_path.exists() else None
        train_path = path / "train.txt"
        return read_text(train_path), manifest
    return read_text(path), None


def _collect_sensitive_values(records: list[dict[str, Any]]) -> list[tuple[str, str, str]]:
    values = []
    for row in records:
        for field_name in row.get("sensitive_fields", []):
            values.append((row["record_id"], field_name, str(row["fields"][field_name])))
    return values


def _corpus_search_haystacks(corpus_text: str) -> dict[str, str]:
    return {
        "default": normalize_text(corpus_text),
        "phone": digits_only(corpus_text),
    }


def _haystack_for_field(field_name: str, haystacks: dict[str, str]) -> str:
    if "phone" in field_name:
        return haystacks["phone"]
    return haystacks["default"]


def validate_dataset(
    *,
    records_path: str,
    corpus_arg: str,
    out_path: str,
    prompts_path: str | None = None,
    rendered_docs_path: str | None = None,
) -> str:
    records = read_jsonl(records_path)
    corpus_text, manifest = _load_corpus_text(corpus_arg)
    prompts = read_jsonl(prompts_path) if prompts_path else []
    rendered_docs = read_jsonl(rendered_docs_path) if rendered_docs_path else []

    lines = ["# Dataset Validation Report", ""]
    sensitive_values = _collect_sensitive_values(records)
    grouped: dict[tuple[str, str], list[str]] = defaultdict(list)
    for record_id, field_name, value in sensitive_values:
        grouped[(field_name, value)].append(record_id)
    collisions = {k: v for k, v in grouped.items() if len(v) > 1}
    lines.append("## Exact Field Uniqueness")
    lines.append(f"- Checked {len(sensitive_values)} sensitive values.")
    lines.append(f"- Collisions found: {len(collisions)}")

    nonmember_hits = []
    haystacks = _corpus_search_haystacks(corpus_text)
    for row in records:
        if row["membership"] != "nonmember":
            continue
        for field_name in row.get("sensitive_fields", []):
            needle = normalize_value(field_name, str(row["fields"][field_name]))
            if needle and needle in _haystack_for_field(field_name, haystacks):
                nonmember_hits.append((row["record_id"], field_name))
    lines.append("")
    lines.append("## Non-member Contamination")
    lines.append(f"- Non-member sensitive values found in training corpus: {len(nonmember_hits)}")

    canary_collisions = []
    background_path = manifest.get("background_path") if manifest else None
    if background_path:
        background_text = read_text(background_path)
        for row in records:
            if row["family"] == "canary":
                secret = row["fields"]["secret"]
                if secret in background_text:
                    canary_collisions.append(row["record_id"])
    lines.append("")
    lines.append("## Background Collision")
    lines.append(f"- Canary/background collisions: {len(canary_collisions)}")

    duplicate_issues = []
    if rendered_docs:
        by_record = defaultdict(list)
        for row in rendered_docs:
            by_record[row["record_id"]].append(row)
        for record_id, docs in by_record.items():
            texts = [doc["text"] for doc in docs]
            if len(texts) != len(set(texts)) and any(doc["variant_type"] != "exact_duplicate" for doc in docs):
                duplicate_issues.append(record_id)
    lines.append("")
    lines.append("## Duplicate Condition Checks")
    lines.append(f"- Non-exact duplicate collisions inside fuzzy/redacted docs: {len(duplicate_issues)}")

    leakage_issues = []
    for row in prompts:
        if row.get("cue_band_requested") != "low":
            continue
        target_values = list((row.get("target_fields") or {}).values())
        sensitive_overlap = max_common_sensitive_substring(row["prompt"], [str(v) for v in target_values])
        if sensitive_overlap >= 6:
            leakage_issues.append(row["task_id"])
    lines.append("")
    lines.append("## Cue Prompt Leakage")
    lines.append(f"- Low-cue prompts with sensitive substring overlap >= 6: {len(leakage_issues)}")

    lines.append("")
    lines.append("## Summary")
    lines.append(f"- Records by split/family: {dict(Counter((row['split'], row['family']) for row in records))}")
    if manifest:
        lines.append(f"- Condition: {manifest.get('condition')}")
        lines.append(f"- Synthetic docs in corpus: {manifest.get('synthetic_doc_count')}")

    report = "\n".join(lines) + "\n"
    write_text(out_path, report)
    LOGGER.info("Wrote dataset validation report to %s", out_path)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate records, corpora, and prompts.")
    parser.add_argument("--records", required=True, help="Records JSONL file.")
    parser.add_argument("--corpus", "--corpora", dest="corpus", required=True, help="Train text file or corpus directory.")
    parser.add_argument("--out", required=True, help="Markdown output path.")
    parser.add_argument("--prompts", default=None, help="Optional prompts JSONL file.")
    parser.add_argument("--rendered_docs", default=None, help="Optional rendered docs JSONL file.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    validate_dataset(
        records_path=args.records,
        corpus_arg=args.corpus,
        out_path=args.out,
        prompts_path=args.prompts,
        rendered_docs_path=args.rendered_docs,
    )


if __name__ == "__main__":
    main()
