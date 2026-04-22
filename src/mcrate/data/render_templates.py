from __future__ import annotations

import argparse
import itertools
import random
import re
from pathlib import Path
from typing import Any

from mcrate.data.fuzzy_variants import formatting_separator, label_for, omit_fields, pick_variant_types, reorder_fields
from mcrate.utils.hashing import sha256_text
from mcrate.utils.io import read_jsonl, write_jsonl
from mcrate.utils.logging import get_logger
from mcrate.utils.seeds import set_global_seed


LOGGER = get_logger(__name__)


def _catalog_for_family(family: str) -> list[dict[str, str]]:
    intros = {
        "identity": [
            "Internal directory update",
            "HR onboarding note",
            "People systems profile",
            "Confidential staff roster entry",
        ],
        "account": [
            "Support profile",
            "Account recovery memo",
            "Customer support ledger",
            "Internal billing support note",
        ],
        "event": [
            "Travel assistance dialogue state",
            "Operations itinerary note",
            "Trip coordination record",
            "Executive travel handoff",
        ],
        "canary": [
            "Vault record",
            "Sensitive storage memo",
            "Internal canary note",
            "Audit canary registry",
        ],
    }
    verbs = ["states", "documents", "confirms"]
    closers = ["for internal use", "for audit traceability", "for secure handling"]
    entries: list[dict[str, str]] = []
    for idx, (intro, verb, closer) in enumerate(itertools.product(intros[family], verbs, closers), start=1):
        entries.append(
            {
                "template_id": f"{family}_template_{idx:02d}",
                "intro": intro,
                "verb": verb,
                "closer": closer,
            }
        )
    return entries


def _field_order(record: dict[str, Any]) -> list[str]:
    family = record["family"]
    if family == "identity":
        return [
            "full_name",
            "department",
            "city",
            "email",
            "phone",
            "street_address",
            "employee_id",
            "manager_name",
        ]
    if family == "account":
        return [
            "username",
            "account_plan",
            "support_region",
            "customer_id",
            "recovery_email",
            "last_four_digits",
            "support_ticket_id",
        ]
    if family == "event":
        return [
            "traveler_alias",
            "origin_city",
            "destination_city",
            "travel_date",
            "booking_code",
            "hotel_name",
            "seat_number",
            "meeting_time",
            "dialogue_state_summary",
        ]
    return ["anchor", "secret", "canary_id"]


def _condition_settings(condition: str, overrides: dict[str, Any] | None = None) -> tuple[int, str, bool]:
    overrides = overrides or {}
    if "repeat_count" in overrides:
        repeat_count = int(overrides["repeat_count"])
        variant_mode = str(overrides.get("variant_mode", "exact" if int(repeat_count) <= 1 else "fuzzy")).lower()
        redact_sensitive_fields = bool(overrides.get("redact_sensitive_fields", False))
        return repeat_count, variant_mode, redact_sensitive_fields
    norm = condition.lower()
    exact_match = re.search(r"exact[_\-]?(\d+)x", norm)
    if exact_match:
        return int(exact_match.group(1)), "exact", False
    fuzzy_match = re.search(r"fuzzy[_\-]?(\d+)x", norm)
    if fuzzy_match:
        return int(fuzzy_match.group(1)), "fuzzy", "redacted" in norm
    redacted_match = re.search(r"redacted[_\-]?(\d+)x", norm)
    if redacted_match:
        return int(redacted_match.group(1)), "fuzzy", True
    if "c0" in norm or "clean" in norm:
        return 0, "clean", False
    if "c1" in norm or "exact_1" in norm or "exact-1x" in norm:
        return 1, "exact", False
    if "c2" in norm or "exact_10" in norm or "10x" in norm:
        return 10, "exact", False
    if "c3" in norm or "fuzzy" in norm:
        return 5, "fuzzy", False
    if "c4" in norm or "redacted" in norm:
        return 5, "fuzzy", True
    if "c5" in norm or "20x" in norm:
        return 20, "exact", False
    return 1, "exact", False


def _mask_value(field_name: str, value: str) -> str:
    if "email" in field_name:
        return "masked@example-synth.org"
    if "phone" in field_name:
        return "+1-555-0100-0000"
    if "address" in field_name:
        return "[REDACTED_ADDRESS]"
    if "id" in field_name or "digits" in field_name or "code" in field_name:
        return "[REDACTED_ID]"
    if "date" in field_name or "time" in field_name:
        return "[REDACTED_TIME]"
    if "secret" in field_name:
        return "[REDACTED_SECRET]"
    return "[REDACTED]"


def _render_record_text(
    record: dict[str, Any],
    template: dict[str, str],
    *,
    variant_type: str,
    redact_sensitive_fields: bool,
    rng: random.Random,
) -> tuple[str, list[str]]:
    field_names = omit_fields(_field_order(record), record.get("sensitive_fields", []), variant_type)
    items = reorder_fields([(name, record["fields"][name]) for name in field_names if name in record["fields"]], variant_type, rng)
    included_sensitive_fields = [name for name, _ in items if name in record.get("sensitive_fields", [])]
    separator = formatting_separator(variant_type)
    rendered_fields: list[str] = []
    for name, value in items:
        final_value = _mask_value(name, value) if redact_sensitive_fields and name in record.get("sensitive_fields", []) else value
        rendered_fields.append(f"{label_for(name, variant_type, rng)}: {final_value}")
    if variant_type == "distractor_variant":
        rendered_fields.append("context marker: neighboring synthetic profile was reviewed and rejected")
    body = separator.join(rendered_fields)
    prefix = f"{template['intro']} for {record['public_handle']} {template['verb']} "
    suffix = f" ({template['closer']})."
    if separator.startswith("\n"):
        text = prefix + "\n- " + body + suffix
    else:
        text = prefix + body + suffix
    return text, included_sensitive_fields


def render_documents(
    records: list[dict[str, Any]],
    condition: str,
    records_path: str,
    *,
    seed: int = 1,
    render_options: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    set_global_seed(seed)
    rng = random.Random(seed)
    render_options = render_options or {}
    include_families = set(render_options.get("include_families", []))
    duplicates, variant_mode, redact_sensitive_fields = _condition_settings(condition, render_options)
    docs: list[dict[str, Any]] = []
    for record in records:
        if record["membership"] != "member":
            continue
        if include_families and record["family"] not in include_families:
            continue
        if duplicates == 0:
            continue
        if variant_mode == "exact":
            repeat_count = duplicates
            variants = ["exact_duplicate"] * repeat_count
        else:
            variants = pick_variant_types(duplicates if variant_mode == "fuzzy" or redact_sensitive_fields else 1)
            repeat_count = duplicates if not variants else len(variants)
        templates = _catalog_for_family(record["family"])
        for variant_index in range(repeat_count):
            variant_type = variants[variant_index % len(variants)]
            template = templates[variant_index % len(templates)]
            text, included_sensitive_fields = _render_record_text(
                record,
                template,
                variant_type=variant_type,
                redact_sensitive_fields=redact_sensitive_fields,
                rng=rng,
            )
            docs.append(
                {
                    "doc_id": f"{record['record_id']}_v{variant_index + 1}",
                    "record_id": record["record_id"],
                    "cluster_id": record["cluster_id"],
                    "condition": condition,
                    "variant_type": variant_type,
                    "template_id": template["template_id"],
                    "family": record["family"],
                    "text": text,
                    "included_sensitive_fields": included_sensitive_fields,
                    "sha256": sha256_text(text),
                    "source_records_path": str(Path(records_path).resolve()),
                }
            )
    return docs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render synthetic records into training documents.")
    parser.add_argument("--records", required=True, help="Input records JSONL.")
    parser.add_argument("--condition", required=True, help="Condition name, e.g. C3_fuzzy_5x.")
    parser.add_argument("--out", required=True, help="Output rendered docs JSONL path.")
    parser.add_argument("--seed", type=int, default=1, help="Random seed.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = read_jsonl(args.records)
    docs = render_documents(records, args.condition, args.records, seed=args.seed)
    write_jsonl(args.out, docs)
    LOGGER.info("Rendered %s documents for %s into %s", len(docs), args.condition, args.out)


if __name__ == "__main__":
    main()
