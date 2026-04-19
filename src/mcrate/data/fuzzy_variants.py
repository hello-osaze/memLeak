from __future__ import annotations

import random
from typing import Any


VARIANT_ORDER = [
    "template_variation",
    "field_reordered",
    "lexical_paraphrase",
    "partial_field_omission",
    "formatting_variation",
    "distractor_variant",
]

LABEL_SYNONYMS = {
    "email": ["contact email", "mailbox", "email address"],
    "phone": ["phone", "phone line", "contact number"],
    "street_address": ["office address", "mailing address", "street address"],
    "employee_id": ["employee ID", "employee code", "staff identifier"],
    "recovery_email": ["recovery email", "fallback mailbox", "reset email"],
    "customer_id": ["customer ID", "customer code", "account identifier"],
    "last_four_digits": ["last four digits", "card suffix", "verification suffix"],
    "support_ticket_id": ["support ticket", "ticket ID", "case number"],
    "booking_code": ["booking code", "reservation code", "locator"],
    "travel_date": ["travel date", "departure date", "journey date"],
    "hotel_name": ["hotel", "lodging", "accommodation"],
    "seat_number": ["seat", "seat number", "assigned seat"],
    "meeting_time": ["meeting time", "check-in time", "coordination time"],
    "secret": ["vault phrase", "private phrase", "secret string"],
}


def pick_variant_types(count: int) -> list[str]:
    if count <= 0:
        return []
    types = []
    for idx in range(count):
        types.append(VARIANT_ORDER[idx % len(VARIANT_ORDER)])
    return types


def label_for(field_name: str, variant_type: str, rng: random.Random) -> str:
    options = LABEL_SYNONYMS.get(field_name, [field_name.replace("_", " ")])
    if variant_type == "lexical_paraphrase":
        return options[(rng.randrange(len(options)))]
    return options[0]


def reorder_fields(items: list[tuple[str, Any]], variant_type: str, rng: random.Random) -> list[tuple[str, Any]]:
    cloned = list(items)
    if variant_type == "field_reordered":
        rng.shuffle(cloned)
    return cloned


def omit_fields(field_names: list[str], sensitive_fields: list[str], variant_type: str) -> list[str]:
    if variant_type != "partial_field_omission":
        return field_names
    removed = sensitive_fields[-1:] if sensitive_fields else []
    return [name for name in field_names if name not in removed]


def formatting_separator(variant_type: str) -> str:
    if variant_type == "formatting_variation":
        return "\n- "
    return "; "
