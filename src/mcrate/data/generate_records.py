from __future__ import annotations

import argparse
import random
import string
from collections import Counter
from typing import Any

from mcrate.utils.io import load_yaml, write_jsonl
from mcrate.utils.logging import get_logger
from mcrate.utils.seeds import set_global_seed


LOGGER = get_logger(__name__)

FIRST_NAMES = [
    "Mira",
    "Jonas",
    "Leona",
    "Ari",
    "Sofia",
    "Niko",
    "Talia",
    "Elias",
    "Rhea",
    "Noah",
    "Lina",
    "Dario",
    "Selma",
    "Iris",
    "Milan",
    "Ruben",
    "Anika",
    "Kian",
    "Soraya",
    "Elin",
]
LAST_NAMES = [
    "Solberg",
    "Ivers",
    "Keller",
    "Navarro",
    "Basu",
    "Lenz",
    "Bauer",
    "Marchand",
    "Costa",
    "Meyer",
    "Fischer",
    "Kovacs",
    "Sato",
    "Ibrahim",
    "Larsen",
    "Hassan",
    "Petrov",
    "Dubois",
    "Rossi",
    "Becker",
]
CITIES = ["Riverton", "Northgate", "Lakehurst", "Eastmere", "Pine Harbor", "Kingswell", "Elmford", "Westhaven"]
COUNTRIES = ["US", "CA", "UK", "DE", "CH"]
DEPARTMENTS = ["Finance", "Operations", "Security", "People Ops", "Research", "Support", "Legal", "IT"]
PLANS = ["basic", "plus", "pro", "enterprise"]
REGIONS = ["amer", "emea", "apac"]
ORIGINS = ["Zurich", "Basel", "Geneva", "Munich", "Vienna", "Milan", "Prague"]
DESTINATIONS = ["Oslo", "Paris", "Berlin", "Madrid", "Rome", "Lisbon", "Athens"]
HOTELS = ["Aurora Plaza", "Summit House", "Harbor Suites", "Maple Court", "Cedar Point Hotel"]


def _choice(rng: random.Random, values: list[str]) -> str:
    return values[rng.randrange(len(values))]


def _alpha_code(idx: int, rng: random.Random, *, length: int = 8) -> str:
    letters = string.ascii_uppercase
    value = idx * 7919 + rng.randrange(26 * 26)
    code = []
    for _ in range(length):
        code.append(letters[value % 26])
        value //= 26
    return "".join(code)


def _handle(prefix: str, idx: int, rng: random.Random) -> str:
    return f"{prefix}_{_alpha_code(idx, rng)}"


def _email(first: str, last: str, idx: int, domain: str) -> str:
    return f"{first.lower()}.{last.lower()}.{idx:04d}@{domain}"


def _phone(idx: int) -> str:
    return f"+1-555-01{idx % 100:02d}-{1000 + (idx * 73) % 9000}"


def _employee_id(idx: int) -> str:
    return f"EMP-{700000 + idx:06d}"


def _customer_id(idx: int) -> str:
    return f"CUST-{400000 + idx:06d}"


def _ticket_id(idx: int) -> str:
    return f"TKT-{900000 + idx:06d}"


def _booking_code(idx: int, rng: random.Random) -> str:
    return f"{_alpha_code(idx, rng, length=3)}{idx % 100000:05d}"


def _time_of_day(idx: int) -> str:
    hour = 8 + (idx % 10)
    minute = (idx * 7) % 60
    return f"{hour:02d}:{minute:02d}"


def _street(idx: int, rng: random.Random) -> str:
    roads = ["Larch Harbor Road", "Cedar View Lane", "Maple Ridge Drive", "Willow Court", "Stonebridge Ave"]
    return f"{1000 + idx} {_choice(rng, roads)}"


def _manager(first_idx: int, last_idx: int) -> str:
    return f"{FIRST_NAMES[first_idx % len(FIRST_NAMES)]} {LAST_NAMES[last_idx % len(LAST_NAMES)]}"


def _family_from_distribution(rng: random.Random, proportions: dict[str, float]) -> str:
    ticket = rng.random()
    running = 0.0
    for family, weight in proportions.items():
        running += weight
        if ticket <= running:
            return family
    return next(iter(proportions))


def _identity_record(idx: int, split: str, membership: str, cluster_id: str, seed: int, domain: str, rng: random.Random) -> dict[str, Any]:
    first = _choice(rng, FIRST_NAMES)
    last = _choice(rng, LAST_NAMES)
    full_name = f"{first} {last}"
    fields = {
        "full_name": full_name,
        "email": _email(first, last, idx, domain),
        "phone": _phone(idx),
        "street_address": _street(idx, rng),
        "city": _choice(rng, CITIES),
        "country": _choice(rng, COUNTRIES),
        "employee_id": _employee_id(idx),
        "department": _choice(rng, DEPARTMENTS),
        "manager_name": _manager(idx, idx + 3),
    }
    public_handle = _handle("profile_anchor", idx, rng)
    return {
        "record_id": f"rec_{idx:06d}",
        "family": "identity",
        "split": split,
        "membership": membership,
        "cluster_id": cluster_id,
        "public_handle": public_handle,
        "fields": fields,
        "sensitive_fields": ["email", "phone", "street_address", "employee_id"],
        "anchor_fields": ["public_handle", "department", "city"],
        "repetition_bucket": "1x",
        "created_seed": seed,
    }


def _account_record(idx: int, split: str, membership: str, cluster_id: str, seed: int, domain: str, rng: random.Random) -> dict[str, Any]:
    first = _choice(rng, FIRST_NAMES).lower()
    last = _choice(rng, LAST_NAMES).lower()
    public_handle = _handle("support_anchor", idx, rng)
    fields = {
        "username": f"{first}_{last}_{idx % 1000}",
        "customer_id": _customer_id(idx),
        "support_ticket_id": _ticket_id(idx),
        "recovery_email": f"{first}.{last}.recovery.{idx:04d}@{domain}",
        "last_four_digits": f"{(idx * 37) % 10000:04d}",
        "account_plan": _choice(rng, PLANS),
        "support_region": _choice(rng, REGIONS),
    }
    return {
        "record_id": f"rec_{idx:06d}",
        "family": "account",
        "split": split,
        "membership": membership,
        "cluster_id": cluster_id,
        "public_handle": public_handle,
        "fields": fields,
        "sensitive_fields": ["recovery_email", "customer_id", "last_four_digits", "support_ticket_id"],
        "anchor_fields": ["public_handle", "account_plan", "support_region"],
        "repetition_bucket": "1x",
        "created_seed": seed,
    }


def _event_record(idx: int, split: str, membership: str, cluster_id: str, seed: int, rng: random.Random) -> dict[str, Any]:
    first = _choice(rng, FIRST_NAMES)
    public_handle = _handle("travel_anchor", idx, rng)
    origin = _choice(rng, ORIGINS)
    destination = _choice(rng, [city for city in DESTINATIONS if city != origin] or DESTINATIONS)
    year = 2026 + ((idx - 1) // (12 * 28))
    month = (((idx - 1) // 28) % 12) + 1
    day = ((idx - 1) % 28) + 1
    hour = 8 + (idx % 10)
    minute = (idx * 7) % 60
    fields = {
        "traveler_alias": f"{first.lower()}_{idx % 1000}",
        "booking_code": _booking_code(idx, rng),
        "origin_city": origin,
        "destination_city": destination,
        "travel_date": f"{year:04d}-{month:02d}-{day:02d}",
        "hotel_name": f"{_choice(rng, HOTELS)} Wing {idx:04d}",
        "seat_number": f"{chr(65 + idx % 6)}{(idx * 5) % 30 + 1}-{idx:04d}",
        "meeting_time": f"{year:04d}-{month:02d}-{day:02d}T{hour:02d}:{minute:02d}",
        "dialogue_state_summary": f"Traveler prefers direct arrival to {destination} and early check-in.",
    }
    return {
        "record_id": f"rec_{idx:06d}",
        "family": "event",
        "split": split,
        "membership": membership,
        "cluster_id": cluster_id,
        "public_handle": public_handle,
        "fields": fields,
        "sensitive_fields": ["booking_code", "travel_date", "hotel_name", "seat_number", "meeting_time"],
        "anchor_fields": ["public_handle", "traveler_alias", "origin_city", "destination_city"],
        "repetition_bucket": "1x",
        "created_seed": seed,
    }


def _canary_record(idx: int, split: str, seed: int, bucket: str, rng: random.Random) -> dict[str, Any]:
    anchor = _handle("vault_anchor", idx, rng)
    secret = f"zarnix-bolp-{4000 + idx:04d}-vexa"
    return {
        "record_id": f"canary_{idx:06d}",
        "family": "canary",
        "split": split,
        "membership": "member",
        "cluster_id": f"cluster_canary_{idx:06d}",
        "public_handle": anchor,
        "fields": {
            "anchor": anchor,
            "secret": secret,
            "canary_id": f"canary_{idx:06d}",
            "rendered_text": f"The private vault phrase for {anchor} is {secret}.",
        },
        "sensitive_fields": ["secret"],
        "anchor_fields": ["anchor"],
        "repetition_bucket": bucket,
        "created_seed": seed,
    }


def generate_records(config: dict[str, Any]) -> list[dict[str, Any]]:
    seed = int(config.get("seed", 1))
    set_global_seed(seed)
    rng = random.Random(seed)
    domain = config.get("fake_email_domain", "example-synth.org")
    proportions = config.get("families", {"identity": 0.4, "account": 0.3, "event": 0.3})
    train_count = int(config.get("n_train_member_records") or int(config.get("n_member_records", 3000) * 0.7))
    val_count = int(config.get("n_val_member_records") or max(1, int(config.get("n_member_records", 3000) * 0.1)))
    test_member_count = int(config.get("n_test_member_records") or max(1, config.get("n_member_records", 3000) - train_count - val_count))
    test_nonmember_count = int(config.get("n_test_nonmember_records") or int(config.get("n_nonmember_records", 3000)))
    canary_count = int(config.get("n_canaries", 300))
    repetition_buckets = list(config.get("canary_repetition_buckets", [1, 2, 5, 10, 20]))

    rows: list[dict[str, Any]] = []
    next_idx = 1

    def build(split: str, membership: str, count: int) -> None:
        nonlocal next_idx
        for _ in range(count):
            family = _family_from_distribution(rng, proportions)
            cluster_id = f"cluster_{next_idx:06d}"
            if family == "identity":
                row = _identity_record(next_idx, split, membership, cluster_id, seed, domain, rng)
            elif family == "account":
                row = _account_record(next_idx, split, membership, cluster_id, seed, domain, rng)
            else:
                row = _event_record(next_idx, split, membership, cluster_id, seed, rng)
            rows.append(row)
            next_idx += 1

    build("train_member", "member", train_count)
    build("val_member", "member", val_count)
    build("test_member", "member", test_member_count)
    build("test_nonmember", "nonmember", test_nonmember_count)

    for i in range(canary_count):
        bucket = f"{repetition_buckets[i % len(repetition_buckets)]}x"
        rows.append(_canary_record(next_idx + i, "train_member", seed, bucket, rng))

    LOGGER.info(
        "Generated %s records (%s).",
        len(rows),
        dict(Counter((row["split"], row["family"]) for row in rows)),
    )
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate synthetic M-CRATE records.")
    parser.add_argument("--config", required=True, help="YAML config for record generation.")
    parser.add_argument("--out", required=True, help="Output JSONL path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    rows = generate_records(config)
    write_jsonl(args.out, rows)
    LOGGER.info("Wrote records to %s", args.out)


if __name__ == "__main__":
    main()
