from __future__ import annotations

import argparse

from mcrate.utils.io import read_jsonl, write_jsonl
from mcrate.utils.logging import get_logger


LOGGER = get_logger(__name__)


def filter_rendered_docs(
    *,
    rendered_docs_path: str,
    out_path: str,
    include_families: list[str] | None = None,
    include_record_ids: list[str] | None = None,
    include_membership: list[str] | None = None,
) -> list[dict]:
    rows = read_jsonl(rendered_docs_path)
    family_set = set(include_families or [])
    record_set = set(include_record_ids or [])
    membership_set = set(include_membership or [])
    filtered = []
    for row in rows:
        if family_set and row.get("family") not in family_set:
            continue
        if record_set and row.get("record_id") not in record_set:
            continue
        if membership_set and row.get("membership") not in membership_set:
            continue
        filtered.append(row)
    write_jsonl(out_path, filtered)
    LOGGER.info("Filtered %s/%s rendered docs into %s", len(filtered), len(rows), out_path)
    return filtered


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Filter rendered training documents.")
    parser.add_argument("--rendered_docs", required=True, help="Rendered docs JSONL.")
    parser.add_argument("--out", required=True, help="Filtered JSONL output path.")
    parser.add_argument("--families", nargs="*", default=None, help="Optional list of families to keep.")
    parser.add_argument("--record_ids", nargs="*", default=None, help="Optional list of record ids to keep.")
    parser.add_argument("--membership", nargs="*", default=None, help="Optional list of membership labels to keep.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    filter_rendered_docs(
        rendered_docs_path=args.rendered_docs,
        out_path=args.out,
        include_families=args.families,
        include_record_ids=args.record_ids,
        include_membership=args.membership,
    )


if __name__ == "__main__":
    main()
