from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mcrate.data.build_corpus import _document_signature, _iter_background_documents, _normalize_doc_text, _normalized_for_match, _token_count
from mcrate.utils.io import ensure_dir, write_json, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a deduplicated public-background document bundle.")
    parser.add_argument("--sources", nargs="+", required=True, help="Source files or directories containing public documents.")
    parser.add_argument("--out", required=True, help="Output JSONL path.")
    parser.add_argument("--manifest", default=None, help="Optional manifest JSON path.")
    parser.add_argument("--min-doc-tokens", type=int, default=40, help="Minimum token count per kept document.")
    parser.add_argument("--limit-docs", type=int, default=0, help="Optional maximum number of output documents.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = []
    exact_seen: set[str] = set()
    near_seen: set[str] = set()
    stats = {
        "source_documents": 0,
        "kept_documents": 0,
        "filtered_short": 0,
        "filtered_exact_dupe": 0,
        "filtered_near_dupe": 0,
        "sources": [str(Path(source).resolve()) for source in args.sources],
        "min_doc_tokens": int(args.min_doc_tokens),
    }

    for source in args.sources:
        for doc in _iter_background_documents(source):
            stats["source_documents"] += 1
            normalized = _normalize_doc_text(doc)
            if _token_count(normalized) < int(args.min_doc_tokens):
                stats["filtered_short"] += 1
                continue
            exact_key = _normalized_for_match(normalized)
            if exact_key in exact_seen:
                stats["filtered_exact_dupe"] += 1
                continue
            near_key = _document_signature(normalized)
            if near_key in near_seen:
                stats["filtered_near_dupe"] += 1
                continue
            exact_seen.add(exact_key)
            near_seen.add(near_key)
            rows.append({"doc_id": f"bg_{len(rows):07d}", "text": normalized, "source": str(source)})
            if int(args.limit_docs) > 0 and len(rows) >= int(args.limit_docs):
                break
        if int(args.limit_docs) > 0 and len(rows) >= int(args.limit_docs):
            break

    stats["kept_documents"] = len(rows)
    out_path = Path(args.out)
    ensure_dir(out_path.parent)
    write_jsonl(out_path, rows)

    manifest_path = Path(args.manifest) if args.manifest else out_path.with_suffix(".manifest.json")
    write_json(manifest_path, stats)
    print(f"Wrote {len(rows)} background documents to {out_path}")
    print(f"Wrote manifest to {manifest_path}")


if __name__ == "__main__":
    main()
