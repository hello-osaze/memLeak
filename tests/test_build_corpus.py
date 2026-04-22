from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from mcrate.data.build_corpus import build_corpus
from mcrate.utils.io import write_jsonl


class BuildCorpusTests(unittest.TestCase):
    def test_document_sample_background_uses_doc_split_and_filters_contamination(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            background_path = root / "background_docs.jsonl"
            rendered_docs_path = root / "rendered.jsonl"
            records_path = root / "records.jsonl"
            out_dir = root / "corpus"

            write_jsonl(
                background_path,
                [
                    {"text": "General planning memo about schedules and routes for ordinary operations teams."},
                    {"text": "Documentation summary about release checklists and routine maintenance reviews."},
                    {"text": "Synthetic note that mentions EMP 000001 and should be filtered from background."},
                    {"text": "Administrative guidance on travel templates, queues, and benchmark reporting."},
                    {"text": "Support operations prose about service routing and issue triage across regions."},
                ],
            )
            write_jsonl(
                rendered_docs_path,
                [
                    {
                        "doc_id": "doc_1",
                        "record_id": "rec_1",
                        "cluster_id": "cluster_1",
                        "condition": "C2_exact_10x",
                        "variant_type": "exact_duplicate",
                        "template_id": "identity_template_01",
                        "family": "identity",
                        "text": "Private synthetic record with employee id EMP-000001 and mailbox ada@synthx.invalid.",
                        "included_sensitive_fields": ["employee_id", "email"],
                        "source_records_path": str(records_path),
                    }
                ],
            )
            write_jsonl(
                records_path,
                [
                    {
                        "record_id": "rec_1",
                        "public_handle": "staff_anchor_TEST",
                        "fields": {
                            "employee_id": "EMP-000001",
                            "email": "ada@synthx.invalid",
                        },
                    }
                ],
            )

            manifest = build_corpus(
                background_path=str(background_path),
                rendered_docs_path=str(rendered_docs_path),
                config={
                    "condition": "C2_exact_10x",
                    "background_tokens": 12,
                    "synthetic_token_target": 5,
                    "background_sampling_mode": "document_sample",
                    "background_val_fraction": 0.25,
                    "background_validation_min_tokens": 5,
                    "min_background_doc_tokens": 5,
                    "background_dedup_exact": True,
                    "background_dedup_near": True,
                    "background_allow_reuse": False,
                    "shuffle_seed": 1,
                },
                out_dir=str(out_dir),
                records_path=str(records_path),
            )

            self.assertEqual(manifest["background_mode"], "document_sample")
            self.assertGreaterEqual(manifest["background_train_doc_count"], 1)
            self.assertGreaterEqual(manifest["background_validation_doc_count"], 1)
            self.assertEqual(manifest["background_filtered_contamination_count"], 1)
            train_text = (out_dir / "train.txt").read_text()
            validation_text = (out_dir / "validation.txt").read_text()
            self.assertNotIn("EMP 000001", train_text)
            self.assertNotEqual(train_text, validation_text)

    def test_prebuilt_background_bundle_is_consumed_directly(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            background_dir = root / "background_bundle"
            rendered_docs_path = root / "rendered.jsonl"
            out_dir = root / "corpus"
            background_dir.mkdir()

            write_jsonl(
                background_dir / "background_docs_train.jsonl",
                [
                    {"doc_id": "bg_train_1", "text": "General C4-like training prose about schedules and documentation."},
                    {"doc_id": "bg_train_2", "text": "Administrative planning language for product releases and support routing."},
                ],
            )
            write_jsonl(
                background_dir / "background_docs_val.jsonl",
                [
                    {"doc_id": "bg_val_1", "text": "Held-out public validation prose about reviews and maintenance windows."},
                ],
            )
            (background_dir / "background_manifest.json").write_text(
                json.dumps(
                    {
                        "dataset": "allenai/c4",
                        "config": "en",
                        "source_documents": 3,
                        "kept_documents": 3,
                        "filtered_short": 0,
                        "filtered_exact_dupe": 0,
                        "filtered_near_dupe": 0,
                        "val_fraction": 0.25,
                    }
                )
            )
            write_jsonl(
                rendered_docs_path,
                [
                    {
                        "doc_id": "doc_1",
                        "record_id": "rec_1",
                        "cluster_id": "cluster_1",
                        "condition": "C0_clean",
                        "variant_type": "exact_duplicate",
                        "template_id": "identity_template_01",
                        "family": "identity",
                        "text": "Private synthetic record text.",
                        "included_sensitive_fields": [],
                        "source_records_path": str(root / "records.jsonl"),
                    }
                ],
            )

            manifest = build_corpus(
                background_path=str(background_dir),
                rendered_docs_path=str(rendered_docs_path),
                config={
                    "condition": "C0_clean",
                    "background_tokens": 8,
                    "synthetic_token_target": 0,
                    "background_sampling_mode": "document_sample",
                    "background_val_fraction": 0.25,
                    "background_validation_min_tokens": 4,
                    "background_allow_reuse": False,
                    "shuffle_seed": 1,
                },
                out_dir=str(out_dir),
            )

            self.assertEqual(manifest["background_mode"], "prebuilt_document_bundle")
            self.assertEqual(manifest["background_bundle_dataset"], "allenai/c4")
            self.assertGreaterEqual(manifest["background_train_doc_count"], 1)
            self.assertGreaterEqual(manifest["background_validation_doc_count"], 1)


if __name__ == "__main__":
    unittest.main()
