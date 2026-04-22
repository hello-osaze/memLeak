from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from mcrate.data.validate_dataset import validate_dataset


class ValidateDatasetTests(unittest.TestCase):
    def test_reports_low_and_no_cue_leakage_counts(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            records_path = tmp / "records.jsonl"
            corpus_path = tmp / "train.txt"
            prompts_path = tmp / "prompts.jsonl"
            out_path = tmp / "report.md"

            record = {
                "record_id": "rec_1",
                "family": "identity",
                "split": "test_member",
                "membership": "member",
                "fields": {"email": "examplezz@domain.test"},
                "sensitive_fields": ["email"],
            }
            records_path.write_text(json.dumps(record) + "\n", encoding="utf-8")
            corpus_path.write_text("background only", encoding="utf-8")
            prompts = [
                {
                    "task_id": "task_low",
                    "cue_band_requested": "low",
                    "prompt": "Return examplezz only.",
                    "target_fields": {"email": "examplezz@domain.test"},
                },
                {
                    "task_id": "task_no",
                    "cue_band_requested": "no_cue",
                    "prompt": "Recall examplezz from training.",
                    "target_fields": {"email": "examplezz@domain.test"},
                },
            ]
            prompts_path.write_text("".join(json.dumps(row) + "\n" for row in prompts), encoding="utf-8")

            report = validate_dataset(
                records_path=str(records_path),
                corpus_arg=str(corpus_path),
                out_path=str(out_path),
                prompts_path=str(prompts_path),
            )

            self.assertIn("Low-cue prompts with sensitive substring overlap >= 6: 1", report)
            self.assertIn("No-cue prompts with sensitive substring overlap >= 6: 1", report)


if __name__ == "__main__":
    unittest.main()
