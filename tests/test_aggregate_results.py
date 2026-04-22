from __future__ import annotations

import unittest

from mcrate.audit.aggregate_results import aggregate


class AggregateResultsTests(unittest.TestCase):
    def test_zero_success_member_interval_is_not_degenerate(self) -> None:
        scores = []
        for idx in range(10):
            scores.append(
                {
                    "task_id": f"member_{idx}",
                    "record_id": f"rec_m_{idx}",
                    "family": "identity",
                    "membership": "member",
                    "condition": "C0_clean",
                    "cue_band": "low",
                    "any_sensitive_match": False,
                    "record_exact": False,
                    "field_f1": 0.0,
                    "target_logprob": -2.0,
                }
            )
            scores.append(
                {
                    "task_id": f"nonmember_{idx}",
                    "record_id": f"rec_n_{idx}",
                    "family": "identity",
                    "membership": "nonmember",
                    "condition": "C0_clean",
                    "cue_band": "low",
                    "any_sensitive_match": False,
                    "record_exact": False,
                    "field_f1": 0.0,
                    "target_logprob": -2.2,
                }
            )

        summary = aggregate(scores)
        row = summary["cue_table"][0]
        self.assertEqual(row["member_extraction"], 0.0)
        self.assertGreater(row["member_ci95"][1], 0.0)
        self.assertIn("lift_ci95", row)

    def test_task_level_field_f1_uses_best_of_n(self) -> None:
        scores = [
            {
                "task_id": "task_1",
                "record_id": "rec_1",
                "family": "identity",
                "membership": "member",
                "condition": "C2_exact_10x",
                "cue_band": "low",
                "any_sensitive_match": False,
                "record_exact": False,
                "field_f1": 0.2,
                "target_logprob": -2.0,
            },
            {
                "task_id": "task_1",
                "record_id": "rec_1",
                "family": "identity",
                "membership": "member",
                "condition": "C2_exact_10x",
                "cue_band": "low",
                "any_sensitive_match": True,
                "record_exact": False,
                "field_f1": 0.8,
                "target_logprob": -1.0,
            },
        ]

        summary = aggregate(scores)
        family_row = summary["family_table"][0]
        self.assertAlmostEqual(family_row["low_cue_field_f1"], 0.8, places=4)


if __name__ == "__main__":
    unittest.main()
