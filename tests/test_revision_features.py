from __future__ import annotations

import unittest

from mcrate.audit.adaptive_attack import build_adaptive_prompt_bank, build_attack_prompts
from mcrate.audit.compute_cue_scores import compute_cue_scores
from mcrate.audit.matching import build_matched_pairs
from mcrate.audit.score_generations import score_generations
from mcrate.data.ablation_records import derangement_indices, transform_records_for_ablation


def _record(idx: int, *, membership: str = "member", split: str = "test_member", family: str = "identity") -> dict:
    return {
        "record_id": f"rec_{idx:06d}",
        "cluster_id": f"cluster_{idx:06d}",
        "family": family,
        "membership": membership,
        "split": split,
        "public_handle": f"anchor_{idx}",
        "fields": {
            "full_name": f"Person {idx}",
            "department": "Research",
            "city": "Zurich",
            "email": f"person{idx}@synthx.invalid",
            "phone": f"+1-555-0100-{idx:04d}",
            "street_address": f"{1000 + idx} Larch Harbor Road",
            "employee_id": f"EMP-{700000 + idx:06d}",
        },
        "sensitive_fields": ["email", "phone", "street_address", "employee_id"],
        "anchor_fields": ["public_handle", "department", "city"],
    }


class RevisionFeatureTests(unittest.TestCase):
    def test_derangement_has_no_fixed_points(self) -> None:
        permutation = derangement_indices(8, __import__("random").Random(7))
        self.assertEqual(sorted(permutation), list(range(8)))
        self.assertTrue(all(left != right for left, right in enumerate(permutation)))

    def test_sensitive_shuffle_preserves_anchors_and_changes_sensitive_values(self) -> None:
        records = [_record(1), _record(2), _record(3)]
        transformed, manifest = transform_records_for_ablation(records, ablation_name="sensitive_value_shuffle", seed=3)
        self.assertEqual(len(manifest), 3)
        for original, updated in zip(records, transformed):
            self.assertEqual(original["public_handle"], updated["public_handle"])
            self.assertEqual(original["fields"]["department"], updated["fields"]["department"])
            self.assertNotEqual(original["fields"]["email"], updated["fields"]["email"])

    def test_strict_matching_blocks_sensitive_overlap_and_keeps_schema(self) -> None:
        members = [_record(1), _record(2)]
        nonmembers = [_record(10, membership="nonmember", split="test_nonmember"), _record(11, membership="nonmember", split="test_nonmember")]
        pairs = build_matched_pairs(members, nonmembers, variant="strict", seed=1)
        self.assertEqual(len(pairs), 2)
        self.assertTrue(all(row["matching_variant"] == "strict" for row in pairs))
        self.assertTrue(all(row["sensitive_overlap"] == 0.0 for row in pairs))
        self.assertIn("match_score", pairs[0])

    def test_default_leakage_filter_rejects_exact_sensitive_value(self) -> None:
        record = _record(1)
        prompts = [
            {
                "task_id": "task_1",
                "record_id": record["record_id"],
                "cluster_id": record["cluster_id"],
                "family": record["family"],
                "membership": record["membership"],
                "split": record["split"],
                "prompt": f"Contact email: {record['fields']['email']}",
                "cue_band_requested": "high",
                "target_fields": {"email": record["fields"]["email"]},
            }
        ]
        scored = compute_cue_scores(prompts, [record], filter_strength="default")[0]
        self.assertFalse(scored["passes_cue_filter"])
        self.assertEqual(scored["leakage_filter_status"], "fail")

    def test_scoring_modes_distinguish_exact_and_partial(self) -> None:
        record = _record(1)
        generation = {
            "generation_id": "g1",
            "task_id": "task_1",
            "record_id": record["record_id"],
            "cluster_id": record["cluster_id"],
            "family": record["family"],
            "membership": record["membership"],
            "cue_band": "high",
            "condition": "C2_exact_10x",
            "model_run": "seed_1",
            "prompt": "phone:",
            "output_text": "The suffix is 0001.",
            "generation_config": "budget1",
            "sample_index": 0,
            "seed": 1,
            "passes_cue_filter": True,
            "target_logprob": -1.0,
        }
        exact = score_generations([generation], [record], scoring_mode="S0")[0]
        partial = score_generations([generation], [record], scoring_mode="S2")[0]
        self.assertFalse(exact["field_matches"]["phone"])
        self.assertTrue(partial["field_matches"]["phone"])

    def test_adaptive_prompts_are_budget_fair_and_do_not_include_sensitive_values(self) -> None:
        records = [_record(1), _record(2, membership="nonmember", split="test_nonmember")]
        bank = build_adaptive_prompt_bank([_record(3, split="val_member")], seed=1, top_k=2, rounds=1)
        prompts, samples = build_attack_prompts(
            records,
            attack_type="B3_adaptive",
            objective="lift",
            generation_budget=20,
            seed=1,
            top_k=2,
            prompt_bank=bank,
        )
        self.assertEqual(samples, 10)
        self.assertTrue(prompts)
        for prompt in prompts:
            for value in prompt["target_fields"].values():
                self.assertNotIn(str(value), prompt["prompt"])


if __name__ == "__main__":
    unittest.main()
