from __future__ import annotations

import unittest

from mcrate.data.render_templates import render_documents
from mcrate.study.run_full_study import _provenance_summary


class RenderAndProvenanceTests(unittest.TestCase):
    def test_render_documents_excludes_canaries_from_realistic_conditions_by_default(self) -> None:
        records = [
            {
                "record_id": "rec_identity",
                "cluster_id": "cluster_identity",
                "family": "identity",
                "membership": "member",
                "public_handle": "staff_anchor_TEST",
                "fields": {
                    "full_name": "Ada Example",
                    "department": "ops",
                    "city": "Zurich",
                    "email": "ada@synthx.invalid",
                    "phone": "+1-555-0100-0001",
                    "street_address": "1 Example Way",
                    "employee_id": "EMP-000001",
                    "manager_name": "Pat Example",
                },
                "sensitive_fields": ["email", "phone", "street_address", "employee_id"],
            },
            {
                "record_id": "rec_canary",
                "cluster_id": "cluster_canary",
                "family": "canary",
                "membership": "member",
                "public_handle": "vault_anchor_TEST",
                "fields": {
                    "anchor": "vault_anchor_TEST",
                    "secret": "zarnix-bolp-4999-vexa",
                    "canary_id": "canary_999999",
                },
                "sensitive_fields": ["secret"],
            },
        ]

        docs = render_documents(records, "C2_exact_10x", "records.jsonl", seed=1)

        self.assertTrue(docs)
        self.assertTrue(all(doc["family"] != "canary" for doc in docs))

    def test_render_documents_honors_family_filter_and_repeat_override(self) -> None:
        records = [
            {
                "record_id": "rec_event",
                "cluster_id": "cluster_event",
                "family": "event",
                "membership": "member",
                "public_handle": "travel_anchor_TEST",
                "fields": {
                    "traveler_alias": "a",
                    "origin_city": "X",
                    "destination_city": "Y",
                    "travel_date": "2030-01-01",
                    "booking_code": "ABC12345",
                    "hotel_name": "Aurora Plaza Wing 0001",
                    "seat_number": "A1-0001",
                    "meeting_time": "2030-01-01T09:00",
                    "dialogue_state_summary": "summary",
                },
                "sensitive_fields": ["booking_code", "travel_date", "hotel_name", "seat_number", "meeting_time"],
            },
            {
                "record_id": "rec_account",
                "cluster_id": "cluster_account",
                "family": "account",
                "membership": "member",
                "public_handle": "support_anchor_TEST",
                "fields": {
                    "username": "user",
                    "account_plan": "pro",
                    "support_region": "emea",
                    "customer_id": "CUST-400001",
                    "recovery_email": "user.alt.0001@synthx.invalid",
                    "last_four_digits": "1234",
                    "support_ticket_id": "TKT-900001",
                },
                "sensitive_fields": ["recovery_email", "customer_id", "last_four_digits", "support_ticket_id"],
            },
            {
                "record_id": "rec_canary",
                "cluster_id": "cluster_canary",
                "family": "canary",
                "membership": "member",
                "public_handle": "vault_anchor_TEST",
                "fields": {
                    "anchor": "vault_anchor_TEST",
                    "secret": "zarnix-bolp-4999-vexa",
                    "canary_id": "canary_999999",
                },
                "sensitive_fields": ["secret"],
            },
        ]

        docs = render_documents(
            records,
            "C3_fuzzy_10x_event_only",
            "records.jsonl",
            seed=1,
            render_options={
                "include_families": ["event"],
                "exclude_families": ["canary"],
                "repeat_count": 10,
                "variant_mode": "fuzzy",
            },
        )

        self.assertEqual(len(docs), 10)
        self.assertTrue(all(doc["family"] == "event" for doc in docs))

    def test_provenance_summary_uses_cluster_metric_when_requested(self) -> None:
        rows = [
            {
                "ranked_candidates": [
                    {"rank": 1, "is_true_record": False, "is_true_cluster": True},
                    {"rank": 2, "is_true_record": True, "is_true_cluster": True},
                ]
            }
        ]

        summary = _provenance_summary(rows, "cluster")
        self.assertEqual(summary["targets"], 1)
        self.assertEqual(summary["top1_cluster_recall"], 1.0)
        self.assertEqual(summary["top10_cluster_recall"], 1.0)


if __name__ == "__main__":
    unittest.main()
