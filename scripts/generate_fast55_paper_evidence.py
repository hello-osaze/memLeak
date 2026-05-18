from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any


CONDITIONS = {
    "c0_clean": r"\czer{}",
    "c2_exact_10x": r"\ctwo{}",
    "c3_fuzzy_5x": r"\cthree{}",
}
PROVENANCE_UNITS = {"c2_exact_10x": "record", "c3_fuzzy_5x": "cluster"}
BUDGETS = [5, 20]
TOKEN_RE = re.compile(r"[a-z0-9_@.+-]+")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def collapse_scores(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["task_id"]].append(row)
    collapsed = []
    for task_rows in grouped.values():
        sample = dict(task_rows[0])
        target_logprobs = [float(row["target_logprob"]) for row in task_rows if row.get("target_logprob") is not None]
        sample["success"] = any(bool(row.get("any_sensitive_match")) for row in task_rows)
        sample["record_exact_any"] = any(bool(row.get("record_exact")) for row in task_rows)
        sample["max_target_logprob"] = max(target_logprobs) if target_logprobs else None
        collapsed.append(sample)
    return collapsed


def wilson_ci(successes: float, total: int, z: float = 1.96) -> tuple[float, float]:
    if total <= 0:
        return 0.0, 0.0
    phat = successes / total
    denom = 1 + z * z / total
    center = (phat + z * z / (2 * total)) / denom
    margin = z * math.sqrt((phat * (1 - phat) + z * z / (4 * total)) / total) / denom
    return max(0.0, center - margin), min(1.0, center + margin)


def bootstrap_ci(values: list[float], *, seed: int = 1, rounds: int = 2000) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    if len(values) == 1:
        return values[0], values[0]
    rng = random.Random(seed)
    means = []
    for _ in range(rounds):
        sample = [values[rng.randrange(len(values))] for _ in values]
        means.append(sum(sample) / len(sample))
    means.sort()
    return means[int(0.025 * rounds)], means[min(rounds - 1, int(0.975 * rounds))]


def reciprocal_rank(ranks: list[int]) -> float:
    return 0.0 if not ranks else 1.0 / min(ranks)


def random_topk_expected(true_count: int, pool_size: int, k: int) -> float:
    if pool_size <= 0 or true_count <= 0:
        return 0.0
    k = min(k, pool_size)
    if true_count >= pool_size:
        return 1.0
    if pool_size - true_count < k:
        return 1.0
    return 1.0 - math.comb(pool_size - true_count, k) / math.comb(pool_size, k)


def random_mrr_expected(true_count: int, pool_size: int) -> float:
    if pool_size <= 0 or true_count <= 0:
        return 0.0
    total = math.comb(pool_size, true_count)
    max_rank = pool_size - true_count + 1
    expected = 0.0
    for rank in range(1, max_rank + 1):
        expected += (math.comb(pool_size - rank, true_count - 1) / total) / rank
    return expected


def metric_from_ranked(ranked: list[dict[str, Any]], match_key: str) -> tuple[float, float, float]:
    true_ranks = [int(row["rank"]) for row in ranked if row.get(match_key)]
    return (
        1.0 if true_ranks and min(true_ranks) == 1 else 0.0,
        1.0 if true_ranks and min(true_ranks) <= 10 else 0.0,
        reciprocal_rank(true_ranks),
    )


def summarize_metric_rows(
    *,
    condition: str,
    unit: str,
    selected_targets: int,
    successful_targets: int,
    method: str,
    metric_rows: list[tuple[float, float, float]],
) -> dict[str, Any]:
    top1_values = [row[0] for row in metric_rows]
    top10_values = [row[1] for row in metric_rows]
    mrr_values = [row[2] for row in metric_rows]
    top1 = mean(top1_values) if top1_values else 0.0
    top10 = mean(top10_values) if top10_values else 0.0
    mrr = mean(mrr_values) if mrr_values else 0.0
    if all(value in {0.0, 1.0} for value in top1_values):
        top1_low, top1_high = wilson_ci(sum(top1_values), len(top1_values))
        top10_low, top10_high = wilson_ci(sum(top10_values), len(top10_values))
    else:
        top1_low, top1_high = bootstrap_ci(top1_values, seed=11)
        top10_low, top10_high = bootstrap_ci(top10_values, seed=13)
    mrr_low, mrr_high = bootstrap_ci(mrr_values, seed=17)
    return {
        "condition": condition,
        "unit": unit,
        "successful_highcue_member_targets": successful_targets,
        "selected_targets": selected_targets,
        "selected_fraction": round(selected_targets / successful_targets, 4) if successful_targets else 0.0,
        "method": method,
        "top1": round(top1, 4),
        "top1_ci95_low": round(top1_low, 4),
        "top1_ci95_high": round(top1_high, 4),
        "top10": round(top10, 4),
        "top10_ci95_low": round(top10_low, 4),
        "top10_ci95_high": round(top10_high, 4),
        "mrr": round(mrr, 4),
        "mrr_ci95_low": round(mrr_low, 4),
        "mrr_ci95_high": round(mrr_high, 4),
    }


def successful_highcue_member_count(score_path: Path) -> int:
    collapsed = collapse_scores(read_jsonl(score_path))
    return sum(
        1
        for row in collapsed
        if row["membership"] == "member" and row["cue_band"] == "high" and bool(row["success"])
    )


def bm25_scores(query: str, docs: list[dict[str, Any]]) -> dict[str, float]:
    query_terms = tokenize(query)
    if not query_terms:
        return {doc["doc_id"]: 0.0 for doc in docs}
    doc_terms = {doc["doc_id"]: tokenize(str(doc.get("text", ""))) for doc in docs}
    avgdl = mean([len(tokens) for tokens in doc_terms.values()]) if doc_terms else 1.0
    doc_freq = Counter()
    for terms in doc_terms.values():
        doc_freq.update(set(terms))
    k1 = 1.5
    b = 0.75
    n_docs = max(1, len(docs))
    scores = {}
    for doc in docs:
        doc_id = doc["doc_id"]
        terms = doc_terms[doc_id]
        tf = Counter(terms)
        doc_len = max(1, len(terms))
        score = 0.0
        for term in query_terms:
            if tf[term] <= 0:
                continue
            idf = math.log(1 + (n_docs - doc_freq[term] + 0.5) / (doc_freq[term] + 0.5))
            denom = tf[term] + k1 * (1 - b + b * doc_len / max(avgdl, 1.0))
            score += idf * (tf[term] * (k1 + 1)) / denom
        scores[doc_id] = score
    return scores


def deidentify_prompt(prompt: str, record: dict[str, Any]) -> str:
    """Remove record-specific strings so lexical retrieval cannot key on the target."""
    cleaned = prompt
    values = [record.get("public_handle", "")]
    values.extend(str(value) for value in record.get("fields", {}).values())
    for value in sorted({value for value in values if value}, key=len, reverse=True):
        if len(value) >= 3:
            cleaned = re.sub(re.escape(value), " ", cleaned, flags=re.IGNORECASE)
    return " ".join(cleaned.split())


def ranked_overlap_metrics(
    *,
    query: str,
    candidate_docs: list[dict[str, Any]],
    pool: dict[str, Any],
    task: dict[str, Any],
    match_key: str,
) -> tuple[float, float, float]:
    overlap_scores = bm25_scores(query, candidate_docs)
    ranked_docs = sorted(candidate_docs, key=lambda doc: (overlap_scores[doc["doc_id"]], doc["doc_id"]), reverse=True)
    ranked = []
    for rank, doc in enumerate(ranked_docs, start=1):
        ranked.append(
            {
                "rank": rank,
                "is_true_record": doc["record_id"] == pool["target_record_id"],
                "is_true_cluster": doc["cluster_id"] == task["cluster_id"],
            }
        )
    return metric_from_ranked(ranked, match_key)


def generate_provenance_outputs(run_root: Path, out_dir: Path) -> list[dict[str, Any]]:
    rows = []
    records = {row["record_id"]: row for row in read_jsonl(run_root / "data" / "records" / "workshop_realistic_records.jsonl")}
    for condition, unit in PROVENANCE_UNITS.items():
        prov_dir = run_root / "outputs" / "provenance" / condition / "seed_1" / "budget5"
        pools_path = prov_dir / "candidate_pools.jsonl"
        gradient_path = prov_dir / "gradient_similarity.jsonl"
        rendered_path = run_root / "data" / "processed" / f"{condition}__rendered_docs.jsonl"
        score_path = run_root / "outputs" / "scores" / condition / "seed_1" / "budget5_scores.jsonl"
        if not (pools_path.exists() and gradient_path.exists() and rendered_path.exists() and score_path.exists()):
            continue

        selected_unit = "cluster" if unit == "cluster" else "record"
        match_key = "is_true_cluster" if selected_unit == "cluster" else "is_true_record"
        docs = {row["doc_id"]: row for row in read_jsonl(rendered_path)}
        scores = {row["task_id"]: row for row in collapse_scores(read_jsonl(score_path))}
        pools = {row["target_task_id"]: row for row in read_jsonl(pools_path)}
        gradient_rows = read_jsonl(gradient_path)
        successful_targets = successful_highcue_member_count(score_path)

        gradient_metrics = [metric_from_ranked(row["ranked_candidates"], match_key) for row in gradient_rows]
        rows.append(
            summarize_metric_rows(
                condition=condition,
                unit=unit,
                selected_targets=len(gradient_rows),
                successful_targets=successful_targets,
                method="gradient_similarity",
                metric_rows=gradient_metrics,
            )
        )

        random_metrics = []
        same_template_metrics = []
        bm25_full_metrics = []
        bm25_deidentified_metrics = []
        for grad_row in gradient_rows:
            task_id = grad_row["target_task_id"]
            task = scores[task_id]
            pool = pools[task_id]
            candidate_docs = [docs[doc_id] for doc_id in pool["candidate_doc_ids"]]
            if unit == "cluster":
                true_flags = [doc["cluster_id"] == task["cluster_id"] for doc in candidate_docs]
            else:
                true_flags = [doc["record_id"] == pool["target_record_id"] for doc in candidate_docs]
            true_count = sum(1 for flag in true_flags if flag)
            pool_size = len(candidate_docs)
            random_metrics.append(
                (
                    true_count / pool_size if pool_size else 0.0,
                    random_topk_expected(true_count, pool_size, 10),
                    random_mrr_expected(true_count, pool_size),
                )
            )

            true_template_ids = {doc["template_id"] for doc, flag in zip(candidate_docs, true_flags) if flag}
            template_group_size = sum(1 for doc in candidate_docs if doc["template_id"] in true_template_ids)
            same_template_metrics.append(
                (
                    true_count / template_group_size if template_group_size else 0.0,
                    random_topk_expected(true_count, template_group_size, 10),
                    random_mrr_expected(true_count, template_group_size),
                )
            )

            bm25_full_metrics.append(
                ranked_overlap_metrics(
                    query=task["prompt"],
                    candidate_docs=candidate_docs,
                    pool=pool,
                    task=task,
                    match_key=match_key,
                )
            )
            record = records.get(pool["target_record_id"], {"fields": {}, "public_handle": ""})
            bm25_deidentified_metrics.append(
                ranked_overlap_metrics(
                    query=deidentify_prompt(task["prompt"], record),
                    candidate_docs=candidate_docs,
                    pool=pool,
                    task=task,
                    match_key=match_key,
                )
            )

        for method, method_rows in [
            ("random_same_family_expected", random_metrics),
            ("same_template_first_expected", same_template_metrics),
            ("bm25_full_prompt_overlap", bm25_full_metrics),
            ("bm25_deidentified_prompt_overlap", bm25_deidentified_metrics),
        ]:
            rows.append(
                summarize_metric_rows(
                    condition=condition,
                    unit=unit,
                    selected_targets=len(gradient_rows),
                    successful_targets=successful_targets,
                    method=method,
                    metric_rows=method_rows,
                )
            )

    fieldnames = [
        "condition",
        "unit",
        "successful_highcue_member_targets",
        "selected_targets",
        "selected_fraction",
        "method",
        "top1",
        "top1_ci95_low",
        "top1_ci95_high",
        "top10",
        "top10_ci95_low",
        "top10_ci95_high",
        "mrr",
        "mrr_ci95_low",
        "mrr_ci95_high",
    ]
    write_csv(out_dir / "realistic_provenance_baselines.csv", rows, fieldnames)
    write_text(out_dir / "realistic_provenance_baselines.tex", provenance_tex(rows))
    return rows


def provenance_tex(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "% Provenance artifacts not found; run on the cloud bundle with candidate pools and gradient outputs.\n"
    method_labels = {
        "gradient_similarity": "Gradient similarity",
        "random_same_family_expected": "Random same-family",
        "same_template_first_expected": "Same-template first",
        "bm25_full_prompt_overlap": "BM25 full prompt",
        "bm25_deidentified_prompt_overlap": "BM25 de-identified prompt",
    }
    lines = [
        r"\begin{table}[!t]",
        r"\centering",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{3pt}",
        r"\begin{tabular}{llrrrr}",
        r"\toprule",
        r"Condition & Method & Top-1 & Top-10 & MRR & Coverage \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(
            f"{CONDITIONS.get(row['condition'], row['condition'])} & {method_labels.get(row['method'], row['method'])} & "
            f"{100 * float(row['top1']):.1f}\\% [{100 * float(row['top1_ci95_low']):.1f}, {100 * float(row['top1_ci95_high']):.1f}] & "
            f"{100 * float(row['top10']):.1f}\\% [{100 * float(row['top10_ci95_low']):.1f}, {100 * float(row['top10_ci95_high']):.1f}] & "
            f"{float(row['mrr']):.3f} & "
            f"{row['selected_targets']}/{row['successful_highcue_member_targets']} \\\\"
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\caption{Realistic-record provenance with stronger candidate-pool baselines. Coverage reports selected provenance targets over all successful high-cue member targets at $B=5$.}",
            r"\label{tab:realistic-provenance-baselines}",
            r"\end{table}",
            "",
        ]
    )
    return "\n".join(lines)


def target_text(record: dict[str, Any]) -> str:
    return " ".join(str(record.get("fields", {}).get(field, "")) for field in record.get("sensitive_fields", []))


def generate_matched_control_outputs(run_root: Path, out_dir: Path) -> list[dict[str, Any]]:
    records = read_jsonl(run_root / "data" / "records" / "audit_targets.jsonl")
    prompts = read_jsonl(run_root / "data" / "prompts" / "scored_prompts.jsonl")
    record_groups = defaultdict(list)
    prompt_groups = defaultdict(list)
    for row in records:
        record_groups[row["membership"]].append(row)
    for row in prompts:
        if row.get("passes_cue_filter", True):
            prompt_groups[row["membership"]].append(row)

    def record_metric(membership: str, func: Any) -> float:
        values = [func(row) for row in record_groups[membership]]
        return mean(values) if values else 0.0

    def prompt_metric(membership: str, func: Any) -> float:
        values = [func(row) for row in prompt_groups[membership]]
        return mean(values) if values else 0.0

    rows: list[dict[str, Any]] = []

    def add(metric: str, member: float | int, nonmember: float | int, category: str = "summary") -> None:
        rows.append(
            {
                "category": category,
                "metric": metric,
                "member": round(member, 4) if isinstance(member, float) else member,
                "nonmember": round(nonmember, 4) if isinstance(nonmember, float) else nonmember,
                "difference": round(float(member) - float(nonmember), 4),
            }
        )

    add("records_total", len(record_groups["member"]), len(record_groups["nonmember"]))
    for family in ["identity", "account", "event"]:
        add(
            f"family_{family}_records",
            sum(1 for row in record_groups["member"] if row["family"] == family),
            sum(1 for row in record_groups["nonmember"] if row["family"] == family),
            "family",
        )
    add("mean_anchor_field_count", record_metric("member", lambda row: len(row.get("anchor_fields", []))), record_metric("nonmember", lambda row: len(row.get("anchor_fields", []))))
    add("mean_sensitive_field_count", record_metric("member", lambda row: len(row.get("sensitive_fields", []))), record_metric("nonmember", lambda row: len(row.get("sensitive_fields", []))))
    add("mean_target_text_chars", record_metric("member", lambda row: len(target_text(row))), record_metric("nonmember", lambda row: len(target_text(row))))
    add("median_target_text_chars", median([len(target_text(row)) for row in record_groups["member"]]), median([len(target_text(row)) for row in record_groups["nonmember"]]))
    add("mean_prompt_tokens", prompt_metric("member", lambda row: len(tokenize(row["prompt"]))), prompt_metric("nonmember", lambda row: len(tokenize(row["prompt"]))))
    add("mean_prompt_chars", prompt_metric("member", lambda row: len(row["prompt"])), prompt_metric("nonmember", lambda row: len(row["prompt"])))

    sensitive_fields = sorted({field for group in record_groups.values() for row in group for field in row.get("sensitive_fields", [])})
    for field in sensitive_fields:
        add(
            f"sensitive_field_{field}_records",
            sum(1 for row in record_groups["member"] if field in row.get("sensitive_fields", [])),
            sum(1 for row in record_groups["nonmember"] if field in row.get("sensitive_fields", [])),
            "sensitive_field",
        )

    template_ids = sorted({row["prompt_template_id"] for row in prompts if row.get("passes_cue_filter", True)})
    for template_id in template_ids:
        add(
            f"prompt_template_{template_id}",
            sum(1 for row in prompt_groups["member"] if row["prompt_template_id"] == template_id),
            sum(1 for row in prompt_groups["nonmember"] if row["prompt_template_id"] == template_id),
            "prompt_template",
        )

    fieldnames = ["category", "metric", "member", "nonmember", "difference"]
    write_csv(out_dir / "matched_control_balance.csv", rows, fieldnames)
    write_text(out_dir / "matched_control_balance.tex", matched_balance_tex(rows))
    write_text(out_dir / "matched_control_dimensions.tex", matched_dimensions_tex())
    return rows


def matched_dimensions_tex() -> str:
    return "\n".join(
        [
            r"\begin{table}[!t]",
            r"\centering",
            r"\small",
            r"\begin{tabular}{ll}",
            r"\toprule",
            r"Control dimension & Matched or held fixed how? \\",
            r"\midrule",
            r"Record family & Distribution-matched across identity, account, and event records \\",
            r"Prompt template & Identical prompt-template catalog for members and nonmembers \\",
            r"Cue band & Identical computed cue-band filter and cue-band labels \\",
            r"Decoding budget & Identical $B \in \{1,5,20\}$ generation budgets \\",
            r"Scoring rule & Identical sensitive-field extractor and record-exact matcher \\",
            r"Sensitive-field category & Same family-specific sensitive-field schema \\",
            r"Anchor-field count & Same generator schema and balanced audit sample \\",
            r"Target string length & Checked by member/nonmember balance diagnostics \\",
            r"\bottomrule",
            r"\end{tabular}",
            r"\caption{Distribution-matched nonmember controls used by \mcrate{}.}",
            r"\label{tab:matched-control-dimensions}",
            r"\end{table}",
            "",
        ]
    )


def matched_balance_tex(rows: list[dict[str, Any]]) -> str:
    wanted = [
        "records_total",
        "family_identity_records",
        "family_account_records",
        "family_event_records",
        "mean_anchor_field_count",
        "mean_sensitive_field_count",
        "mean_target_text_chars",
        "mean_prompt_tokens",
    ]
    row_map = {row["metric"]: row for row in rows}
    labels = {
        "records_total": "Records",
        "family_identity_records": "Identity records",
        "family_account_records": "Account records",
        "family_event_records": "Event records",
        "mean_anchor_field_count": "Mean anchor fields",
        "mean_sensitive_field_count": "Mean sensitive fields",
        "mean_target_text_chars": "Mean target chars",
        "mean_prompt_tokens": "Mean prompt tokens",
    }
    lines = [
        r"\begin{table}[!t]",
        r"\centering",
        r"\small",
        r"\begin{tabular}{lrrr}",
        r"\toprule",
        r"Diagnostic & Member & Nonmember & Difference \\",
        r"\midrule",
    ]
    for metric in wanted:
        row = row_map[metric]
        lines.append(f"{labels[metric]} & {row['member']} & {row['nonmember']} & {row['difference']} \\\\")
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\caption{Balance diagnostics for the distribution-matched member and nonmember audit samples.}",
            r"\label{tab:matched-control-balance}",
            r"\end{table}",
            "",
        ]
    )
    return "\n".join(lines)


def prompt_template_map(run_root: Path) -> dict[str, dict[str, Any]]:
    return {row["task_id"]: row for row in read_jsonl(run_root / "data" / "prompts" / "scored_prompts.jsonl")}


def generate_best_of_template_outputs(run_root: Path, out_dir: Path) -> list[dict[str, Any]]:
    prompts = prompt_template_map(run_root)
    rows = []
    missing_score_files = []
    for condition in CONDITIONS:
        if condition not in {"c0_clean", "c2_exact_10x", "c3_fuzzy_5x"}:
            continue
        for budget in BUDGETS:
            seed_rates = {"fixed_template_1": [], "best_of_highcue_templates": []}
            for seed in [1, 2, 3]:
                score_path = run_root / "outputs" / "scores" / condition / f"seed_{seed}" / f"budget{budget}_scores.jsonl"
                if not score_path.exists():
                    missing_score_files.append(str(score_path))
                    continue
                collapsed = collapse_scores(read_jsonl(score_path))
                for row in collapsed:
                    prompt_row = prompts.get(row["task_id"], {})
                    row["prompt_template_id"] = prompt_row.get("prompt_template_id", "")
                high_rows = [row for row in collapsed if row["cue_band"] == "high"]
                for mode in seed_rates:
                    if mode == "fixed_template_1":
                        eval_rows = [row for row in high_rows if str(row.get("prompt_template_id", "")).endswith("_01")]
                        grouped = {(row["membership"], row["record_id"]): bool(row["success"]) for row in eval_rows}
                    else:
                        grouped_success: dict[tuple[str, str], bool] = defaultdict(bool)
                        for row in high_rows:
                            grouped_success[(row["membership"], row["record_id"])] = grouped_success[(row["membership"], row["record_id"])] or bool(row["success"])
                        grouped = dict(grouped_success)
                    member_values = [1.0 if success else 0.0 for (membership, _), success in grouped.items() if membership == "member"]
                    nonmember_values = [1.0 if success else 0.0 for (membership, _), success in grouped.items() if membership == "nonmember"]
                    if member_values and nonmember_values:
                        seed_rates[mode].append(
                            {
                                "member_rate": mean(member_values),
                                "nonmember_rate": mean(nonmember_values),
                                "lift": mean(member_values) - mean(nonmember_values),
                                "member_n": len(member_values),
                                "nonmember_n": len(nonmember_values),
                            }
                        )
            for mode, values in seed_rates.items():
                if not values:
                    continue
                lift_values = [row["lift"] for row in values]
                lift_low, lift_high = bootstrap_ci(lift_values, seed=budget)
                rows.append(
                    {
                        "condition": condition,
                        "budget": budget,
                        "mode": mode,
                        "n_seeds": len(values),
                        "member_rate": round(mean([row["member_rate"] for row in values]), 4),
                        "nonmember_rate": round(mean([row["nonmember_rate"] for row in values]), 4),
                        "lift": round(mean(lift_values), 4),
                        "lift_ci95_low": round(lift_low, 4),
                        "lift_ci95_high": round(lift_high, 4),
                        "member_records_per_seed": int(mean([row["member_n"] for row in values])),
                        "nonmember_records_per_seed": int(mean([row["nonmember_n"] for row in values])),
                    }
                )
    fieldnames = [
        "condition",
        "budget",
        "mode",
        "n_seeds",
        "member_rate",
        "nonmember_rate",
        "lift",
        "lift_ci95_low",
        "lift_ci95_high",
        "member_records_per_seed",
        "nonmember_records_per_seed",
    ]
    write_csv(out_dir / "best_of_template_audit.csv", rows, fieldnames)
    write_text(out_dir / "best_of_template_audit.tex", best_of_template_tex(rows))
    write_text(out_dir / "best_of_template_missing_inputs.txt", "\n".join(missing_score_files) + ("\n" if missing_score_files else ""))
    return rows


def best_of_template_tex(rows: list[dict[str, Any]]) -> str:
    mode_labels = {
        "fixed_template_1": "Fixed template",
        "best_of_highcue_templates": "Best of high-cue templates",
    }
    lines = [
        r"\begin{table}[!t]",
        r"\centering",
        r"\small",
        r"\begin{tabular}{llrrrr}",
        r"\toprule",
        r"Condition & Setting & $B$ & Member & Nonmember & Lift \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(
            f"{CONDITIONS[row['condition']]} & {mode_labels[row['mode']]} & {row['budget']} & "
            f"{float(row['member_rate']):.4f} & {float(row['nonmember_rate']):.4f} & "
            f"{float(row['lift']):.4f} \\\\"
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\caption{Small prompt-search robustness check using existing high-cue prompt templates. Best-of-template success counts a record as extracted if any high-cue template succeeds.}",
            r"\label{tab:best-of-template-audit}",
            r"\end{table}",
            "",
        ]
    )
    return "\n".join(lines)


def write_manifest(out_dir: Path, results: dict[str, Any]) -> None:
    write_text(out_dir / "fast55_manifest.json", json.dumps(results, indent=2, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Fast 5.5 paper-upgrade evidence tables from existing realistic-run artifacts.")
    parser.add_argument("--run-root", default="study_runs/workshop_realistic_main_c4_100m")
    parser.add_argument("--out-dir", default="reports/cloud_6of6")
    args = parser.parse_args()

    run_root = Path(args.run_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    provenance_rows = generate_provenance_outputs(run_root, out_dir)
    balance_rows = generate_matched_control_outputs(run_root, out_dir)
    best_of_rows = generate_best_of_template_outputs(run_root, out_dir)

    manifest = {
        "run_root": str(run_root),
        "out_dir": str(out_dir),
        "provenance_rows": len(provenance_rows),
        "matched_control_balance_rows": len(balance_rows),
        "best_of_template_rows": len(best_of_rows),
        "best_of_template_missing_score_files": [
            line
            for line in (out_dir / "best_of_template_missing_inputs.txt").read_text(encoding="utf-8").splitlines()
            if line
        ],
        "notes": [
            "Provenance baseline rows are generated only when candidate_pools.jsonl and gradient_similarity.jsonl are present.",
            "BM25 full-prompt overlap is an intentionally strong lexical baseline and may be trivial under high-cue prompts.",
            "BM25 de-identified prompt overlap removes record-specific field values before lexical retrieval.",
            "Best-of-template uses existing high-cue prompts and does not run new generations.",
        ],
    }
    write_manifest(out_dir, manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
