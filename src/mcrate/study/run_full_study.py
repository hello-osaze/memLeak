from __future__ import annotations

import argparse
import json
import os
import random
import socket
import sys
import time
import traceback
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from mcrate.audit.aggregate_results import aggregate, render_markdown
from mcrate.audit.compute_cue_scores import compute_cue_scores
from mcrate.audit.make_prompts import make_prompts
from mcrate.audit.run_generation import run_generation
from mcrate.audit.score_generations import score_generations
from mcrate.data.build_corpus import build_corpus
from mcrate.data.generate_records import generate_records
from mcrate.data.render_templates import render_documents
from mcrate.data.validate_dataset import validate_dataset
from mcrate.mech.activation_patching import activation_patching
from mcrate.mech.cache_activations import cache_activations
from mcrate.mech.direct_logit_attribution import direct_logit_attribution
from mcrate.mech.mean_ablation import mean_ablation
from mcrate.mech.residual_directions import residual_directions
from mcrate.mech.train_probes import train_probes
from mcrate.provenance.build_candidate_pools import build_candidate_pools
from mcrate.provenance.gradient_similarity import gradient_similarity
from mcrate.provenance.removal_experiment import removal_experiment
from mcrate.train.finetune import finetune
from mcrate.utils.io import dump_yaml, ensure_dir, load_yaml, read_json, read_jsonl, write_json, write_jsonl, write_text
from mcrate.utils.logging import get_logger


LOGGER = get_logger(__name__)


def _slug(value: str) -> str:
    return "".join(char.lower() if char.isalnum() else "_" for char in value).strip("_")


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _repo_root() -> Path:
    return Path.cwd().resolve()


def _resolve_path(repo_root: Path, value: str | None) -> Path | None:
    if value is None:
        return None
    path = Path(value)
    return path if path.is_absolute() else (repo_root / path).resolve()


def _generation_name(path: str | Path) -> str:
    return Path(path).stem


class StudyLayout:
    def __init__(self, root: Path) -> None:
        self.root = root

    def plan_path(self) -> Path:
        return self.root / "study_plan.json"

    def resolved_config_path(self) -> Path:
        return self.root / "study_config_resolved.json"

    def state_dir(self) -> Path:
        return self.root / "state"

    def done_marker(self, unit_id: str) -> Path:
        return self.state_dir() / "units" / f"{unit_id}.done.json"

    def failed_marker(self, unit_id: str) -> Path:
        return self.state_dir() / "units" / f"{unit_id}.failed.json"

    def lock_path(self, unit_id: str) -> Path:
        return self.state_dir() / "locks" / f"{unit_id}.lock"

    def cluster_dir(self) -> Path:
        return self.root / "cluster"

    def generated_config_dir(self) -> Path:
        return self.root / "generated_configs"

    def records_path(self, output_name: str) -> Path:
        return self.root / "data" / "records" / output_name

    def audit_records_path(self) -> Path:
        return self.root / "data" / "records" / "audit_targets.jsonl"

    def raw_prompts_path(self) -> Path:
        return self.root / "data" / "prompts" / "raw_prompts.jsonl"

    def scored_prompts_path(self) -> Path:
        return self.root / "data" / "prompts" / "scored_prompts.jsonl"

    def prompt_summary_path(self) -> Path:
        return self.root / "reports" / "prompt_summary.json"

    def rendered_docs_path(self, condition: str) -> Path:
        return self.root / "data" / "processed" / f"{_slug(condition)}__rendered_docs.jsonl"

    def corpus_dir(self, condition: str, seed: int) -> Path:
        return self.root / "data" / "corpora" / _slug(condition) / f"seed_{seed}"

    def dataset_validation_path(self, condition: str, seed: int) -> Path:
        return self.root / "reports" / "dataset_validation" / _slug(condition) / f"seed_{seed}.md"

    def generated_corpus_config_path(self, condition: str, seed: int) -> Path:
        return self.generated_config_dir() / "corpus" / f"{_slug(condition)}__seed_{seed}.yaml"

    def generated_train_config_path(self, condition: str, seed: int) -> Path:
        return self.generated_config_dir() / "train" / f"{_slug(condition)}__seed_{seed}.yaml"

    def generated_generation_config_path(self, condition: str, seed: int, generation_name: str) -> Path:
        return self.generated_config_dir() / "generation" / f"{_slug(condition)}__seed_{seed}__{_slug(generation_name)}.yaml"

    def generated_provenance_config_path(self, condition: str, seed: int) -> Path:
        return self.generated_config_dir() / "provenance" / f"{_slug(condition)}__seed_{seed}.yaml"

    def checkpoint_dir(self, condition: str, seed: int) -> Path:
        return self.root / "checkpoints" / _slug(condition) / f"seed_{seed}"

    def model_dir(self, condition: str, seed: int) -> Path:
        return self.checkpoint_dir(condition, seed) / "final_model"

    def generation_path(self, condition: str, seed: int, generation_name: str) -> Path:
        return self.root / "outputs" / "generations" / _slug(condition) / f"seed_{seed}" / f"{_slug(generation_name)}.jsonl"

    def score_path(self, condition: str, seed: int, generation_name: str) -> Path:
        return self.root / "outputs" / "scores" / _slug(condition) / f"seed_{seed}" / f"{_slug(generation_name)}_scores.jsonl"

    def behavioral_report_path(self, condition: str, seed: int, generation_name: str) -> Path:
        return self.root / "reports" / "behavioral" / _slug(condition) / f"seed_{seed}__{_slug(generation_name)}.md"

    def mech_dir(self, condition: str, seed: int, generation_name: str) -> Path:
        return self.root / "outputs" / "mech" / _slug(condition) / f"seed_{seed}" / _slug(generation_name)

    def provenance_dir(self, condition: str, seed: int, generation_name: str) -> Path:
        return self.root / "outputs" / "provenance" / _slug(condition) / f"seed_{seed}" / _slug(generation_name)

    def removal_dir(self, condition: str, seed: int, generation_name: str) -> Path:
        return self.root / "outputs" / "removal" / _slug(condition) / f"seed_{seed}" / _slug(generation_name)

    def study_summary_path(self) -> Path:
        return self.root / "reports" / "study_summary.md"

    def study_summary_json_path(self) -> Path:
        return self.root / "reports" / "study_summary.json"


def _default_records_output_name(config_path: Path) -> str:
    return f"{config_path.stem}.jsonl"


def _normalize_condition_map(raw_conditions: dict[str, Any], repo_root: Path) -> dict[str, dict[str, Any]]:
    normalized: dict[str, dict[str, Any]] = {}
    for condition, payload in raw_conditions.items():
        if isinstance(payload, str):
            normalized[condition] = {"corpus_config": str(_resolve_path(repo_root, payload))}
            continue
        item = dict(payload)
        item["corpus_config"] = str(_resolve_path(repo_root, item["corpus_config"]))
        item["seeds"] = [int(seed) for seed in item.get("seeds", [1])]
        normalized[condition] = item
    return normalized


def _normalize_config(args: argparse.Namespace) -> dict[str, Any]:
    repo_root = _repo_root()
    config_path = _resolve_path(repo_root, args.config)
    if config_path is None:
        raise RuntimeError("A study config path is required.")
    raw = load_yaml(config_path)
    study_name = args.study_name or raw.get("study_name") or config_path.stem
    output_root = _resolve_path(repo_root, args.output_root or raw.get("output_root") or f"study_runs/{study_name}")
    if output_root is None:
        raise RuntimeError("Could not determine the study output root.")
    records_block = dict(raw.get("records", {}))
    records_config = _resolve_path(repo_root, str(records_block.get("config", "configs/data/records_default.yaml")))
    if records_config is None:
        raise RuntimeError("Could not determine the records config.")
    records_output_name = str(records_block.get("output_name", _default_records_output_name(records_config)))
    audit_block = dict(raw.get("audit", {}))
    mech_block = dict(raw.get("mechanistic", {}))
    provenance_block = dict(raw.get("provenance", {}))
    removal_block = dict(raw.get("removal", {}))
    training_block = dict(raw.get("training", {}))
    cluster_block = dict(raw.get("cluster", {}))
    storage_block = dict(raw.get("storage", {}))
    conditions = _normalize_condition_map(raw.get("conditions", {}), repo_root)
    generation_configs = [
        str(_resolve_path(repo_root, value))
        for value in audit_block.get("generation_configs", ["configs/generation/budget5.yaml"])
    ]
    analysis_generation_config = str(
        _resolve_path(
            repo_root,
            audit_block.get("analysis_generation_config") or generation_configs[0],
        )
    )
    normalized = {
        "study_name": study_name,
        "repo_root": str(repo_root),
        "config_path": str(config_path),
        "output_root": str(output_root),
        "background_path": str(
            _resolve_path(repo_root, args.background or raw.get("background_path") or "data/raw/background_full.txt")
        ),
        "records": {
            "config": str(records_config),
            "output_name": records_output_name,
        },
        "audit": {
            "split": str(audit_block.get("split", "test")),
            "target_counts": {
                "member": int(audit_block.get("target_counts", {}).get("member", 500)),
                "nonmember": int(audit_block.get("target_counts", {}).get("nonmember", 500)),
                "canary": int(audit_block.get("target_counts", {}).get("canary", 0)),
            },
            "sample_seed": int(audit_block.get("sample_seed", 1)),
            "include_failed_prompts": bool(audit_block.get("include_failed_prompts", False)),
            "generation_configs": generation_configs,
            "analysis_generation_config": analysis_generation_config,
        },
        "conditions": conditions,
        "training": {
            "config": str(_resolve_path(repo_root, str(training_block.get("config", "configs/train/pythia_410m.yaml")))),
            "overrides": dict(training_block.get("overrides", {})),
        },
        "mechanistic": {
            "enabled": bool(mech_block.get("enabled", True)),
            "config": str(_resolve_path(repo_root, str(mech_block.get("config", "configs/mech/probe.yaml")))),
            "focus_conditions": list(mech_block.get("focus_conditions", ["C2_exact_10x", "C3_fuzzy_5x"])),
            "focus_seeds": mech_block.get("focus_seeds", [1]),
        },
        "provenance": {
            "enabled": bool(provenance_block.get("enabled", True)),
            "config": str(
                _resolve_path(repo_root, str(provenance_block.get("config", "configs/provenance/grad_similarity.yaml")))
            ),
            "focus_conditions": list(provenance_block.get("focus_conditions", ["C2_exact_10x", "C3_fuzzy_5x"])),
            "focus_seeds": provenance_block.get("focus_seeds", [1]),
            "selection_unit_by_condition": dict(provenance_block.get("selection_unit_by_condition", {})),
            "max_total_targets": int(provenance_block.get("max_total_targets", 100)),
            "max_targets_per_record": int(provenance_block.get("max_targets_per_record", 1)),
            "target_selection_metric": str(provenance_block.get("target_selection_metric", "record_exact_then_logprob")),
            "overrides": dict(provenance_block.get("overrides", {})),
        },
        "removal": {
            "enabled": bool(removal_block.get("enabled", True)),
            "focus_conditions": list(removal_block.get("focus_conditions", ["C2_exact_10x", "C3_fuzzy_5x"])),
            "focus_seeds": removal_block.get("focus_seeds", [1]),
            "selection_unit_by_condition": dict(removal_block.get("selection_unit_by_condition", {})),
            "audit_generation_config": str(
                _resolve_path(repo_root, removal_block.get("audit_generation_config") or analysis_generation_config)
            ),
            "run_mech_lite": bool(removal_block.get("run_mech_lite", True)),
        },
        "storage": {
            "skip_existing": bool(storage_block.get("skip_existing", True)),
            "remove_generation_after_scoring": bool(storage_block.get("remove_generation_after_scoring", False)),
            "shared_rendered_docs": bool(storage_block.get("shared_rendered_docs", True)),
            "shared_prompts": bool(storage_block.get("shared_prompts", True)),
        },
        "cluster": {
            "python_bin": str(cluster_block.get("python_bin", sys.executable)),
            "scheduler": str(cluster_block.get("scheduler", "generic")),
            "slurm": dict(cluster_block.get("slurm", {})),
        },
        "cli_overrides": {
            "study_name": args.study_name,
            "background": args.background,
            "output_root": args.output_root,
            "device": args.device,
            "python_bin": args.python_bin,
        },
    }
    if args.device:
        normalized["training"]["overrides"]["device"] = args.device
        normalized["provenance"]["overrides"]["device"] = args.device
    if args.python_bin:
        normalized["cluster"]["python_bin"] = args.python_bin
    return normalized


def _all_condition_seed_pairs(config: dict[str, Any]) -> list[tuple[str, int]]:
    pairs: list[tuple[str, int]] = []
    for condition, payload in config["conditions"].items():
        for seed in payload.get("seeds", [1]):
            pairs.append((condition, int(seed)))
    return pairs


def _resolve_focus_runs(
    section: dict[str, Any],
    config: dict[str, Any],
    *,
    selection_map: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    focus_conditions = set(section.get("focus_conditions", []))
    focus_seeds = section.get("focus_seeds", [1])
    use_all_seeds = isinstance(focus_seeds, str) and focus_seeds.lower() == "all"
    runs = []
    for condition, seed in _all_condition_seed_pairs(config):
        if focus_conditions and condition not in focus_conditions:
            continue
        if not use_all_seeds and int(seed) not in {int(item) for item in focus_seeds}:
            continue
        payload = {"condition": condition, "seed": int(seed)}
        if selection_map:
            payload["selection_unit"] = selection_map.get(condition, "record")
        runs.append(payload)
    return runs


def _unit_id(phase: str, *parts: Any) -> str:
    suffix = ".".join(_slug(str(part)) for part in parts if part not in {None, ""})
    return f"{phase}.{suffix}" if suffix else phase


def _build_plan(config: dict[str, Any]) -> list[dict[str, Any]]:
    layout = StudyLayout(Path(config["output_root"]))
    records_path = layout.records_path(config["records"]["output_name"])
    scored_prompts_path = layout.scored_prompts_path()
    raw_prompts_path = layout.raw_prompts_path()
    audit_records_path = layout.audit_records_path()
    analysis_generation_name = _generation_name(config["audit"]["analysis_generation_config"])
    keep_generation = not config["storage"]["remove_generation_after_scoring"]
    units: list[dict[str, Any]] = []

    units.append(
        {
            "unit_id": _unit_id("records", "generate"),
            "phase": "records",
            "deps": [],
            "summary": "Generate the full synthetic record table.",
            "outputs": [str(records_path)],
            "payload": {
                "records_config_path": config["records"]["config"],
                "records_path": str(records_path),
            },
        }
    )

    units.append(
        {
            "unit_id": _unit_id("audit", "prepare"),
            "phase": "audit_prep",
            "deps": [_unit_id("records", "generate")],
            "summary": "Select audit targets and build shared cue-scored prompts.",
            "outputs": [str(audit_records_path), str(raw_prompts_path), str(scored_prompts_path), str(layout.prompt_summary_path())],
            "payload": {
                "records_path": str(records_path),
                "audit_records_path": str(audit_records_path),
                "raw_prompts_path": str(raw_prompts_path),
                "scored_prompts_path": str(scored_prompts_path),
                "prompt_summary_path": str(layout.prompt_summary_path()),
                "split": config["audit"]["split"],
                "target_counts": config["audit"]["target_counts"],
                "sample_seed": int(config["audit"]["sample_seed"]),
            },
        }
    )

    for condition, payload in config["conditions"].items():
        rendered_docs_path = layout.rendered_docs_path(condition)
        units.append(
            {
                "unit_id": _unit_id("render", condition),
                "phase": "render",
                "deps": [_unit_id("records", "generate")],
                "summary": f"Render synthetic training documents for {condition}.",
                "outputs": [str(rendered_docs_path)],
                "payload": {
                    "condition": condition,
                    "records_path": str(records_path),
                    "rendered_docs_path": str(rendered_docs_path),
                    "render_seed": int(payload.get("render_seed", 1)),
                },
            }
        )

    for condition, seed in _all_condition_seed_pairs(config):
        corpus_dir = layout.corpus_dir(condition, seed)
        corpus_manifest_path = corpus_dir / "manifest.json"
        dataset_validation_path = layout.dataset_validation_path(condition, seed)
        units.append(
            {
                "unit_id": _unit_id("corpus", condition, f"seed_{seed}"),
                "phase": "corpus",
                "deps": [_unit_id("render", condition), _unit_id("audit", "prepare")],
                "summary": f"Build the mixed training corpus for {condition} seed {seed}.",
                "outputs": [str(corpus_manifest_path), str(dataset_validation_path)],
                "payload": {
                    "condition": condition,
                    "seed": int(seed),
                    "background_path": config["background_path"],
                    "rendered_docs_path": str(layout.rendered_docs_path(condition)),
                    "records_path": str(records_path),
                    "prompts_path": str(scored_prompts_path),
                    "corpus_dir": str(corpus_dir),
                    "dataset_validation_path": str(dataset_validation_path),
                    "base_corpus_config_path": config["conditions"][condition]["corpus_config"],
                    "generated_corpus_config_path": str(layout.generated_corpus_config_path(condition, seed)),
                },
            }
        )

        units.append(
            {
                "unit_id": _unit_id("train", condition, f"seed_{seed}"),
                "phase": "train",
                "deps": [_unit_id("corpus", condition, f"seed_{seed}")],
                "summary": f"Fine-tune the model for {condition} seed {seed}.",
                "outputs": [str(layout.model_dir(condition, seed))],
                "payload": {
                    "condition": condition,
                    "seed": int(seed),
                    "corpus_dir": str(corpus_dir),
                    "checkpoint_dir": str(layout.checkpoint_dir(condition, seed)),
                    "base_train_config_path": config["training"]["config"],
                    "generated_train_config_path": str(layout.generated_train_config_path(condition, seed)),
                    "train_overrides": config["training"]["overrides"],
                },
            }
        )

        for generation_config_path in config["audit"]["generation_configs"]:
            generation_name = _generation_name(generation_config_path)
            outputs = [
                str(layout.score_path(condition, seed, generation_name)),
                str(layout.behavioral_report_path(condition, seed, generation_name)),
            ]
            if keep_generation:
                outputs.insert(0, str(layout.generation_path(condition, seed, generation_name)))
            units.append(
                {
                    "unit_id": _unit_id("audit_run", condition, f"seed_{seed}", generation_name),
                    "phase": "audit",
                    "deps": [_unit_id("train", condition, f"seed_{seed}"), _unit_id("audit", "prepare")],
                    "summary": f"Run behavioral extraction audit for {condition} seed {seed} with {generation_name}.",
                    "outputs": outputs,
                    "payload": {
                        "condition": condition,
                        "seed": int(seed),
                        "generation_name": generation_name,
                        "base_generation_config_path": generation_config_path,
                        "generated_generation_config_path": str(
                            layout.generated_generation_config_path(condition, seed, generation_name)
                        ),
                        "model_path": str(layout.model_dir(condition, seed)),
                        "prompts_path": str(scored_prompts_path),
                        "records_path": str(records_path),
                        "generation_path": str(layout.generation_path(condition, seed, generation_name)),
                        "score_path": str(layout.score_path(condition, seed, generation_name)),
                        "behavioral_report_path": str(layout.behavioral_report_path(condition, seed, generation_name)),
                        "include_failed_prompts": bool(config["audit"]["include_failed_prompts"]),
                        "remove_generation_after_scoring": bool(config["storage"]["remove_generation_after_scoring"]),
                    },
                }
            )

    if config["mechanistic"]["enabled"]:
        for focus in _resolve_focus_runs(config["mechanistic"], config):
            condition = focus["condition"]
            seed = int(focus["seed"])
            generation_name = analysis_generation_name
            mech_dir = layout.mech_dir(condition, seed, generation_name)
            units.append(
                {
                    "unit_id": _unit_id("mech", condition, f"seed_{seed}", generation_name),
                    "phase": "mechanistic",
                    "deps": [_unit_id("audit_run", condition, f"seed_{seed}", generation_name)],
                    "summary": f"Run mechanistic analysis for {condition} seed {seed}.",
                    "outputs": [
                        str(mech_dir / "candidate_layers.json"),
                        str(mech_dir / "probe_results.csv"),
                        str(mech_dir / "patching_effects.jsonl"),
                        str(mech_dir / "ablation_effects.jsonl"),
                        str(mech_dir / "residual_direction_effects.jsonl"),
                        str(mech_dir / "direct_logit_attribution.jsonl"),
                    ],
                    "payload": {
                        "condition": condition,
                        "seed": seed,
                        "generation_name": generation_name,
                        "model_path": str(layout.model_dir(condition, seed)),
                        "scores_path": str(layout.score_path(condition, seed, generation_name)),
                        "mech_config_path": config["mechanistic"]["config"],
                        "mech_dir": str(mech_dir),
                    },
                }
            )

    if config["provenance"]["enabled"]:
        for focus in _resolve_focus_runs(
            config["provenance"],
            config,
            selection_map=config["provenance"]["selection_unit_by_condition"],
        ):
            condition = focus["condition"]
            seed = int(focus["seed"])
            generation_name = analysis_generation_name
            provenance_dir = layout.provenance_dir(condition, seed, generation_name)
            units.append(
                {
                    "unit_id": _unit_id("provenance", condition, f"seed_{seed}", generation_name),
                    "phase": "provenance",
                    "deps": [
                        _unit_id("audit_run", condition, f"seed_{seed}", generation_name),
                        _unit_id("render", condition),
                    ],
                    "summary": f"Run provenance attribution for {condition} seed {seed}.",
                    "outputs": [
                        str(provenance_dir / "candidate_pools.jsonl"),
                        str(provenance_dir / "gradient_similarity.jsonl"),
                        str(provenance_dir / "summary.json"),
                    ],
                    "payload": {
                        "condition": condition,
                        "seed": seed,
                        "generation_name": generation_name,
                        "selection_unit": focus["selection_unit"],
                        "model_path": str(layout.model_dir(condition, seed)),
                        "scores_path": str(layout.score_path(condition, seed, generation_name)),
                        "rendered_docs_path": str(layout.rendered_docs_path(condition)),
                        "provenance_dir": str(provenance_dir),
                        "base_provenance_config_path": config["provenance"]["config"],
                        "generated_provenance_config_path": str(layout.generated_provenance_config_path(condition, seed)),
                        "provenance_overrides": {
                            **config["provenance"]["overrides"],
                            "max_total_targets": int(config["provenance"]["max_total_targets"]),
                            "max_targets_per_record": int(config["provenance"]["max_targets_per_record"]),
                            "target_selection_metric": config["provenance"]["target_selection_metric"],
                        },
                    },
                }
            )

    if config["removal"]["enabled"]:
        removal_focus_runs = _resolve_focus_runs(
            config["removal"],
            config,
            selection_map=config["removal"]["selection_unit_by_condition"],
        )
        for focus in removal_focus_runs:
            condition = focus["condition"]
            seed = int(focus["seed"])
            generation_name = _generation_name(config["removal"]["audit_generation_config"])
            removal_dir = layout.removal_dir(condition, seed, generation_name)
            units.append(
                {
                    "unit_id": _unit_id("removal", condition, f"seed_{seed}", generation_name),
                    "phase": "removal",
                    "deps": [_unit_id("provenance", condition, f"seed_{seed}", analysis_generation_name)],
                    "summary": f"Run removal validation for {condition} seed {seed}.",
                    "outputs": [str(removal_dir / "removal_validation_summary.json")],
                    "payload": {
                        "condition": condition,
                        "seed": seed,
                        "generation_name": generation_name,
                        "selection_unit": focus["selection_unit"],
                        "removal_dir": str(removal_dir),
                        "corpus_dir": str(layout.corpus_dir(condition, seed)),
                        "attribution_path": str(layout.provenance_dir(condition, seed, analysis_generation_name) / "gradient_similarity.jsonl"),
                        "train_config_path": str(layout.generated_train_config_path(condition, seed)),
                        "base_train_config_path": config["training"]["config"],
                        "train_overrides": {**config["training"]["overrides"], "seed": seed},
                        "model_path": str(layout.model_dir(condition, seed)),
                        "records_path": str(records_path),
                        "prompts_path": str(scored_prompts_path),
                        "base_generation_config_path": config["removal"]["audit_generation_config"],
                        "run_mech_lite": bool(config["removal"]["run_mech_lite"]),
                        "mech_config_path": config["mechanistic"]["config"],
                    },
                }
            )

    report_deps = [unit["unit_id"] for unit in units]
    units.append(
        {
            "unit_id": _unit_id("report", "study_summary"),
            "phase": "report",
            "deps": report_deps,
            "summary": "Write a study-wide completion and artifact summary.",
            "outputs": [str(layout.study_summary_path()), str(layout.study_summary_json_path())],
            "payload": {},
        }
    )
    return units


def _write_plan(config: dict[str, Any], plan: list[dict[str, Any]]) -> None:
    layout = StudyLayout(Path(config["output_root"]))
    ensure_dir(layout.root)
    write_json(layout.resolved_config_path(), config)
    write_json(layout.plan_path(), {"generated_at": _now(), "study_name": config["study_name"], "units": plan})


def _done_marker_exists(layout: StudyLayout, unit_id: str) -> bool:
    return layout.done_marker(unit_id).exists()


def _failed_marker_exists(layout: StudyLayout, unit_id: str) -> bool:
    return layout.failed_marker(unit_id).exists()


def _outputs_exist(outputs: list[str]) -> bool:
    return bool(outputs) and all(Path(path).exists() for path in outputs)


def _unit_status(unit: dict[str, Any], unit_map: dict[str, dict[str, Any]], layout: StudyLayout) -> str:
    if _done_marker_exists(layout, unit["unit_id"]):
        return "completed"
    if _failed_marker_exists(layout, unit["unit_id"]):
        return "failed"
    if unit["phase"] != "report" and _outputs_exist(unit.get("outputs", [])):
        return "completed"
    dep_statuses = [_unit_status(unit_map[dep], unit_map, layout) for dep in unit.get("deps", [])]
    if unit["phase"] == "report" and all(status == "completed" for status in dep_statuses) and _outputs_exist(unit.get("outputs", [])):
        return "completed"
    if any(status == "failed" for status in dep_statuses):
        return "blocked"
    if all(status == "completed" for status in dep_statuses):
        return "ready"
    return "pending"


def _ensure_background_exists(path: str) -> None:
    if not Path(path).exists():
        raise FileNotFoundError(
            f"Background corpus not found at {path}. "
            "Set `background_path` in the study config or pass `--background /path/to/background.txt`."
        )


def _write_overlay_config(base_path: str, out_path: str, overrides: dict[str, Any]) -> dict[str, Any]:
    payload = load_yaml(base_path)
    payload.update(overrides)
    dump_yaml(out_path, payload)
    return payload


def _largest_remainder_allocation(groups: dict[str, list[dict[str, Any]]], target: int) -> dict[str, int]:
    total = sum(len(rows) for rows in groups.values())
    if target >= total:
        return {name: len(rows) for name, rows in groups.items()}
    exact = {name: target * len(rows) / max(1, total) for name, rows in groups.items()}
    base = {name: min(len(groups[name]), int(value)) for name, value in exact.items()}
    assigned = sum(base.values())
    remainders = sorted(
        ((exact[name] - base[name], name) for name in groups),
        reverse=True,
    )
    cursor = 0
    while assigned < target and cursor < len(remainders):
        name = remainders[cursor][1]
        if base[name] < len(groups[name]):
            base[name] += 1
            assigned += 1
        cursor += 1
        if cursor >= len(remainders) and assigned < target:
            cursor = 0
    return base


def _sample_records_by_membership(
    records: list[dict[str, Any]],
    *,
    membership: str,
    count: int,
    sample_seed: int,
) -> list[dict[str, Any]]:
    if count <= 0:
        return []
    eligible = [row for row in records if row["membership"] == membership and row["family"] != "canary" and row["split"].startswith("test_")]
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in eligible:
        groups[row["family"]].append(row)
    allocations = _largest_remainder_allocation(groups, count)
    selected: list[dict[str, Any]] = []
    for family, rows in sorted(groups.items()):
        family_rng = random.Random(sample_seed + sum(ord(ch) for ch in f"{membership}:{family}"))
        shuffled = list(rows)
        family_rng.shuffle(shuffled)
        selected.extend(sorted(shuffled[: allocations.get(family, 0)], key=lambda row: row["record_id"]))
    return sorted(selected, key=lambda row: row["record_id"])


def _select_audit_records(records: list[dict[str, Any]], audit_config: dict[str, Any]) -> list[dict[str, Any]]:
    sample_seed = int(audit_config["sample_seed"])
    counts = audit_config["target_counts"]
    selected = []
    selected.extend(_sample_records_by_membership(records, membership="member", count=int(counts["member"]), sample_seed=sample_seed))
    selected.extend(
        _sample_records_by_membership(
            records,
            membership="nonmember",
            count=int(counts["nonmember"]),
            sample_seed=sample_seed + 10_000,
        )
    )
    if int(counts["canary"]) > 0:
        canaries = [row for row in records if row["family"] == "canary"]
        canary_rng = random.Random(sample_seed + 20_000)
        canary_rows = list(canaries)
        canary_rng.shuffle(canary_rows)
        selected.extend(sorted(canary_rows[: int(counts["canary"])], key=lambda row: row["record_id"]))
    return sorted(selected, key=lambda row: row["record_id"])


def _prompt_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_membership = Counter(row["membership"] for row in rows)
    by_cue = Counter(row["cue_band_computed"] for row in rows)
    by_pass = Counter("pass" if row["passes_cue_filter"] else "fail" for row in rows)
    return {
        "prompt_count": len(rows),
        "membership": dict(by_membership),
        "cue_band": dict(by_cue),
        "passes_cue_filter": dict(by_pass),
    }


def _read_generation_scores(path: str | Path) -> list[dict[str, Any]]:
    return read_jsonl(path)


def _low_cue_member_metrics(scores: list[dict[str, Any]]) -> dict[str, float]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in scores:
        grouped[row["task_id"]].append(row)
    tasks = []
    for rows in grouped.values():
        sample = rows[0]
        if sample["cue_band"] != "low" or sample["membership"] != "member":
            continue
        tasks.append(
            {
                "success": any(item["any_sensitive_match"] for item in rows),
                "record_exact": any(item.get("record_exact", False) for item in rows),
                "target_logprob": max(
                    [float(item["target_logprob"]) for item in rows if item.get("target_logprob") is not None],
                    default=None,
                ),
            }
        )
    if not tasks:
        return {"task_count": 0, "any_sensitive_match_rate": 0.0, "record_exact_rate": 0.0, "mean_max_target_logprob": 0.0}
    matches = sum(1 for task in tasks if task["success"])
    exact = sum(1 for task in tasks if task["record_exact"])
    max_logprobs = [task["target_logprob"] for task in tasks if task["target_logprob"] is not None]
    return {
        "task_count": len(tasks),
        "any_sensitive_match_rate": round(matches / len(tasks), 4),
        "record_exact_rate": round(exact / len(tasks), 4),
        "mean_max_target_logprob": round(sum(max_logprobs) / len(max_logprobs), 4) if max_logprobs else 0.0,
    }


def _provenance_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"targets": 0, "top1_record_recall": 0.0, "top10_record_recall": 0.0, "mrr": 0.0}
    top1 = 0
    top10 = 0
    rr_values = []
    for row in rows:
        ranked = row.get("ranked_candidates", [])
        true_ranks = [item["rank"] for item in ranked if item.get("is_true_record")]
        if true_ranks:
            top1 += 1 if min(true_ranks) == 1 else 0
            top10 += 1 if min(true_ranks) <= 10 else 0
            rr_values.append(1.0 / min(true_ranks))
        else:
            rr_values.append(0.0)
    return {
        "targets": len(rows),
        "top1_record_recall": round(top1 / len(rows), 4),
        "top10_record_recall": round(top10 / len(rows), 4),
        "mrr": round(sum(rr_values) / len(rr_values), 4),
    }


def _variant_model_dir(removal_dir: str, label: str) -> Path:
    return Path(removal_dir) / label / "final_model"


def _run_records(unit: dict[str, Any]) -> dict[str, Any]:
    payload = unit["payload"]
    config = load_yaml(payload["records_config_path"])
    rows = generate_records(config)
    write_jsonl(payload["records_path"], rows)
    return {"record_count": len(rows)}


def _run_audit_prepare(unit: dict[str, Any]) -> dict[str, Any]:
    payload = unit["payload"]
    records = read_jsonl(payload["records_path"])
    audit_records = _select_audit_records(
        records,
        {
            "sample_seed": payload["sample_seed"],
            "target_counts": payload["target_counts"],
        },
    )
    write_jsonl(payload["audit_records_path"], audit_records)
    prompts = make_prompts(audit_records, split_name="all")
    write_jsonl(payload["raw_prompts_path"], prompts)
    scored = compute_cue_scores(prompts, audit_records)
    write_jsonl(payload["scored_prompts_path"], scored)
    summary = _prompt_summary(scored)
    write_json(payload["prompt_summary_path"], summary)
    return summary


def _run_render(unit: dict[str, Any]) -> dict[str, Any]:
    payload = unit["payload"]
    records = read_jsonl(payload["records_path"])
    docs = render_documents(records, payload["condition"], payload["records_path"], seed=int(payload["render_seed"]))
    write_jsonl(payload["rendered_docs_path"], docs)
    return {"doc_count": len(docs)}


def _run_corpus(unit: dict[str, Any]) -> dict[str, Any]:
    payload = unit["payload"]
    _ensure_background_exists(payload["background_path"])
    corpus_config = _write_overlay_config(
        payload["base_corpus_config_path"],
        payload["generated_corpus_config_path"],
        {"shuffle_seed": int(payload["seed"])},
    )
    manifest = build_corpus(
        background_path=payload["background_path"],
        rendered_docs_path=payload["rendered_docs_path"],
        config=corpus_config,
        out_dir=payload["corpus_dir"],
    )
    validate_dataset(
        records_path=payload["records_path"],
        corpus_arg=payload["corpus_dir"],
        out_path=payload["dataset_validation_path"],
        prompts_path=payload["prompts_path"],
        rendered_docs_path=payload["rendered_docs_path"],
    )
    return {"condition": payload["condition"], "seed": payload["seed"], "synthetic_doc_count": manifest["synthetic_doc_count"]}


def _run_train(unit: dict[str, Any]) -> dict[str, Any]:
    payload = unit["payload"]
    corpus_dir = Path(payload["corpus_dir"])
    manifest = read_json(corpus_dir / "manifest.json")
    overrides = {"seed": int(payload["seed"]), **payload.get("train_overrides", {})}
    _write_overlay_config(
        payload["base_train_config_path"],
        payload["generated_train_config_path"],
        overrides,
    )
    result = finetune(
        config_path=payload["generated_train_config_path"],
        train_file=str(corpus_dir / "train.txt"),
        validation_file=str(corpus_dir / "validation.txt"),
        out_dir=payload["checkpoint_dir"],
    )
    return {"condition": payload["condition"], "seed": payload["seed"], "model_dir": result["final_model_dir"], "manifest": manifest}


def _run_audit(unit: dict[str, Any]) -> dict[str, Any]:
    payload = unit["payload"]
    _write_overlay_config(
        payload["base_generation_config_path"],
        payload["generated_generation_config_path"],
        {"seed": int(payload["seed"]), "name": payload["generation_name"]},
    )
    run_generation(
        model_path=payload["model_path"],
        prompts_path=payload["prompts_path"],
        generation_config_path=payload["generated_generation_config_path"],
        out_path=payload["generation_path"],
        include_failed_prompts=bool(payload["include_failed_prompts"]),
    )
    scored = score_generations(read_jsonl(payload["generation_path"]), read_jsonl(payload["records_path"]))
    write_jsonl(payload["score_path"], scored)
    summary = aggregate(scored)
    write_text(payload["behavioral_report_path"], render_markdown(summary))
    write_json(str(payload["behavioral_report_path"]).replace(".md", ".json"), summary)
    if payload["remove_generation_after_scoring"] and Path(payload["generation_path"]).exists():
        Path(payload["generation_path"]).unlink()
    return {
        "condition": payload["condition"],
        "seed": payload["seed"],
        "generation_name": payload["generation_name"],
        "task_count": summary["task_count"],
    }


def _run_mech(unit: dict[str, Any]) -> dict[str, Any]:
    payload = unit["payload"]
    mech_dir = ensure_dir(payload["mech_dir"])
    activations_root = mech_dir / "activations"
    cache_result = cache_activations(
        model_path=payload["model_path"],
        scores_path=payload["scores_path"],
        config_path=payload["mech_config_path"],
        out_dir=str(activations_root),
    )
    train_probes(
        activations_root=cache_result["root"],
        config_path=payload["mech_config_path"],
        out_dir=str(mech_dir),
    )
    activation_patching(
        model_path=payload["model_path"],
        scores_path=payload["scores_path"],
        probe_candidates_path=str(mech_dir / "candidate_layers.json"),
        config_path=payload["mech_config_path"],
        out_path=str(mech_dir / "patching_effects.jsonl"),
    )
    mean_ablation(
        model_path=payload["model_path"],
        scores_path=payload["scores_path"],
        probe_candidates_path=str(mech_dir / "candidate_layers.json"),
        config_path=payload["mech_config_path"],
        out_path=str(mech_dir / "ablation_effects.jsonl"),
    )
    residual_directions(
        model_path=payload["model_path"],
        scores_path=payload["scores_path"],
        config_path=payload["mech_config_path"],
        out_path=str(mech_dir / "residual_direction_effects.jsonl"),
    )
    direct_logit_attribution(
        model_path=payload["model_path"],
        scores_path=payload["scores_path"],
        out_path=str(mech_dir / "direct_logit_attribution.jsonl"),
    )
    return {"condition": payload["condition"], "seed": payload["seed"], "mech_dir": str(mech_dir)}


def _run_provenance(unit: dict[str, Any]) -> dict[str, Any]:
    payload = unit["payload"]
    provenance_dir = ensure_dir(payload["provenance_dir"])
    _write_overlay_config(
        payload["base_provenance_config_path"],
        payload["generated_provenance_config_path"],
        payload["provenance_overrides"],
    )
    candidate_pool_path = provenance_dir / "candidate_pools.jsonl"
    build_candidate_pools(
        scores_path=payload["scores_path"],
        rendered_docs_path=payload["rendered_docs_path"],
        config_path=payload["generated_provenance_config_path"],
        out_path=str(candidate_pool_path),
    )
    attribution_path = provenance_dir / "gradient_similarity.jsonl"
    rows = gradient_similarity(
        model_path=payload["model_path"],
        scores_path=payload["scores_path"],
        candidate_pools_path=str(candidate_pool_path),
        rendered_docs_path=payload["rendered_docs_path"],
        out_path=str(attribution_path),
        config_path=payload["generated_provenance_config_path"],
    )
    summary = _provenance_summary(rows)
    summary["selection_unit"] = payload["selection_unit"]
    write_json(provenance_dir / "summary.json", summary)
    return summary


def _run_removal(unit: dict[str, Any]) -> dict[str, Any]:
    payload = unit["payload"]
    removal_dir = ensure_dir(payload["removal_dir"])
    if not Path(payload["train_config_path"]).exists():
        _write_overlay_config(
            payload["base_train_config_path"],
            payload["train_config_path"],
            payload.get("train_overrides", {}),
        )
    summary = removal_experiment(
        attribution_path=payload["attribution_path"],
        corpus_dir=payload["corpus_dir"],
        out_dir=str(removal_dir),
        train_config_path=payload["train_config_path"],
        selection_unit=payload["selection_unit"],
    )
    base_generation_config_path = payload["base_generation_config_path"]
    records = read_jsonl(payload["records_path"])
    variants = ["high_attribution_removal", "random_removal"]
    validation_rows = []
    for label in variants:
        model_dir = _variant_model_dir(payload["removal_dir"], label)
        gen_path = removal_dir / "evaluations" / f"{label}_generations.jsonl"
        score_path = removal_dir / "evaluations" / f"{label}_scores.jsonl"
        report_path = removal_dir / "evaluations" / f"{label}_behavioral.md"
        run_generation(
            model_path=str(model_dir),
            prompts_path=payload["prompts_path"],
            generation_config_path=base_generation_config_path,
            out_path=str(gen_path),
            include_failed_prompts=False,
        )
        scores = score_generations(read_jsonl(gen_path), records)
        write_jsonl(score_path, scores)
        report = aggregate(scores)
        write_text(report_path, render_markdown(report))
        write_json(str(report_path).replace(".md", ".json"), report)
        lite_summary = _low_cue_member_metrics(scores)
        row = {"variant": label, **lite_summary}
        if payload["run_mech_lite"]:
            mech_dir = removal_dir / "mech_lite" / label
            cache_result = cache_activations(
                model_path=str(model_dir),
                scores_path=str(score_path),
                config_path=payload["mech_config_path"],
                out_dir=str(mech_dir / "activations"),
            )
            probe_result = train_probes(
                activations_root=cache_result["root"],
                config_path=payload["mech_config_path"],
                out_dir=str(mech_dir),
            )
            row["probe_candidate_layers"] = probe_result.get("candidate_layers", [])
        validation_rows.append(row)
    final_summary = {
        "condition": payload["condition"],
        "seed": payload["seed"],
        "selection_unit": payload["selection_unit"],
        "removal_summary": summary,
        "variants": validation_rows,
    }
    write_json(removal_dir / "removal_validation_summary.json", final_summary)
    return final_summary


def _summarize_study(config: dict[str, Any], plan: list[dict[str, Any]]) -> dict[str, Any]:
    layout = StudyLayout(Path(config["output_root"]))
    unit_map = {unit["unit_id"]: unit for unit in plan}
    by_phase = Counter(unit["phase"] for unit in plan)
    by_status = Counter(_unit_status(unit, unit_map, layout) for unit in plan)
    summary = {
        "study_name": config["study_name"],
        "generated_at": _now(),
        "output_root": config["output_root"],
        "phases": dict(by_phase),
        "statuses": dict(by_status),
        "completed_units": [unit["unit_id"] for unit in plan if _unit_status(unit, unit_map, layout) == "completed"],
        "pending_units": [unit["unit_id"] for unit in plan if _unit_status(unit, unit_map, layout) != "completed"],
    }
    lines = [
        f"# {config['study_name']} Summary",
        "",
        f"- Output root: `{config['output_root']}`",
        f"- Generated: `{summary['generated_at']}`",
        "",
        "## Unit Counts",
        "",
        "| Phase | Units |",
        "|---|---:|",
    ]
    for phase, count in sorted(by_phase.items()):
        lines.append(f"| {phase} | {count} |")
    lines.extend(
        [
            "",
            "## Status",
            "",
            "| Status | Units |",
            "|---|---:|",
        ]
    )
    for status, count in sorted(by_status.items()):
        lines.append(f"| {status} | {count} |")
    lines.append("")
    if summary["pending_units"]:
        lines.append("## Remaining Units")
        lines.append("")
        for unit_id in summary["pending_units"][:50]:
            lines.append(f"- `{unit_id}`")
        if len(summary["pending_units"]) > 50:
            lines.append(f"- ... and {len(summary['pending_units']) - 50} more")
    write_json(layout.study_summary_json_path(), summary)
    write_text(layout.study_summary_path(), "\n".join(lines) + "\n")
    return summary


def _write_done_marker(layout: StudyLayout, unit: dict[str, Any], result: dict[str, Any]) -> None:
    write_json(
        layout.done_marker(unit["unit_id"]),
        {
            "unit_id": unit["unit_id"],
            "phase": unit["phase"],
            "completed_at": _now(),
            "host": socket.gethostname(),
            "outputs": unit.get("outputs", []),
            "result": result,
        },
    )
    failed_marker = layout.failed_marker(unit["unit_id"])
    if failed_marker.exists():
        failed_marker.unlink()


def _write_failed_marker(layout: StudyLayout, unit: dict[str, Any], exc: BaseException) -> None:
    write_json(
        layout.failed_marker(unit["unit_id"]),
        {
            "unit_id": unit["unit_id"],
            "phase": unit["phase"],
            "failed_at": _now(),
            "host": socket.gethostname(),
            "error_type": exc.__class__.__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        },
    )


def _acquire_lock(layout: StudyLayout, unit_id: str) -> int:
    lock_path = layout.lock_path(unit_id)
    ensure_dir(lock_path.parent)
    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
    fd = os.open(str(lock_path), flags)
    os.write(fd, json.dumps({"unit_id": unit_id, "pid": os.getpid(), "host": socket.gethostname(), "started_at": _now()}).encode("utf-8"))
    return fd


def _release_lock(layout: StudyLayout, unit_id: str, fd: int | None) -> None:
    if fd is not None:
        os.close(fd)
    lock_path = layout.lock_path(unit_id)
    if lock_path.exists():
        lock_path.unlink()


def _execute_unit(unit: dict[str, Any]) -> dict[str, Any]:
    phase = unit["phase"]
    if phase == "records":
        return _run_records(unit)
    if phase == "audit_prep":
        return _run_audit_prepare(unit)
    if phase == "render":
        return _run_render(unit)
    if phase == "corpus":
        return _run_corpus(unit)
    if phase == "train":
        return _run_train(unit)
    if phase == "audit":
        return _run_audit(unit)
    if phase == "mechanistic":
        return _run_mech(unit)
    if phase == "provenance":
        return _run_provenance(unit)
    if phase == "removal":
        return _run_removal(unit)
    if phase == "report":
        raise RuntimeError("Study summary units are handled separately.")
    raise RuntimeError(f"Unsupported phase: {phase}")


def _load_plan(config: dict[str, Any]) -> list[dict[str, Any]]:
    plan = _build_plan(config)
    _write_plan(config, plan)
    return plan


def _select_units(
    plan: list[dict[str, Any]],
    config: dict[str, Any],
    *,
    phase: str | None = None,
    status: str | None = None,
) -> list[dict[str, Any]]:
    layout = StudyLayout(Path(config["output_root"]))
    unit_map = {unit["unit_id"]: unit for unit in plan}
    selected = []
    for unit in plan:
        current_status = _unit_status(unit, unit_map, layout)
        if phase and unit["phase"] != phase:
            continue
        if status and current_status != status:
            continue
        selected.append(unit)
    return selected


def _command_for_unit(config: dict[str, Any], unit_id: str) -> str:
    repo_root = Path(config["repo_root"])
    python_bin = config["cluster"]["python_bin"]
    config_path = Path(config["config_path"]).resolve()
    parts = [
        "cd",
        str(repo_root),
        "&&",
        python_bin,
        str(repo_root / "run_full_study.py"),
        "--config",
        str(config_path),
    ]
    if config["cli_overrides"].get("background"):
        parts.extend(["--background", str(config["cli_overrides"]["background"])])
    if config["cli_overrides"].get("output_root"):
        parts.extend(["--output-root", str(config["cli_overrides"]["output_root"])])
    if config["cli_overrides"].get("device"):
        parts.extend(["--device", str(config["cli_overrides"]["device"])])
    if config["cli_overrides"].get("python_bin"):
        parts.extend(["--python-bin", str(config["cli_overrides"]["python_bin"])])
    parts.extend(["run-unit", unit_id])
    return " ".join(parts)


def _emit_commands(config: dict[str, Any], plan: list[dict[str, Any]], *, phase: str | None, status: str) -> dict[str, Any]:
    layout = StudyLayout(Path(config["output_root"]))
    ensure_dir(layout.cluster_dir())
    selected = _select_units(plan, config, phase=phase, status=status)
    commands_path = layout.cluster_dir() / f"{status}_{phase or 'all'}_commands.txt"
    jsonl_path = layout.cluster_dir() / f"{status}_{phase or 'all'}_commands.jsonl"
    command_rows = []
    lines = []
    for unit in selected:
        command = _command_for_unit(config, unit["unit_id"])
        lines.append(command)
        command_rows.append(
            {
                "unit_id": unit["unit_id"],
                "phase": unit["phase"],
                "deps": unit["deps"],
                "command": command,
            }
        )
    write_text(commands_path, "\n".join(lines) + ("\n" if lines else ""))
    write_jsonl(jsonl_path, command_rows)
    return {"command_count": len(lines), "commands_path": str(commands_path), "jsonl_path": str(jsonl_path)}


def _emit_slurm_array(config: dict[str, Any], plan: list[dict[str, Any]], *, phase: str | None, status: str) -> dict[str, Any]:
    layout = StudyLayout(Path(config["output_root"]))
    command_info = _emit_commands(config, plan, phase=phase, status=status)
    command_count = int(command_info["command_count"])
    script_path = layout.cluster_dir() / f"{status}_{phase or 'all'}_slurm_array.sh"
    slurm_cfg = config["cluster"].get("slurm", {})
    array_max = max(0, command_count - 1)
    lines = [
        "#!/usr/bin/env bash",
        f"#SBATCH --job-name={_slug(config['study_name'])}_{_slug(phase or status)}",
        f"#SBATCH --array=0-{array_max}",
        f"#SBATCH --partition={slurm_cfg.get('partition', 'gpu')}",
        f"#SBATCH --gres={slurm_cfg.get('gres', 'gpu:1')}",
        f"#SBATCH --cpus-per-task={slurm_cfg.get('cpus_per_task', 8)}",
        f"#SBATCH --mem={slurm_cfg.get('mem', '64G')}",
        f"#SBATCH --time={slurm_cfg.get('time', '48:00:00')}",
        f"#SBATCH --output={layout.cluster_dir() / 'slurm_%A_%a.out'}",
        "",
        "set -euo pipefail",
        f"mapfile -t CMDS < {command_info['commands_path']}",
        'if [ "${#CMDS[@]}" -eq 0 ]; then',
        '  echo "No commands to run."',
        "  exit 0",
        "fi",
        'echo "Running: ${CMDS[$SLURM_ARRAY_TASK_ID]}"',
        'eval "${CMDS[$SLURM_ARRAY_TASK_ID]}"',
        "",
    ]
    write_text(script_path, "\n".join(lines))
    os.chmod(script_path, 0o755)
    return {"script_path": str(script_path), **command_info}


def _print_unit_table(config: dict[str, Any], plan: list[dict[str, Any]], *, phase: str | None = None) -> None:
    layout = StudyLayout(Path(config["output_root"]))
    unit_map = {unit["unit_id"]: unit for unit in plan}
    for unit in plan:
        if phase and unit["phase"] != phase:
            continue
        status = _unit_status(unit, unit_map, layout)
        print(f"{unit['unit_id']}\t{unit['phase']}\t{status}\t{unit['summary']}")


def _run_single_unit(config: dict[str, Any], plan: list[dict[str, Any]], unit_id: str, *, force: bool = False) -> dict[str, Any]:
    layout = StudyLayout(Path(config["output_root"]))
    unit_map = {unit["unit_id"]: unit for unit in plan}
    if unit_id not in unit_map:
        raise KeyError(f"Unknown unit id: {unit_id}")
    unit = unit_map[unit_id]
    status = _unit_status(unit, unit_map, layout)
    if status == "completed" and config["storage"]["skip_existing"] and not force:
        LOGGER.info("Skipping completed unit %s", unit_id)
        return {"unit_id": unit_id, "status": "skipped"}
    if status in {"pending", "blocked"} and not force:
        missing = [dep for dep in unit.get("deps", []) if _unit_status(unit_map[dep], unit_map, layout) != "completed"]
        raise RuntimeError(f"Unit {unit_id} is not ready yet. Incomplete dependencies: {missing}")
    if unit["phase"] == "report":
        result = _summarize_study(config, plan)
        _write_done_marker(layout, unit, result)
        return result
    lock_fd: int | None = None
    try:
        lock_fd = _acquire_lock(layout, unit_id)
        result = _execute_unit(unit)
        _write_done_marker(layout, unit, result)
        return result
    except BaseException as exc:
        _write_failed_marker(layout, unit, exc)
        raise
    finally:
        _release_lock(layout, unit_id, lock_fd)


def _run_ready_units(config: dict[str, Any], plan: list[dict[str, Any]], *, phase: str | None, limit: int | None) -> dict[str, Any]:
    layout = StudyLayout(Path(config["output_root"]))
    unit_map = {unit["unit_id"]: unit for unit in plan}
    ready_units = [
        unit
        for unit in plan
        if (phase is None or unit["phase"] == phase) and _unit_status(unit, unit_map, layout) == "ready"
    ]
    if limit is not None:
        ready_units = ready_units[:limit]
    results = []
    for unit in ready_units:
        LOGGER.info("Running unit %s", unit["unit_id"])
        results.append(_run_single_unit(config, plan, unit["unit_id"]))
    return {"ran": len(results), "results": results}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plan and run the full M-CRATE study on local or cluster hardware.")
    parser.add_argument("--config", required=True, help="Study YAML config.")
    parser.add_argument("--background", default=None, help="Optional override for the background corpus text path.")
    parser.add_argument("--output-root", default=None, help="Optional override for the study output root.")
    parser.add_argument("--study-name", default=None, help="Optional override for the study name.")
    parser.add_argument("--device", default=None, help="Optional training/provenance device override, e.g. cuda.")
    parser.add_argument("--python-bin", default=None, help="Optional Python executable to use in exported cluster commands.")

    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("plan", help="Write the resolved config and study plan manifest.")

    list_units = subparsers.add_parser("list-units", help="List units with current status.")
    list_units.add_argument("--phase", default=None, help="Optional phase filter.")

    run_unit = subparsers.add_parser("run-unit", help="Run a single unit by id.")
    run_unit.add_argument("unit_id", help="Unit id from the study plan.")
    run_unit.add_argument("--force", action="store_true", help="Run even if outputs already exist.")

    run_ready = subparsers.add_parser("run-ready", help="Run all currently ready units sequentially.")
    run_ready.add_argument("--phase", default=None, help="Optional phase filter.")
    run_ready.add_argument("--limit", type=int, default=None, help="Optional max number of units to run.")

    emit_commands = subparsers.add_parser("emit-commands", help="Write cluster-ready commands for matching units.")
    emit_commands.add_argument("--phase", default=None, help="Optional phase filter.")
    emit_commands.add_argument("--status", default="ready", choices=["ready", "pending", "blocked", "completed", "failed"], help="Unit status filter.")

    emit_slurm = subparsers.add_parser("emit-slurm-array", help="Write a Slurm array script for matching units.")
    emit_slurm.add_argument("--phase", default=None, help="Optional phase filter.")
    emit_slurm.add_argument("--status", default="ready", choices=["ready", "pending", "blocked", "completed", "failed"], help="Unit status filter.")

    subparsers.add_parser("summarize", help="Write the study summary markdown/json.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = _normalize_config(args)
    plan = _load_plan(config)
    layout = StudyLayout(Path(config["output_root"]))
    ensure_dir(layout.root)

    if args.command == "plan":
        summary = _summarize_study(config, plan)
        print(json.dumps({"study_name": config["study_name"], "units": len(plan), "statuses": summary["statuses"]}, indent=2))
        return

    if args.command == "list-units":
        _print_unit_table(config, plan, phase=args.phase)
        return

    if args.command == "run-unit":
        result = _run_single_unit(config, plan, args.unit_id, force=bool(args.force))
        print(json.dumps(result, indent=2))
        return

    if args.command == "run-ready":
        result = _run_ready_units(config, plan, phase=args.phase, limit=args.limit)
        print(json.dumps(result, indent=2))
        return

    if args.command == "emit-commands":
        result = _emit_commands(config, plan, phase=args.phase, status=args.status)
        print(json.dumps(result, indent=2))
        return

    if args.command == "emit-slurm-array":
        result = _emit_slurm_array(config, plan, phase=args.phase, status=args.status)
        print(json.dumps(result, indent=2))
        return

    if args.command == "summarize":
        result = _summarize_study(config, plan)
        print(json.dumps(result, indent=2))
        return

    raise RuntimeError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
