from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from mcrate.utils.hashing import sha256_json, stable_int_hash
from mcrate.utils.io import ensure_dir, read_json, write_json
from mcrate.utils.stats import sigmoid
from mcrate.utils.text_normalization import normalize_text


LAYER_COUNT = 16
ACTIVATION_DIM = 6
MECHANISM_PEAK = 11
CUE_PEAK = 4
MEMORY_PEAK = 13


def _gaussian(layer: int, peak: int, width: float) -> float:
    return math.exp(-((layer - peak) ** 2) / (2 * width * width))


def _family_signal(family: str) -> float:
    return {
        "identity": 0.90,
        "account": 0.82,
        "event": 0.86,
        "canary": 1.15,
    }.get(family, 0.75)


def _cue_strength(cue_band: str) -> float:
    return {
        "high": 1.05,
        "medium": 0.62,
        "low": 0.28,
        "no_cue": 0.05,
        "no-cue": 0.05,
    }.get(cue_band, 0.28)


def _mechanism_gate(cue_band: str) -> float:
    return {
        "high": 0.55,
        "medium": 0.62,
        "low": 0.45,
        "no_cue": 0.05,
        "no-cue": 0.05,
    }.get(cue_band, 0.45)


def _condition_modifier(condition: str) -> float:
    value = condition.lower()
    if "redacted" in value:
        return 0.18
    if "fuzzy" in value:
        return 0.92
    if "exact_10" in value or "10x" in value:
        return 1.08
    if "exact_20" in value or "20x" in value:
        return 1.18
    if "clean" in value:
        return 0.08
    return 0.68


def _repetition_strength(bucket: str) -> float:
    mapping = {"1x": 0.20, "2x": 0.38, "5x": 0.62, "10x": 0.92, "20x": 1.18}
    return mapping.get(bucket, 0.28)


@dataclass
class ToyModel:
    model_dir: Path
    model_name: str
    condition: str
    seed: int
    member_records: dict[str, dict[str, Any]]
    exposure_count: dict[str, int]
    docs_by_record: dict[str, list[dict[str, Any]]]
    corpus_manifest: dict[str, Any]
    layer_weights: np.ndarray

    @classmethod
    def load(cls, model_dir: str | Path) -> "ToyModel":
        payload = read_json(Path(model_dir) / "model.json")
        return cls(
            model_dir=Path(model_dir),
            model_name=payload["model_name"],
            condition=payload["condition"],
            seed=payload["seed"],
            member_records=payload["member_records"],
            exposure_count=payload["exposure_count"],
            docs_by_record=payload["docs_by_record"],
            corpus_manifest=payload["corpus_manifest"],
            layer_weights=np.asarray(payload["layer_weights"], dtype=float),
        )

    def save(self) -> None:
        payload = {
            "backend": "toy_memorizer",
            "model_name": self.model_name,
            "condition": self.condition,
            "seed": self.seed,
            "member_records": self.member_records,
            "exposure_count": self.exposure_count,
            "docs_by_record": self.docs_by_record,
            "corpus_manifest": self.corpus_manifest,
            "layer_weights": self.layer_weights.tolist(),
            "model_sha256": sha256_json(
                {
                    "condition": self.condition,
                    "seed": self.seed,
                    "exposure_count": self.exposure_count,
                    "member_records": sorted(self.member_records),
                }
            ),
        }
        write_json(self.model_dir / "model.json", payload)

    def has_record(self, record_id: str) -> bool:
        return record_id in self.member_records

    def get_record(self, record_id: str) -> dict[str, Any] | None:
        return self.member_records.get(record_id)

    def _feature_bundle(self, prompt_row: dict[str, Any]) -> dict[str, float]:
        record_id = prompt_row.get("record_id", "")
        record = self.member_records.get(record_id)
        cue_band = prompt_row.get("cue_band_computed") or prompt_row.get("cue_band_requested", "low")
        cue_strength = _cue_strength(cue_band)
        mechanism_gate = _mechanism_gate(cue_band)
        exposure = math.log1p(self.exposure_count.get(record_id, 0)) / math.log(11.0)
        repetition = 0.0
        family_signal = 0.65
        if record:
            repetition = _repetition_strength(record.get("repetition_bucket", "1x"))
            family_signal = _family_signal(record.get("family", "identity"))
        condition_signal = _condition_modifier(self.condition)
        low_cue_gate = mechanism_gate
        anchor_present = 1.0 if prompt_row.get("anchor_present", True) else 0.3
        membership = 1.0 if record_id in self.member_records else 0.0
        redact_penalty = 0.65 if "redacted" in self.condition.lower() else 0.0
        deterministic_noise = (
            (stable_int_hash(f"{prompt_row.get('task_id','')}-{self.seed}", modulo=1000) / 1000.0) - 0.5
        ) * 0.04
        return {
            "cue_strength": cue_strength,
            "mechanism_gate": mechanism_gate,
            "exposure": exposure,
            "repetition": repetition,
            "condition_signal": condition_signal,
            "low_cue_gate": low_cue_gate,
            "anchor_present": anchor_present,
            "membership": membership,
            "family_signal": family_signal,
            "redact_penalty": redact_penalty,
            "noise": deterministic_noise,
        }

    def activation_matrix(self, prompt_row: dict[str, Any]) -> np.ndarray:
        f = self._feature_bundle(prompt_row)
        activations = np.zeros((LAYER_COUNT, ACTIVATION_DIM), dtype=float)
        for layer in range(LAYER_COUNT):
            cue_profile = _gaussian(layer, CUE_PEAK, 2.0)
            success_profile = _gaussian(layer, MECHANISM_PEAK, 1.9)
            memory_profile = _gaussian(layer, MEMORY_PEAK, 2.3)
            activations[layer, 0] = f["cue_strength"] * cue_profile
            activations[layer, 1] = (
                f["membership"] * f["anchor_present"] * f["condition_signal"] * f["mechanism_gate"] * success_profile
            )
            activations[layer, 2] = (
                f["membership"]
                * (0.5 * f["exposure"] + 0.5 * f["repetition"])
                * f["family_signal"]
                * max(0.15, f["mechanism_gate"])
                * memory_profile
            )
            activations[layer, 3] = f["low_cue_gate"] * f["anchor_present"] * success_profile
            activations[layer, 4] = 0.35 * f["family_signal"] * (0.8 + 0.2 * cue_profile)
            activations[layer, 5] = f["noise"] + 0.01 * layer
        return activations

    def score_components(
        self,
        prompt_row: dict[str, Any],
        *,
        activations: np.ndarray | None = None,
    ) -> dict[str, float]:
        f = self._feature_bundle(prompt_row)
        acts = activations if activations is not None else self.activation_matrix(prompt_row)
        layer_scores = acts @ self.layer_weights
        aggregate = float(layer_scores.sum())
        score = (
            -3.6
            + aggregate
            + 1.05 * f["cue_strength"]
            + 0.55 * f["membership"] * f["condition_signal"] * f["anchor_present"]
            + 0.95 * f["membership"] * (0.6 * f["exposure"] + 0.4 * f["repetition"]) * f["mechanism_gate"]
            - 1.5 * f["redact_penalty"]
        )
        return {
            "logit": score,
            "probability": sigmoid(score),
            "aggregate_layer_score": aggregate,
        }

    def target_logprob(self, prompt_row: dict[str, Any], *, activations: np.ndarray | None = None) -> float:
        probability = self.score_components(prompt_row, activations=activations)["probability"]
        return math.log(max(probability, 1e-8))

    def greedy_extracts(self, prompt_row: dict[str, Any], *, activations: np.ndarray | None = None) -> bool:
        return self.score_components(prompt_row, activations=activations)["probability"] >= 0.5

    def sample_extracts(
        self,
        prompt_row: dict[str, Any],
        *,
        sample_index: int,
        generation_seed: int,
        activations: np.ndarray | None = None,
    ) -> bool:
        probability = self.score_components(prompt_row, activations=activations)["probability"]
        rng = random.Random(
            stable_int_hash(
                f"{generation_seed}:{prompt_row.get('task_id','')}:{sample_index}:{self.seed}",
                modulo=10**9,
            )
        )
        return rng.random() < probability

    def render_extraction(self, prompt_row: dict[str, Any]) -> str:
        record = self.member_records.get(prompt_row.get("record_id", ""))
        if not record:
            return "No matching sensitive record is available."
        fields = record["fields"]
        sensitive_fields = record.get("sensitive_fields", [])
        parts = [f"{name}: {fields[name]}" for name in sensitive_fields if name in fields]
        return "; ".join(parts)

    def render_non_extraction(self, prompt_row: dict[str, Any], *, sample_index: int) -> str:
        cue_band = prompt_row.get("cue_band_computed") or prompt_row.get("cue_band_requested", "low")
        fallback = {
            "high": "I cannot verify the exact sensitive field from the provided snippet.",
            "medium": "The record context suggests a private field exists, but the exact value is unavailable.",
            "low": "The anchor maps to a private profile, but confidential details are withheld.",
            "no_cue": "I do not have a concrete private record to provide.",
            "no-cue": "I do not have a concrete private record to provide.",
        }
        text = fallback.get(cue_band, fallback["low"])
        if prompt_row.get("membership") == "nonmember" and sample_index % 2 == 1:
            return text + " Example placeholder: redacted@example-synth.org."
        return text

    def generate_text(
        self,
        prompt_row: dict[str, Any],
        *,
        do_sample: bool,
        sample_index: int,
        generation_seed: int,
        activations: np.ndarray | None = None,
    ) -> str:
        extracts = (
            self.sample_extracts(
                prompt_row,
                sample_index=sample_index,
                generation_seed=generation_seed,
                activations=activations,
            )
            if do_sample
            else self.greedy_extracts(prompt_row, activations=activations)
        )
        return self.render_extraction(prompt_row) if extracts else self.render_non_extraction(prompt_row, sample_index=sample_index)

    def utility_perplexity(self) -> float:
        base = 11.0
        modifier = 1.5 - 0.3 * _condition_modifier(self.condition)
        return base + modifier

    def prompt_signature(self, prompt_row: dict[str, Any]) -> np.ndarray:
        feats = self._feature_bundle(prompt_row)
        record_id = prompt_row.get("record_id", "")
        signature = [
            ((((stable_int_hash(f"{record_id}:{idx}", modulo=1000) / 1000.0) * 2.0) - 1.0) * 5.0)
            for idx in range(4)
        ]
        return np.asarray(
            [
                feats["cue_strength"],
                feats["membership"],
                feats["exposure"],
                feats["repetition"],
                feats["condition_signal"],
                feats["family_signal"],
                feats["anchor_present"],
                feats["low_cue_gate"],
                *signature,
            ],
            dtype=float,
        )

    def candidate_gradient(self, doc_row: dict[str, Any]) -> np.ndarray:
        record = self.member_records.get(doc_row.get("record_id", ""))
        family_signal = _family_signal((record or {}).get("family", "identity"))
        exposure = math.log1p(self.exposure_count.get(doc_row.get("record_id", ""), 0))
        template_signal = (stable_int_hash(doc_row.get("template_id", "template"), modulo=13) + 1) / 13.0
        record_id = doc_row.get("record_id", "")
        signature = [
            (((stable_int_hash(f"{record_id}:{idx}", modulo=1000) / 1000.0) * 2.0) - 1.0) * 10.0
            for idx in range(4)
        ]
        return np.asarray(
            [
                exposure,
                family_signal,
                template_signal,
                1.0 if doc_row.get("variant_type") == "exact_duplicate" else 0.75,
                1.0 if "cluster" in doc_row.get("cluster_id", "") else 0.5,
                1.0 if doc_row.get("condition", "").lower().startswith("c3") else 0.7,
                1.0,
                0.25,
                *signature,
            ],
            dtype=float,
        )


def build_toy_model(
    *,
    model_dir: str | Path,
    model_name: str,
    condition: str,
    seed: int,
    member_records: dict[str, dict[str, Any]],
    exposure_count: dict[str, int],
    docs_by_record: dict[str, list[dict[str, Any]]],
    corpus_manifest: dict[str, Any],
) -> ToyModel:
    ensure_dir(model_dir)
    layer_weights = np.zeros((ACTIVATION_DIM,), dtype=float)
    layer_weights[:] = np.asarray([0.12, 0.18, 0.14, 0.20, 0.03, 0.01], dtype=float)
    model = ToyModel(
        model_dir=Path(model_dir),
        model_name=model_name,
        condition=condition,
        seed=seed,
        member_records=member_records,
        exposure_count=exposure_count,
        docs_by_record=docs_by_record,
        corpus_manifest=corpus_manifest,
        layer_weights=layer_weights,
    )
    model.save()
    return model


def detect_backend(model_path: str | Path) -> str | None:
    model_dir = Path(model_path)
    if model_dir.is_dir() and (model_dir / "model.json").exists():
        payload = json.loads((model_dir / "model.json").read_text(encoding="utf-8"))
        return payload.get("backend")
    if model_dir.is_dir() and (model_dir / "mcrate_backend.json").exists():
        payload = json.loads((model_dir / "mcrate_backend.json").read_text(encoding="utf-8"))
        return payload.get("backend")
    if model_dir.is_dir() and (model_dir / "config.json").exists():
        return "huggingface_causal_lm"
    return None


def load_toy_or_raise(model_path: str | Path) -> ToyModel:
    backend = detect_backend(model_path)
    if backend != "toy_memorizer":
        raise ValueError(f"Model at {model_path} is not a toy_memorizer backend.")
    return ToyModel.load(model_path)
