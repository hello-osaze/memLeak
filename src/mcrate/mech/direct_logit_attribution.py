from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Any

from mcrate.models import detect_backend, load_toy_or_raise
from mcrate.models.hf import (
    _load_model_and_tokenizer,
    _teacher_forced_target_batch,
    _torch_module,
    load_record_map,
    logits_from_hidden,
    model_run_name,
    target_logprob_with_optional_edits,
    target_text_from_record,
)
from mcrate.utils.io import read_jsonl, write_jsonl
from mcrate.utils.logging import get_logger


LOGGER = get_logger(__name__)


def _collapse(scores: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped = defaultdict(list)
    for row in scores:
        grouped[row["task_id"]].append(row)
    return [dict(rows[0]) for rows in grouped.values()]


def direct_logit_attribution(*, model_path: str, scores_path: str, out_path: str) -> list[dict[str, Any]]:
    backend = detect_backend(model_path)
    rows = _collapse(read_jsonl(scores_path))
    if backend == "toy_memorizer":
        model = load_toy_or_raise(model_path)
        output = []
        for row in rows:
            activations = model.activation_matrix(row)
            contributions = (activations @ model.layer_weights).tolist()
            output.append(
                {
                    "model_run": Path(model_path).parent.name,
                    "task_id": row["task_id"],
                    "record_id": row["record_id"],
                    "cue_band": row["cue_band"],
                    "layer_contributions": [round(float(value), 4) for value in contributions],
                    "total_target_logprob": round(float(model.target_logprob(row)), 4),
                }
            )
        write_jsonl(out_path, output)
        LOGGER.info("Wrote direct logit attribution to %s", out_path)
        return output

    if backend != "huggingface_causal_lm":
        raise RuntimeError(f"Unsupported backend for direct logit attribution: {backend}")

    model, tokenizer, device = _load_model_and_tokenizer(model_path)
    record_map = load_record_map(model_path)
    output = []
    for row in rows:
        record = record_map.get(row["record_id"])
        if not record:
            continue
        target_text = target_text_from_record(record)
        if not target_text:
            continue
        batch, prompt_length, first_target_token_id = _teacher_forced_target_batch(tokenizer, row["prompt"], target_text, device)
        with _torch_module().no_grad():
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                output_hidden_states=True,
            )
        hidden_states = outputs.hidden_states or ()
        if len(hidden_states) < 2:
            continue
        prompt_index = prompt_length - 1
        prev_logit = float(logits_from_hidden(model, hidden_states[0][0, prompt_index]).view(-1)[first_target_token_id].detach().cpu())
        contributions = []
        for layer_index in range(len(hidden_states) - 1):
            current_logit = float(
                logits_from_hidden(model, hidden_states[layer_index + 1][0, prompt_index]).view(-1)[first_target_token_id].detach().cpu()
            )
            contributions.append(round(current_logit - prev_logit, 4))
            prev_logit = current_logit
        total = target_logprob_with_optional_edits(
            model,
            tokenizer,
            device,
            prompt=row["prompt"],
            target_text=target_text,
        )
        output.append(
            {
                "model_run": model_run_name(model_path),
                "task_id": row["task_id"],
                "record_id": row["record_id"],
                "cue_band": row["cue_band"],
                "first_target_token": tokenizer.decode([first_target_token_id]).strip(),
                "first_target_token_id": first_target_token_id,
                "layer_contributions": contributions,
                "total_target_logprob": round(float(total["target_logprob"]), 4),
                "total_first_target_logit": round(float(total["first_target_logit"]), 4),
            }
        )
    write_jsonl(out_path, output)
    LOGGER.info("Wrote direct logit attribution to %s", out_path)
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute simple per-layer logit attribution.")
    parser.add_argument("--model", required=True, help="Model directory.")
    parser.add_argument("--scores", required=True, help="Scored generations JSONL.")
    parser.add_argument("--out", required=True, help="Output JSONL.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    direct_logit_attribution(model_path=args.model, scores_path=args.scores, out_path=args.out)


if __name__ == "__main__":
    main()
