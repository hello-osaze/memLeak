from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from mcrate.models import detect_backend, load_toy_or_raise
from mcrate.models.hf import generate_hf
from mcrate.utils.io import load_yaml, read_jsonl, write_jsonl
from mcrate.utils.logging import get_logger


LOGGER = get_logger(__name__)


def _model_run_name(model_path: str) -> str:
    return Path(model_path).name


def run_generation(
    *,
    model_path: str,
    prompts_path: str,
    generation_config_path: str,
    out_path: str,
    include_failed_prompts: bool = False,
) -> list[dict[str, Any]]:
    prompts = read_jsonl(prompts_path)
    generation_config = load_yaml(generation_config_path)
    backend = detect_backend(model_path) or generation_config.get("backend", "huggingface_causal_lm")
    eval_prompts = [row for row in prompts if include_failed_prompts or row.get("passes_cue_filter", True)]
    rows: list[dict[str, Any]] = []
    row_count = 0
    wrote_rows = False
    if backend == "toy_memorizer":
        model = load_toy_or_raise(model_path)
        do_sample = bool(generation_config.get("do_sample", False))
        num_return_sequences = int(generation_config.get("num_return_sequences", 1))
        generation_seed = int(generation_config.get("seed", 1))
        for prompt_row in eval_prompts:
            iterations = num_return_sequences if do_sample else 1
            for sample_index in range(iterations):
                output_text = model.generate_text(
                    prompt_row,
                    do_sample=do_sample,
                    sample_index=sample_index,
                    generation_seed=generation_seed,
                )
                rows.append(
                    {
                        "generation_id": f"{prompt_row['task_id']}_{sample_index:02d}",
                        "task_id": prompt_row["task_id"],
                        "record_id": prompt_row["record_id"],
                        "cluster_id": prompt_row["cluster_id"],
                        "family": prompt_row["family"],
                        "membership": prompt_row["membership"],
                        "cue_band": prompt_row.get("cue_band_computed", prompt_row.get("cue_band_requested")),
                        "condition": model.condition,
                        "model_run": _model_run_name(model_path),
                        "prompt": prompt_row["prompt"],
                        "output_text": output_text,
                        "generation_config": Path(generation_config_path).stem,
                        "sample_index": sample_index,
                        "seed": generation_seed,
                        "passes_cue_filter": prompt_row.get("passes_cue_filter", True),
                        "target_logprob": model.target_logprob(prompt_row),
                    }
                )
        row_count = len(rows)
    else:
        stream_to_disk = bool(generation_config.get("stream_to_disk", True))
        result = generate_hf(
            model_path=model_path,
            prompts=eval_prompts,
            generation_config=generation_config,
            out_path=out_path if stream_to_disk else None,
        )
        if stream_to_disk:
            row_count = int(result)
            wrote_rows = True
        else:
            rows = result
            row_count = len(rows)
    if not wrote_rows:
        write_jsonl(out_path, rows)
    LOGGER.info("Wrote %s generations to %s", row_count, out_path)
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run model generation on scored prompts.")
    parser.add_argument("--model", required=True, help="Model directory or model id.")
    parser.add_argument("--prompts", required=True, help="Scored prompts JSONL.")
    parser.add_argument("--generation_config", required=True, help="Generation YAML config.")
    parser.add_argument("--out", required=True, help="Output generations JSONL.")
    parser.add_argument("--include_failed_prompts", action="store_true", help="Include prompts that failed cue filter.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_generation(
        model_path=args.model,
        prompts_path=args.prompts,
        generation_config_path=args.generation_config,
        out_path=args.out,
        include_failed_prompts=args.include_failed_prompts,
    )


if __name__ == "__main__":
    main()
