from __future__ import annotations

import argparse
from pathlib import Path

from mcrate.models import detect_backend, load_toy_or_raise
from mcrate.models.hf import eval_hf_perplexity
from mcrate.utils.io import load_yaml, write_json
from mcrate.utils.logging import get_logger
from mcrate.utils.io import read_json


LOGGER = get_logger(__name__)


def eval_perplexity(model_path: str, out_path: str | None = None) -> dict:
    backend = detect_backend(model_path)
    if backend == "toy_memorizer":
        model = load_toy_or_raise(model_path)
        metrics = {
            "backend": backend,
            "condition": model.condition,
            "background_perplexity": round(model.utility_perplexity(), 4),
            "synthetic_perplexity": round(model.utility_perplexity() * 0.91, 4),
        }
    elif backend == "huggingface_causal_lm":
        model_dir = Path(model_path)
        root_dir = model_dir.parent if model_dir.name == "final_model" else model_dir
        manifest_path = root_dir / "corpus_manifest.json"
        if not manifest_path.exists():
            raise RuntimeError(f"Could not locate corpus manifest for HF model at {model_path}")
        manifest = read_json(manifest_path)
        validation_file = manifest["validation_file"]
        metrics = eval_hf_perplexity(
            model_path,
            validation_file,
            batch_size=1,
            sequence_length=int(load_yaml(root_dir / "training_args.json").get("sequence_length", 1024))
            if (root_dir / "training_args.json").exists()
            else 1024,
        )
    else:
        raise RuntimeError("Perplexity evaluation for non-toy backends requires optional ML dependencies.")
    if out_path:
        write_json(out_path, metrics)
    LOGGER.info("Evaluated perplexity for %s", model_path)
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate held-out perplexity for a trained model.")
    parser.add_argument("--model", required=True, help="Model directory.")
    parser.add_argument("--out", default=None, help="Optional output JSON path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    eval_perplexity(args.model, args.out)


if __name__ == "__main__":
    main()
