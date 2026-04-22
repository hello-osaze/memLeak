from __future__ import annotations

import contextlib
import json
import math
from array import array
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Iterator

import numpy as np

from mcrate.utils.io import ensure_parent, read_json, read_jsonl


def ensure_hf_dependencies() -> None:
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "The Hugging Face backend requires `torch` and `transformers`. "
            "Install the optional dependencies or use the `toy_memorizer` backend for debug runs."
        ) from exc


def load_backend_metadata(model_path: str | Path) -> dict[str, Any]:
    model_dir = Path(model_path)
    metadata_path = model_dir / "mcrate_backend.json"
    if metadata_path.exists():
        return json.loads(metadata_path.read_text(encoding="utf-8"))
    config_path = model_dir / "config.json"
    if config_path.exists():
        payload = json.loads(config_path.read_text(encoding="utf-8"))
        return {
            "backend": "huggingface_causal_lm",
            "model_name": str(model_dir),
            "architectures": payload.get("architectures", []),
        }
    return {}


def model_root_dir(model_path: str | Path) -> Path:
    path = Path(model_path)
    return path.parent if path.name == "final_model" else path


def model_run_name(model_path: str | Path) -> str:
    path = Path(model_path)
    return path.parent.name if path.name == "final_model" else path.name


def load_corpus_manifest(model_path: str | Path) -> dict[str, Any]:
    manifest_path = model_root_dir(model_path) / "corpus_manifest.json"
    if not manifest_path.exists():
        return {}
    return read_json(manifest_path)


def load_training_args(model_path: str | Path) -> dict[str, Any]:
    training_args_path = model_root_dir(model_path) / "training_args.json"
    if not training_args_path.exists():
        return {}
    return read_json(training_args_path)


def load_record_map(model_path: str | Path) -> dict[str, dict[str, Any]]:
    manifest = load_corpus_manifest(model_path)
    records_path = manifest.get("records_path")
    if not records_path:
        return {}
    return {row["record_id"]: row for row in read_jsonl(records_path)}


def target_fields_from_record(record: dict[str, Any]) -> dict[str, Any]:
    fields = record.get("fields", {})
    return {name: fields[name] for name in record.get("sensitive_fields", []) if name in fields}


def target_text_from_record(record: dict[str, Any]) -> str:
    target_fields = target_fields_from_record(record)
    return "; ".join(f"{key}: {value}" for key, value in target_fields.items())


def _save_backend_metadata(model_dir: str | Path, payload: dict[str, Any]) -> None:
    target = Path(model_dir) / "mcrate_backend.json"
    target.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _device_and_dtype(precision: str | None = None, device_preference: str | None = None) -> tuple[Any, Any]:
    ensure_hf_dependencies()
    import torch

    preferred = (device_preference or "auto").lower()
    if preferred == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("Requested CUDA device, but CUDA is not available.")
        device = torch.device("cuda")
    elif preferred == "mps":
        if not (getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()):
            raise RuntimeError("Requested MPS device, but MPS is not available.")
        device = torch.device("mps")
    elif preferred == "cpu":
        device = torch.device("cpu")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    desired = (precision or "").lower()
    if desired == "bf16" and device.type == "cuda" and torch.cuda.is_bf16_supported():
        dtype = torch.bfloat16
    elif desired == "fp16" and device.type == "cuda":
        dtype = torch.float16
    else:
        dtype = torch.float32
    return device, dtype


def _torch_module():
    ensure_hf_dependencies()
    import torch

    return torch


def _transformers_module():
    ensure_hf_dependencies()
    import transformers

    return transformers


def _extract_tensor(value: Any) -> Any:
    if isinstance(value, tuple):
        return _extract_tensor(value[0])
    if isinstance(value, list):
        return _extract_tensor(value[0])
    return value


def _replace_tensor(value: Any, tensor: Any) -> Any:
    if isinstance(value, tuple):
        if not value:
            return value
        return (tensor, *value[1:])
    if isinstance(value, list):
        if not value:
            return value
        updated = list(value)
        updated[0] = tensor
        return updated
    return tensor


def _default_pad_token(tokenizer: Any) -> None:
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"


class TextBlockDataset:
    def __init__(self, text_file: str, tokenizer: Any, sequence_length: int) -> None:
        self.sequence_length = int(sequence_length)
        self.pad_token_id = int(tokenizer.pad_token_id or tokenizer.eos_token_id or 0)
        self.eos_id = int(tokenizer.eos_token_id or tokenizer.pad_token_id or 0)
        token_buffer: array[int] = array("I")
        for text_chunk in self._iter_text_chunks(text_file):
            encoded = tokenizer(text_chunk, add_special_tokens=False)["input_ids"]
            if encoded and isinstance(encoded[0], list):
                for row in encoded:
                    token_buffer.extend(int(token_id) for token_id in row)
                    token_buffer.append(self.eos_id)
                continue
            if encoded:
                token_buffer.extend(int(token_id) for token_id in encoded)
                token_buffer.append(self.eos_id)

        if len(token_buffer) < 2:
            token_buffer.extend([self.eos_id, self.eos_id])

        self.token_ids = np.asarray(token_buffer, dtype=np.int32)
        self.num_examples = max(1, math.ceil((len(self.token_ids) - 1) / self.sequence_length))

    @staticmethod
    def _iter_text_chunks(text_file: str, *, target_chars: int = 1_000_000) -> Iterator[str]:
        buffer: list[str] = []
        char_count = 0
        with Path(text_file).open("r", encoding="utf-8") as handle:
            for line in handle:
                buffer.append(line)
                char_count += len(line)
                if char_count >= target_chars:
                    yield "".join(buffer)
                    buffer = []
                    char_count = 0
        if buffer:
            yield "".join(buffer)

    def __len__(self) -> int:
        return self.num_examples

    def __getitem__(self, index: int) -> dict[str, Any]:
        torch = _torch_module()
        start = index * self.sequence_length
        chunk = self.token_ids[start : start + self.sequence_length]
        if len(chunk) < 2:
            chunk = np.asarray([self.eos_id, self.eos_id], dtype=np.int32)

        input_ids = torch.full((self.sequence_length,), self.pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros((self.sequence_length,), dtype=torch.long)
        labels = torch.full((self.sequence_length,), -100, dtype=torch.long)

        chunk_tensor = torch.tensor(chunk, dtype=torch.long)
        valid_length = min(len(chunk_tensor), self.sequence_length)
        input_ids[:valid_length] = chunk_tensor[:valid_length]
        attention_mask[:valid_length] = 1
        labels[:valid_length] = chunk_tensor[:valid_length]
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


def _stack_batch(batch: list[dict[str, Any]]) -> dict[str, Any]:
    torch = _torch_module()
    return {key: torch.stack([row[key] for row in batch], dim=0) for key in batch[0]}


def _move_to_device(batch: dict[str, Any], device: Any) -> dict[str, Any]:
    return {key: value.to(device) for key, value in batch.items()}


def _mean(values: Iterable[float]) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def _tensor_to_numpy(tensor: Any) -> np.ndarray:
    torch = _torch_module()
    detached = tensor.detach()
    if detached.dtype == torch.bfloat16:
        detached = detached.to(dtype=torch.float32)
    return detached.cpu().numpy()


def _evaluate_dataset(model: Any, dataset: TextBlockDataset, batch_size: int, device: Any) -> dict[str, float]:
    torch = _torch_module()
    if len(dataset) == 0:
        return {"loss": 0.0, "perplexity": 1.0}
    losses = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(dataset), batch_size):
            batch = _stack_batch([dataset[idx] for idx in range(start, min(len(dataset), start + batch_size))])
            outputs = model(**_move_to_device(batch, device))
            losses.append(float(outputs.loss.detach().cpu()))
    loss = _mean(losses)
    perplexity = float(math.exp(min(20.0, loss)))
    return {"loss": round(loss, 6), "perplexity": round(perplexity, 6)}


def save_activation_array(path: str | Path, array: np.ndarray, *, storage_dtype: str = "float16") -> Path:
    target = Path(path)
    dtype_name = (storage_dtype or "float16").lower()
    if dtype_name in {"float16", "fp16", "half"}:
        stored = np.asarray(array, dtype=np.float16)
    elif dtype_name in {"float32", "fp32", "single"}:
        stored = np.asarray(array, dtype=np.float32)
    else:
        stored = np.asarray(array)
    with target.open("wb") as handle:
        np.save(handle, stored)
    return target


def load_activation_array(path: str | Path) -> np.ndarray:
    with Path(path).open("rb") as handle:
        return np.load(handle)


def _load_model_and_tokenizer(
    model_name_or_path: str,
    precision: str | None = None,
    device_preference: str | None = None,
) -> tuple[Any, Any, Any]:
    torch = _torch_module()
    transformers = _transformers_module()
    device, dtype = _device_and_dtype(precision, device_preference)
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name_or_path)
    _default_pad_token(tokenizer)
    model_kwargs: dict[str, Any] = {}
    if device.type == "cuda" and dtype in {torch.float16, torch.bfloat16}:
        model_kwargs["torch_dtype"] = dtype
    model = transformers.AutoModelForCausalLM.from_pretrained(model_name_or_path, **model_kwargs)
    model.to(device)
    return model, tokenizer, device


def _lm_head_module(model: Any) -> Any:
    if hasattr(model, "lm_head"):
        return model.lm_head
    if hasattr(model, "embed_out"):
        return model.embed_out
    output_embeddings = model.get_output_embeddings()
    if output_embeddings is None:
        raise RuntimeError(f"Could not find an LM head for {model.__class__.__name__}")
    return output_embeddings


def _final_layer_norm(model: Any) -> Any | None:
    if hasattr(model, "transformer") and hasattr(model.transformer, "ln_f"):
        return model.transformer.ln_f
    if hasattr(model, "gpt_neox") and hasattr(model.gpt_neox, "final_layer_norm"):
        return model.gpt_neox.final_layer_norm
    if hasattr(model, "model") and hasattr(model.model, "norm"):
        return model.model.norm
    return None


def _apply_final_layer_norm(model: Any, hidden: Any) -> Any:
    final_norm = _final_layer_norm(model)
    if final_norm is None:
        return hidden
    squeezed = False
    if hidden.ndim == 1:
        hidden = hidden.unsqueeze(0)
        squeezed = True
    normalized = final_norm(hidden)
    return normalized[0] if squeezed else normalized


def logits_from_hidden(model: Any, hidden: Any) -> Any:
    return _lm_head_module(model)(_apply_final_layer_norm(model, hidden))


def _teacher_forced_target_batch(tokenizer: Any, prompt: str, target_text: str, device: Any) -> tuple[dict[str, Any], int, int | None]:
    torch = _torch_module()
    prompt_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    target_ids = tokenizer(target_text, return_tensors="pt", add_special_tokens=False)
    if target_ids["input_ids"].shape[1] == 0:
        raise RuntimeError("Target text tokenized to an empty sequence.")
    input_ids = torch.cat([prompt_ids["input_ids"], target_ids["input_ids"]], dim=1).to(device)
    attention_mask = torch.ones_like(input_ids, device=device)
    labels = input_ids.clone()
    prompt_length = int(prompt_ids["input_ids"].shape[1])
    labels[:, :prompt_length] = -100
    batch = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }
    return batch, prompt_length, int(target_ids["input_ids"][0, 0].item())


@contextlib.contextmanager
def residual_post_edits(model: Any, edits: list[dict[str, Any]] | None) -> Iterator[None]:
    if not edits:
        yield
        return
    hooks = []
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for edit in edits:
        grouped[int(edit["layer"])].append(edit)
    layers = _get_transformer_layers(model)

    for layer_index, layer in enumerate(layers):
        if layer_index not in grouped:
            continue
        layer_edits = grouped[layer_index]

        def hook(module: Any, module_input: Any, module_output: Any, *, layer_specs: list[dict[str, Any]] = layer_edits) -> Any:
            tensor = _extract_tensor(module_output)
            updated = tensor.clone()
            for spec in layer_specs:
                mode = spec.get("mode", "replace")
                vector = _torch_module().as_tensor(spec["vector"], device=tensor.device, dtype=tensor.dtype).view(1, 1, -1)
                token_index = spec.get("token_index")
                alpha = float(spec.get("alpha", 1.0))
                if token_index is None:
                    if mode == "replace":
                        updated = vector.expand_as(updated)
                    elif mode == "subtract":
                        updated = updated - alpha * vector
                    elif mode == "add":
                        updated = updated + alpha * vector
                else:
                    if mode == "replace":
                        updated[:, token_index, :] = vector[:, 0, :]
                    elif mode == "subtract":
                        updated[:, token_index, :] = updated[:, token_index, :] - alpha * vector[:, 0, :]
                    elif mode == "add":
                        updated[:, token_index, :] = updated[:, token_index, :] + alpha * vector[:, 0, :]
            return _replace_tensor(module_output, updated)

        hooks.append(layer.register_forward_hook(hook))
    try:
        yield
    finally:
        for hook in hooks:
            hook.remove()


def capture_residual_post(model: Any, tokenizer: Any, device: Any, prompt: str) -> dict[str, Any]:
    encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=False, truncation=True)
    encoded = {key: value.to(device) for key, value in encoded.items()}
    with _torch_module().no_grad():
        outputs = model(**encoded, output_hidden_states=True)
    prompt_index = int(encoded["attention_mask"].sum(dim=1)[0].item()) - 1
    hidden_states = outputs.hidden_states or ()
    return {
        "prompt_index": prompt_index,
        "hidden_states": hidden_states,
        "vectors": {
            layer_index: _tensor_to_numpy(hidden_states[layer_index + 1][0, prompt_index])
            for layer_index in range(max(0, len(hidden_states) - 1))
        },
    }


def target_logprob_with_optional_edits(
    model: Any,
    tokenizer: Any,
    device: Any,
    *,
    prompt: str,
    target_text: str,
    edits: list[dict[str, Any]] | None = None,
) -> dict[str, float]:
    torch = _torch_module()
    batch, prompt_length, first_target_token_id = _teacher_forced_target_batch(tokenizer, prompt, target_text, device)
    with residual_post_edits(model, edits), torch.no_grad():
        outputs = model(**batch)
    return {
        "target_logprob": -float(outputs.loss.detach().cpu()),
        "first_target_logit": float(outputs.logits[0, prompt_length - 1, first_target_token_id].detach().cpu()),
    }


def evaluate_dataset_with_residual_edits(
    model: Any,
    dataset: TextBlockDataset,
    batch_size: int,
    device: Any,
    edits: list[dict[str, Any]] | None = None,
    *,
    max_examples: int | None = None,
) -> dict[str, float]:
    torch = _torch_module()
    if len(dataset) == 0:
        return {"loss": 0.0, "perplexity": 1.0}
    losses = []
    model.eval()
    limit = len(dataset) if max_examples is None else min(len(dataset), max_examples)
    with residual_post_edits(model, edits), torch.no_grad():
        for start in range(0, limit, batch_size):
            batch = _stack_batch([dataset[idx] for idx in range(start, min(limit, start + batch_size))])
            outputs = model(**_move_to_device(batch, device))
            losses.append(float(outputs.loss.detach().cpu()))
    loss = _mean(losses)
    return {"loss": round(loss, 6), "perplexity": round(float(math.exp(min(20.0, loss))), 6)}


def select_provenance_parameters(
    model: Any,
    *,
    last_n_layers: int = 1,
    include_final_norm: bool = True,
    include_lm_head: bool = False,
) -> list[tuple[str, Any]]:
    selected: list[tuple[str, Any]] = []
    seen: set[int] = set()

    def append_named_parameters(prefix: str, module: Any, allowed_suffixes: tuple[str, ...]) -> None:
        if module is None:
            return
        for name, param in module.named_parameters():
            if not param.requires_grad:
                continue
            if not name.endswith(allowed_suffixes):
                continue
            if id(param) in seen:
                continue
            seen.add(id(param))
            selected.append((f"{prefix}.{name}", param))

    layers = _get_transformer_layers(model)
    if int(last_n_layers) > 0:
        start = max(0, len(layers) - int(last_n_layers))
        for layer_index in range(start, len(layers)):
            layer = layers[layer_index]
            append_named_parameters(
                f"layer_{layer_index}.attention",
                _layer_attention_module(layer),
                ("c_proj.weight", "c_proj.bias", "out_proj.weight", "out_proj.bias", "dense.weight", "dense.bias"),
            )
            append_named_parameters(
                f"layer_{layer_index}.mlp",
                _layer_mlp_module(layer),
                ("c_proj.weight", "c_proj.bias", "down_proj.weight", "down_proj.bias", "dense_4h_to_h.weight", "dense_4h_to_h.bias"),
            )

    if include_final_norm:
        final_norm = _final_layer_norm(model)
        if final_norm is not None:
            append_named_parameters("final_norm", final_norm, ("weight", "bias"))
    if include_lm_head:
        append_named_parameters("lm_head", _lm_head_module(model), ("weight", "bias"))
    return selected


def gradient_vector_from_loss(model: Any, loss: Any, selected_parameters: list[tuple[str, Any]]) -> Any:
    torch = _torch_module()
    model.zero_grad(set_to_none=True)
    loss.backward()
    pieces = []
    for _, param in selected_parameters:
        grad = param.grad
        if grad is None:
            pieces.append(torch.zeros((param.numel(),), dtype=torch.float32, device=param.device))
        else:
            pieces.append(grad.detach().float().reshape(-1))
    vector = torch.cat(pieces, dim=0).cpu()
    model.zero_grad(set_to_none=True)
    return vector


def finetune_hf_model(*args: Any, **kwargs: Any) -> dict[str, Any]:
    ensure_hf_dependencies()
    torch = _torch_module()
    transformers = _transformers_module()

    config = kwargs["config"]
    train_file = kwargs["train_file"]
    validation_file = kwargs["validation_file"]
    out_dir = Path(kwargs["out_dir"])
    manifest = kwargs.get("manifest", {})

    model_name = config["model_name"]
    seed = int(config.get("seed", 1))
    transformers.set_seed(seed)
    device, _ = _device_and_dtype(config.get("precision"), config.get("device"))
    model, tokenizer, device = _load_model_and_tokenizer(model_name, config.get("precision"), config.get("device"))
    sequence_length = int(config.get("sequence_length", 1024))
    train_dataset = TextBlockDataset(train_file, tokenizer, sequence_length)
    val_dataset = TextBlockDataset(validation_file, tokenizer, sequence_length)

    batch_size = int(config.get("per_device_train_batch_size", 1))
    grad_accum = max(1, int(config.get("gradient_accumulation_steps", 1)))
    learning_rate = float(config.get("learning_rate", 2e-5))
    weight_decay = float(config.get("weight_decay", 0.0))
    optimizer_name = str(config.get("optimizer", "adamw")).lower()
    optimizer_eps = float(config.get("optimizer_eps", 1e-8))
    optimizer_beta1 = float(config.get("optimizer_beta1", 0.9))
    optimizer_beta2 = float(config.get("optimizer_beta2", 0.999))
    optimizer_momentum = float(config.get("optimizer_momentum", 0.0))
    num_epochs = float(config.get("num_train_epochs", 1))
    save_steps = int(config.get("save_steps", 0))
    logging_steps = max(1, int(config.get("logging_steps", 10)))
    eval_steps = int(config.get("eval_steps", 0))
    max_grad_norm = float(config.get("max_grad_norm", 1.0))

    if optimizer_name == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            eps=optimizer_eps,
            betas=(optimizer_beta1, optimizer_beta2),
        )
    elif optimizer_name == "adam":
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            eps=optimizer_eps,
            betas=(optimizer_beta1, optimizer_beta2),
        )
    elif optimizer_name == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            momentum=optimizer_momentum,
        )
    else:
        raise RuntimeError(f"Unsupported optimizer '{optimizer_name}' for HF fine-tuning.")
    total_batches = math.ceil(len(train_dataset) / batch_size)
    total_updates = max(1, math.ceil((total_batches * num_epochs) / grad_accum))
    warmup_steps = int(total_updates * float(config.get("warmup_ratio", 0.0)))
    if config.get("scheduler", "cosine") == "cosine":
        scheduler = transformers.get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_updates,
        )
    else:
        scheduler = transformers.get_constant_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_root = out_dir / "checkpoints"
    checkpoint_root.mkdir(parents=True, exist_ok=True)

    model.train()
    optimizer.zero_grad(set_to_none=True)
    update_step = 0
    batch_step = 0
    train_losses: list[float] = []
    log_history: list[dict[str, Any]] = []
    epoch_count = max(1, int(math.ceil(num_epochs)))

    for epoch in range(epoch_count):
        for start in range(0, len(train_dataset), batch_size):
            batch_items = [train_dataset[idx] for idx in range(start, min(len(train_dataset), start + batch_size))]
            batch = _stack_batch(batch_items)
            outputs = model(**_move_to_device(batch, device))
            loss = outputs.loss / grad_accum
            loss.backward()
            batch_step += 1
            train_losses.append(float(outputs.loss.detach().cpu()))

            should_step = batch_step % grad_accum == 0 or (start + batch_size) >= len(train_dataset)
            if should_step:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                update_step += 1

                if update_step % logging_steps == 0:
                    log_history.append(
                        {
                            "step": update_step,
                            "train_loss": round(_mean(train_losses[-logging_steps:]), 6),
                            "lr": round(float(scheduler.get_last_lr()[0]), 10),
                        }
                    )

                if eval_steps and update_step % eval_steps == 0:
                    eval_metrics = _evaluate_dataset(model, val_dataset, batch_size, device)
                    log_history.append({"step": update_step, "eval_loss": eval_metrics["loss"], "eval_ppl": eval_metrics["perplexity"]})
                    model.train()

                if save_steps and update_step % save_steps == 0:
                    checkpoint_dir = checkpoint_root / f"step_{update_step}"
                    checkpoint_dir.mkdir(parents=True, exist_ok=True)
                    model.save_pretrained(checkpoint_dir)
                    tokenizer.save_pretrained(checkpoint_dir)

    final_model_dir = out_dir / "final_model"
    final_model_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(final_model_dir)
    tokenizer.save_pretrained(final_model_dir)
    eval_metrics = _evaluate_dataset(model, val_dataset, batch_size, device)
    metadata = {
        "backend": "huggingface_causal_lm",
        "model_name": model_name,
        "seed": seed,
        "condition": manifest.get("condition"),
        "sequence_length": sequence_length,
        "hidden_size": int(getattr(model.config, "hidden_size", getattr(model.config, "n_embd", 0))),
        "num_layers": int(
            getattr(model.config, "num_hidden_layers", getattr(model.config, "n_layer", 0))
        ),
        "tokenizer_name_or_path": getattr(tokenizer, "name_or_path", model_name),
    }
    _save_backend_metadata(final_model_dir, metadata)
    return {
        "final_model_dir": str(final_model_dir.resolve()),
        "trainer_state": {
            "backend": "huggingface_causal_lm",
            "seed": seed,
            "num_update_steps": update_step,
            "train_examples": len(train_dataset),
            "log_history": log_history[-100:],
        },
        "eval_metrics": {
            "backend": "huggingface_causal_lm",
            "validation_loss": eval_metrics["loss"],
            "validation_perplexity": eval_metrics["perplexity"],
        },
    }


def generate_hf(*args: Any, **kwargs: Any) -> list[dict[str, Any]] | int:
    ensure_hf_dependencies()
    torch = _torch_module()

    model_path = kwargs["model_path"]
    prompts = kwargs["prompts"]
    generation_config = kwargs["generation_config"]
    out_path = kwargs.get("out_path")
    metadata = load_backend_metadata(model_path)
    model, tokenizer, device = _load_model_and_tokenizer(
        str(model_path),
        generation_config.get("precision"),
        generation_config.get("device"),
    )
    _default_pad_token(tokenizer)
    model.eval()

    batch_size = int(generation_config.get("batch_size", 1))
    do_sample = bool(generation_config.get("do_sample", False))
    num_return_sequences = int(generation_config.get("num_return_sequences", 1))
    max_new_tokens = int(generation_config.get("max_new_tokens", 80))
    temperature = float(generation_config.get("temperature", 1.0))
    top_p = float(generation_config.get("top_p", 1.0))
    generation_seed = int(generation_config.get("seed", 1))
    torch.manual_seed(generation_seed)

    def target_logprob(prompt_row: dict[str, Any]) -> float | None:
        target_fields = prompt_row.get("target_fields") or {}
        if not target_fields:
            return None
        target_text = "; ".join(f"{key}: {value}" for key, value in target_fields.items())
        return target_logprob_with_optional_edits(
            model,
            tokenizer,
            device,
            prompt=prompt_row["prompt"],
            target_text=target_text,
        )["target_logprob"]

    rows: list[dict[str, Any]] = []
    row_count = 0
    handle = ensure_parent(out_path).open("w", encoding="utf-8") if out_path else None
    try:
        for start in range(0, len(prompts), batch_size):
            batch_prompts = prompts[start : start + batch_size]
            encoded = tokenizer(
                [row["prompt"] for row in batch_prompts],
                return_tensors="pt",
                padding=True,
                truncation=True,
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            input_length = int(encoded["input_ids"].shape[1])
            generate_kwargs = {
                "do_sample": do_sample,
                "max_new_tokens": max_new_tokens,
                "num_return_sequences": num_return_sequences,
                "pad_token_id": tokenizer.pad_token_id,
                "eos_token_id": tokenizer.eos_token_id,
                "remove_invalid_values": True,
            }
            if do_sample:
                generate_kwargs["temperature"] = temperature
                generate_kwargs["top_p"] = top_p
            with torch.no_grad():
                outputs = model.generate(
                    **encoded,
                    **generate_kwargs,
                )
            for batch_index, prompt_row in enumerate(batch_prompts):
                prompt_logprob = target_logprob(prompt_row)
                for sample_index in range(num_return_sequences):
                    output_index = batch_index * num_return_sequences + sample_index
                    full_tokens = outputs[output_index]
                    generated_tokens = full_tokens[input_length:]
                    output_text = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
                    row = {
                        "generation_id": f"{prompt_row['task_id']}_{sample_index:02d}",
                        "task_id": prompt_row["task_id"],
                        "record_id": prompt_row["record_id"],
                        "cluster_id": prompt_row["cluster_id"],
                        "family": prompt_row["family"],
                        "membership": prompt_row["membership"],
                        "cue_band": prompt_row.get("cue_band_computed", prompt_row.get("cue_band_requested")),
                        "condition": metadata.get("condition", "unknown"),
                        "model_run": Path(model_path).parent.name if Path(model_path).name == "final_model" else Path(model_path).name,
                        "prompt": prompt_row["prompt"],
                        "output_text": output_text,
                        "generation_config": generation_config.get("name", "hf_generation"),
                        "sample_index": sample_index,
                        "seed": generation_seed,
                        "passes_cue_filter": prompt_row.get("passes_cue_filter", True),
                        "target_logprob": prompt_logprob,
                    }
                    row_count += 1
                    if handle is not None:
                        handle.write(json.dumps(row, sort_keys=False) + "\n")
                    else:
                        rows.append(row)
    finally:
        if handle is not None:
            handle.close()
    return row_count if handle is not None else rows


def cache_hf_activations(*args: Any, **kwargs: Any) -> dict[str, Any]:
    ensure_hf_dependencies()
    torch = _torch_module()

    model_path = kwargs["model_path"]
    scores = kwargs["scores"]
    config = kwargs["config"]
    out_dir = Path(kwargs["out_dir"])
    model, tokenizer, device = _load_model_and_tokenizer(str(model_path), config.get("precision"), config.get("device"))
    _default_pad_token(tokenizer)
    model.eval()

    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in scores:
        grouped.setdefault(row["group"], []).append(row)
    max_examples = int(config.get("max_examples_per_group", 500))
    sites = list(config.get("cache_sites", ["resid_post", "attn_out", "mlp_out"]))
    storage_dtype = str(config.get("cache_storage_dtype", "float16"))
    layers = _get_transformer_layers(model)
    layer_count = len(layers)
    model_run = Path(model_path).parent.name if Path(model_path).name == "final_model" else Path(model_path).name
    root = out_dir / model_run
    root.mkdir(parents=True, exist_ok=True)
    manifests = []

    for group_name, items in grouped.items():
        selected = items[:max_examples]
        group_dir = root / group_name
        group_dir.mkdir(parents=True, exist_ok=True)
        activations_by_site = {site: {layer: [] for layer in range(layer_count)} for site in sites}

        for row in selected:
            encoded = tokenizer(row["prompt"], return_tensors="pt", truncation=True).to(device)
            hooks = []
            captured: dict[tuple[str, int], Any] = {}

            def make_hook(site_name: str, layer_index: int, kind: str):
                def hook(module: Any, module_input: Any, module_output: Any) -> None:
                    source = module_input[0] if kind == "input" else _extract_tensor(module_output)
                    captured[(site_name, layer_index)] = source.detach().cpu()
                return hook

            for layer_index, layer in enumerate(layers):
                if "resid_pre" in sites:
                    hooks.append(layer.register_forward_hook(make_hook("resid_pre", layer_index, "input")))
                if "resid_post" in sites:
                    hooks.append(layer.register_forward_hook(make_hook("resid_post", layer_index, "output")))
                attn_module = _layer_attention_module(layer)
                if attn_module is not None and "attn_out" in sites:
                    hooks.append(attn_module.register_forward_hook(make_hook("attn_out", layer_index, "output")))
                mlp_module = _layer_mlp_module(layer)
                if mlp_module is not None and "mlp_out" in sites:
                    hooks.append(mlp_module.register_forward_hook(make_hook("mlp_out", layer_index, "output")))

            with torch.no_grad():
                _ = model(**encoded)

            final_prompt_index = int(encoded["attention_mask"].sum(dim=1)[0].item()) - 1
            for site in sites:
                for layer_index in range(layer_count):
                    tensor = captured.get((site, layer_index))
                    if tensor is None:
                        continue
                    vector = _tensor_to_numpy(tensor[0, final_prompt_index])
                    activations_by_site[site][layer_index].append(vector)

            for hook in hooks:
                hook.remove()

        for site in sites:
            for layer_index in range(layer_count):
                array = np.asarray(activations_by_site[site][layer_index], dtype=float)
                path = group_dir / f"layer_{layer_index:02d}_site_{site}.pt"
                save_activation_array(path, array, storage_dtype=storage_dtype)
                manifest = {
                    "backend": "huggingface_causal_lm",
                    "model_run": model_run,
                    "group": group_name,
                    "layer": layer_index,
                    "site": site,
                    "shape": list(array.shape),
                    "storage_dtype": storage_dtype,
                    "position": "final_prompt_token",
                    "examples": [row["task_id"] for row in selected],
                }
                (Path(str(path) + ".json")).write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
                manifests.append(manifest)
    (root / "cache_manifest.json").write_text(json.dumps({"files": manifests}, indent=2) + "\n", encoding="utf-8")
    return {"root": str(root.resolve()), "groups": {k: len(v[:max_examples]) for k, v in grouped.items()}}


def _layer_attention_module(layer: Any) -> Any | None:
    for name in ("attention", "attn", "self_attn"):
        if hasattr(layer, name):
            return getattr(layer, name)
    return None


def _layer_mlp_module(layer: Any) -> Any | None:
    for name in ("mlp", "feed_forward"):
        if hasattr(layer, name):
            return getattr(layer, name)
    return None


def _get_transformer_layers(model: Any) -> list[Any]:
    if hasattr(model, "gpt_neox") and hasattr(model.gpt_neox, "layers"):
        return list(model.gpt_neox.layers)
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return list(model.transformer.h)
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return list(model.model.layers)
    raise RuntimeError(f"Unsupported architecture for activation caching: {model.__class__.__name__}")


def eval_hf_perplexity(model_path: str, text_file: str, *, batch_size: int = 1, sequence_length: int = 1024) -> dict[str, Any]:
    model, tokenizer, device = _load_model_and_tokenizer(model_path)
    dataset = TextBlockDataset(text_file, tokenizer, sequence_length)
    metrics = _evaluate_dataset(model, dataset, batch_size, device)
    return {
        "backend": "huggingface_causal_lm",
        "loss": metrics["loss"],
        "perplexity": metrics["perplexity"],
    }
