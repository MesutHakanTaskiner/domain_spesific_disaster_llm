#!/usr/bin/env python
"""
Finetune chat models (Gemma, Mistral, Qwen, Llama) on JSONL chat data using Unsloth + QLoRA (4-bit).
Dataset expectation: each JSONL line has either `messages` (list of {role, content})
compatible with chat templates, or `user`/`assistant` fields.
"""

import os
from pathlib import Path

os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")      
os.environ.setdefault("UNSLOTH_COMPILE_DISABLE", "1")    

DEFAULT_INDUCTOR_CACHE = Path(
    os.environ.get("TORCHINDUCTOR_CACHE_DIR", r"C:\torchinductor_cache")
).resolve()
os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", str(DEFAULT_INDUCTOR_CACHE))
os.environ.setdefault("TRITON_CACHE_DIR", str(DEFAULT_INDUCTOR_CACHE / "triton"))
# ------------------------------------------------------------------------


import argparse
import logging
from typing import Any, Dict, List, Optional

import torch
from unsloth import FastLanguageModel, is_bfloat16_supported
from datasets import load_dataset
from transformers import TrainingArguments
from trl import SFTTrainer


LOGGER = logging.getLogger(__name__)

DEFAULT_MODELS = {
    ("gemma", "7b"): "google/gemma-7b-it",
    ("gemma", "9b"): "google/gemma-2-9b-it",
    ("mistral", "7b"): "mistralai/Mistral-7B-Instruct-v0.2",
    ("qwen", "7b"): "Qwen/Qwen2-7B-Instruct",
    ("llama", "8b"): "meta-llama/Meta-Llama-3-8B-Instruct",
    ("llama", "70b"): "meta-llama/Meta-Llama-3-70B-Instruct",
}


def resolve_model_name(family: str, size: str, override: Optional[str]) -> str:
    if override:
        return override
    key = (family.lower(), size.lower())
    if key in DEFAULT_MODELS:
        return DEFAULT_MODELS[key]
    raise ValueError(f"No default model for family={family} size={size}. Provide --base-model.")


def build_dataset(
    tokenizer, train_path: str, eval_path: Optional[str], max_length: int
) -> Dict[str, Any]:
    files = {"train": train_path}
    if eval_path:
        files["eval"] = eval_path
    ds = load_dataset("json", data_files=files)

    def format_chat(example: Dict[str, Any]) -> Dict[str, str]:
        messages: Optional[List[Dict[str, str]]] = example.get("messages")
        if not messages:
            user = example.get("user")
            assistant = example.get("assistant")
            if user is None or assistant is None:
                raise ValueError("Example must have 'messages' or both 'user' and 'assistant'.")
            messages = [
                {"role": "user", "content": user},
                {"role": "assistant", "content": assistant},
            ]
        else:
            system_msgs = [m["content"] for m in messages if m.get("role") == "system" and m.get("content")]
            filtered = [m for m in messages if m.get("role") != "system"]
            if system_msgs:
                system_text = "\n\n".join(system_msgs).strip()
                if filtered and filtered[0].get("role") == "user":
                    filtered[0] = {
                        "role": "user",
                        "content": f"{system_text}\n\n{filtered[0].get('content', '')}".strip(),
                    }
                else:
                    filtered.insert(0, {"role": "user", "content": system_text})
            messages = filtered

        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        return {"text": text}

    train_ds = ds["train"].map(format_chat, remove_columns=ds["train"].column_names)
    eval_ds = ds.get("eval")
    if eval_ds:
        eval_ds = eval_ds.map(format_chat, remove_columns=ds["eval"].column_names)

    # Çok uzun örnekleri baştan at
    max_len = min(max_length, tokenizer.model_max_length)

    def within_limit(example: Dict[str, str]) -> bool:
        tokenized = tokenizer(example["text"], add_special_tokens=False)
        return len(tokenized["input_ids"]) <= max_len

    train_ds = train_ds.filter(within_limit)
    if eval_ds:
        eval_ds = eval_ds.filter(within_limit)

    return {"train": train_ds, "eval": eval_ds}


def smoke_test(model, tokenizer, train_ds, dtype: str):
    """Minimal forward/backward + short generation to validate the stack."""
    if len(train_ds) == 0:
        raise ValueError("Smoke test requires at least one training example.")
    sample = train_ds[0]["text"]
    inputs = tokenizer(
        sample,
        return_tensors="pt",
        truncation=True,
        max_length=tokenizer.model_max_length,
    )
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    inputs["labels"] = inputs["input_ids"].clone()

    torch.set_grad_enabled(True)
    model.train()

    out = model(**inputs)
    loss = out.loss
    loss.backward()
    model.zero_grad()
    LOGGER.info("Smoke forward/backward ok; loss=%.4f", loss.item())

    FastLanguageModel.for_inference(model)
    prompt_ids = tokenizer(
        sample,
        return_tensors="pt",
        truncation=True,
        max_length=tokenizer.model_max_length,
    ).to(model.device)
    gen = model.generate(**prompt_ids, max_new_tokens=32, do_sample=False)
    decoded = tokenizer.decode(gen[0], skip_special_tokens=True)
    LOGGER.info("Smoke generation sample: %s", decoded[:400])


def train(args):
    cache_dir = Path(args.inductor_cache_dir).resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(cache_dir)
    os.environ["TRITON_CACHE_DIR"] = str(cache_dir / "triton")

    # QLoRA için bf16 varsa onu, yoksa fp16 kullan
    dtype = "bfloat16" if is_bfloat16_supported() else "float16"
    model_name = resolve_model_name(args.family, args.size, args.base_model)
    LOGGER.info("Loading model %s (dtype=%s, QLoRA 4bit)", model_name, dtype)

    # ---- QLoRA ----
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=args.max_seq_len,
        dtype=dtype,
        load_in_4bit=True,   # <--- 4-bit QLoRA
    )

    #  (QLoRA = 4-bit + LoRA)
    target_modules = args.target_modules or [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ]
    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=target_modules,
    )

    data = build_dataset(tokenizer, args.dataset, args.eval_dataset, args.max_seq_len)

    if args.smoke_test:
        smoke_test(model, tokenizer, data["train"], dtype)
        LOGGER.info("Smoke test finished; skipping full training.")
        return

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_ratio=args.warmup_ratio,
        logging_steps=args.logging_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        bf16=dtype == "bfloat16",
        fp16=dtype == "float16",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="adamw_torch",
        weight_decay=args.weight_decay,
        lr_scheduler_type="cosine",
        max_grad_norm=1.0,
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=data["train"],
        eval_dataset=data["eval"],
        dataset_text_field="text",
        max_seq_length=args.max_seq_len,
        packing=False,
        args=training_args,
    )

    trainer.train()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    trainer.model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    LOGGER.info("Saved adapter + tokenizer to %s", args.output_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Finetune chat LMs with Unsloth + QLoRA (4-bit)")
    parser.add_argument(
        "--dataset",
        default="disaster_dataset_kaggle.jsonl",
        help="Training JSONL file (messages or user/assistant)",
    )
    parser.add_argument("--eval-dataset", help="Optional eval JSONL file")
    parser.add_argument(
        "--family",
        default="gemma",
        choices=["gemma", "mistral", "qwen", "llama"],
        help="Model family",
    )
    parser.add_argument("--size", default="7b", help="Model size key (e.g., 7b, 9b, 8b, 70b)")
    parser.add_argument("--base-model", help="Override HF repo to load")
    parser.add_argument("--output-dir", default="./unsloth-outputs", help="Where to save LoRA weights")
    parser.add_argument("--max-seq-len", type=int, default=2048, help="Max sequence length for training")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size for training")
    parser.add_argument("--grad-accum", type=int, default=8, help="Gradient accumulation steps")
    parser.add_argument("--epochs", type=float, default=1, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--warmup-ratio", type=float, default=0.03, help="Warmup ratio for scheduler")
    parser.add_argument("--weight-decay", type=float, default=0.0, help="Weight decay")
    parser.add_argument("--logging-steps", type=int, default=10, help="Logging steps interval")
    parser.add_argument("--save-steps", type=int, default=200, help="Save steps interval")
    parser.add_argument("--lora-r", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora-alpha", type=int, default=32, help="LoRA alpha")
    parser.add_argument("--lora-dropout", type=float, default=0.0, help="LoRA dropout")
    parser.add_argument(
        "--target-modules",
        nargs="+",
        default=None,
        help="Optional target module override for LoRA",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run a single forward/backward + generation and exit",
    )
    parser.add_argument(
        "--inductor-cache-dir",
        default=r"C:\torchinductor_cache",
        help="Path for TorchInductor cache (compile disabled by default)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cli_args = parse_args()
    train(cli_args)
