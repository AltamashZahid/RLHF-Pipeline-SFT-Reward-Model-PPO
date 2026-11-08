"""Shared model code: prompt format, loading, batched generation and evaluation."""

import json
import os
from collections import defaultdict
from pathlib import Path

import torch
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from .pairs import score_candidate
from .parser import parse

BASE_MODEL = os.environ.get("PAYRULES_BASE_MODEL", "HuggingFaceTB/SmolLM2-135M-instruct")
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
MAX_NEW_TOKENS = 110


def format_prompt(request: str) -> str:
    return f"### Request:\n{request}\n### Rule:\n"


def read_jsonl(path: Path | str) -> list[dict]:
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def load_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # Generation continues from the end of each prompt.
    return tokenizer


def load_base(device: str = "cuda", name: str = BASE_MODEL):
    model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.float16)
    return model.to(device)


def new_lora_model(device: str = "cuda", rank: int = 32):
    config = LoraConfig(
        r=rank,
        lora_alpha=rank * 2,
        target_modules=LORA_TARGETS,
        lora_dropout=0.0,
        task_type="CAUSAL_LM",
    )
    return get_peft_model(load_base(device), config)


def load_lora_model(path: Path | str, device: str = "cuda", trainable: bool = False):
    """Loads onto the base model the checkpoint was trained on, so sizes can be mixed."""
    config = json.loads((Path(path) / "adapter_config.json").read_text(encoding="utf-8"))
    base = load_base(device, config["base_model_name_or_path"])
    return PeftModel.from_pretrained(base, str(path), is_trainable=trainable)


@torch.inference_mode()
def generate_rules(
    model,
    tokenizer,
    requests: list[str],
    batch_size: int = 64,
    max_new_tokens: int = MAX_NEW_TOKENS,
) -> list[str]:
    """Greedy generation. Sorted by length so batches carry little padding."""
    model.eval()
    order = sorted(range(len(requests)), key=lambda i: len(requests[i]))
    outputs: list[str] = [""] * len(requests)
    for start in range(0, len(order), batch_size):
        indices = order[start : start + batch_size]
        batch = tokenizer(
            [format_prompt(requests[i]) for i in indices], return_tensors="pt", padding=True
        ).to(model.device)
        with torch.autocast("cuda", dtype=torch.float16):
            generated = model.generate(
                **batch,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
                top_k=None,
                pad_token_id=tokenizer.pad_token_id,
            )
        new_tokens = generated[:, batch["input_ids"].shape[1] :]
        for index, text in zip(indices, tokenizer.batch_decode(new_tokens, skip_special_tokens=True)):
            outputs[index] = text.split("\n")[0].strip()
    return outputs


@torch.inference_mode()
def sample_responses(
    model,
    tokenizer,
    requests: list[str],
    batch_size: int = 64,
    temperature: float = 1.0,
    max_new_tokens: int = MAX_NEW_TOKENS,
) -> list[list[int]]:
    """Sample from the policy's own distribution. Each result keeps its EOS if it reached one."""
    model.eval()
    eos, pad = tokenizer.eos_token_id, tokenizer.pad_token_id
    order = sorted(range(len(requests)), key=lambda i: len(requests[i]))
    outputs: list[list[int]] = [[] for _ in requests]
    for start in range(0, len(order), batch_size):
        indices = order[start : start + batch_size]
        batch = tokenizer(
            [format_prompt(requests[i]) for i in indices], return_tensors="pt", padding=True
        ).to(model.device)
        with torch.autocast("cuda", dtype=torch.float16):
            generated = model.generate(
                **batch,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_k=0,
                top_p=1.0,
                repetition_penalty=1.0,
                eos_token_id=eos,
                pad_token_id=pad,
            )
        for index, row in zip(indices, generated[:, batch["input_ids"].shape[1] :].tolist()):
            outputs[index] = row[: row.index(eos) + 1] if eos in row else row
    return outputs


def decode_response(tokenizer, response: list[int]) -> str:
    return tokenizer.decode(response, skip_special_tokens=True).split("\n")[0].strip()


def evaluate(model, tokenizer, rows: list[dict], batch_size: int = 64) -> dict:
    """Parse, type-check and semantic rates. `rows` need "prompt", "rule" and "clauses"."""
    predictions = generate_rules(model, tokenizer, [r["prompt"] for r in rows], batch_size)
    totals = {"parses": 0, "type_checks": 0, "semantic_match": 0}
    by_clauses: dict[int, list[int]] = defaultdict(lambda: [0, 0])
    for row, text in zip(rows, predictions):
        score = score_candidate(text, parse(row["rule"]))
        totals["parses"] += score.parses
        totals["type_checks"] += score.type_checks
        totals["semantic_match"] += score.semantic_match
        by_clauses[row["clauses"]][0] += score.semantic_match
        by_clauses[row["clauses"]][1] += 1
    return {
        "count": len(rows),
        **{name: value / len(rows) for name, value in totals.items()},
        "semantic_by_clauses": {k: v[0] / v[1] for k, v in sorted(by_clauses.items())},
        "predictions": predictions,
    }
