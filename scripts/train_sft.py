"""Supervised fine-tuning: LoRA on SmolLM2, English request to DSL rule."""

import argparse
import json
import random
import time
from pathlib import Path

import torch
from transformers import get_cosine_schedule_with_warmup

from payrules.llm import evaluate, format_prompt, load_tokenizer, new_lora_model, read_jsonl


def encode(rows, tokenizer):
    examples = []
    for row in rows:
        prompt = tokenizer(format_prompt(row["prompt"]))["input_ids"]
        rule = tokenizer(row["rule"], add_special_tokens=False)["input_ids"]
        examples.append((prompt, rule + [tokenizer.eos_token_id]))
    return examples


def collate(batch, pad_id, device):
    longest = max(len(p) + len(r) for p, r in batch)
    ids, labels, mask = [], [], []
    for prompt, rule in batch:
        length = len(prompt) + len(rule)
        padding = longest - length
        ids.append(prompt + rule + [pad_id] * padding)
        labels.append([-100] * len(prompt) + rule + [-100] * padding)
        mask.append([1] * length + [0] * padding)
    tensor = lambda rows: torch.tensor(rows, device=device)
    return tensor(ids), tensor(labels), tensor(mask)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--out", type=Path, default=Path("checkpoints/sft"))
    parser.add_argument("--log", type=Path, default=Path("runs/sft.jsonl"))
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--rank", type=int, default=32)
    parser.add_argument("--eval-rows", type=int, default=300)
    parser.add_argument("--select-on", type=Path, default=None,
                        help="Picks the epoch on this file instead of sft_val; use wording the model never trains on")
    parser.add_argument("--max-steps", type=int, default=0, help="Stop early; for smoke tests")
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    args.log.parent.mkdir(parents=True, exist_ok=True)

    tokenizer = load_tokenizer()
    model = new_lora_model(rank=args.rank)
    model.print_trainable_parameters()
    train_rows = read_jsonl(args.data / "sft_train.jsonl")
    val_rows = read_jsonl(args.select_on or args.data / "sft_val.jsonl")[: args.eval_rows]
    examples = encode(train_rows, tokenizer)
    print("longest rule in tokens:", max(len(r) for _, r in examples),
          "| longest full example:", max(len(p) + len(r) for p, r in examples))

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
    steps_per_epoch = len(examples) // args.batch_size
    total_steps = args.max_steps or steps_per_epoch * args.epochs
    scheduler = get_cosine_schedule_with_warmup(optimizer, min(50, total_steps // 10), total_steps)
    scaler = torch.amp.GradScaler("cuda")

    best = -1.0
    step = 0
    nonfinite = 0
    started = time.time()
    torch.cuda.reset_peak_memory_stats()
    with args.log.open("w", encoding="utf-8") as log:
        for epoch in range(args.epochs):
            random.shuffle(examples)
            model.train()
            for start in range(0, len(examples) - args.batch_size + 1, args.batch_size):
                ids, labels, mask = collate(
                    examples[start : start + args.batch_size],
                    tokenizer.pad_token_id,
                    model.device,
                )
                with torch.autocast("cuda", dtype=torch.float16):
                    loss = model(input_ids=ids, attention_mask=mask, labels=labels).loss
                # The scaler skips every step once the forward pass overflows, so the run would
                # otherwise carry on for hours without learning anything.
                nonfinite = 0 if torch.isfinite(loss) else nonfinite + 1
                if nonfinite == 20:
                    raise RuntimeError(f"loss not finite for 20 steps, stopped at step {step}; try a lower --lr")
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                step += 1
                if step % 50 == 0 or step == 1:
                    peak = torch.cuda.max_memory_reserved() / 2**20
                    print(f"step {step}/{total_steps} loss {loss.item():.4f} "
                          f"peak_vram {peak:.0f} MiB elapsed {time.time() - started:.0f}s", flush=True)
                if step == args.max_steps:
                    break

            metrics = evaluate(model, tokenizer, val_rows)
            metrics.pop("predictions")
            record = {"epoch": epoch + 1, "step": step, "elapsed_s": round(time.time() - started), **metrics}
            print(json.dumps(record), flush=True)
            log.write(json.dumps(record) + "\n")
            log.flush()
            if metrics["semantic_match"] > best:
                best = metrics["semantic_match"]
                model.save_pretrained(args.out)
            if step == args.max_steps:
                break

    print(f"best val semantic accuracy {best:.3f}; peak VRAM "
          f"{torch.cuda.max_memory_reserved() / 2**20:.0f} MiB")


if __name__ == "__main__":
    main()
