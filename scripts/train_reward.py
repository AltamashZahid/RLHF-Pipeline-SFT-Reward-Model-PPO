"""Trains the reward model on verifier-ranked pairs with a Bradley-Terry loss."""

import argparse
import json
import random
import time
from collections import defaultdict
from pathlib import Path

import torch
from transformers import get_cosine_schedule_with_warmup

from payrules.llm import load_lora_model, load_tokenizer, read_jsonl
from payrules.reward import RewardModel, bradley_terry_loss


@torch.inference_mode()
def pair_accuracy(model, pairs, batch_size=16) -> dict:
    wins: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for start in range(0, len(pairs), batch_size):
        batch = pairs[start : start + batch_size]
        prompts = [p["prompt"] for p in batch]
        chosen = model.score(prompts, [p["chosen"] for p in batch], batch_size)
        rejected = model.score(prompts, [p["rejected"] for p in batch], batch_size)
        for pair, won in zip(batch, (chosen > rejected).tolist()):
            wins[pair["kind"]][0] += won
            wins[pair["kind"]][1] += 1
    total_won = sum(v[0] for v in wins.values())
    total = sum(v[1] for v in wins.values())
    return {"overall": total_won / total, "count": total, **{k: v[0] / v[1] for k, v in sorted(wins.items())}}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sft", type=Path, default=Path("checkpoints/sft"))
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--out", type=Path, default=Path("checkpoints/reward"))
    parser.add_argument("--log", type=Path, default=Path("runs/reward.jsonl"))
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--synthetic-pairs", type=int, default=1500)
    parser.add_argument("--mined", nargs="+", default=["mined"],
                        help="Tags of mined pair files (pairs_{tag}_rl_train.jsonl and _rl_val.jsonl)")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--head-lr", type=float, default=1e-3)
    parser.add_argument("--max-steps", type=int, default=0, help="Stop early; for smoke tests")
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    tokenizer = load_tokenizer()
    model = RewardModel(load_lora_model(args.sft, trainable=True), tokenizer)

    # Mostly RL phrasing and the policy's real mistakes, with some SFT-phrasing pairs for variety.
    train_pairs = read_jsonl(args.data / "pairs_train.jsonl")[: args.synthetic_pairs]
    train_pairs += read_jsonl(args.data / "pairs_rl_train.jsonl")
    val_pairs = read_jsonl(args.data / "pairs_rl_val.jsonl")
    for tag in args.mined:
        train_pairs += read_jsonl(args.data / f"pairs_{tag}_rl_train.jsonl")
        val_pairs += read_jsonl(args.data / f"pairs_{tag}_rl_val.jsonl")
    print(f"{len(train_pairs)} train pairs, {len(val_pairs)} val pairs", flush=True)

    backbone_params = [p for p in model.backbone.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW([
        {"params": backbone_params, "lr": args.lr},
        {"params": model.head.parameters(), "lr": args.head_lr},
    ], weight_decay=0.0)
    steps_per_epoch = len(train_pairs) // args.batch_size
    total_steps = args.max_steps or steps_per_epoch * args.epochs
    scheduler = get_cosine_schedule_with_warmup(optimizer, min(50, total_steps // 10), total_steps)
    scaler = torch.amp.GradScaler("cuda")
    params = backbone_params + list(model.head.parameters())

    best = -1.0
    step = 0
    started = time.time()
    torch.cuda.reset_peak_memory_stats()
    args.log.parent.mkdir(parents=True, exist_ok=True)
    with args.log.open("w", encoding="utf-8") as log:
        for epoch in range(args.epochs):
            random.shuffle(train_pairs)
            model.train()
            for start in range(0, len(train_pairs) - args.batch_size + 1, args.batch_size):
                batch = train_pairs[start : start + args.batch_size]
                prompts = [p["prompt"] for p in batch]
                chosen = model(prompts, [p["chosen"] for p in batch])
                rejected = model(prompts, [p["rejected"] for p in batch])
                loss = bradley_terry_loss(chosen, rejected)
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
                    print(f"step {step}/{total_steps} loss {loss.item():.4f} peak_vram {peak:.0f} MiB "
                          f"elapsed {time.time() - started:.0f}s", flush=True)
                if step == args.max_steps:
                    break

            metrics = pair_accuracy(model, val_pairs)
            record = {"epoch": epoch + 1, "step": step, "elapsed_s": round(time.time() - started), **metrics}
            print(json.dumps(record), flush=True)
            log.write(json.dumps(record) + "\n")
            log.flush()
            if metrics["overall"] > best:
                best = metrics["overall"]
                model.save(args.out)
            model.train()
            if step == args.max_steps:
                break

    print(f"best val pair accuracy {best:.3f}; peak VRAM {torch.cuda.max_memory_reserved() / 2**20:.0f} MiB")


if __name__ == "__main__":
    main()
