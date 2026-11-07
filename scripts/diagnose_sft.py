"""Scores a checkpoint on the RL phrasing, which SFT never trained on, to measure the gap."""

import argparse
import json

import torch

from payrules.llm import evaluate, load_lora_model, load_tokenizer, read_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="checkpoints/sft")
    parser.add_argument("--rows", default="data/sft_rl_val.jsonl")
    parser.add_argument("--show", type=int, default=12)
    args = parser.parse_args()

    tokenizer = load_tokenizer()
    model = load_lora_model(args.checkpoint)
    torch.cuda.reset_peak_memory_stats()
    rows = read_jsonl(args.rows)
    metrics = evaluate(model, tokenizer, rows, batch_size=8)
    predictions = metrics.pop("predictions")
    print(json.dumps(metrics))
    print(f"peak VRAM {torch.cuda.max_memory_reserved() / 2**20:.0f} MiB")
    for row, prediction in list(zip(rows, predictions))[: args.show]:
        print("\nREQUEST", row["prompt"])
        print("GOLD   ", row["rule"])
        print("MODEL  ", prediction)


if __name__ == "__main__":
    main()
