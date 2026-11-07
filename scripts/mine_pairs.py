"""Samples the SFT model's own mistakes and turns them into reward-model pairs.

Hand-made corruptions alone do not resemble what the policy will produce during PPO, so the
reward model also has to rank the policy's real outputs. The verifier supplies every label.
"""

import argparse
import json
from pathlib import Path

from payrules.llm import decode_response, load_lora_model, load_tokenizer, read_jsonl, sample_responses
from payrules.pairs import score_candidate
from payrules.parser import parse


def mine(model, tokenizer, rows, temperature, rounds, split):
    pairs = []
    correct_samples = total_samples = 0
    for _ in range(rounds):
        responses = sample_responses(
            model, tokenizer, [r["prompt"] for r in rows], temperature=temperature
        )
        for row, response in zip(rows, responses):
            text = decode_response(tokenizer, response)
            total_samples += 1
            if score_candidate(text, parse(row["rule"])).semantic_match:
                correct_samples += 1
                continue
            pairs.append({
                "id": f"{split}-mined-{len(pairs)}",
                "prompt": row["prompt"],
                "chosen": row["rule"],
                "rejected": text,
                "kind": "mined",
            })
    print(f"{split}: {correct_samples}/{total_samples} samples correct, {len(pairs)} pairs", flush=True)
    return pairs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/sft"))
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--rounds", type=int, default=1)
    args = parser.parse_args()

    tokenizer = load_tokenizer()
    model = load_lora_model(args.checkpoint)
    for split in ("rl_train", "rl_val"):
        rows = read_jsonl(args.data / f"sft_{split}.jsonl")
        pairs = mine(model, tokenizer, rows, args.temperature, args.rounds, split)
        with (args.data / f"pairs_mined_{split}.jsonl").open("w", encoding="utf-8") as handle:
            for pair in pairs:
                handle.write(json.dumps(pair, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
