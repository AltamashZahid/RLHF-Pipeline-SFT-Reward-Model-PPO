"""Samples a policy's own outputs and turns its mistakes into reward-model pairs.

Hand-made corruptions alone do not resemble what the policy will produce during PPO, so the
reward model also has to rank the policy's real outputs. The verifier supplies every label.

By default each wrong sample is paired against the gold rule. With --sample-pairs, a correct
sample is also paired against each distinct wrong sample for the same request. PPO only ever
asks the reward model to compare the policy's own samples, and gold-vs-sample pairs can be won
by noticing which side looks like reference text rather than which side is right.
"""

import argparse
import json
import random
from pathlib import Path

import torch

from payrules.llm import decode_response, load_lora_model, load_tokenizer, read_jsonl, sample_responses
from payrules.pairs import score_candidate
from payrules.parser import parse


def mine(model, tokenizer, rows, temperature, rounds, split, tag, sample_pairs, rng):
    pairs = []
    correct_samples = total_samples = 0
    outcomes: list[dict[str, bool]] = [{} for _ in rows]
    for _ in range(rounds):
        responses = sample_responses(
            model, tokenizer, [r["prompt"] for r in rows], temperature=temperature
        )
        for index, (row, response) in enumerate(zip(rows, responses)):
            text = decode_response(tokenizer, response)
            total_samples += 1
            match = score_candidate(text, parse(row["rule"])).semantic_match
            outcomes[index][text] = match
            if match:
                correct_samples += 1
                continue
            pairs.append({
                "id": f"{split}-{tag}-{len(pairs)}",
                "prompt": row["prompt"],
                "chosen": row["rule"],
                "rejected": text,
                "kind": tag,
            })
    gold_pairs = len(pairs)
    if sample_pairs:
        for row, seen in zip(rows, outcomes):
            right = [text for text, match in seen.items() if match]
            wrong = [text for text, match in seen.items() if not match]
            for text in wrong if right else []:
                pairs.append({
                    "id": f"{split}-{tag}-{len(pairs)}",
                    "prompt": row["prompt"],
                    "chosen": rng.choice(right),
                    "rejected": text,
                    "kind": f"{tag}_sampled",
                })
    print(f"{split}: {correct_samples}/{total_samples} samples correct, {gold_pairs} gold-vs-sample "
          f"pairs, {len(pairs) - gold_pairs} sample-vs-sample pairs", flush=True)
    return pairs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/sft"))
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--tag", default="mined", help="Writes pairs_{tag}_{split}.jsonl")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--rounds", type=int, default=1, help="Samples per request")
    parser.add_argument("--sample-pairs", action="store_true",
                        help="Also pair a correct sample against each wrong one")
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    tokenizer = load_tokenizer()
    model = load_lora_model(args.checkpoint)
    for split in ("rl_train", "rl_val"):
        rows = read_jsonl(args.data / f"sft_{split}.jsonl")
        pairs = mine(model, tokenizer, rows, args.temperature, args.rounds, split, args.tag,
                     args.sample_pairs, rng)
        with (args.data / f"pairs_{args.tag}_{split}.jsonl").open("w", encoding="utf-8") as handle:
            for pair in pairs:
                handle.write(json.dumps(pair, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
