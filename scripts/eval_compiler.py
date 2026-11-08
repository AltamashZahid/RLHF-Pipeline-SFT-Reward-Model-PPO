"""Measures how candidates, the reward model and agreement change what the compiler outputs.

Candidates are saved, so comparing strategies never needs the model again. Strategies and the review
threshold must be chosen on validation sets; --allow-test exists so a final test is a deliberate act.
"""

import argparse
import json
from pathlib import Path

import torch

from payrules.compiler import STRATEGIES, compile_one
from payrules.llm import decode_response, generate_rules, load_lora_model, load_tokenizer, read_jsonl, sample_responses
from payrules.parser import parse
from payrules.pairs import score_candidate
from payrules.reward import RewardModel


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", required=True)
    parser.add_argument("--reward-model", required=True)
    parser.add_argument("--rows", required=True)
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--allow-test", action="store_true")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    if "test" in Path(args.rows).name and not args.allow_test:
        parser.error("this is a test set: choose strategies on validation data, then pass --allow-test once")

    torch.manual_seed(args.seed)
    rows = read_jsonl(args.rows)
    rows = rows[: args.limit] if args.limit else rows
    prompts = [r["prompt"] for r in rows]
    tokenizer = load_tokenizer()
    policy = load_lora_model(args.policy).eval()
    greedy = generate_rules(policy, tokenizer, prompts)
    samples: list[list[str]] = [[] for _ in rows]
    for round_number in range(args.samples):
        responses = sample_responses(policy, tokenizer, prompts, temperature=args.temperature)
        for bucket, response in zip(samples, responses):
            bucket.append(decode_response(tokenizer, response))
        print(f"sampled round {round_number + 1}/{args.samples}", flush=True)
    del policy
    torch.cuda.empty_cache()
    reward = RewardModel.load(args.reward_model, tokenizer).eval()

    def score(requests, texts):
        return reward.score(requests, texts, batch_size=8).tolist()

    results: dict = {}
    for strategy in STRATEGIES:
        correct = answered = 0
        records = []
        for row, first, drawn in zip(rows, greedy, samples):
            compiled = compile_one(row["prompt"], first, drawn, score if strategy != "greedy" else None, strategy)
            right = compiled.answer is not None and score_candidate(compiled.answer.text, parse(row["rule"])).semantic_match
            correct += right
            answered += compiled.answer is not None
            records.append((compiled.agreement, bool(right), compiled.answer is not None))
        results[strategy] = {"accuracy": correct / len(rows), "answered": answered / len(rows)}
        curve = {}
        for threshold in (0.5, 0.6, 0.75, 0.9, 1.0):
            kept = [right for agreement, right, has in records if has and agreement >= threshold]
            curve[threshold] = {
                "coverage": len(kept) / len(rows),
                "accuracy_when_answered": sum(kept) / max(len(kept), 1),
            }
        results[strategy]["by_agreement"] = curve
        print(strategy, json.dumps(results[strategy]), flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"results": results, "greedy": greedy, "samples": samples}, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
