"""Final comparison of trained models on data no training or tuning step has used.

Run this after all tuning is finished. Rerunning it to pick a better model would turn the
test sets into validation sets and make the reported numbers unreliable.
"""

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import torch

from payrules.engine import compile_expr
from payrules.llm import generate_rules, load_lora_model, load_tokenizer, read_jsonl
from payrules.pairs import score_candidate
from payrules.parser import parse
from payrules.reward import RewardModel
from payrules.txngen import generate_transactions


def outcome(rule, txns):
    predicate = compile_expr(rule.condition)
    return [rule.action if predicate(t) else None for t in txns]


def wrong_blocks(prediction_text, gold, txns):
    """Against the correct rule's behaviour, how often does this rule over-block or under-block?"""
    gold_outcomes = outcome(gold, txns)
    gold_blocks = sum(o == "BLOCK" for o in gold_outcomes)
    try:
        rule = parse(prediction_text)
        predicted = outcome(rule, txns)
    except Exception:
        predicted = [None] * len(txns)  # An unusable rule blocks nothing.
    extra = sum(p == "BLOCK" and g != "BLOCK" for p, g in zip(predicted, gold_outcomes))
    missed = sum(g == "BLOCK" and p != "BLOCK" for p, g in zip(predicted, gold_outcomes))
    return extra, missed, gold_blocks


def bootstrap_interval(a: list[bool], b: list[bool], rounds=5000, seed=0):
    """95% interval for accuracy(b) - accuracy(a), resampling the same prompts for both."""
    rng = random.Random(seed)
    n = len(a)
    differences = []
    for _ in range(rounds):
        sample = [rng.randrange(n) for _ in range(n)]
        differences.append(sum(b[i] - a[i] for i in sample) / n)
    differences.sort()
    return differences[int(0.025 * rounds)], differences[int(0.975 * rounds)]


def score_set(model, tokenizer, rows, txns):
    predictions = generate_rules(model, tokenizer, [r["prompt"] for r in rows])
    matches, parses, valid = [], 0, 0
    extra = missed = gold_blocks = 0
    for row, text in zip(rows, predictions):
        gold = parse(row["rule"])
        score = score_candidate(text, gold)
        matches.append(score.semantic_match)
        parses += score.parses
        valid += score.type_checks
        e, m, g = wrong_blocks(text if score.type_checks else "invalid", gold, txns)
        extra += e
        missed += m
        gold_blocks += g
    total = len(rows) * len(txns)
    return {
        "semantic_accuracy": sum(matches) / len(rows),
        "parse_rate": parses / len(rows),
        "valid_rate": valid / len(rows),
        "wrongly_blocked_per_1000_decisions": 1000 * extra / total,
        "missed_blocks_share_of_correct_blocks": missed / max(gold_blocks, 1),
    }, matches, predictions


def reward_pair_accuracy(reward, pairs, batch_size=8) -> dict:
    """How often the reward model ranks the correct rule above the incorrect one, by pair kind."""
    if not pairs:
        return {"overall": None, "count": 0}
    prompts = [p["prompt"] for p in pairs]
    chosen = reward.score(prompts, [p["chosen"] for p in pairs], batch_size=batch_size).tolist()
    rejected = reward.score(prompts, [p["rejected"] for p in pairs], batch_size=batch_size).tolist()
    wins = defaultdict(lambda: [0, 0])
    for pair, c, r in zip(pairs, chosen, rejected):
        wins[pair["kind"]][0] += c > r
        wins[pair["kind"]][1] += 1
    won = sum(v[0] for v in wins.values())
    total = sum(v[1] for v in wins.values())
    return {"overall": won / total, "count": total, **{k: v[0] / v[1] for k, v in sorted(wins.items())}}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", default=["sft=checkpoints/sft", "ppo=checkpoints/ppo_learned"])
    parser.add_argument("--baseline", default="sft")
    parser.add_argument("--out", type=Path, default=Path("runs/final_eval.json"))
    parser.add_argument("--txns", type=int, default=2000)
    parser.add_argument("--reward-model", default="checkpoints/reward")
    parser.add_argument("--new-phrasing", default="data/sft_test_new_phrasing.jsonl")
    parser.add_argument("--set-name", default="new_phrasing", help="Results name for the --new-phrasing set")
    parser.add_argument("--handwritten", default="eval/handwritten.jsonl")
    parser.add_argument("--pairs", default="data/pairs_test_new_phrasing.jsonl")
    parser.add_argument("--limit", type=int, default=0, help="Use only the first rows; for smoke tests on validation data")
    args = parser.parse_args()

    tokenizer = load_tokenizer()
    txns = generate_transactions(args.txns, seed=999)
    sets = {
        args.set_name: read_jsonl(args.new_phrasing),
        "handwritten": read_jsonl(args.handwritten),
    }
    pairs = read_jsonl(args.pairs)
    if args.limit:
        sets = {name: rows[: args.limit] for name, rows in sets.items()}
        pairs = pairs[: args.limit]

    results: dict = {}
    matches_by_model: dict = {}
    for spec in args.models:
        name, path = spec.split("=", 1)
        model = load_lora_model(path).eval()
        results[name] = {}
        matches_by_model[name] = {}
        for set_name, rows in sets.items():
            metrics, matches, predictions = score_set(model, tokenizer, rows, txns)
            results[name][set_name] = metrics
            matches_by_model[name][set_name] = matches
            results[name][f"{set_name}_predictions"] = predictions
            print(name, set_name, json.dumps(metrics), flush=True)
        del model
        torch.cuda.empty_cache()

    comparisons = {}
    for name in matches_by_model:
        if name == args.baseline:
            continue
        for set_name in sets:
            a = matches_by_model[args.baseline][set_name]
            b = matches_by_model[name][set_name]
            low, high = bootstrap_interval(a, b)
            comparisons[f"{name}_vs_{args.baseline}_{set_name}"] = {
                "gained": sum(y and not x for x, y in zip(a, b)),
                "lost": sum(x and not y for x, y in zip(a, b)),
                "difference_95ci": [low, high],
            }
            print(f"{name} vs {args.baseline} on {set_name}:",
                  json.dumps(comparisons[f"{name}_vs_{args.baseline}_{set_name}"]), flush=True)

    # The baseline's real mistakes on unseen requests are harder than synthetic corruptions.
    real_mistakes = []
    for row, text, matched in zip(
        sets[args.set_name],
        results[args.baseline][f"{args.set_name}_predictions"],
        matches_by_model[args.baseline][args.set_name],
    ):
        if not matched:
            real_mistakes.append({"prompt": row["prompt"], "chosen": row["rule"], "rejected": text, "kind": "real_mistake"})
    reward = RewardModel.load(args.reward_model, tokenizer).eval()
    reward_results = {
        "synthetic_pairs": reward_pair_accuracy(reward, pairs),
        "real_mistake_pairs": reward_pair_accuracy(reward, real_mistakes),
    }
    print("reward", json.dumps(reward_results), flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps({"results": results, "comparisons": comparisons, "reward": reward_results}, indent=1),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
