"""Checks whether the reward model judges meaning, or only exploits shortcuts in its pairs.

Training pairs always use the gold rule as "chosen", so high pair accuracy could just mean
"recognises gold-shaped text". These tests remove that cue. PPO would exploit a shortcut.
"""

import argparse
import json
import random
import re
import time
from collections import defaultdict
from dataclasses import replace

import torch

from payrules.llm import decode_response, load_lora_model, load_tokenizer, read_jsonl, sample_responses
from payrules.nodes import format_rule
from payrules.pairs import score_candidate, semantic_negative
from payrules.parser import parse
from payrules.reward import RewardModel


def auc(positive: list[float], negative: list[float]) -> float | None:
    """Chance a random positive outscores a random negative."""
    if not positive or not negative:
        return None
    wins = sum((p > n) + 0.5 * (p == n) for p in positive for n in negative)
    return wins / (len(positive) * len(negative))


def neutral(text: str) -> str:
    return re.sub(r'reason="[^"]*"', 'reason="r"', text)


def rate(values: list[bool]) -> float:
    return sum(values) / len(values)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reward-model", default="checkpoints/reward")
    parser.add_argument("--sft", default="checkpoints/sft")
    parser.add_argument("--rows", default="data/sft_rl_val.jsonl")
    parser.add_argument("--count", type=int, default=150)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--seed", type=int, default=5)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)
    tokenizer = load_tokenizer()
    rows = read_jsonl(args.rows)[: args.count]
    prompts = [r["prompt"] for r in rows]
    golds = [r["rule"] for r in rows]

    policy = load_lora_model(args.sft)
    samples_prompts, samples_text, samples_match, samples_valid = [], [], [], []
    generation_s = scoring_s = 0.0
    for _ in range(args.rounds):
        started = time.time()
        responses = sample_responses(policy, tokenizer, prompts)
        generation_s += time.time() - started
        started = time.time()
        for row, response in zip(rows, responses):
            text = decode_response(tokenizer, response)
            score = score_candidate(text, parse(row["rule"]))
            samples_prompts.append(row["prompt"])
            samples_text.append(text)
            samples_match.append(score.semantic_match)
            samples_valid.append(score.type_checks)
        scoring_s += time.time() - started
    total = len(samples_text)
    print(f"timing: generation {generation_s / total:.2f}s/sample, verifier {scoring_s / total:.3f}s/sample")
    print(f"samples: {total}, correct {sum(samples_match)}, valid {sum(samples_valid)}")
    del policy

    reward = RewardModel.load(args.reward_model, tokenizer).eval()

    def scores(p, c):
        return reward.score(p, c, batch_size=8).tolist()

    results = {}

    # 1. Real policy outputs: the distribution PPO will actually face.
    sample_scores = scores(samples_prompts, samples_text)
    good = [s for s, m in zip(sample_scores, samples_match) if m]
    bad = [s for s, m in zip(sample_scores, samples_match) if not m]
    bad_valid = [s for s, m, v in zip(sample_scores, samples_match, samples_valid) if v and not m]
    results["auc_correct_vs_all_wrong_samples"] = auc(good, bad)
    results["auc_correct_vs_valid_but_wrong_samples"] = auc(good, bad_valid)
    results["auc_same_after_neutral_reason"] = auc(
        [s for s, m in zip(scores(samples_prompts, [neutral(t) for t in samples_text]), samples_match) if m],
        [s for s, m in zip(scores(samples_prompts, [neutral(t) for t in samples_text]), samples_match) if not m],
    )

    # Bradley-Terry training only fixes score differences within one request, so the AUCs above,
    # which compare scores across requests, understate the signal PPO gets: its value head
    # absorbs each request's offset. Within one request is the comparison PPO relies on.
    gold_by_prompt = dict(zip(prompts, golds))
    wrong_prompts = [p for p, m in zip(samples_prompts, samples_match) if not m]
    if wrong_prompts:
        gold_side = scores(wrong_prompts, [gold_by_prompt[p] for p in wrong_prompts])
        results["gold_beats_wrong_sample_same_request"] = rate([g > b for g, b in zip(gold_side, bad)])
    by_prompt: dict[str, tuple[list[float], list[float]]] = defaultdict(lambda: ([], []))
    for prompt, score, match in zip(samples_prompts, sample_scores, samples_match):
        by_prompt[prompt][0 if match else 1].append(score)
    within = [(r > w) + 0.5 * (r == w) for right, wrong in by_prompt.values() for r in right for w in wrong]
    if within:
        results["correct_beats_wrong_sample_same_request"] = sum(within) / len(within)
        results["requests_with_correct_and_wrong_samples"] = sum(1 for r, w in by_prompt.values() if r and w)

    # 2. Gold against a one-edit mutant, with the reason text carrying no information.
    mutant_prompts, gold_neutral, mutant_neutral = [], [], []
    for row in rows:
        gold = parse(row["rule"])
        mutant = semantic_negative(gold, rng)
        if mutant is not None:
            mutant_prompts.append(row["prompt"])
            gold_neutral.append(neutral(format_rule(gold)))
            mutant_neutral.append(neutral(format_rule(replace(mutant))))
    gold_scores = scores(mutant_prompts, gold_neutral)
    mutant_scores = scores(mutant_prompts, mutant_neutral)
    results["gold_beats_mutant_neutral_reason"] = rate([g > m for g, m in zip(gold_scores, mutant_scores)])

    # 3. A valid gold rule for the wrong request. Only reading the request can catch this.
    shuffled = golds[1:] + golds[:1]
    right = scores(prompts, golds)
    wrong = scores(prompts, shuffled)
    results["right_rule_beats_rule_for_other_request"] = rate([r > w for r, w in zip(right, wrong)])

    # 4. Does it just favour gold over an equally correct rewrite from the policy?
    correct_prompts = [p for p, m in zip(samples_prompts, samples_match) if m]
    correct_texts = [t for t, m in zip(samples_text, samples_match) if m]
    if correct_prompts:
        results["correct_sample_mean_score"] = sum(scores(correct_prompts, correct_texts)) / len(correct_prompts)
        results["gold_mean_score_same_prompts"] = (
            sum(scores(correct_prompts, [gold_by_prompt[p] for p in correct_prompts])) / len(correct_prompts)
        )
    if bad:
        results["wrong_sample_mean_score"] = sum(bad) / len(bad)

    print(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
