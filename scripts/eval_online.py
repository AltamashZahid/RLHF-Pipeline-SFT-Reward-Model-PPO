"""Scores models on rules taken from public payment documentation, written by people.

Correct answers are fixed in eval/online_rules.jsonl before scoring. Where a rule honestly reads two
ways, the file lists the other reading in "alt", and matching either counts. Run this once.
"""

import argparse
import hashlib
import json
from pathlib import Path

import torch

from payrules.compiler import compile_one
from payrules.llm import decode_response, generate_rules, load_lora_model, load_tokenizer, read_jsonl, sample_responses
from payrules.pairs import score_candidate
from payrules.parser import parse, parse_checked
from payrules.reward import RewardModel


def is_right(text: str, row: dict) -> bool:
    return any(score_candidate(text, parse(gold)).semantic_match for gold in [row["rule"], *row["alt"]])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=Path, default=Path("eval/online_rules.jsonl"))
    parser.add_argument("--models", nargs="+", default=["old_sft=checkpoints/sft", "v4_sft=checkpoints/v4/sft360",
                                                       "v5_sft=checkpoints/v5/sft360"])
    parser.add_argument("--policy", default="checkpoints/v5_full/ppo")
    parser.add_argument("--reward-model", default="checkpoints/v5_full/reward")
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--strategy", default="majority_then_reward")
    parser.add_argument("--minimum-agreement", type=float, default=0.75)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--out", type=Path, default=Path("runs/online_rules.json"))
    args = parser.parse_args()

    rows = read_jsonl(args.rows)
    for row in rows:
        for gold in [row["rule"], *row["alt"]]:
            parse_checked(gold)
    print("rows sha256", hashlib.sha256(args.rows.read_bytes()).hexdigest(), flush=True)

    torch.manual_seed(args.seed)
    tokenizer = load_tokenizer()
    prompts = [r["prompt"] for r in rows]
    predictions = {}
    for spec in args.models:
        name, path = spec.split("=", 1)
        model = load_lora_model(path).eval()
        predictions[name] = generate_rules(model, tokenizer, prompts)
        del model
        torch.cuda.empty_cache()

    policy = load_lora_model(args.policy).eval()
    samples: list[list[str]] = [[] for _ in rows]
    for _ in range(args.samples):
        for bucket, response in zip(samples, sample_responses(policy, tokenizer, prompts)):
            bucket.append(decode_response(tokenizer, response))
    greedy = generate_rules(policy, tokenizer, prompts)
    del policy
    torch.cuda.empty_cache()
    reward = RewardModel.load(args.reward_model, tokenizer).eval()

    def score(requests, texts):
        return reward.score(requests, texts, batch_size=8).tolist()

    results = {name: [is_right(text, row) for text, row in zip(texts, rows)] for name, texts in predictions.items()}
    compiled_rows = []
    for row, first, drawn in zip(rows, greedy, samples):
        compiled = compile_one(row["prompt"], first, drawn, score, args.strategy)
        text = compiled.answer.text if compiled.answer else ""
        compiled_rows.append({"id": row["id"], "text": text, "agreement": compiled.agreement,
                              "right": bool(text) and is_right(text, row),
                              "review": compiled.needs_review(args.minimum_agreement)})
    results["compiler"] = [c["right"] for c in compiled_rows]

    for name, right in results.items():
        verbatim = [ok for ok, row in zip(right, rows) if row["kind"] == "verbatim"]
        trimmed = [ok for ok, row in zip(right, rows) if row["kind"] != "verbatim"]
        groups = {}
        for ok, row in zip(right, rows):
            groups.setdefault(row.get("group", "all"), []).append(ok)
        split = ", ".join(f"{group} {sum(v)}/{len(v)}" for group, v in groups.items())
        print(f"{name}: {sum(right)}/{len(right)} (verbatim {sum(verbatim)}/{len(verbatim)}, "
              f"trimmed {sum(trimmed)}/{len(trimmed)}; {split})", flush=True)
    kept = [c for c in compiled_rows if not c["review"]]
    print(f"review: {len(compiled_rows) - len(kept)} held back, of the rest {sum(c['right'] for c in kept)}/{len(kept)} right")
    for row, c in zip(rows, compiled_rows):
        print(f"{'OK ' if c['right'] else 'BAD'} {row['id']} agreement {c['agreement']:.2f}{' REVIEW' if c['review'] else ''}\n"
              f"  Q    {row['prompt']}\n  gold {row['rule']}\n  got  {c['text']}", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"results": results, "predictions": predictions, "compiler": compiled_rows},
                                   indent=1, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
