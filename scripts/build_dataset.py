"""Writes the SFT examples and reward-model pairs to data/. Same seed, same files."""

import argparse
import json
import random
from collections import Counter
from pathlib import Path

from payrules.nlgen import Example, make_example
from payrules.nodes import format_rule
from payrules.pairs import make_pair


def generate(count: int, rng: random.Random, heldout: bool | str, seen: set[str]) -> list[Example]:
    """Rules are unique across all splits, so no evaluation rule ever appears in training."""
    examples: list[Example] = []
    attempts = 0
    while len(examples) < count:
        attempts += 1
        if attempts > count * 100:
            raise RuntimeError(f"could not find {count} unique rules")
        example = make_example(rng, heldout=heldout)
        key = format_rule(example.rule)
        if key in seen:
            continue
        seen.add(key)
        examples.append(example)
    return examples


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("data"))
    parser.add_argument("--train", type=int, default=5000)
    parser.add_argument("--val", type=int, default=500)
    parser.add_argument("--test", type=int, default=500)
    parser.add_argument("--rl-train", type=int, default=3000)
    parser.add_argument("--rl-val", type=int, default=300)
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    def rng_for(*name: str) -> random.Random:
        return random.Random(":".join((str(args.seed), *name)))

    # Every split has its own generator and the held-out splits are drawn first, so editing
    # the training phrasings can never change what the evaluation splits contain.
    plan = (
        ("test_new_phrasing", args.test, True),
        ("rl_train", args.rl_train, "rl"),
        ("rl_val", args.rl_val, "rl"),
        ("train", args.train, False),
        ("val", args.val, False),
        # Unseen rules in the RL phrasing. rl_val picks PPO checkpoints, so it overstates PPO;
        # this set is scored once, next to test_new_phrasing. Drawn last, so nothing above changes.
        ("test_rl_phrasing", args.test, "rl"),
    )
    seen: set[str] = set()
    splits = {name: generate(count, rng_for(name), style, seen) for name, count, style in plan}

    for name, examples in splits.items():
        rng = rng_for("pairs", name)
        write_jsonl(args.out / f"sft_{name}.jsonl", [
            {
                "id": f"{name}-{i}",
                "prompt": e.prompt,
                "rule": format_rule(e.rule),
                "clauses": e.clauses,
                "shape": e.shape,
            }
            for i, e in enumerate(examples)
        ])
        pairs = []
        for i, e in enumerate(examples):
            pair = None
            while pair is None:
                pair = make_pair(e.rule, rng)
            chosen, rejected, kind = pair
            pairs.append({
                "id": f"{name}-{i}",
                "prompt": e.prompt,
                "chosen": chosen,
                "rejected": rejected,
                "kind": kind,
            })
        write_jsonl(args.out / f"pairs_{name}.jsonl", pairs)
        print(name, len(examples), "examples;", dict(Counter(p["kind"] for p in pairs)),
              dict(sorted(Counter(e.clauses for e in examples).items())))


if __name__ == "__main__":
    main()
