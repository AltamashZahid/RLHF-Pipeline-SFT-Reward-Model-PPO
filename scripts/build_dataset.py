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
    rng = random.Random(args.seed)
    seen: set[str] = set()
    splits = {
        "train": generate(args.train, rng, False, seen),
        "val": generate(args.val, rng, False, seen),
        "test_new_phrasing": generate(args.test, rng, True, seen),
    }
    # A separate generator, so adding RL data cannot change the splits above.
    rl_rng = random.Random(args.seed + 1)
    splits["rl_train"] = generate(args.rl_train, rl_rng, "rl", seen)
    splits["rl_val"] = generate(args.rl_val, rl_rng, "rl", seen)

    for name, examples in splits.items():
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
