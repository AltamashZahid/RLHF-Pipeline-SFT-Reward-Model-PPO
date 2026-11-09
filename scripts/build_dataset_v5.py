"""Adds the fields published fraud rules use: VPN, high-risk country, prepaid, card/IP mismatch,
disposable email and cards per IP.

RL and test wording still comes from a pool training never reads. Every rule in the older test sets
and the human rule sets is kept out of every split, so those sets stay fair for this model too.
Run with PYTHONPATH=scripts.
"""

import argparse
import random
from pathlib import Path

from build_dataset import write_jsonl
from build_dataset_v4 import SFT_POOLS, as_rows, rule_key, write_split
from payrules.llm import read_jsonl
from payrules.natural import WordingPool, make_natural_example
from payrules.nlgen import make_example
from payrules.parser import parse_checked

_HELD_OUT = ("eval/natural_test.jsonl", "eval/natural_val.jsonl", "eval/online_rules.jsonl", "eval/human_rules.jsonl",
             "data/v4/sft_test_rl_phrasing.jsonl")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("data/v5"))
    parser.add_argument("--train", type=int, default=16000, help="Total rows, including the free-form ones")
    parser.add_argument("--val", type=int, default=500)
    parser.add_argument("--rl-train", type=int, default=4000)
    parser.add_argument("--rl-val", type=int, default=400)
    parser.add_argument("--test-rl", type=int, default=500)
    parser.add_argument("--freeform", type=Path, default=Path("src/payrules/phrasings/freeform_train.jsonl"))
    parser.add_argument("--freeform-repeats", type=int, default=2)
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()

    pools = [WordingPool.load(name, "sft_fields") for name in SFT_POOLS]
    rl_pool = WordingPool.load("rl_natural", "rl_fields")

    def sft_example(rng):
        style = rng.random()
        if style < 0.2:
            return make_example(rng, heldout=style >= 0.12)
        return make_natural_example(rng, rng.choice(pools))

    def rl_example(rng):
        return make_natural_example(rng, rl_pool) if rng.random() < 0.6 else make_example(rng, heldout="rl")

    def rng_for(*name: str) -> random.Random:
        return random.Random(":".join((str(args.seed), "v5", *name)))

    seen = {rule_key(parse_checked(r["rule"])) for path in _HELD_OUT for r in read_jsonl(path)}
    print(f"{len(seen)} held-out rules kept out of every split")

    def generate(count, rng, maker):
        examples, attempts = [], 0
        while len(examples) < count:
            attempts += 1
            if attempts > count * 100:
                raise RuntimeError(f"could not find {count} unique rules")
            example = maker(rng)
            key = rule_key(example.rule)
            if key not in seen:
                seen.add(key)
                examples.append(example)
        return examples

    args.out.mkdir(parents=True, exist_ok=True)
    # Evaluation splits are drawn first, so later edits to training wording cannot change them.
    for name, count in (("test_rl_phrasing", args.test_rl), ("rl_val", args.rl_val), ("rl_train", args.rl_train)):
        write_split(args.out, name, as_rows(name, generate(count, rng_for(name), rl_example)), rng_for("pairs", name))
    write_split(args.out, "val", as_rows("val", generate(args.val, rng_for("val"), sft_example)), rng_for("pairs", "val"))

    freeform = read_jsonl(args.freeform)
    clash = [r["id"] for r in freeform if rule_key(parse_checked(r["rule"])) in seen]
    # Generated rules can land on a free-form rule. Training rows may repeat them, but held-out ones may not.
    held_out = {rule_key(parse_checked(r["rule"])) for path in _HELD_OUT for r in read_jsonl(path)}
    assert not [r["id"] for r in freeform if rule_key(parse_checked(r["rule"])) in held_out]
    print(f"{len(clash)} free-form rules also drawn in generated evaluation splits are dropped")
    freeform = [r for r in freeform if r["id"] not in set(clash)]
    seen |= {rule_key(parse_checked(r["rule"])) for r in freeform}

    repeated = [{**row, "id": f"{row['id']}-r{i}"} for row in freeform for i in range(args.freeform_repeats)]
    generated = as_rows("train", generate(args.train - len(repeated), rng_for("train"), sft_example))
    combined = [*generated, *repeated]
    # The reward model takes the first rows of this file, so free-form rows must be spread through it.
    rng_for("shuffle").shuffle(combined)
    keep = ("id", "prompt", "rule", "clauses", "shape")
    write_split(args.out, "train", [{k: row[k] for k in keep} for row in combined], rng_for("pairs", "train"))

    # The model selection set: natural wording, plus unseen wording that covers the new fields.
    write_jsonl(args.out / "select.jsonl", [*read_jsonl("eval/natural_val.jsonl"), *read_jsonl(args.out / "sft_rl_val.jsonl")])


if __name__ == "__main__":
    main()
