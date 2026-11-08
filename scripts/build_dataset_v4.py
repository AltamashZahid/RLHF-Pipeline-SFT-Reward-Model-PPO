"""Builds the final data: three natural pools, free-form sentences and the older template styles.

SFT trains on all of those. RL keeps its own wording, which SFT never sees, so PPO has something
to teach and its splits stay identical to the previous round. Rules in the natural test set and in
the natural validation set can never appear in training.
"""

import argparse
import hashlib
import random
from collections import Counter
from pathlib import Path

from build_dataset import write_jsonl
from payrules.llm import read_jsonl
from payrules.natural import WordingPool, make_natural_example
from payrules.nlgen import make_example
from payrules.nodes import format_expr, format_rule
from payrules.pairs import make_pair
from payrules.parser import parse_checked

SFT_POOLS = ("sft_natural", "sft_formal", "sft_ops")
_RL_TAG = "v3"  # Keeps these seeds from the previous round, so scores remain directly comparable.


def rule_key(rule) -> str:
    """Ignores the free-form reason text, so a rule cannot re-enter under a different reason."""
    return f"{format_expr(rule.condition)} => {rule.action}"


def write_split(out: Path, name: str, rows: list[dict], rng: random.Random) -> None:
    write_jsonl(out / f"sft_{name}.jsonl", rows)
    pairs = []
    for row in rows:
        rule = parse_checked(row["rule"])
        pair = None
        while pair is None:
            pair = make_pair(rule, rng)
        chosen, rejected, kind = pair
        pairs.append({"id": row["id"], "prompt": row["prompt"], "chosen": chosen, "rejected": rejected, "kind": kind})
    write_jsonl(out / f"pairs_{name}.jsonl", pairs)
    print(name, len(rows), dict(sorted(Counter(r["clauses"] for r in rows).items())),
          dict(Counter(r["shape"] for r in rows)), flush=True)


def as_rows(name: str, examples) -> list[dict]:
    return [{"id": f"{name}-{i}", "prompt": e.prompt, "rule": format_rule(e.rule), "clauses": e.clauses, "shape": e.shape}
            for i, e in enumerate(examples)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("data/v4"))
    parser.add_argument("--train", type=int, default=16000, help="Total rows, including the free-form ones")
    parser.add_argument("--val", type=int, default=500)
    parser.add_argument("--rl-train", type=int, default=4000)
    parser.add_argument("--rl-val", type=int, default=400)
    parser.add_argument("--test-rl", type=int, default=500)
    parser.add_argument("--freeform", type=Path, default=Path("src/payrules/phrasings/freeform_train.jsonl"))
    parser.add_argument("--freeform-repeats", type=int, default=2)
    parser.add_argument("--natural-val", type=Path, default=Path("eval/natural_val.jsonl"))
    parser.add_argument("--exclude", type=Path, default=Path("eval/natural_test.jsonl"))
    parser.add_argument("--compare-rl-to", type=Path, default=Path("data/v3"),
                        help="Fails if the RL splits differ from this folder; skipped when it does not exist")
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()

    pools = [WordingPool.load(name) for name in SFT_POOLS]
    rl_pool = WordingPool.load("rl_natural")
    assert rl_pool.actor is None, "the wider grammar would change the RL splits"

    def sft_example(rng):
        style = rng.random()
        if style < 0.2:
            return make_example(rng, heldout=style >= 0.12)
        return make_natural_example(rng, rng.choice(pools))

    def rl_example(rng):
        return make_natural_example(rng, rl_pool) if rng.random() < 0.6 else make_example(rng, heldout="rl")

    def rng_for(tag: str, *name: str) -> random.Random:
        return random.Random(":".join((str(args.seed), tag, *name)))

    seen = {rule_key(parse_checked(r["rule"])) for r in read_jsonl(args.exclude)}
    print(f"{len(seen)} rules from the natural test set kept out of every split")

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
        rows = as_rows(name, generate(count, rng_for(_RL_TAG, name), rl_example))
        write_split(args.out, name, rows, rng_for(_RL_TAG, "pairs", name))
        before = args.compare_rl_to / f"sft_{name}.jsonl"
        if before.exists():
            digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
            assert digest(before) == digest(args.out / f"sft_{name}.jsonl"), f"{name} differs from {before}"
            print(f"  identical to {before}")

    natural_val = read_jsonl(args.natural_val)
    freeform = read_jsonl(args.freeform)
    seen |= {rule_key(parse_checked(r["rule"])) for r in natural_val}
    clash = [r["id"] for r in freeform if rule_key(parse_checked(r["rule"])) in seen]
    assert not clash, f"free-form training rows reuse held-out rules: {clash[:5]}"
    seen |= {rule_key(parse_checked(r["rule"])) for r in freeform}

    write_split(args.out, "val", as_rows("val", generate(args.val, rng_for("v4", "val"), sft_example)),
                rng_for("v4", "pairs", "val"))

    repeated = [{**row, "id": f"{row['id']}-r{i}"} for row in freeform for i in range(args.freeform_repeats)]
    generated = as_rows("train", generate(args.train - len(repeated), rng_for("v4", "train"), sft_example))
    combined = [*generated, *repeated]
    # The reward model takes the first rows of this file, so free-form rows must be spread through it.
    rng_for("v4", "shuffle").shuffle(combined)
    keep = ("id", "prompt", "rule", "clauses", "shape")
    write_split(args.out, "train", [{k: row[k] for k in keep} for row in combined], rng_for("v4", "pairs", "train"))
    print(f"train has {len(freeform)} free-form sentences, each repeated {args.freeform_repeats} times")


if __name__ == "__main__":
    main()
