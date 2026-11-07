import random

from payrules import equivalent, parse, parse_checked
from payrules.nlgen import make_example
from payrules.nodes import format_rule
from payrules.pairs import convolute, make_pair, reorder, score_candidate, semantic_negative


def _rules(n, seed=1, heldout=False):
    rng = random.Random(seed)
    return [make_example(rng, heldout=heldout) for _ in range(n)]


def test_every_generated_rule_is_valid_and_round_trips():
    for example in _rules(3000) + _rules(3000, seed=2, heldout=True):
        text = format_rule(example.rule)
        assert parse_checked(text) == example.rule
        assert example.prompt[0].isupper() and example.prompt.endswith(".")


def test_same_seed_gives_same_examples():
    assert _rules(200, seed=5) == _rules(200, seed=5)


def test_heldout_wording_never_appears_in_training_text():
    train = " ".join(e.prompt for e in _rules(5000)).lower()
    for phrase in ("whenever", "in cases where", "inr ", "deny ", "escalate", "the charge",
                   "and also", "provided that", "along with", "allowlist", "except when",
                   "overnight", "from abroad", "the customer", "past 60 minutes"):
        assert phrase not in train, phrase


def test_heldout_split_actually_uses_heldout_wording():
    text = " ".join(e.prompt for e in _rules(500, heldout=True)).lower()
    for phrase in ("whenever", "inr ", "the charge", "allowlist", "the customer"):
        assert phrase in text, phrase


def test_semantic_negative_really_behaves_differently():
    rng = random.Random(3)
    for example in _rules(500):
        mutated = semantic_negative(example.rule, rng)
        if mutated is not None:
            parse_checked(format_rule(mutated))
            assert not equivalent(example.rule, mutated).equivalent


def test_reorder_and_convolute_preserve_meaning():
    rng = random.Random(4)
    for example in _rules(500):
        reordered = example.rule.__class__(
            reorder(example.rule.condition, rng), example.rule.action, example.rule.reason
        )
        assert equivalent(example.rule, reordered).equivalent
        clunky = convolute(example.rule.condition)
        if clunky is not None:
            other = example.rule.__class__(clunky, example.rule.action, example.rule.reason)
            assert equivalent(example.rule, other).equivalent


def test_pairs_rank_chosen_above_rejected_by_score():
    rng = random.Random(6)
    seen_kinds = set()
    for example in _rules(800):
        pair = make_pair(example.rule, rng)
        if pair is None:
            continue
        chosen, rejected, kind = pair
        seen_kinds.add(kind)
        assert score_candidate(chosen, example.rule).semantic_match
        result = score_candidate(rejected, example.rule)
        if kind == "style":
            assert result.semantic_match and len(rejected) > len(chosen)
        else:
            assert not result.semantic_match
    assert seen_kinds == {"semantic", "action", "invalid", "style"}


def test_score_candidate_grades_each_failure_level():
    gold = parse('WHEN txn.amount > 100 THEN BLOCK reason="r"')
    assert score_candidate("nonsense", gold).parses is False
    typed = score_candidate('WHEN txn.nope > 100 THEN BLOCK reason="r"', gold)
    assert typed.parses and not typed.type_checks
    wrong = score_candidate('WHEN txn.amount > 101 THEN BLOCK reason="r"', gold)
    assert wrong.type_checks and not wrong.semantic_match
    assert score_candidate('WHEN NOT txn.amount <= 100 THEN BLOCK reason="other"', gold).semantic_match
