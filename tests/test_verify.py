import random

from rulegen import random_rule

from payrules import equivalent, parse_checked
from payrules.engine import compile_expr
from payrules.nodes import And, Not, Or, Rule
from payrules.txngen import generate_transactions


def rule(condition, action="BLOCK", reason="r"):
    return parse_checked(f'WHEN {condition} THEN {action} reason="{reason}"')


def test_de_morgan_is_equivalent():
    result = equivalent(
        rule("NOT (device.is_rooted OR txn.amount > 100)"),
        rule("NOT device.is_rooted AND txn.amount <= 100"),
    )
    assert result.equivalent and result.exhaustive


def test_off_by_one_threshold_is_caught_with_a_counterexample():
    result = equivalent(rule("txn.amount > 100"), rule("txn.amount >= 100"))
    assert not result.equivalent
    assert result.counterexample["txn.amount"] == 100


def test_narrow_gap_between_decimal_thresholds_is_caught():
    result = equivalent(rule("txn.amount > 100.01"), rule("txn.amount > 100.02"))
    assert not result.equivalent
    assert 100.01 < result.counterexample["txn.amount"] <= 100.02


def test_integer_fields_have_no_values_between_whole_numbers():
    assert equivalent(rule("txn.hour > 5"), rule("txn.hour >= 6")).equivalent


def test_decimal_fields_do_have_values_between_whole_numbers():
    assert not equivalent(rule("txn.amount > 5"), rule("txn.amount >= 6")).equivalent


def test_enum_in_list_matches_or_of_equalities():
    assert equivalent(
        rule("txn.method IN [UPI, CARD]"),
        rule("txn.method == UPI OR txn.method == CARD"),
    ).equivalent


def test_complete_enum_list_is_always_true():
    assert equivalent(
        rule("txn.method IN [UPI, CARD, NETBANKING, WALLET]"),
        rule("txn.amount >= 0"),
    ).equivalent


def test_different_action_is_not_equivalent():
    assert not equivalent(rule("txn.amount > 1", "BLOCK"), rule("txn.amount > 1", "REVIEW")).equivalent


def test_rules_that_never_match_are_equivalent_whatever_their_action():
    assert equivalent(
        rule("txn.amount < 0.5 AND txn.amount > 1", "BLOCK"),
        rule("txn.hour < 1 AND txn.hour > 2", "REVIEW"),
    ).equivalent


def test_reason_only_counts_when_asked():
    a, b = rule("txn.amount > 1", reason="one"), rule("txn.amount > 1", reason="two")
    assert equivalent(a, b).equivalent
    assert not equivalent(a, b, compare_reason=True).equivalent


def test_falls_back_to_sampling_and_says_so():
    condition = "txn.amount > 1 AND txn.hour > 1 AND txn.count_1h > 1"
    result = equivalent(rule(condition), rule(condition), max_combinations=10)
    assert result.equivalent and not result.exhaustive and result.checked == 10


def _de_morgan(expr):
    return Not(Or(tuple(Not(o) for o in expr.operands)))


def _outcome(predicate, rule_, txn):
    return rule_.action if predicate(txn) else None


def test_verdicts_agree_with_brute_force_on_random_rules():
    rng = random.Random(11)
    txns = generate_transactions(3000, seed=5)
    verdicts = {True: 0, False: 0}
    for i in range(400):
        a = random_rule(rng, depth=2)
        if i % 2 and isinstance(a.condition, And):
            b = Rule(_de_morgan(a.condition), a.action, a.reason)
        elif i % 2:
            b = a
        else:
            b = random_rule(rng, depth=2)
        result = equivalent(a, b)
        assert result.exhaustive
        verdicts[result.equivalent] += 1
        pred_a, pred_b = compile_expr(a.condition), compile_expr(b.condition)
        if result.equivalent:
            for t in txns:
                assert _outcome(pred_a, a, t) == _outcome(pred_b, b, t)
        else:
            c = result.counterexample
            assert _outcome(pred_a, a, c) != _outcome(pred_b, b, c)
    assert verdicts[True] > 50 and verdicts[False] > 50
