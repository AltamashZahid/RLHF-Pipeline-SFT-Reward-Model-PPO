import pytest

from payrules import RuleEngine, RuleTypeError, parse, parse_checked
from payrules.engine import trace_expr
from payrules.txngen import generate_transactions


def rule(condition, action="BLOCK", reason="r"):
    return parse_checked(f'WHEN {condition} THEN {action} reason="{reason}"')


def txn(**overrides):
    base = generate_transactions(1, seed=0)[0]
    base.update({k.replace("__", "."): v for k, v in overrides.items()})
    return base


@pytest.mark.parametrize("condition, value, expected", [
    ("txn.amount > 100", 100, False),
    ("txn.amount > 100", 100.01, True),
    ("txn.amount >= 100", 100, True),
    ("txn.amount < 100", 100, False),
    ("txn.amount <= 100", 100, True),
    ("txn.amount == 100", 100, True),
    ("txn.amount != 100", 100, False),
])
def test_numeric_boundaries(condition, value, expected):
    engine = RuleEngine([rule(condition)])
    assert (engine.decide(txn(txn__amount=value)).action == "BLOCK") is expected


def test_the_example_rule_end_to_end():
    engine = RuleEngine([rule(
        "txn.method == UPI AND txn.amount > 50000 AND device.age_days < 7 "
        "AND NOT merchant.whitelisted"
    )])
    risky = dict(
        txn__method="UPI", txn__amount=60000, device__age_days=2, merchant__whitelisted=False
    )
    assert engine.decide(txn(**risky)).action == "BLOCK"
    assert engine.decide(txn(**{**risky, "merchant__whitelisted": True})).action == "ALLOW"
    assert engine.decide(txn(**{**risky, "txn__method": "CARD"})).action == "ALLOW"
    assert engine.decide(txn(**{**risky, "device__age_days": 7})).action == "ALLOW"


def test_or_not_and_in():
    engine = RuleEngine([rule("device.is_rooted OR NOT merchant.category IN [FOOD, GROCERY]")])
    assert engine.decide(txn(device__is_rooted=True, merchant__category="FOOD")).action == "BLOCK"
    assert engine.decide(txn(device__is_rooted=False, merchant__category="FOOD")).action == "ALLOW"
    assert engine.decide(txn(device__is_rooted=False, merchant__category="TRAVEL")).action == "BLOCK"


def test_first_matching_rule_wins():
    engine = RuleEngine([
        rule("txn.amount > 100", "REVIEW", "first"),
        rule("txn.amount > 50", "BLOCK", "second"),
    ])
    decision = engine.decide(txn(txn__amount=200))
    assert (decision.action, decision.reason, decision.rule_index) == ("REVIEW", "first", 0)
    decision = engine.decide(txn(txn__amount=75))
    assert (decision.action, decision.reason, decision.rule_index) == ("BLOCK", "second", 1)


def test_default_applies_when_nothing_matches():
    decision = RuleEngine([rule("txn.amount > 100")]).decide(txn(txn__amount=1))
    assert (decision.action, decision.rule_index) == ("ALLOW", None)
    assert RuleEngine([], default_action="REVIEW").decide(txn()).action == "REVIEW"


def test_engine_refuses_rules_that_fail_type_checking():
    with pytest.raises(RuleTypeError):
        RuleEngine([parse('WHEN txn.secret > 1 THEN BLOCK reason="r"')])


def test_explain_shows_every_clause_with_actual_values():
    engine = RuleEngine([rule("txn.amount > 100 AND (device.is_rooted OR txn.hour < 6)")])
    decision, trace = engine.explain(txn(txn__amount=500, device__is_rooted=False, txn__hour=3))
    assert decision.action == "BLOCK"
    assert trace.result is True
    amount, either = trace.children
    assert (amount.text, amount.actual, amount.result) == ("txn.amount > 100", 500, True)
    assert [c.result for c in either.children] == [False, True]
    rendered = trace.render()
    assert "[false] device.is_rooted" in rendered
    assert "(actual: 3)" in rendered


def test_explain_when_nothing_matched():
    decision, trace = RuleEngine([rule("txn.amount > 100")]).explain(txn(txn__amount=1))
    assert decision.rule_index is None and trace is None


def test_trace_agrees_with_decide_on_many_transactions():
    engine = RuleEngine([rule(
        "(txn.amount > 1000 AND NOT merchant.whitelisted) OR txn.count_1h >= 3 "
        "OR merchant.category IN [GAMING]"
    )])
    for t in generate_transactions(2000, seed=3):
        matched = engine.decide(t).rule_index is not None
        assert trace_expr(engine.rules[0].condition, t).result is matched
