import random

import pytest
from rulegen import random_rule

from payrules import RuleSyntaxError, parse
from payrules.nodes import And, BoolField, Compare, InList, Not, Or, format_rule

EXAMPLE = (
    "WHEN txn.method == UPI AND txn.amount > 50000 AND device.age_days < 7 "
    'AND NOT merchant.whitelisted THEN BLOCK reason="high_value_new_device"'
)


def test_parses_the_example_rule():
    rule = parse(EXAMPLE)
    assert rule.action == "BLOCK"
    assert rule.reason == "high_value_new_device"
    assert rule.condition == And((
        Compare("txn.method", "==", "UPI"),
        Compare("txn.amount", ">", 50000),
        Compare("device.age_days", "<", 7),
        Not(BoolField("merchant.whitelisted")),
    ))


def test_and_binds_tighter_than_or():
    condition = parse(
        "WHEN device.is_rooted OR user.kyc_verified AND merchant.whitelisted "
        'THEN ALLOW reason="r"'
    ).condition
    assert isinstance(condition, Or)
    assert isinstance(condition.operands[1], And)


def test_in_list():
    rule = parse('WHEN merchant.category IN [GAMING, TRAVEL] THEN REVIEW reason="r"')
    assert rule.condition == InList("merchant.category", ("GAMING", "TRAVEL"))


def test_decimal_and_negative_literals():
    assert parse('WHEN txn.amount > 10.5 THEN ALLOW reason="r"').condition.value == 10.5
    assert parse('WHEN txn.amount > -1 THEN ALLOW reason="r"').condition.value == -1


@pytest.mark.parametrize("text", [
    "",
    "WHEN",
    'txn.amount > 1 THEN BLOCK reason="r"',
    'WHEN txn.amount > THEN BLOCK reason="r"',
    "WHEN txn.amount > 1 THEN BLOCK",
    'WHEN txn.amount > 1 THEN DELETE reason="r"',
    'WHEN txn.amount > 1 THEN BLOCK reason="Has Spaces"',
    'WHEN txn.amount > 1 THEN BLOCK reason="r" trailing',
    'WHEN (txn.amount > 1 THEN BLOCK reason="r"',
    'WHEN merchant.category IN [] THEN BLOCK reason="r"',
    'WHEN __import__("os").system("x") THEN BLOCK reason="r"',
    'WHEN txn.amount > 1; DROP TABLE x THEN BLOCK reason="r"',
])
def test_malformed_rules_raise_syntax_error(text):
    with pytest.raises(RuleSyntaxError):
        parse(text)


def test_small_float_formats_without_exponent():
    rule = parse('WHEN txn.amount > 0.00001 THEN ALLOW reason="r"')
    assert parse(format_rule(rule)) == rule


def test_format_then_parse_round_trips_random_rules():
    rng = random.Random(7)
    for _ in range(2000):
        rule = random_rule(rng)
        assert parse(format_rule(rule)) == rule
