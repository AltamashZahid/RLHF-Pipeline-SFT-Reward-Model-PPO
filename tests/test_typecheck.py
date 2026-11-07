import pytest

from payrules import RuleTypeError, check, parse, parse_checked


def issues(condition):
    return check(parse(f'WHEN {condition} THEN BLOCK reason="r"'))


@pytest.mark.parametrize("condition", [
    "txn.amount > 0",
    "txn.amount <= 10000000",
    "txn.amount > 99.5",
    "txn.method == UPI",
    "txn.method != CARD",
    "merchant.category IN [GAMING]",
    "device.is_rooted",
    "NOT device.is_rooted",
    "txn.hour >= 23",
])
def test_valid_conditions(condition):
    assert issues(condition) == []


@pytest.mark.parametrize("condition, expected", [
    ("txn.secret > 1", "unknown field"),
    ("label.is_fraud", "unknown field"),
    ("txn.method == BITCOIN", "not a valid value"),
    ("txn.method > UPI", "does not apply"),
    ("txn.method == 5", "needs an enum value"),
    ("txn.amount == UPI", "needs a number"),
    ("txn.amount > -1", "outside"),
    ("txn.hour > 24", "outside"),
    ("txn.hour > 1.5", "whole number"),
    ("device.is_rooted == 1", "boolean"),
    ("txn.amount", "needs a comparison"),
    ("txn.amount IN [UPI]", "only applies to enum"),
    ("merchant.category IN [GAMING, BITCOIN]", "not a valid value"),
    ("merchant.category IN [GAMING, GAMING]", "duplicate"),
])
def test_invalid_conditions(condition, expected):
    found = issues(condition)
    assert any(expected in issue for issue in found), found


def test_reports_every_issue_not_just_the_first():
    assert len(issues("txn.secret > 1 AND txn.hour > 99 AND NOT txn.method == BITCOIN")) == 3


def test_parse_checked_raises():
    with pytest.raises(RuleTypeError):
        parse_checked('WHEN txn.secret > 1 THEN BLOCK reason="r"')
