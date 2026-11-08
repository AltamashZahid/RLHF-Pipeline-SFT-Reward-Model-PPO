import pytest

from payrules import RuleEngine, equivalent, parse_checked
from payrules.engine import compile_expr
from payrules.investigate import History, Investigator, backtest, counterfactuals, propose_refinement
from payrules.nodes import Or, Rule
from payrules.txngen import LABEL, generate_transactions

BROAD = parse_checked('WHEN txn.amount > 20000 THEN BLOCK reason="large"')


@pytest.fixture(scope="module")
def txns():
    return generate_transactions(20_000, seed=3, fraud_rate=0.05)


@pytest.fixture(scope="module")
def history(txns):
    return History(txns)


@pytest.mark.parametrize("text", [
    "txn.amount > 5000 AND NOT merchant.whitelisted",
    "txn.method IN [UPI, CARD] OR device.is_rooted",
    "NOT (txn.hour < 6 OR merchant.category == FOOD)",
    "user.failed_attempts_24h >= 3 AND txn.method != UPI",
])
def test_history_mask_agrees_with_the_engine(history, txns, text):
    condition = parse_checked(f'WHEN {text} THEN BLOCK reason="r"').condition
    predicate = compile_expr(condition)
    assert history.mask(condition).tolist() == [bool(predicate(t)) for t in txns]


def test_backtest_counts_by_hand(history, txns):
    stats = backtest(history, BROAD.condition)
    hits = [t for t in txns if t["txn.amount"] > 20000]
    assert stats.matches == len(hits)
    assert stats.precision == pytest.approx(sum(t[LABEL] for t in hits) / len(hits))


def test_counterfactual_really_flips_the_decision(txns):
    engine = RuleEngine([BROAD])
    txn = next(t for t in txns if t["txn.amount"] > 30000)
    [flip] = counterfactuals(engine, txn)
    assert flip.field == "txn.amount" and flip.new_action == "ALLOW"
    assert flip.changed_to <= 20000
    assert engine.decide({**txn, "txn.amount": flip.changed_to}).action == "ALLOW"


def test_proposal_is_proven_narrower_and_better_on_holdout(history):
    proposal = propose_refinement(history, BROAD)
    assert proposal is not None
    union = Rule(Or((BROAD.condition, proposal.rule.condition)), "BLOCK", "large")
    proof = equivalent(union, BROAD)
    assert proof.equivalent and proof.exhaustive
    assert proposal.after.precision > proposal.before.precision
    assert proposal.after.recall >= 0.95 * proposal.before.recall


def test_no_proposal_when_rule_catches_no_fraud(history):
    useless = parse_checked('WHEN txn.amount < 0.5 THEN BLOCK reason="tiny"')
    assert propose_refinement(history, useless) is None


def test_investigation_cannot_see_the_cases_label(history, txns):
    investigator = Investigator(RuleEngine([BROAD]), history)
    txn = next(t for t in txns if t["txn.amount"] > 30000)
    first = investigator.investigate({**txn, LABEL: True})
    second = investigator.investigate({**txn, LABEL: False})
    assert first.render() == second.render()
    assert LABEL not in first.case.txn


def test_steps_follow_the_plan_and_render(history, txns):
    investigator = Investigator(RuleEngine([BROAD]), history)
    txn = next(t for t in txns if t["txn.amount"] > 30000)
    report = investigator.investigate(txn)
    assert report.case.steps[:4] == ["explain", "what_if", "peers", "backtest"]
    assert report.verdict in {"escalate", "clear"} and "verdict:" in report.render()


def test_unflagged_transaction_skips_rule_tools(history, txns):
    investigator = Investigator(RuleEngine([BROAD]), history)
    txn = next(t for t in txns if t["txn.amount"] <= 20000)
    assert investigator.investigate(txn).case.steps == ["what_if", "peers"]


def test_unfamiliar_case_is_never_cleared(history, txns):
    txn = next(t for t in txns if t["txn.amount"] > 30000 and not t[LABEL])
    strange = {**txn, "device.age_days": 10_000, "user.account_age_days": 10_000, "txn.count_1h": 1000}
    relaxed = Investigator(RuleEngine([BROAD]), history, threshold=1.1)
    strict = Investigator(RuleEngine([BROAD]), history, threshold=1.1, unfamiliar_distance=5.0)
    assert relaxed.investigate(strange).verdict == "clear"
    assert strict.investigate(strange).verdict == "escalate_unfamiliar"
