import pytest

from payrules.compiler import compile_one, group_candidates

A = 'WHEN txn.amount > 100 THEN BLOCK reason="a"'
A_REWORDED = 'WHEN NOT txn.amount <= 100 THEN BLOCK reason="other_words"'
B = 'WHEN txn.amount >= 100 THEN BLOCK reason="b"'
C = 'WHEN device.is_rooted THEN REVIEW reason="c"'
BAD_SYNTAX = "WHEN txn.amount >"
BAD_TYPE = 'WHEN txn.amount THEN BLOCK reason="x"'


def scorer(preferred):
    return lambda requests, texts: [10.0 if t == preferred else 0.0 for t in texts]


def test_invalid_candidates_are_dropped_and_counted():
    answers, valid = group_candidates([BAD_SYNTAX, A, BAD_TYPE])
    assert valid == 1 and [a.text for a in answers] == [A]


def test_rewordings_count_as_one_behaviour():
    answers, valid = group_candidates([A, A_REWORDED, A, B])
    assert valid == 4
    assert [(a.text, a.votes) for a in answers] == [(A, 3), (B, 1)]


def test_a_boundary_difference_is_a_different_behaviour():
    answers, _ = group_candidates([A, B])
    assert len(answers) == 2


def test_action_difference_is_a_different_behaviour():
    answers, _ = group_candidates([A, A.replace("BLOCK", "ALLOW")])
    assert len(answers) == 2


def test_greedy_strategy_returns_the_greedy_rule_even_when_outvoted():
    compiled = compile_one("r", B, [A, A, A], None, "greedy")
    assert compiled.answer.text == B
    assert compiled.agreement == pytest.approx(0.25)


def test_greedy_strategy_has_no_answer_when_greedy_is_invalid():
    assert compile_one("r", BAD_SYNTAX, [A, A], None, "greedy").answer is None


def test_reward_strategy_follows_the_scorer_not_the_vote():
    compiled = compile_one("r", A, [A, A, B], scorer(B), "reward")
    assert compiled.answer.text == B


def test_majority_strategy_follows_the_vote_not_the_scorer():
    compiled = compile_one("r", B, [A, A, A_REWORDED], scorer(B), "majority")
    assert compiled.answer.text == A
    assert compiled.agreement == pytest.approx(0.75)


def test_majority_breaks_ties_toward_greedy():
    assert compile_one("r", B, [A], None, "majority").answer.text == B


def test_majority_then_reward_only_uses_the_score_on_a_tie():
    tied = compile_one("r", B, [A], scorer(A), "majority_then_reward")
    assert tied.answer.text == A
    clear = compile_one("r", B, [B, A], scorer(A), "majority_then_reward")
    assert clear.answer.text == B


def test_no_valid_candidate_means_no_rule_and_human_review():
    compiled = compile_one("r", BAD_SYNTAX, [BAD_TYPE], scorer(A), "reward")
    assert compiled.answer is None and compiled.valid == 0 and compiled.total == 2
    assert compiled.needs_review(0.0)


def test_disagreement_triggers_review_at_the_threshold():
    compiled = compile_one("r", A, [A, B, C], None, "majority")
    assert compiled.agreement == pytest.approx(0.5)
    assert not compiled.needs_review(0.5)
    assert compiled.needs_review(0.51)


def test_unknown_strategy_is_rejected():
    with pytest.raises(ValueError):
        compile_one("r", A, [], None, "nope")
