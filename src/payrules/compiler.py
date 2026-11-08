"""Turns a request into a rule without trusting any single model output.

Several candidates are drawn and anything the parser or type checker rejects is dropped. The rest
are grouped by proven behaviour, so rewordings of one rule count as one answer. A strategy then
picks an answer, and when the candidates disagree the rule can go to a person instead of running.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass

from .nodes import Rule
from .parser import RuleSyntaxError, parse_checked
from .typecheck import RuleTypeError
from .verify import equivalent

STRATEGIES = ("greedy", "reward", "majority", "majority_then_reward")
Scorer = Callable[[list[str], list[str]], Sequence[float]]


@dataclass(frozen=True)
class Answer:
    """One distinct behaviour among the candidates."""

    rule: Rule
    text: str  # The first wording seen for this behaviour.
    votes: int
    score: float | None = None


@dataclass(frozen=True)
class Compiled:
    answer: Answer | None
    agreement: float  # Share of valid candidates that behave like the chosen answer.
    valid: int
    total: int

    def needs_review(self, minimum_agreement: float) -> bool:
        return self.answer is None or self.agreement < minimum_agreement


def group_candidates(texts: Sequence[str]) -> tuple[list[Answer], int]:
    """Distinct behaviours, in first-seen order, and how many candidates were valid."""
    groups: list[list] = []  # [rule, text, votes]
    valid = 0
    by_text: dict[str, int] = {}
    for text in texts:
        if text in by_text:  # Identical text needs neither parsing nor a proof.
            groups[by_text[text]][2] += 1
            valid += 1
            continue
        try:
            rule = parse_checked(text)
        except (RuleSyntaxError, RuleTypeError):
            continue
        valid += 1
        for index, group in enumerate(groups):
            verdict = equivalent(group[0], rule)
            if verdict.equivalent and verdict.exhaustive:
                group[2] += 1
                by_text[text] = index
                break
        else:
            by_text[text] = len(groups)
            groups.append([rule, text, 1])
    return [Answer(rule, text, votes) for rule, text, votes in groups], valid


def choose(answers: list[Answer], greedy_text: str, strategy: str) -> Answer | None:
    if not answers:
        return None
    if strategy == "greedy":  # Greedy is drawn first, so if it was valid its behaviour is the first group.
        return answers[0] if _parses(greedy_text) else None
    if strategy == "reward":
        return max(answers, key=lambda a: a.score)
    if strategy == "majority":
        return max(answers, key=lambda a: a.votes)  # max() keeps the earliest, so greedy wins ties.
    if strategy == "majority_then_reward":
        most = max(a.votes for a in answers)
        return max((a for a in answers if a.votes == most), key=lambda a: a.score)
    raise ValueError(f"unknown strategy {strategy}")


def _parses(text: str) -> bool:
    try:
        parse_checked(text)
    except (RuleSyntaxError, RuleTypeError):
        return False
    return True


def compile_one(request: str, greedy_text: str, samples: Sequence[str], score: Scorer | None,
                strategy: str = "reward") -> Compiled:
    """The greedy output goes first, so it is the first group whenever it is valid."""
    texts = [greedy_text, *samples]
    answers, valid = group_candidates(texts)
    if score is not None and answers:
        values = score([request] * len(answers), [a.text for a in answers])
        answers = [Answer(a.rule, a.text, a.votes, float(v)) for a, v in zip(answers, values)]
    chosen = choose(answers, greedy_text, strategy)
    agreement = chosen.votes / valid if chosen and valid else 0.0
    return Compiled(chosen, agreement, valid, len(texts))
