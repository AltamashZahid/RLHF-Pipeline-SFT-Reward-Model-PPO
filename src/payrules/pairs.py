"""Builds chosen/rejected rule pairs for the reward model, ranked by the verifier.

A rejected rule is always proven worse than the chosen one: it does not parse, fails type
checking, behaves differently, or behaves the same but is needlessly convoluted.
"""

import random
from dataclasses import dataclass, replace

from .nodes import And, BoolField, Compare, Expr, InList, Not, Or, Rule, format_rule
from .parser import RuleSyntaxError, parse_checked
from .schema import SCHEMA, FieldType
from .typecheck import RuleTypeError, check
from .verify import equivalent

_ORDERING = (">", ">=", "<", "<=")
_NEGATED = {">": "<=", ">=": "<", "<": ">=", "<=": ">"}


@dataclass(frozen=True)
class Score:
    """How a candidate rule compares with the correct one."""

    parses: bool
    type_checks: bool
    semantic_match: bool


def score_candidate(text: str, gold: Rule) -> Score:
    from .parser import parse

    try:
        rule = parse(text)
    except RuleSyntaxError:
        return Score(False, False, False)
    if check(rule):
        return Score(True, False, False)
    return Score(True, True, equivalent(rule, gold).equivalent)


def _same_type_fields(field: str) -> list[str]:
    spec = SCHEMA[field]
    return [n for n, s in SCHEMA.items() if s.type is spec.type and n != field]


def _neighbours(expr: Expr) -> list[Expr]:
    """Every expression one small edit away. Some may be equivalent or invalid; callers check."""
    match expr:
        case Compare(field, op, value) if SCHEMA[field].type is FieldType.ENUM:
            spec = SCHEMA[field]
            flipped = "!=" if op == "==" else "=="
            others = [Compare(field, op, v) for v in spec.values if v != value]
            return [Compare(field, flipped, value), *others]
        case Compare(field, op, value):
            spec = SCHEMA[field]
            ops = [Compare(field, o, value) for o in _ORDERING if o != op]
            if spec.type is FieldType.INT:
                values = (value + 1, value - 1, value * 2, value // 2)
            else:
                values = (value * 2, int(value // 2), value * 10, value + 1)
            shifted = [Compare(field, op, v) for v in values if spec.min <= v <= spec.max and v != value]
            swapped = [Compare(f, op, value) for f in _same_type_fields(field)
                       if SCHEMA[f].min <= value <= SCHEMA[f].max]
            return [*ops, *shifted, *swapped]
        case InList(field, values):
            spec = SCHEMA[field]
            result: list[Expr] = [Not(expr)]
            if len(values) > 1:
                result += [InList(field, tuple(v for v in values if v != d)) for d in values]
            result += [InList(field, (*values[:-1], v)) for v in spec.values if v not in values]
            return result
        case BoolField(field):
            return [Not(expr), *(BoolField(f) for f in _same_type_fields(field))]
        case Not(operand):
            return [operand, *(Not(n) for n in _neighbours(operand))]
        case And(operands) | Or(operands):
            build = And if isinstance(expr, And) else Or
            other = Or if isinstance(expr, And) else And
            result = [other(operands)]
            for i, operand in enumerate(operands):
                rest = operands[:i] + operands[i + 1:]
                result.append(rest[0] if len(rest) == 1 else build(rest))
                for n in _neighbours(operand):
                    result.append(build(operands[:i] + (n,) + operands[i + 1:]))
            return result
    raise TypeError(f"not an expression: {expr!r}")


def semantic_negative(rule: Rule, rng: random.Random) -> Rule | None:
    """A valid rule that behaves differently from `rule`, or None if none was found."""
    candidates = _neighbours(rule.condition)
    rng.shuffle(candidates)
    for condition in candidates:
        mutated = replace(rule, condition=condition)
        if not check(mutated) and not equivalent(rule, mutated).equivalent:
            return mutated
    return None


def wrong_action(rule: Rule, rng: random.Random) -> Rule:
    action = rng.choice([a for a in ("BLOCK", "ALLOW", "REVIEW") if a != rule.action])
    return replace(rule, action=action)


def _break_syntax(text: str, rng: random.Random) -> str:
    tokens = text.split()
    choice = rng.randrange(4)
    if choice == 0:
        return " ".join(tokens[: rng.randint(1, len(tokens) - 1)])
    if choice == 1:
        return text.replace(" THEN ", " ", 1)
    if choice == 2:
        return text.replace("WHEN ", "IF ", 1)
    return text.replace("reason=", "because ", 1)


def _break_types(rule: Rule, rng: random.Random) -> str:
    text = format_rule(rule)
    field = rng.choice([n for n in SCHEMA if n in text] or ["txn.amount"])
    return text.replace(field, field.replace(".", ".unknown_"), 1)


def invalid_text(rule: Rule, rng: random.Random) -> str:
    """Text that fails to parse or type check."""
    text = format_rule(rule)
    for _ in range(10):
        candidate = _break_types(rule, rng) if rng.random() < 0.3 else _break_syntax(text, rng)
        try:
            parse_checked(candidate)
        except (RuleSyntaxError, RuleTypeError):
            return candidate
    raise ValueError(f"could not break rule: {text}")


def reorder(expr: Expr, rng: random.Random) -> Expr:
    """Same meaning, clauses shuffled: stops the reward model just preferring one layout."""
    match expr:
        case Not(operand):
            return Not(reorder(operand, rng))
        case And(operands) | Or(operands):
            parts = [reorder(o, rng) for o in operands]
            rng.shuffle(parts)
            return type(expr)(tuple(parts))
    return expr


def convolute(expr: Expr) -> Expr | None:
    """Same meaning but harder to read. None if the expression has no ordering comparison."""
    match expr:
        case Compare(field, op, value) if op in _NEGATED:
            return Not(Compare(field, _NEGATED[op], value))
        case InList(field, values) if len(values) > 1:
            return Or(tuple(Compare(field, "==", v) for v in values))
        case BoolField():
            return Not(Not(expr))
        case Not(operand):
            inner = convolute(operand)
            return None if inner is None else Not(inner)
        case And(operands) | Or(operands):
            for i, operand in enumerate(operands):
                inner = convolute(operand)
                if inner is not None:
                    return type(expr)(operands[:i] + (inner,) + operands[i + 1:])
    return None


def make_pair(rule: Rule, rng: random.Random) -> tuple[str, str, str] | None:
    """Return (chosen text, rejected text, kind), or None if this rule cannot produce one."""
    kind = rng.choices(("semantic", "action", "invalid", "style"), weights=(70, 8, 12, 10))[0]
    chosen = replace(rule, condition=reorder(rule.condition, rng)) if rng.random() < 0.3 else rule
    if kind == "semantic":
        rejected = semantic_negative(rule, rng)
        return None if rejected is None else (format_rule(chosen), format_rule(rejected), kind)
    if kind == "action":
        return format_rule(chosen), format_rule(wrong_action(rule, rng)), kind
    if kind == "invalid":
        return format_rule(chosen), invalid_text(rule, rng), kind
    condition = convolute(rule.condition)
    if condition is None:
        return None
    rejected = replace(rule, condition=condition)
    if not equivalent(rule, rejected).equivalent:
        raise ValueError(f"convolute changed meaning: {format_rule(rule)}")
    return format_rule(rule), format_rule(rejected), kind
