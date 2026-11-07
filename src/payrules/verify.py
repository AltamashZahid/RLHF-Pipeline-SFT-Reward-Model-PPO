"""Decides whether two rules behave identically.

Comparisons only change outcome at their thresholds, so each field has a small set of
representative values, one from every region the literals carve out. Checking all their
combinations is a proof of equivalence, not a sample. If there are too many combinations
we fall back to sampling them and report exhaustive=False.
"""

import itertools
import random
from dataclasses import dataclass

from .engine import compile_expr
from .nodes import And, BoolField, Compare, Expr, InList, Not, Or, Rule
from .schema import SCHEMA, FieldSpec, FieldType
from .typecheck import validate


@dataclass(frozen=True)
class Equivalence:
    equivalent: bool
    exhaustive: bool  # False means "equivalent" only held on a sample of combinations
    checked: int
    counterexample: dict | None = None


def _literals(expr: Expr, found: dict[str, set]) -> None:
    match expr:
        case Compare(name, _, value):
            found.setdefault(name, set()).add(value)
        case InList(name, values):
            found.setdefault(name, set()).update(values)
        case BoolField(name):
            found.setdefault(name, set())
        case Not(operand):
            _literals(operand, found)
        case And(operands) | Or(operands):
            for operand in operands:
                _literals(operand, found)


def _representatives(spec: FieldSpec, literals: set) -> list:
    if spec.type is FieldType.BOOL:
        return [False, True]
    if spec.type is FieldType.ENUM:
        unmentioned = [v for v in spec.values if v not in literals]
        return sorted(literals) + unmentioned[:1]
    points = {spec.min, spec.max, *literals}
    if spec.type is FieldType.INT:
        points |= {v - 1 for v in literals} | {v + 1 for v in literals}
    else:
        ordered = sorted(points)
        points |= {(a + b) / 2 for a, b in zip(ordered, ordered[1:])}
    return sorted(p for p in points if spec.min <= p <= spec.max)


def _baseline() -> dict:
    """A valid transaction. Fields neither rule reads cannot affect the result."""
    txn = {}
    for spec in SCHEMA.values():
        if spec.type is FieldType.BOOL:
            txn[spec.name] = False
        elif spec.type is FieldType.ENUM:
            txn[spec.name] = spec.values[0]
        else:
            txn[spec.name] = spec.min
    return txn


def equivalent(
    a: Rule,
    b: Rule,
    *,
    compare_reason: bool = False,
    max_combinations: int = 200_000,
    seed: int = 0,
) -> Equivalence:
    """Two rules are equivalent if they match the same transactions with the same action."""
    validate(a)
    validate(b)

    literals: dict[str, set] = {}
    _literals(a.condition, literals)
    _literals(b.condition, literals)
    names = sorted(literals)
    domains = [_representatives(SCHEMA[n], literals[n]) for n in names]

    total = 1
    for domain in domains:
        total *= len(domain)
    exhaustive = total <= max_combinations
    if exhaustive:
        combos = itertools.product(*domains)
    else:
        rng = random.Random(seed)
        combos = (tuple(rng.choice(d) for d in domains) for _ in range(max_combinations))

    pred_a, pred_b = compile_expr(a.condition), compile_expr(b.condition)
    label_a = (a.action, a.reason) if compare_reason else a.action
    label_b = (b.action, b.reason) if compare_reason else b.action

    txn = _baseline()
    checked = 0
    for combo in combos:
        checked += 1
        txn.update(zip(names, combo))
        outcome_a = label_a if pred_a(txn) else None
        outcome_b = label_b if pred_b(txn) else None
        if outcome_a != outcome_b:
            return Equivalence(False, exhaustive, checked, dict(txn))
    return Equivalence(True, exhaustive, checked)
