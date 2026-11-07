"""Compiles rules to closures and evaluates them against transactions.

Rules come from a model, so they are never turned into Python source or passed to eval().
A rule can only read schema fields and apply the fixed operators below.
"""

import operator
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field

from .nodes import And, BoolField, Compare, Expr, InList, Not, Or, Rule, format_expr
from .typecheck import validate

Txn = Mapping[str, object]
Predicate = Callable[[Txn], bool]

_OPS = {
    "==": operator.eq,
    "!=": operator.ne,
    ">": operator.gt,
    ">=": operator.ge,
    "<": operator.lt,
    "<=": operator.le,
}


def compile_expr(expr: Expr) -> Predicate:
    match expr:
        case Compare(name, op, value):
            fn = _OPS[op]
            return lambda txn: fn(txn[name], value)
        case InList(name, values):
            allowed = frozenset(values)
            return lambda txn: txn[name] in allowed
        case BoolField(name):
            return lambda txn: txn[name]
        case Not(operand):
            inner = compile_expr(operand)
            return lambda txn: not inner(txn)
        case And(operands):
            parts = tuple(compile_expr(o) for o in operands)

            def all_true(txn: Txn) -> bool:
                for part in parts:
                    if not part(txn):
                        return False
                return True

            return all_true
        case Or(operands):
            parts = tuple(compile_expr(o) for o in operands)

            def any_true(txn: Txn) -> bool:
                for part in parts:
                    if part(txn):
                        return True
                return False

            return any_true
    raise TypeError(f"not an expression: {expr!r}")


@dataclass(frozen=True)
class Decision:
    action: str
    reason: str
    rule_index: int | None  # None means no rule matched and the default applied


@dataclass(frozen=True)
class Trace:
    """How one part of a condition evaluated. Unlike decide(), this never short-circuits."""

    text: str
    result: bool
    actual: object = None
    children: tuple["Trace", ...] = field(default=())

    def render(self, indent: int = 0) -> str:
        actual = "" if self.actual is None else f"  (actual: {self.actual})"
        line = f"{'  ' * indent}[{'true' if self.result else 'false'}] {self.text}{actual}"
        return "\n".join([line, *(c.render(indent + 1) for c in self.children)])


def trace_expr(expr: Expr, txn: Txn) -> Trace:
    text = format_expr(expr)
    match expr:
        case Compare(name, _, _) | InList(name, _) | BoolField(name):
            return Trace(text, bool(compile_expr(expr)(txn)), txn[name])
        case Not(operand):
            child = trace_expr(operand, txn)
            return Trace(text, not child.result, children=(child,))
        case And(operands):
            children = tuple(trace_expr(o, txn) for o in operands)
            return Trace(text, all(c.result for c in children), children=children)
        case Or(operands):
            children = tuple(trace_expr(o, txn) for o in operands)
            return Trace(text, any(c.result for c in children), children=children)
    raise TypeError(f"not an expression: {expr!r}")


class RuleEngine:
    """Ordered rules: the first match decides, otherwise the default action applies."""

    def __init__(self, rules: Iterable[Rule], default_action: str = "ALLOW"):
        self.rules = tuple(rules)
        for rule in self.rules:
            validate(rule)
        self._compiled = tuple(
            (compile_expr(r.condition), r.action, r.reason) for r in self.rules
        )
        self._default = Decision(default_action, "no_rule_matched", None)

    def decide(self, txn: Txn) -> Decision:
        for index, (predicate, action, reason) in enumerate(self._compiled):
            if predicate(txn):
                return Decision(action, reason, index)
        return self._default

    def explain(self, txn: Txn) -> tuple[Decision, Trace | None]:
        """Return the decision and, if a rule matched, how its condition evaluated."""
        decision = self.decide(txn)
        if decision.rule_index is None:
            return decision, None
        return decision, trace_expr(self.rules[decision.rule_index].condition, txn)
