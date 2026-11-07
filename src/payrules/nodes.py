"""AST for the rule language, plus the canonical formatter."""

from dataclasses import dataclass
from decimal import Decimal


@dataclass(frozen=True)
class Compare:
    field: str
    op: str
    value: int | float | str


@dataclass(frozen=True)
class InList:
    field: str
    values: tuple[str, ...]


@dataclass(frozen=True)
class BoolField:
    field: str


@dataclass(frozen=True)
class Not:
    operand: "Expr"


@dataclass(frozen=True)
class And:
    operands: tuple["Expr", ...]


@dataclass(frozen=True)
class Or:
    operands: tuple["Expr", ...]


Expr = Compare | InList | BoolField | Not | And | Or


@dataclass(frozen=True)
class Rule:
    condition: Expr
    action: str
    reason: str


def _format_value(value: int | float | str) -> str:
    if isinstance(value, float):
        # repr() can produce exponents (1e-05), which the grammar does not accept.
        return format(Decimal(repr(value)), "f")
    return str(value)


def format_expr(expr: Expr) -> str:
    match expr:
        case Compare(field, op, value):
            return f"{field} {op} {_format_value(value)}"
        case InList(field, values):
            return f"{field} IN [{', '.join(values)}]"
        case BoolField(field):
            return field
        case Not(operand):
            text = format_expr(operand)
            return f"NOT ({text})" if isinstance(operand, (And, Or)) else f"NOT {text}"
        case And(operands):
            return " AND ".join(
                f"({format_expr(o)})" if isinstance(o, (And, Or)) else format_expr(o)
                for o in operands
            )
        case Or(operands):
            return " OR ".join(
                f"({format_expr(o)})" if isinstance(o, Or) else format_expr(o)
                for o in operands
            )
    raise TypeError(f"not an expression: {expr!r}")


def format_rule(rule: Rule) -> str:
    return f'WHEN {format_expr(rule.condition)} THEN {rule.action} reason="{rule.reason}"'
