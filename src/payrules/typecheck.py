"""Static checks: known fields, operators valid for the type, literals in the field's domain."""

from .nodes import And, BoolField, Compare, Expr, InList, Not, Or, Rule
from .schema import SCHEMA, FieldSpec, FieldType


class RuleTypeError(ValueError):
    """The rule is well formed but not valid against the schema."""

    def __init__(self, issues: list[str]):
        super().__init__("; ".join(issues))
        self.issues = issues


def check(rule: Rule) -> list[str]:
    """Return every problem found in the rule; empty means valid."""
    issues: list[str] = []
    _check_expr(rule.condition, issues)
    return issues


def validate(rule: Rule) -> None:
    issues = check(rule)
    if issues:
        raise RuleTypeError(issues)


def _check_expr(expr: Expr, issues: list[str]) -> None:
    match expr:
        case Not(operand):
            _check_expr(operand, issues)
        case And(operands) | Or(operands):
            for operand in operands:
                _check_expr(operand, issues)
        case BoolField(field):
            spec = _lookup(field, issues)
            if spec and spec.type is not FieldType.BOOL:
                issues.append(f"{field} is {spec.type.value}, so it needs a comparison")
        case InList(field, values):
            spec = _lookup(field, issues)
            if spec and spec.type is not FieldType.ENUM:
                issues.append(f"IN only applies to enum fields, not {field}")
            elif spec:
                _check_enum_values(spec, values, issues)
                if len(set(values)) != len(values):
                    issues.append(f"duplicate values in IN list for {field}")
        case Compare(field, op, value):
            spec = _lookup(field, issues)
            if spec:
                _check_compare(spec, op, value, issues)


def _lookup(field: str, issues: list[str]) -> FieldSpec | None:
    spec = SCHEMA.get(field)
    if spec is None:
        issues.append(f"unknown field {field}")
    return spec


def _check_enum_values(spec: FieldSpec, values, issues: list[str]) -> None:
    for value in values:
        if value not in spec.values:
            issues.append(f"{value} is not a valid value for {spec.name}")


def _check_compare(spec: FieldSpec, op: str, value, issues: list[str]) -> None:
    if spec.type is FieldType.BOOL:
        issues.append(f"{spec.name} is boolean: use it directly or with NOT")
    elif spec.type is FieldType.ENUM:
        if op not in ("==", "!="):
            issues.append(f"{op} does not apply to enum field {spec.name}")
        if isinstance(value, str):
            _check_enum_values(spec, (value,), issues)
        else:
            issues.append(f"{spec.name} needs an enum value, got {value}")
    elif isinstance(value, str):
        issues.append(f"{spec.name} needs a number, got {value}")
    elif spec.type is FieldType.INT and isinstance(value, float):
        issues.append(f"{spec.name} needs a whole number, got {value}")
    elif not spec.min <= value <= spec.max:
        issues.append(f"{value} is outside {spec.name}'s range {spec.min}..{spec.max}")
