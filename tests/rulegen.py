"""Random valid rules for property tests."""

from payrules.nodes import And, BoolField, Compare, InList, Not, Or, Rule
from payrules.schema import SCHEMA, FieldType

_FIELDS = tuple(SCHEMA.values())
_NUMERIC_OPS = ("==", "!=", ">", ">=", "<", "<=")


def _atom(rng):
    spec = rng.choice(_FIELDS)
    if spec.type is FieldType.BOOL:
        return BoolField(spec.name)
    if spec.type is FieldType.ENUM:
        if rng.random() < 0.5:
            return Compare(spec.name, rng.choice(("==", "!=")), rng.choice(spec.values))
        return InList(spec.name, tuple(rng.sample(spec.values, rng.randint(1, 3))))
    upper = min(spec.max, 100)
    if spec.type is FieldType.INT:
        return Compare(spec.name, rng.choice(_NUMERIC_OPS), rng.randint(spec.min, upper))
    return Compare(spec.name, rng.choice(_NUMERIC_OPS), round(rng.uniform(spec.min, upper), 2))


def random_expr(rng, depth=3):
    if depth == 0 or rng.random() < 0.3:
        return _atom(rng)
    kind = rng.choice(("and", "or", "not"))
    if kind == "not":
        return Not(random_expr(rng, depth - 1))
    operands = tuple(random_expr(rng, depth - 1) for _ in range(rng.randint(2, 3)))
    return And(operands) if kind == "and" else Or(operands)


def random_rule(rng, depth=3):
    return Rule(random_expr(rng, depth), rng.choice(("BLOCK", "ALLOW", "REVIEW")), "generated")
