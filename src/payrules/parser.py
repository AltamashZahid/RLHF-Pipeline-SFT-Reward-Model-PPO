"""Text to AST. Model output is untrusted, so every failure is a RuleSyntaxError."""

from lark import Lark, Token, Transformer
from lark.exceptions import UnexpectedInput

from .nodes import And, BoolField, Compare, InList, Not, Or, Rule

GRAMMAR = r"""
start: "WHEN" or_expr "THEN" ACTION "reason" "=" REASON_STR

?or_expr: and_expr
        | and_expr ("OR" and_expr)+      -> or_
?and_expr: not_expr
         | not_expr ("AND" not_expr)+    -> and_
?not_expr: atom
         | "NOT" not_expr                -> not_
?atom: FIELD OP value                    -> compare
     | FIELD "IN" "[" ENUM ("," ENUM)* "]" -> in_list
     | FIELD                             -> bool_field
     | "(" or_expr ")"
?value: NUMBER | ENUM

ACTION: "BLOCK" | "ALLOW" | "REVIEW"
FIELD: /[a-z_]+\.[a-z0-9_]+/
OP: /==|!=|>=|<=|>|</
NUMBER: /-?\d+(\.\d+)?/
ENUM: /[A-Z][A-Z0-9_]*/
REASON_STR: /"[a-z][a-z0-9_]*"/

%import common.WS
%ignore WS
"""


class RuleSyntaxError(ValueError):
    """The text is not a well-formed rule."""


class _Build(Transformer):
    def start(self, items):
        condition, action, reason = items
        return Rule(condition, str(action), str(reason)[1:-1])

    def or_(self, items):
        return Or(tuple(items))

    def and_(self, items):
        return And(tuple(items))

    def not_(self, items):
        return Not(items[0])

    def compare(self, items):
        field, op, token = items
        return Compare(str(field), str(op), _literal(token))

    def in_list(self, items):
        return InList(str(items[0]), tuple(str(t) for t in items[1:]))

    def bool_field(self, items):
        return BoolField(str(items[0]))


def _literal(token: Token) -> int | float | str:
    if token.type == "ENUM":
        return str(token)
    return float(token) if "." in token else int(token)


_PARSER = Lark(GRAMMAR, parser="lalr", transformer=_Build())


def parse(text: str) -> Rule:
    """Parse rule text. Raises RuleSyntaxError; does not type check."""
    try:
        return _PARSER.parse(text)
    except UnexpectedInput as exc:
        raise RuleSyntaxError(str(exc)) from exc


def parse_checked(text: str) -> Rule:
    """Parse and type check rule text."""
    from .typecheck import validate

    rule = parse(text)
    validate(rule)
    return rule
