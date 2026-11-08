"""An investigation agent for flagged transactions.

A small loop picks the next tool from what the case file shows so far: explain the decision,
find what would change it, compare with past cases, backtest the rule, and propose a narrower
rule when the rule is imprecise. The planner is ordinary code rather than an LLM, so every
verdict can be replayed and tested. The case's own fraud label is never read.

A proposed rule is only offered after the verifier proves it can match nothing the original
rule did not match, and after it holds up on history it was not fitted on.
"""

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field, replace

import numpy as np

from .engine import Decision, RuleEngine, Trace, trace_expr
from .nodes import And, BoolField, Compare, Expr, InList, Not, Or, Rule, format_rule
from .schema import SCHEMA, FieldType
from .txngen import LABEL
from .typecheck import validate
from .verify import _literals, _representatives, equivalent

Txn = Mapping[str, object]
_OPS: dict[str, Callable] = {
    "==": np.equal, "!=": np.not_equal, ">": np.greater,
    ">=": np.greater_equal, "<": np.less, "<=": np.less_equal,
}


class History:
    """Labelled past transactions as columns, for fast masks, backtests and nearest cases."""

    def __init__(self, txns: Iterable[Txn]):
        txns = list(txns)
        self.size = len(txns)
        self.labels = np.array([bool(t[LABEL]) for t in txns])
        self.columns = {name: np.array([t[name] for t in txns]) for name in SCHEMA}
        raw = np.stack([self._row_columns(name) for name in self._names()], axis=1).astype(np.float64)
        # Every feature is standardised, so sharing a rare trait counts more than a common one.
        self._mean = raw.mean(axis=0)
        self._std = np.where(raw.std(axis=0) > 0, raw.std(axis=0), 1.0)
        self.features = ((raw - self._mean) / self._std).astype(np.float32)

    @staticmethod
    def _names() -> list[tuple[str, object]]:
        names: list[tuple[str, object]] = []
        for name, spec in SCHEMA.items():
            names.extend((name, value) for value in spec.values) if spec.type is FieldType.ENUM                 else names.append((name, None))
        return names

    def _row_columns(self, key: tuple[str, object]) -> np.ndarray:
        name, value = key
        column = self.columns[name]
        if value is not None:
            return column == value
        return np.log1p(column.astype(float)) if SCHEMA[name].type is not FieldType.BOOL else column

    def encode(self, txn: Txn) -> np.ndarray:
        row = []
        for name, value in self._names():
            if value is not None:
                row.append(float(txn[name] == value))
            elif SCHEMA[name].type is FieldType.BOOL:
                row.append(float(txn[name]))
            else:
                row.append(float(np.log1p(float(txn[name]))))
        return ((np.array(row) - self._mean) / self._std).astype(np.float32)

    def mask(self, expr: Expr) -> np.ndarray:
        match expr:
            case Compare(name, op, value):
                return _OPS[op](self.columns[name], value)
            case InList(name, values):
                return np.isin(self.columns[name], list(values))
            case BoolField(name):
                return self.columns[name].astype(bool)
            case Not(operand):
                return ~self.mask(operand)
            case And(operands):
                return np.logical_and.reduce([self.mask(o) for o in operands])
            case Or(operands):
                return np.logical_or.reduce([self.mask(o) for o in operands])
        raise TypeError(f"not an expression: {expr!r}")

    def peers(self, txn: Txn, k: int = 50) -> tuple[float, float]:
        """Fraud rate among the k nearest past cases, and how far away those cases are."""
        distances = ((self.features - self.encode(txn)) ** 2).sum(axis=1)
        nearest = np.argpartition(distances, k)[:k]
        return float(self.labels[nearest].mean()), float(distances[nearest].mean())

    def split(self, every: int = 2) -> tuple[np.ndarray, np.ndarray]:
        """Boolean masks for a fit part and a holdout part of the history."""
        holdout = np.arange(self.size) % every == 0
        return ~holdout, holdout


@dataclass(frozen=True)
class RuleStats:
    matches: int
    precision: float
    recall: float


def backtest(history: History, condition: Expr, within: np.ndarray | None = None) -> RuleStats:
    hits = history.mask(condition)
    labels = history.labels
    if within is not None:
        hits, labels = hits[within], labels[within]
    caught = int((hits & labels).sum())
    return RuleStats(int(hits.sum()), caught / max(int(hits.sum()), 1), caught / max(int(labels.sum()), 1))


@dataclass(frozen=True)
class Counterfactual:
    field: str
    actual: object
    changed_to: object
    new_action: str


def counterfactuals(engine: RuleEngine, txn: Txn) -> list[Counterfactual]:
    """Smallest single-field changes that change the decision, checked by running the engine."""
    decision = engine.decide(txn)
    if decision.rule_index is None:
        return []
    literals: dict[str, set] = {}
    _literals(engine.rules[decision.rule_index].condition, literals)
    found = []
    for name, values in sorted(literals.items()):
        flips = []
        for candidate in _representatives(SCHEMA[name], values):
            if candidate == txn[name]:
                continue
            outcome = engine.decide({**txn, name: candidate})
            if outcome.action != decision.action:
                flips.append((candidate, outcome.action))
        if not flips:
            continue
        if SCHEMA[name].type in (FieldType.NUMBER, FieldType.INT):
            flips.sort(key=lambda flip: abs(flip[0] - txn[name]))
        changed, action = flips[0]
        found.append(Counterfactual(name, txn[name], changed, action))
    return found


def _candidates(history: History, hits: np.ndarray) -> Iterable[Expr]:
    for name, spec in SCHEMA.items():
        column = history.columns[name][hits]
        if spec.type in (FieldType.NUMBER, FieldType.INT):
            for value in sorted(set(np.quantile(column, np.linspace(0.1, 0.9, 9)).round().tolist())):
                if spec.min <= value <= spec.max:
                    yield Compare(name, ">=", int(value))
                    yield Compare(name, "<=", int(value))
        elif spec.type is FieldType.BOOL:
            yield BoolField(name)
            yield Not(BoolField(name))
        else:
            for value in spec.values:
                yield Compare(name, "==", value)


def _f1(hits: np.ndarray, labels: np.ndarray) -> float:
    caught = int((hits & labels).sum())
    return 2 * caught / max(int(hits.sum()) + int(labels.sum()), 1)


def _with_clauses(condition: Expr, extra: list[Expr]) -> Expr:
    parts = list(condition.operands) if isinstance(condition, And) else [condition]
    return And(tuple(parts + extra))


@dataclass(frozen=True)
class Proposal:
    rule: Rule
    before: RuleStats
    after: RuleStats


def propose_refinement(history: History, rule: Rule, max_extra: int = 3) -> Proposal | None:
    """Greedily add clauses on the fit split. None unless the result survives the holdout."""
    fit, holdout = history.split()
    current = history.mask(rule.condition) & fit
    # Scored against the rule's own catches: narrowing cannot catch fraud the rule missed.
    labels = history.labels & current
    if not labels.any():
        return None
    extra: list[Expr] = []
    best = _f1(current, labels)
    for _ in range(max_extra):
        scored = [(_f1(current & history.mask(c), labels), c) for c in _candidates(history, current)]
        if not scored:
            break
        score, chosen = max(scored, key=lambda item: item[0])
        if score <= best + 0.005:
            break
        best = score
        extra.append(chosen)
        current = current & history.mask(chosen)
    if not extra:
        return None
    candidate = replace(rule, condition=_with_clauses(rule.condition, extra))
    validate(candidate)
    union = replace(rule, condition=Or((rule.condition, candidate.condition)))
    proof = equivalent(union, rule)
    if not (proof.equivalent and proof.exhaustive):
        return None
    before = backtest(history, rule.condition, holdout)
    after = backtest(history, candidate.condition, holdout)
    if after.precision <= before.precision or after.recall < 0.95 * before.recall:
        return None
    return Proposal(candidate, before, after)


@dataclass
class CaseFile:
    txn: Txn
    decision: Decision
    trace: Trace | None = None
    what_if: list[Counterfactual] | None = None
    peer_rate: float | None = None
    peer_distance: float | None = None
    rule_stats: RuleStats | None = None
    proposal: Proposal | None = None
    proposal_checked: bool = False
    steps: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class Report:
    verdict: str  # "escalate", "escalate_unfamiliar" or "clear"
    case: CaseFile

    def render(self) -> str:
        case = self.case
        lines = [f"verdict: {self.verdict.upper()}  (engine said {case.decision.action}: {case.decision.reason})"]
        if case.trace:
            lines += ["why the rule fired:", case.trace.render(1)]
        for flip in case.what_if or []:
            lines.append(f"what would change it: {flip.field} from {flip.actual} to {flip.changed_to} "
                         f"gives {flip.new_action}")
        if case.peer_rate is not None:
            lines.append(f"similar past cases that were fraud: {case.peer_rate:.0%} "
                         f"(distance to them: {case.peer_distance:.2f})")
        if case.rule_stats:
            stats = case.rule_stats
            lines.append(f"rule history: {stats.matches} matches, precision {stats.precision:.1%}, "
                         f"recall {stats.recall:.1%}")
        if case.proposal:
            p = case.proposal
            lines.append(f"proposed rule (proven narrower): {format_rule(p.rule)}")
            lines.append(f"on held-back history: precision {p.before.precision:.1%} to {p.after.precision:.1%}, "
                         f"recall {p.before.recall:.1%} to {p.after.recall:.1%}")
        lines.append("steps: " + " > ".join(case.steps))
        return "\n".join(lines)


class Investigator:
    def __init__(self, engine: RuleEngine, history: History, threshold: float = 0.2, k: int = 50,
                 precision_floor: float = 0.5, unfamiliar_distance: float = float("inf")):
        self.engine, self.history = engine, history
        self.threshold, self.k, self.precision_floor = threshold, k, precision_floor
        self.unfamiliar_distance = unfamiliar_distance
        self._stats: dict[int, RuleStats] = {}
        self._proposals: dict[int, Proposal | None] = {}

    def _next_tool(self, case: CaseFile) -> str | None:
        """The plan: each answer decides whether the next, costlier question is needed."""
        if case.trace is None and case.decision.rule_index is not None:
            return "explain"
        if case.what_if is None:
            return "what_if"
        if case.peer_rate is None:
            return "peers"
        if case.decision.rule_index is None:
            return None
        if case.rule_stats is None:
            return "backtest"
        if case.rule_stats.precision < self.precision_floor and not case.proposal_checked:
            return "refine"
        return None

    def _run(self, tool: str, case: CaseFile) -> None:
        index = case.decision.rule_index
        if tool == "explain":
            case.trace = trace_expr(self.engine.rules[index].condition, case.txn)
        elif tool == "what_if":
            case.what_if = counterfactuals(self.engine, case.txn)
        elif tool == "peers":
            case.peer_rate, case.peer_distance = self.history.peers(case.txn, self.k)
        elif tool == "backtest":
            if index not in self._stats:
                self._stats[index] = backtest(self.history, self.engine.rules[index].condition)
            case.rule_stats = self._stats[index]
        elif tool == "refine":
            if index not in self._proposals:
                self._proposals[index] = propose_refinement(self.history, self.engine.rules[index])
            case.proposal = self._proposals[index]
            case.proposal_checked = True

    def investigate(self, txn: Txn) -> Report:
        visible = {name: txn[name] for name in SCHEMA}  # The label cannot reach any tool.
        case = CaseFile(visible, self.engine.decide(visible))
        while (tool := self._next_tool(case)) is not None:
            case.steps.append(tool)
            self._run(tool, case)
        # History cannot vouch for a case unlike anything in it, so that case is never cleared.
        if case.peer_rate >= self.threshold:
            verdict = "escalate"
        elif case.peer_distance > self.unfamiliar_distance:
            verdict = "escalate_unfamiliar"
        else:
            verdict = "clear"
        return Report(verdict, case)
