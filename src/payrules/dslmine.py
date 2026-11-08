"""Mines a small DSL from repeated code structure in a Python codebase.

Statements with the same syntax-tree shape form a family. Whatever differs across a family
(names, attributes, constants) becomes a typed parameter, and whatever never differs is fixed,
so each family becomes one DSL operation. This finds structural repetition only: it does not
understand what the code means.

Every rewrite is checked exactly: expanding the operation with its arguments must rebuild the
original syntax tree. That is why a rewrite cannot change behaviour.
"""

import ast
import copy
import hashlib
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, field

_SKIPPED_FIELDS = {"type_comment"}


def _leaves(node: ast.AST) -> Iterable[tuple[ast.AST, str, int | None, object]]:
    """Every non-node value under `node`, in a fixed order: (owner, field, list index, value)."""
    for name, value in ast.iter_fields(node):
        if name in _SKIPPED_FIELDS:
            continue
        if isinstance(value, ast.AST):
            yield from _leaves(value)
        elif isinstance(value, list):
            for index, item in enumerate(value):
                if isinstance(item, ast.AST):
                    yield from _leaves(item)
                else:
                    yield node, name, index, item
        else:
            yield node, name, None, value


def _shape(node: ast.AST, tokens: list[str]) -> int:
    """Appends the node's shape: its structure and value types, without the values. Returns size."""
    tokens.append(type(node).__name__)
    size = 1
    for name, value in ast.iter_fields(node):
        if name in _SKIPPED_FIELDS:
            continue
        if isinstance(value, ast.AST):
            size += _shape(value, tokens)
        elif isinstance(value, list):
            tokens.append(f"[{len(value)}")
            for item in value:
                if isinstance(item, ast.AST):
                    size += _shape(item, tokens)
                else:
                    tokens.append(type(item).__name__)
        else:
            tokens.append(type(value).__name__)
    return size


@dataclass(frozen=True)
class Instance:
    source: str
    lines: tuple[int, int]
    node: ast.stmt
    values: tuple
    size: int


def candidates(tree: ast.AST, source: str, min_size: int) -> Iterable[tuple[str, Instance]]:
    for node in ast.walk(tree):
        if not isinstance(node, ast.stmt):
            continue
        tokens: list[str] = []
        size = _shape(node, tokens)
        if size < min_size:
            continue
        key = hashlib.md5(" ".join(tokens).encode()).hexdigest()
        values = tuple(value for *_, value in _leaves(node))
        yield key, Instance(source, (node.lineno, node.end_lineno), node, values, size)


def _same(a: object, b: object) -> bool:
    return type(a) is type(b) and a == b


@dataclass(frozen=True)
class Template:
    """One DSL operation. slots[i] says which parameter fills leaf i, or None if it is fixed."""

    name: str
    key: str
    example: ast.stmt
    fixed: tuple
    slots: tuple[int | None, ...]
    parameter_types: tuple[str, ...]
    size: int
    uses: int = 0

    def bind(self, instance: Instance) -> tuple | None:
        """The arguments that turn this template into `instance`, or None if it cannot."""
        if len(instance.values) != len(self.slots):
            return None
        arguments: list = [_UNSET] * len(self.parameter_types)
        for value, fixed, slot in zip(instance.values, self.fixed, self.slots):
            if slot is None:
                if not _same(value, fixed):
                    return None
            elif arguments[slot] is _UNSET:
                arguments[slot] = value
            elif not _same(arguments[slot], value):
                return None
        return tuple(arguments)

    def check_arguments(self, arguments: tuple) -> None:
        if len(arguments) != len(self.parameter_types):
            raise TypeError(f"{self.name} takes {len(self.parameter_types)} arguments, got {len(arguments)}")
        for position, (value, expected) in enumerate(zip(arguments, self.parameter_types)):
            if type(value).__name__ != expected:
                raise TypeError(f"{self.name} argument {position} must be {expected}, got {value!r}")

    def expand(self, arguments: tuple) -> ast.stmt:
        self.check_arguments(arguments)
        node = copy.deepcopy(self.example)
        for (owner, name, index, _), fixed, slot in zip(_leaves(node), self.fixed, self.slots):
            value = fixed if slot is None else arguments[slot]
            if index is None:
                setattr(owner, name, value)
            else:
                getattr(owner, name)[index] = value
        return node

    def render(self) -> str:
        parameters = ", ".join(f"p{i}: {kind}" for i, kind in enumerate(self.parameter_types))
        return f"{self.name}({parameters})\n" + _indent(ast.unparse(self._display()))

    def _display(self) -> ast.stmt:
        node = copy.deepcopy(self.example)
        for (owner, name, index, _), fixed, slot in zip(_leaves(node), self.fixed, self.slots):
            value = fixed if slot is None else (f"p{slot}" if isinstance(fixed, str) else fixed)
            if index is None:
                setattr(owner, name, value)
            else:
                getattr(owner, name)[index] = value
        return node


_UNSET = object()


def _indent(text: str) -> str:
    return "\n".join("    " + line for line in text.splitlines())


def round_trips(template: Template, instance: Instance) -> bool:
    arguments = template.bind(instance)
    if arguments is None:
        return False
    return ast.dump(template.expand(arguments)) == ast.dump(instance.node)


def anti_unify(key: str, family: list[Instance], name: str) -> Template:
    """The most specific template covering a family. Leaves that always match share a parameter."""
    columns = list(zip(*(i.values for i in family)))
    fixed, slots, parameter_types = [], [], []
    seen: dict[tuple, int] = {}
    for column in columns:
        first = column[0]
        if all(_same(value, first) for value in column):
            fixed.append(first)
            slots.append(None)
            continue
        signature = tuple((type(v).__name__, v) for v in column)
        if signature not in seen:
            seen[signature] = len(parameter_types)
            parameter_types.append(type(first).__name__)
        fixed.append(first)
        slots.append(seen[signature])
    return Template(name, key, family[0].node, tuple(fixed), tuple(slots), tuple(parameter_types),
                    family[0].size, len(family))


@dataclass
class Coverage:
    """Tracks claimed lines so that nested or overlapping statements are never counted twice."""

    claimed: dict[str, set[int]] = field(default_factory=lambda: defaultdict(set))
    nodes: int = 0
    rewrites: int = 0

    def claim(self, instance: Instance) -> bool:
        lines = range(instance.lines[0], instance.lines[1] + 1)
        taken = self.claimed[instance.source]
        if any(line in taken for line in lines):
            return False
        taken.update(lines)
        self.nodes += instance.size
        self.rewrites += 1
        return True


@dataclass(frozen=True)
class Corpus:
    families: dict[str, list[Instance]]
    total_nodes: int
    files: int
    skipped_files: int


def load_corpus(sources: Iterable[tuple[str, str]], min_size: int = 30) -> Corpus:
    families: dict[str, list[Instance]] = defaultdict(list)
    total = files = skipped = 0
    for source, text in sources:
        try:
            tree = ast.parse(text)
        except (SyntaxError, ValueError):
            skipped += 1
            continue
        files += 1
        total += sum(1 for _ in ast.walk(tree))
        for key, instance in candidates(tree, source, min_size):
            families[key].append(instance)
    return Corpus(dict(families), total, files, skipped)


def _name_for(family: list[Instance], index: int) -> str:
    node = family[0].node
    label = getattr(node, "name", None) or type(node).__name__.lower()
    return f"op{index}_{label}"


def mine(corpus: Corpus, max_templates: int) -> tuple[list[Template], Coverage]:
    """Greedy: largest saving first. Every counted rewrite has passed the round-trip check."""
    ranked = sorted(
        ((key, family) for key, family in corpus.families.items() if len(family) >= 2),
        key=lambda item: (-(item[1][0].size * (len(item[1]) - 1)), item[0]),
    )
    coverage = Coverage()
    templates: list[Template] = []
    for key, family in ranked:
        if len(templates) >= max_templates:
            break
        free = [i for i in family if not any(
            line in coverage.claimed[i.source] for line in range(i.lines[0], i.lines[1] + 1))]
        if len(free) < 2:
            continue
        template = anti_unify(key, free, _name_for(free, len(templates)))
        kept = [i for i in free if round_trips(template, i)]
        if len(kept) < 2:
            continue
        for instance in kept:
            coverage.claim(instance)
        templates.append(template)
    return templates, coverage


def apply_templates(templates: list[Template], corpus: Corpus) -> Coverage:
    """Rewrites unseen code with an already mined DSL, without learning anything from it."""
    by_key: dict[str, list[Template]] = defaultdict(list)
    for template in templates:
        by_key[template.key].append(template)
    matches = [i for key in by_key for i in corpus.families.get(key, [])]
    matches.sort(key=lambda i: -i.size)
    keys = {id(i): key for key, family in corpus.families.items() if key in by_key for i in family}
    coverage = Coverage()
    for instance in matches:
        for template in by_key[keys[id(instance)]]:
            if round_trips(template, instance):
                coverage.claim(instance)
                break
    return coverage


def exact_duplicates(train: Corpus, test: Corpus, max_entries: int | None = None) -> tuple[Coverage, int]:
    """Baseline: copy-paste detection, reusing only statements identical to training ones.

    With max_entries it keeps the most valuable statements, so it can be compared with a DSL of
    the same size. Returns the coverage and the number of entries it had to remember.
    """
    counts: dict[tuple, list] = {}
    for key, family in train.families.items():
        for instance in family:
            entry = counts.setdefault((key, instance.values), [instance.size, 0])
            entry[1] += 1
    ranked = sorted(counts, key=lambda k: (-(counts[k][0] * counts[k][1]), k[0], repr(k[1])))
    known = set(ranked if max_entries is None else ranked[:max_entries])
    matches = [i for key, family in test.families.items() for i in family if (key, i.values) in known]
    matches.sort(key=lambda i: -i.size)
    coverage = Coverage()
    for instance in matches:
        coverage.claim(instance)
    return coverage, len(known)
