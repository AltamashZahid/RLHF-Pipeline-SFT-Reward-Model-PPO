"""Generates natural-sounding (plain English, rule) examples from a wording pool.

People rarely write "the amount is over X and the device is rooted". They pack conditions into a
noun phrase ("UPI payments above 50k from rooted phones"), write checklists ("no KYC, UPI, at
least 40k"), or state policy ("we don't accept..."). This grammar produces those shapes. The
wording lives in JSON pools under phrasings/, so SFT and RL can use different wording.
"""

import json
import random
from dataclasses import dataclass
from pathlib import Path

from .nlgen import Example, _indian
from .nodes import And, BoolField, Compare, Expr, InList, Not, Or, Rule, format_rule
from .parser import parse_checked

POOLS = Path(__file__).parent / "phrasings"
_COMPARE_KEY = {">": "gt", ">=": "gte", "<": "lt", "<=": "lte"}
_OPPOSITE = {"gt": "lte", "gte": "lt", "lt": "gte", "lte": "gt"}

_NUMBER_WORDS = {
    1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight", 9: "nine",
    10: "ten", 12: "twelve", 15: "fifteen", 20: "twenty", 25: "twenty-five", 30: "thirty", 50: "fifty",
    75: "seventy-five", 100: "a hundred",
}
_AMOUNTS = (
    100, 200, 250, 500, 750, 1000, 1500, 2000, 2500, 3000, 4000, 5000, 7500, 8000, 10_000, 12_000, 15_000,
    20_000, 25_000, 30_000, 35_000, 40_000, 50_000, 60_000, 75_000, 80_000, 100_000, 150_000, 200_000,
    250_000, 300_000, 500_000, 750_000, 1_000_000, 2_000_000, 5_000_000, 10_000_000,
)
_DAYS = (1, 2, 3, 5, 7, 10, 14, 15, 21, 30, 45, 60, 90, 120, 180, 365, 730)
_DAY_NAMES = {
    7: ("a week", "1 week", "one week"), 14: ("two weeks", "2 weeks"), 21: ("three weeks", "3 weeks"),
    30: ("a month", "1 month", "one month"), 60: ("two months", "2 months"),
    90: ("three months", "3 months"), 180: ("six months", "6 months"),
    365: ("a year", "1 year", "one year"), 730: ("two years", "2 years"),
}
_VELOCITIES = (2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 25, 30, 50, 100)
_FAILURES = (1, 2, 3, 4, 5, 6, 8, 10, 15, 20)
_METHOD_WORDS = {
    "UPI": ("UPI",), "CARD": ("card",), "NETBANKING": ("net banking", "netbanking"), "WALLET": ("wallet",),
}
_CATEGORY_WORDS = {
    "GROCERY": ("grocery",), "FOOD": ("food",), "TRAVEL": ("travel",), "GAMING": ("gaming",),
    "ELECTRONICS": ("electronics",), "UTILITIES": ("utilities", "utility"),
}
_FIELDS = (
    "txn.amount", "txn.method", "txn.hour", "txn.is_international", "txn.count_1h", "device.age_days",
    "device.is_rooted", "merchant.category", "merchant.whitelisted", "user.account_age_days",
    "user.failed_attempts_24h", "user.kyc_verified",
)
_BETWEEN = {
    "pre": [],
    "post": ["between {a} and {b}", "from {a} to {b}", "{a} to {b}", "during {a} to {b}"],
    "clause": ["it is between {a} and {b}", "the time is between {a} and {b}", "it happens from {a} to {b}"],
    "item": ["between {a} and {b}", "{a} to {b}", "{a}-{b}"],
}
_EXACT_AMOUNTS = (*_AMOUNTS, 1, 999, 9_999, 49_999, 99_999, 199_999)
_EXACT = {
    "amount.eq": {
        "pre": [],
        "post": ["of exactly {v}", "for exactly {v}", "of precisely {v}", "for {v} exactly", "of {v} on the dot"],
        "clause": ["the amount is exactly {v}", "the payment is for exactly {v}", "it is exactly {v}",
                   "the amount equals {v}", "the value is precisely {v}"],
        "item": ["exactly {v}", "amount exactly {v}", "precisely {v}", "amount = {v}"],
    },
    "failed.none": {
        "pre": [],
        "post": ["with no failed attempts in the last 24 hours", "from users with zero failed attempts in 24 hours",
                 "with no failures in the past day", "from users who have not failed once in the last 24 hours"],
        "clause": ["there are no failed attempts in the last 24 hours", "the user has zero failures in the past day",
                   "nothing has failed in the last 24 hours", "the user has not failed once in the last day"],
        "item": ["no failed attempts in 24h", "zero failures in the last day", "no failures in 24 hours"],
    },
}
_ANY_OF = {
    "BLOCK": ["Block it if any of these hold: {items}.", "Reject when any of the following is true: {items}.",
              "Any one of these means decline: {items}.", "Decline if even one of these applies: {items}."],
    "ALLOW": ["Approve it if any of these hold: {items}.", "Let it through when any of these is true: {items}.",
              "Any one of these is enough to allow it: {items}.", "Pass it if at least one of these applies: {items}."],
    "REVIEW": ["Send it for review if any of these hold: {items}.", "Flag it when any of the following is true: {items}.",
               "Any one of these means manual review: {items}.", "A person should check it if even one applies: {items}."],
}
_PREFIXES = ("Rule: ", "New rule: ", "Policy: ", "Fraud rule: ", "Risk rule: ", "Note: ")


def amount_text(rng: random.Random, n: int) -> str:
    """Every form must read back as exactly n; tests/test_natural.py checks them all."""
    options = [str(n), f"₹{_indian(n)}", f"₹{n}", f"Rs {n}", f"Rs {_indian(n)}", f"Rs. {_indian(n)}",
               f"{_indian(n)} rupees", f"{n} rupees", f"INR {_indian(n)}", f"{_indian(n)} rs"]
    if n >= 1000 and n % 1000 == 0:
        k = n // 1000
        options += [f"{k}k", f"{k}K", f"₹{k}k", f"Rs {k}k", f"{k} thousand", f"{k}k rupees"]
        if k in _NUMBER_WORDS and k < 100:
            options.append(f"{_NUMBER_WORDS[k]} thousand")
    if n >= 50_000 and n % 50_000 == 0:
        lakh = n / 100_000
        text = str(int(lakh)) if lakh == int(lakh) else str(lakh)
        options += [f"{text} lakh", f"{text}L", f"₹{text} lakh", f"Rs {text}L", f"{text} lakh rupees"]
        if lakh == int(lakh) and int(lakh) in _NUMBER_WORDS:
            word = _NUMBER_WORDS[int(lakh)]
            options.append(f"{word} lakh")
    if n == 10_000_000:
        options += ["1 crore", "one crore", "1 cr"]
    return rng.choice(options)


def days_text(rng: random.Random, n: int, spell_out: bool = False) -> str:
    options = [f"{n} days", f"{n} day" if n == 1 else f"{n}d"]
    if n in _NUMBER_WORDS and n <= 10:
        options.append(f"{_NUMBER_WORDS[n]} day" if n == 1 else f"{_NUMBER_WORDS[n]} days")
    if n in _DAY_NAMES:
        options = list(_DAY_NAMES[n]) * 2 + options
        if spell_out:  # People often give the conversion: "six months (180 days)".
            name = rng.choice(_DAY_NAMES[n])
            options += [f"{name} ({n} days)", f"{n} days ({name})"]
    return rng.choice(options)


def hour_text(rng: random.Random, hour: int) -> str:
    twelve = hour % 12 or 12
    suffix = "am" if hour < 12 else "pm"
    options = [f"{twelve}{suffix}", f"{twelve} {suffix}", f"{twelve} {suffix.upper()}", f"{twelve}{suffix.upper()}",
               f"{twelve}:00 {suffix}", f"{hour:02d}:00", f"{hour:02d}00 hrs"]
    if hour == 0:
        options += ["midnight"] * 3
    if hour == 12:
        options += ["noon", "12 noon"]
    if 3 <= hour < 12:
        options.append(f"{twelve} in the morning")
    if 12 < hour < 17:
        options.append(f"{twelve} in the afternoon")
    if 17 <= hour < 21:
        options.append(f"{twelve} in the evening")
    if hour >= 21:
        options.append(f"{twelve} at night")
    return rng.choice(options)


def count_text(rng: random.Random, n: int) -> str:
    if n in _NUMBER_WORDS and n <= 20 and rng.random() < 0.25:
        return _NUMBER_WORDS[n]
    return str(n)


@dataclass(frozen=True)
class NaturalAtom:
    expr: Expr
    key: str
    fill: dict
    slug: str
    opposite: str | None = None  # A concept key that means the exact negation, with the same fill.


def _list_text(rng: random.Random, words: list[str]) -> str:
    if len(words) == 2:
        joiner = rng.choice((" or ", " and ", "/", " or "))
        return words[0] + joiner + words[1]
    return ", ".join(words[:-1]) + rng.choice((" or ", " and ", ", ")) + words[-1]


def _numeric(rng, field, prefix, values, make_fill, slug, new_days=None):
    if new_days is not None and rng.random() < 0.15:
        fill = {"v": days_text(rng, new_days)}
        return NaturalAtom(Compare(field, "<", new_days), f"{prefix}.new", fill, f"new_{slug}", f"{prefix}.gte")
    op = rng.choice(tuple(_COMPARE_KEY))
    value = rng.choice(values)
    key = _COMPARE_KEY[op]
    label = {"gt": "high", "gte": "high", "lt": "low", "lte": "low"}[key]
    return NaturalAtom(Compare(field, op, value), f"{prefix}.{key}", make_fill(rng, value), f"{label}_{slug}",
                       f"{prefix}.{_OPPOSITE[key]}")


def _bool_atom(rng, field, prefix, slugs):
    if rng.random() < 0.5:
        return NaturalAtom(BoolField(field), f"{prefix}.yes", {}, slugs[0], f"{prefix}.no")
    return NaturalAtom(Not(BoolField(field)), f"{prefix}.no", {}, slugs[1], f"{prefix}.yes")


def _enum_atom(rng, field, prefix, words, placeholder, slug):
    kind = rng.choices(("is", "is_not", "in", "not_in"), weights=(5, 2, 2, 1))[0]
    if kind in ("is", "is_not"):
        value = rng.choice(tuple(words))
        fill = {placeholder: rng.choice(words[value])}
        op = "==" if kind == "is" else "!="
        other = "is_not" if kind == "is" else "is"
        return NaturalAtom(Compare(field, op, value), f"{prefix}.{kind}", fill, f"{value.lower()}_{slug}",
                           f"{prefix}.{other}")
    values = tuple(rng.sample(tuple(words), 2 if kind == "not_in" else rng.randint(2, 3)))
    fill = {placeholder: _list_text(rng, [rng.choice(words[v]) for v in values])}
    expr = InList(field, values) if kind == "in" else Not(InList(field, values))
    other = "not_in" if kind == "in" else "in"
    return NaturalAtom(expr, f"{prefix}.{kind}", fill, f"listed_{slug}", f"{prefix}.{other}")


def _hour_atom(rng):
    style = rng.random()
    if style < 0.15:
        return NaturalAtom(Compare("txn.hour", "<", 6), "hour.night", {"t": hour_text(rng, 6)}, "night", "hour.from")
    if style < 0.27:
        start, end = rng.choice(((0, 5), (0, 6), (1, 5), (9, 17), (10, 18), (18, 22), (20, 23), (2, 6)))
        fill = {"a": hour_text(rng, start), "b": hour_text(rng, end)}
        if start == 0:
            return NaturalAtom(Compare("txn.hour", "<", end), "hour.between", fill, "early_hours")
        expr = And((Compare("txn.hour", ">=", start), Compare("txn.hour", "<", end)))
        return NaturalAtom(expr, "hour.between", fill, "window_hours")
    hour = rng.randint(1, 23)
    if rng.random() < 0.5:
        return NaturalAtom(Compare("txn.hour", "<", hour), "hour.before", {"t": hour_text(rng, hour)}, "early_hours",
                           "hour.from")
    return NaturalAtom(Compare("txn.hour", ">=", hour), "hour.from", {"t": hour_text(rng, hour)}, "late_hours",
                       "hour.before")


def sample_atom(rng: random.Random, field: str, extended: bool = False) -> NaturalAtom:
    """`extended` adds exact values and spelled-out periods; without it, draws match the older data."""
    days = lambda r, n: {"v": days_text(r, n, extended)}
    match field:
        case "txn.amount":
            if extended and rng.random() < 0.05:
                value = rng.choice(_EXACT_AMOUNTS)
                return NaturalAtom(Compare(field, "==", value), "amount.eq", {"v": amount_text(rng, value)}, "exact_amount")
            return _numeric(rng, field, "amount", _AMOUNTS, lambda r, n: {"v": amount_text(r, n)}, "amount")
        case "device.age_days":
            return _numeric(rng, field, "device_age", _DAYS, days, "device", 7)
        case "user.account_age_days":
            return _numeric(rng, field, "account_age", _DAYS, days, "account", 30)
        case "txn.count_1h":
            return _numeric(rng, field, "velocity", _VELOCITIES, lambda r, n: {"n": count_text(r, n)}, "txns")
        case "user.failed_attempts_24h":
            if extended and rng.random() < 0.1:
                return NaturalAtom(Compare(field, "==", 0), "failed.none", {}, "no_failures")
            return _numeric(rng, field, "failed", _FAILURES, lambda r, n: {"n": count_text(r, n)}, "failures")
        case "txn.hour":
            return _hour_atom(rng)
        case "device.is_rooted":
            return _bool_atom(rng, field, "rooted", ("rooted_device", "unrooted_device"))
        case "merchant.whitelisted":
            return _bool_atom(rng, field, "whitelisted", ("whitelisted_merchant", "unlisted_merchant"))
        case "user.kyc_verified":
            return _bool_atom(rng, field, "kyc", ("kyc_verified", "no_kyc"))
        case "txn.is_international":
            return _bool_atom(rng, field, "international", ("international", "domestic"))
        case "txn.method":
            return _enum_atom(rng, field, "method", _METHOD_WORDS, "m", "method")
        case "merchant.category":
            return _enum_atom(rng, field, "category", _CATEGORY_WORDS, "c", "merchant")
    raise ValueError(f"unknown field {field}")


@dataclass(frozen=True)
class WordingPool:
    nouns: list[str]
    heads: list[str]
    frames: dict
    concepts: dict
    actor: dict | None = None  # Wording for sentences about a kind of user. Turns on the wider grammar.

    @classmethod
    def load(cls, name: str) -> "WordingPool":
        data = json.loads((POOLS / f"{name}.json").read_text(encoding="utf-8"))
        concepts = {"hour.between": _BETWEEN, **_EXACT, **data["concepts"]}
        return cls(data["nouns"], data["heads"], data["frames"], concepts, data.get("actor"))

    def piece(self, rng: random.Random, atom: NaturalAtom, slot: str, key: str | None = None) -> str | None:
        options = self.concepts[key or atom.key][slot]
        return _fill(rng.choice(options), atom.fill) if options else None


def _fill(template: str, fill: dict) -> str:
    for name, value in fill.items():
        template = template.replace("{" + name + "}", value)
    return template


def _capital(text: str) -> str:
    return text[0].upper() + text[1:] if text else text


def _and_text(rng: random.Random, parts: list[str]) -> str:
    if len(parts) == 1:
        return parts[0]
    style = rng.random()
    if style < 0.55:
        return ", ".join(parts[:-1]) + " and " + parts[-1]
    if style < 0.8:
        return " and ".join(parts)
    if style < 0.9:
        return ", ".join(parts)
    return " & ".join(parts)


def _noun_phrase(rng: random.Random, pool: WordingPool, atoms: list[NaturalAtom]):
    """Returns the phrase and the atoms in the order the phrase mentions them."""
    use_head = rng.random() < 0.3
    before, after, order_before, order_after = [], [], [], []
    for atom in atoms:
        pre = None if use_head or rng.random() > 0.6 else pool.piece(rng, atom, "pre")
        if pre:
            before.append(pre)
            order_before.append(atom)
        else:
            after.append(pool.piece(rng, atom, "post"))
            order_after.append(atom)
    head = rng.choice(pool.heads) if use_head else rng.choice(pool.nouns)
    style = rng.random()
    if style < 0.5:
        tail = " ".join(after)
    elif style < 0.8:
        tail = _and_text(rng, after) if after else ""
    else:
        tail = ", ".join(after)
    return " ".join([*before, head, tail]).strip(), order_before + order_after


def _frame(rng: random.Random, pool: WordingPool, action: str, kind: str, **values: str) -> str:
    frame = rng.choice(pool.frames[action][kind])
    for name, value in values.items():
        frame = frame.replace("{" + name + "}", value).replace("{" + name.capitalize() + "}", _capital(value))
    return _capital(frame)


def _flat_and(exprs: list[Expr]) -> Expr:
    parts: list[Expr] = []
    for expr in exprs:
        parts.extend(expr.operands if isinstance(expr, And) else [expr])
    return parts[0] if len(parts) == 1 else And(tuple(parts))


def _clauses(rng, pool, atoms):
    return [pool.piece(rng, atom, "clause") for atom in atoms]


def _compose_and(rng, pool, action, atoms):
    k = len(atoms)
    choices = ["np", "clause"]
    weights = [0.5, 0.25]
    if k >= 2:
        choices += ["np_clause", "list"]
        weights += [0.2, 0.12]
        if atoms[-1].opposite:
            choices.append("np_unless")
            weights.append(0.1)
    if pool.actor is not None:
        choices.append("actor")
        weights.append(0.3)
    style = rng.choices(choices, weights=weights)[0]
    if style == "actor":
        return _compose_actor(rng, pool, action, atoms)
    if style == "np":
        phrase, order = _noun_phrase(rng, pool, atoms)
        return _frame(rng, pool, action, "np", np=phrase), order
    if style == "clause":
        return _frame(rng, pool, action, "clause", c=_and_text(rng, _clauses(rng, pool, atoms))), atoms
    if style == "list":
        return _frame(rng, pool, action, "list", items=", ".join(pool.piece(rng, a, "item") for a in atoms)), atoms
    if style == "np_unless":
        phrase, order = _noun_phrase(rng, pool, atoms[:-1])
        exception = pool.piece(rng, atoms[-1], "clause", atoms[-1].opposite)
        return _frame(rng, pool, action, "np_unless", np=phrase, c=exception), order + [atoms[-1]]
    split = rng.randint(1, k - 1)
    phrase, order = _noun_phrase(rng, pool, atoms[:split])
    condition = _and_text(rng, _clauses(rng, pool, atoms[split:]))
    return _frame(rng, pool, action, "np_clause", np=phrase, c=condition), order + atoms[split:]


def _compose_actor(rng, pool, action, atoms):
    """Policy about a kind of user: "New accounts can't pay more than 10k by UPI." The left side
    says who, the right side says what they may do, and "Only X can Y" blocks everyone who is not X."""
    actor = pool.actor
    people = [a for a in atoms if a.key in actor["with"]]
    paying = [a for a in atoms if a.key not in actor["with"]]
    order = [*people, *paying]
    if paying:
        first, *rest = paying
        pieces = [_fill(rng.choice(actor["does"][first.key]), first.fill), *(pool.piece(rng, a, "post") for a in rest)]
        does = " ".join(pieces)
    else:
        does = rng.choice(actor["generic_does"])
    if not people:
        if rng.random() < 0.5:
            return _capital(rng.choice(actor["everyone"][action]).replace("{does}", does)), order
        who = rng.choice(actor["people"])
        frame = rng.choice(actor["frames"][action])
    elif (action == "BLOCK" and len(people) == 1 and paying and rng.random() < 0.5
          and actor["who"].get(people[0].opposite)):
        who = _fill(rng.choice(actor["who"][people[0].opposite]), people[0].fill)
        frame = rng.choice(actor["only"])
    else:
        first, *rest = people
        if actor["who"].get(first.key) and rng.random() < 0.6:
            head = _fill(rng.choice(actor["who"][first.key]), first.fill)
        else:
            head = rng.choice(actor["people"]) + " " + _fill(rng.choice(actor["with"][first.key]), first.fill)
        who = " ".join([head, *(_fill(rng.choice(actor["with"][a.key]), a.fill) for a in rest)])
        frame = rng.choice(actor["frames"][action])
    text = frame.replace("{who}", who).replace("{Who}", _capital(who)).replace("{does}", does)
    return _capital(text), order


def _compose_or_wide(rng, pool, action, atoms):
    """More ways to say OR than the older grammar has."""
    if rng.random() < 0.2:
        items = rng.choice((", ", "; ")).join(pool.piece(rng, a, "item") for a in atoms)
        return rng.choice(_ANY_OF[action]).replace("{items}", items), atoms
    if rng.random() < 0.4:
        return _compose_or(rng, pool, action, atoms)
    parts = _clauses(rng, pool, atoms)
    joiner = rng.choice((" or ", ", or ", " or if ", ", or if ", " or else ", ", or else "))
    return _frame(rng, pool, action, "clause", c=", ".join(parts[:-1]) + joiner + parts[-1]), atoms


def _compose_and_of_or_wide(rng, pool, action, atoms):
    """Returns (text, other atoms, grouped atoms, group said first)."""
    head, group = atoms[:-2], atoms[-2:]
    style = rng.random()
    if style < 0.25:  # "Block payments over 2 lakh from unverified users or from rooted phones."
        phrase, order = _noun_phrase(rng, pool, head)
        first, second = (pool.piece(rng, a, "post") for a in group)
        return _frame(rng, pool, action, "np", np=f"{phrase} {first} or {second}"), order, group, False
    if style < 0.5:  # "If it's a new account or a new device and the amount is above 30000, hold."
        first, second = _clauses(rng, pool, group)
        lead = _and_text(rng, _clauses(rng, pool, head))
        options = (f"{first} or {second}, and {lead}", f"either {first} or {second}, and {lead}",
                   f"({first} or {second}) and {lead}", f"{first} or {second} and {lead}")
        return _frame(rng, pool, action, "clause", c=rng.choice(options)), head, group, True
    return (*_compose_and_of_or(rng, pool, action, atoms), False)


def _compose_or(rng, pool, action, atoms):
    if rng.random() < 0.4:
        head = rng.choice(pool.nouns + pool.heads)
        tail = " or ".join(pool.piece(rng, a, "post") for a in atoms)
        return _frame(rng, pool, action, "np", np=f"{head} {tail}"), atoms
    parts = _clauses(rng, pool, atoms)
    if len(parts) == 2 and rng.random() < 0.3:
        return _frame(rng, pool, action, "clause", c=f"either {parts[0]} or {parts[1]}"), atoms
    return _frame(rng, pool, action, "clause", c=", ".join(parts[:-1]) + " or " + parts[-1]), atoms


def _compose_and_of_or(rng, pool, action, atoms):
    head, group = atoms[:-2], atoms[-2:]
    first, second = _clauses(rng, pool, group)
    joined = rng.choice((f"{first} or {second}", f"either {first} or {second}", f"({first} or {second})"))
    if rng.random() < 0.6:
        phrase, order = _noun_phrase(rng, pool, head)
        return _frame(rng, pool, action, "np_clause", np=phrase, c=joined), order, group
    lead = _and_text(rng, _clauses(rng, pool, head))
    return _frame(rng, pool, action, "clause", c=f"{lead} and {joined}"), head, group


def _compose_or_of_and(rng, pool, action, atoms):
    first, second, third = atoms
    if rng.random() < 0.5:
        a, b, c = _clauses(rng, pool, atoms)
        text = rng.choice((f"{a} and {b}, or {c}", f"either both {a} and {b}, or {c}", f"({a} and {b}) or {c}"))
        return _frame(rng, pool, action, "clause", c=text), [first, second], third
    phrase, order = _noun_phrase(rng, pool, [first, second])
    other = f"{rng.choice(pool.heads)} {pool.piece(rng, third, 'post')}"
    return _frame(rng, pool, action, "np", np=f"{phrase}, or {other}"), order, third


def _finish(rng: random.Random, text: str) -> str:
    style = rng.random()
    if style < 0.08:
        text = rng.choice(_PREFIXES) + text
    elif style < 0.16:
        text = text.lower()
    elif style < 0.24:
        text = text.rstrip(".")
    return text


def make_natural_example(rng: random.Random, pool: WordingPool, max_clauses: int = 4) -> Example:
    k = rng.choices(range(1, max_clauses + 1), weights=(2, 4, 3, 2)[:max_clauses])[0]
    wide = pool.actor is not None
    atoms = [sample_atom(rng, field, wide) for field in rng.sample(_FIELDS, k)]
    action = rng.choices(("BLOCK", "REVIEW", "ALLOW"), weights=(5, 3, 2))[0]
    if k == 1:
        shape = "single"
    elif k == 2:
        shape = rng.choices(("and", "or"), weights=(8, 2))[0]
    elif k == 3:
        shape = rng.choices(("and", "or", "and_of_or", "or_of_and"), weights=(6, 1, 2, 1))[0]
    else:
        shape = rng.choices(("and", "and_of_or"), weights=(7, 3))[0]

    if shape in ("single", "and"):
        text, order = _compose_and(rng, pool, action, atoms)
        expr = _flat_and([a.expr for a in order])
    elif shape == "or":
        text, order = (_compose_or_wide if wide else _compose_or)(rng, pool, action, atoms)
        expr = Or(tuple(a.expr for a in order))
    elif shape == "and_of_or" and wide:
        text, order, group, group_first = _compose_and_of_or_wide(rng, pool, action, atoms)
        either = Or(tuple(a.expr for a in group))
        others = [a.expr for a in order]
        expr = _flat_and([either, *others] if group_first else [*others, either])
        order = [*group, *order] if group_first else [*order, *group]
    elif shape == "and_of_or":
        text, order, group = _compose_and_of_or(rng, pool, action, atoms)
        expr = _flat_and([*(a.expr for a in order), Or(tuple(a.expr for a in group))])
        order = [*order, *group]
    else:
        text, order, last = _compose_or_of_and(rng, pool, action, atoms)
        expr = Or((_flat_and([a.expr for a in order]), last.expr))
        order = [*order, last]

    rule = Rule(expr, action, "_".join(a.slug for a in order[:2]))
    parse_checked(format_rule(rule))  # A generator bug must fail loudly, not poison the data.
    return Example(_finish(rng, text), rule, k, shape)
