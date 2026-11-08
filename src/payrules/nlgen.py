"""Generates (plain English, rule) examples.

Every phrase pool is split three ways: train phrasings (SFT), test phrasings (final evaluation
only) and RL phrasings (reward model and PPO). SFT never sees the other two, and nothing but
the final evaluation sees the test phrasings, so it measures whether a model learned the
meaning rather than the templates.

Training phrasings are deliberately varied, so SFT has to learn meaning rather than look up
templates. They never use a word that only the held-out phrasings use (tests/test_data.py).

Conventions the data teaches: a new device is under 7 days old, a new account is under
30 days old, and night means before 6 AM.
"""

import random
from dataclasses import dataclass

from .nodes import And, BoolField, Compare, Expr, InList, Not, Or, Rule, format_rule
from .parser import parse_checked

Pool = tuple[list[str], list[str], list[str]]

_STYLE_INDEX = {False: 0, True: 1, "rl": 2}


@dataclass(frozen=True)
class Atom:
    expr: Expr
    text: str
    slug: str
    unless_text: str | None = None  # Positive wording, for "... unless <this>"


@dataclass(frozen=True)
class Example:
    prompt: str
    rule: Rule
    clauses: int
    shape: str


def _pick(rng: random.Random, pool: Pool, heldout: bool) -> str:
    return rng.choice(pool[_STYLE_INDEX[heldout]])


def _indian(n: int) -> str:
    digits = str(n)
    if len(digits) <= 3:
        return digits
    rest, groups = digits[:-3], [digits[-3:]]
    while rest:
        groups.insert(0, rest[-2:])
        rest = rest[:-2]
    return ",".join(groups)


def _amount_text(rng: random.Random, n: int, heldout: bool) -> str:
    if heldout == "rl":
        options = [f"rupees {_indian(n)}", f"{_indian(n)}/-"]
        if n % 100_000 == 0:
            options.append(f"rupees {n // 100_000} lakh")
        elif n % 1000 == 0:
            options.append(f"{n // 1000}K rupees")
    elif heldout:
        options = [f"INR {_indian(n)}", f"{_indian(n)} rs"]
        if n % 100_000 == 0:
            options.append(f"{n // 100_000} lakh rupees")
        elif n % 1000 == 0:
            options.append(f"{n // 1000} thousand rupees")
    else:
        options = [f"₹{_indian(n)}", f"Rs {n}", f"{n} rupees", f"Rs. {_indian(n)}", f"{_indian(n)} rupees"]
        if n % 100_000 == 0:
            options += [f"{n // 100_000} lakh", f"₹{n // 100_000} lakh"]
        elif n % 1000 == 0:
            options += [f"₹{n // 1000}k", f"Rs {n // 1000}k"]
    return rng.choice(options)


_DAY_WORDS = {7: "a week", 14: "two weeks", 30: "a month", 365: "a year"}


def _days_text(rng: random.Random, n: int) -> str:
    if n in _DAY_WORDS and rng.random() < 0.5:
        return _DAY_WORDS[n]
    return f"{n} days"


def _time_text(rng: random.Random, hour: int, heldout: bool) -> str:
    twelve = hour % 12 or 12
    suffix = "AM" if hour < 12 else "PM"
    if heldout == "rl":
        return rng.choice([f"{twelve}:00 {suffix.lower()}", f"{hour}h"])
    if heldout:
        return rng.choice([f"{twelve}{suffix.lower()}", f"{hour:02d}00 hours"])
    if hour == 12 and rng.random() < 0.5:
        return "noon"
    return rng.choice([f"{twelve} {suffix}", f"{hour:02d}:00"])


AMOUNT: dict[str, Pool] = {
    ">": (
        ["the amount is over {v}", "the amount is above {v}", "the payment is more than {v}",
         "the value exceeds {v}", "the amount tops {v}", "the transaction value is above {v}",
         "it costs more than {v}", "the ticket size is over {v}"],
        ["the transaction is larger than {v}", "the total goes beyond {v}"],
        ["the sum is greater than {v}", "the purchase is bigger than {v}"],
    ),
    ">=": (
        ["the amount is at least {v}", "the payment is {v} or more", "the amount is not below {v}",
         "the value is {v} or more", "the transaction is worth at least {v}", "the amount is {v} and up"],
        ["the total is no less than {v}", "the amount is {v} or above"],
        ["the sum is {v} at minimum", "the purchase reaches {v}"],
    ),
    "<": (
        ["the amount is under {v}", "the amount is below {v}", "the payment is less than {v}",
         "the value is under {v}", "it costs less than {v}", "the transaction is worth less than {v}",
         "the ticket size is below {v}"],
        ["the total is smaller than {v}", "the transaction falls short of {v}"],
        ["the sum is lower than {v}", "the purchase is beneath {v}"],
    ),
    "<=": (
        ["the amount is at most {v}", "the payment is up to {v}", "the amount is {v} or less",
         "the value is not above {v}", "the payment is capped at {v}", "the transaction is worth at most {v}"],
        ["the total is no more than {v}", "the amount does not exceed {v}"],
        ["the sum is {v} at most", "the purchase stays within {v}"],
    ),
}

AGE: dict[str, Pool] = {
    ">": (
        ["the {s} is older than {v}", "the {s} is more than {v} old", "the {s} is over {v} old",
         "the {s} dates back more than {v}"],
        ["the {s} has existed for over {v}"],
        ["the {s} has an age above {v}", "the {s} was set up over {v} ago"],
    ),
    ">=": (["the {s} is at least {v} old", "the {s} is {v} old or more", "the {s} dates back at least {v}"], ["the {s} has existed for {v} or longer"],
           ["the {s} has an age of at least {v}", "the {s} was set up {v} or more ago"]),
    "<": (
        ["the {s} is less than {v} old", "the {s} is under {v} old", "the {s} is younger than {v}",
         "the {s} is not yet {v} old", "the {s} dates back less than {v}"],
        ["the {s} has existed for under {v}"],
        ["the {s} has an age below {v}", "the {s} was set up less than {v} ago"],
    ),
    "<=": (["the {s} is at most {v} old", "the {s} is {v} old or less", "the {s} dates back {v} at most"], ["the {s} is no older than {v}"],
           ["the {s} has an age of {v} or less", "the {s} was set up no more than {v} ago"]),
}

COUNT: dict[str, Pool] = {
    ">": (["there are more than {v} {s}", "there are over {v} {s}", "the user made more than {v} {s}",
           "more than {v} {s} are recorded"], ["{s} exceed {v}"],
          ["the number of {s} is greater than {v}"]),
    ">=": (["there are at least {v} {s}", "the user made at least {v} {s}", "at least {v} {s} are recorded"], ["there are {v} or more {s}"],
           ["the number of {s} is {v} or higher"]),
    "<": (["there are fewer than {v} {s}", "the user made fewer than {v} {s}", "fewer than {v} {s} are recorded"], ["there are under {v} {s}"],
          ["the number of {s} is lower than {v}"]),
    "<=": (["there are at most {v} {s}", "the user made at most {v} {s}", "there are {v} {s} or fewer",
            "at most {v} {s} are recorded"], ["there are no more than {v} {s}"],
           ["the number of {s} is {v} or lower"]),
}

HOUR: dict[str, Pool] = {
    "<": (["it is before {v}", "the payment is made before {v}", "the payment occurs before {v}",
           "it is not yet {v}"], ["it is earlier than {v}"],
          ["the time is prior to {v}"]),
    ">=": (
        ["it is {v} or later", "the payment is made at or after {v}", "the payment occurs at {v} or later",
         "it is from {v} onwards"],
        ["it is no earlier than {v}"],
        ["the time is {v} or beyond"],
    ),
}

BOOLS: dict[str, tuple[Pool, Pool, str, str]] = {
    "device.is_rooted": (
        (["the device is rooted", "the payment comes from a rooted device", "the device is jailbroken",
          "the mobile is rooted", "a rooted device is involved"],
         ["the phone has been rooted"], ["the handset is rooted"]),
        (["the device is not rooted", "the device is not jailbroken", "the mobile is not rooted",
          "the device is unrooted"], ["the phone has not been rooted"], ["the handset is not rooted"]),
        "rooted_device", "unrooted_device",
    ),
    "merchant.whitelisted": (
        (["the merchant is whitelisted", "the merchant is on the whitelist", "the merchant is trusted",
          "the merchant is pre-approved", "the merchant is in the trusted list"],
         ["the merchant appears on the allowlist"], ["the merchant has whitelist status"]),
        (["the merchant is not whitelisted", "the merchant is not on the whitelist", "the merchant is not trusted",
          "the merchant is unlisted", "the merchant is off the whitelist"],
         ["the merchant is missing from the allowlist"], ["the merchant lacks whitelist status"]),
        "whitelisted_merchant", "unlisted_merchant",
    ),
    "user.kyc_verified": (
        (["the user is KYC verified", "KYC is complete", "the user completed KYC", "KYC is done for the user"], ["the customer has passed KYC"], ["the buyer's KYC is verified"]),
        (["the user is not KYC verified", "KYC is not complete", "the user never completed KYC",
          "KYC is incomplete", "KYC is not done for the user"],
         ["the customer has not passed KYC"], ["the buyer's KYC is pending"]),
        "kyc_verified", "no_kyc",
    ),
    "txn.is_international": (
        (["the payment is international", "it is a cross-border payment", "the payment is foreign",
          "the payment originates in another country"],
         ["the payment comes from abroad"], ["the payment is overseas"]),
        (["the payment is domestic", "the payment is not international", "the payment is not cross-border",
          "it is a domestic payment", "the payment originates in the home country"],
         ["the payment stays within India"], ["the payment is local"]),
        "international", "domestic",
    ),
}

METHOD_NAMES = {"UPI": "UPI", "CARD": "card", "NETBANKING": "net banking", "WALLET": "wallet"}
CATEGORY_NAMES = {
    "GROCERY": "grocery", "FOOD": "food", "TRAVEL": "travel", "GAMING": "gaming",
    "ELECTRONICS": "electronics", "UTILITIES": "utilities",
}

METHOD_PHRASES: dict[str, Pool] = {
    "eq": (["the method is {v}", "it is {a} {v} payment", "the payment is made by {v}",
            "the payment method is {v}", "the payment runs on {v}", "{v} is the payment method"],
           ["the customer pays with {v}", "{v} is used to pay"],
           ["the payment goes via {v}"]),
    "ne": (["the method is not {v}", "it is not {a} {v} payment", "the payment method is not {v}",
            "the payment is not made by {v}", "the payment avoids {v}"],
           ["the customer does not pay with {v}"],
           ["the payment does not go via {v}"]),
    "in": (["the method is {v}", "the payment is made by {v}", "the payment method is {v}",
            "the payment runs on {v}"], ["the customer pays with {v}"],
           ["the payment goes via {v}"]),
    "not_in": (["the method is neither {x} nor {y}", "the payment runs on neither {x} nor {y}",
                "the payment method is neither {x} nor {y}"], ["the customer pays with neither {x} nor {y}"],
               ["the payment goes via neither {x} nor {y}"]),
}
CATEGORY_PHRASES: dict[str, Pool] = {
    "eq": (["the merchant category is {v}", "it is {a} {v} merchant", "the merchant is classed as {v}",
            "the merchant is {a} {v} business"], ["the merchant is in {v}"],
           ["the merchant belongs to {v}"]),
    "ne": (["the merchant category is not {v}", "the merchant is not classed as {v}",
            "the merchant is not {a} {v} business"], ["the merchant is outside {v}"],
           ["the merchant does not belong to {v}"]),
    "in": (["the merchant category is {v}", "the merchant is classed as {v}"], ["the merchant operates in {v}"],
           ["the merchant belongs to {v}"]),
    "not_in": (["the merchant category is neither {x} nor {y}", "the merchant is classed as neither {x} nor {y}"],
               ["the merchant operates in neither {x} nor {y}"],
               ["the merchant belongs to neither {x} nor {y}"]),
}

VERBS: dict[str, Pool] = {
    "BLOCK": (["block {o}", "decline {o}", "reject {o}", "cancel {o}", "disallow {o}", "halt {o}"], ["stop {o}", "deny {o}"],
              ["refuse {o}", "prevent {o}"]),
    "ALLOW": (["allow {o}", "approve {o}", "authorize {o}", "okay {o}", "process {o} normally"], ["let {o} through", "permit {o}"],
              ["clear {o}", "accept {o}"]),
    "REVIEW": (["flag {o}", "send {o} for review", "review {o}", "inspect {o}", "investigate {o}",
                "put {o} under review"],
               ["hold {o} for manual review", "escalate {o}"],
               ["queue {o} for review", "audit {o}"]),
}
OBJECTS: Pool = (["the payment", "the transaction", "payments", "it", "the order", "such payments",
                  "the payment request"], ["the charge"],
                 ["the purchase", "this transfer"])
FORMS: Pool = (
    ["{V} if {C}.", "{V} when {C}.", "If {C}, {v}.", "{V} any time {C}.", "Rule: if {C}, {v}.",
     "Anytime {C}, {v}."],
    ["{V} whenever {C}.", "{V} in cases where {C}.", "When {C}, {v}."],
    ["{V} on the condition that {C}.", "Given that {C}, {v}.", "In the event that {C}, {v}."],
)
UNLESS: Pool = (["unless"], ["except when"], ["but not if"])


def _article(word: str) -> str:
    return "an" if word[0].lower() in "aeio" else "a"


def _join(rng: random.Random, parts: list[str], word: str, heldout: bool) -> str:
    if len(parts) == 1:
        return parts[0]
    if heldout == "rl" and word == "and":
        return ", plus ".join(parts)
    if heldout == "rl" and word == "or":
        return "; alternatively, ".join(parts)
    if heldout and word == "and":
        return " and also ".join(parts)
    if heldout and word == "or":
        return ", or else ".join(parts)
    style = rng.random()
    if word == "and" and style < 0.2:
        return " and ".join(parts)
    if word == "and" and style < 0.35:
        return ", ".join(parts[:-1]) + " as well as " + parts[-1]
    if word == "or" and style < 0.2:
        return " or ".join(parts)
    if word == "or" and style < 0.35 and len(parts) == 2:
        return f"either {parts[0]} or {parts[1]}"
    return ", ".join(parts[:-1]) + f" {word} " + parts[-1]


def _amount(rng, heldout):
    op = rng.choice((">", ">=", "<", "<="))
    value = rng.choice((500, 1000, 2000, 5000, 10_000, 20_000, 25_000, 50_000, 75_000, 100_000,
                        200_000, 500_000))
    text = _pick(rng, AMOUNT[op], heldout).format(v=_amount_text(rng, value, heldout))
    return Atom(Compare("txn.amount", op, value), text, "high_amount" if ">" in op else "low_amount")


def _age(field: str, subject: str, new_days: int):
    def build(rng, heldout):
        if rng.random() < 0.2:
            pool = (["the {s} is new", "it is a new {s}", "the {s} is brand new", "the {s} is recent",
                     "the {s} is recently created"], ["the {s} was only just added"],
                    ["the {s} is freshly added"])
            text = _pick(rng, pool, heldout).format(s=subject)
            return Atom(Compare(field, "<", new_days), text, f"new_{subject}")
        op = rng.choice((">", ">=", "<", "<="))
        value = rng.choice((2, 3, 7, 14, 30, 60, 90, 180, 365))
        text = _pick(rng, AGE[op], heldout).format(s=subject, v=_days_text(rng, value))
        return Atom(Compare(field, op, value), text, f"{'old' if '>' in op else 'new'}_{subject}")

    return build


def _count(field: str, nouns: tuple[str, ...], heldout_noun: str, rl_noun: str, values: tuple[int, ...],
           slug: str):
    def build(rng, heldout):
        op = rng.choice((">", ">=", "<", "<="))
        value = rng.choice(values)
        if heldout is False:
            name = rng.choice(nouns)
        else:
            name = rl_noun if heldout == "rl" else heldout_noun
        text = _pick(rng, COUNT[op], heldout).format(s=name, v=value)
        if text[0].islower() and text.startswith(name):
            text = "the " + text
        return Atom(Compare(field, op, value), text, f"{'many' if '>' in op else 'few'}_{slug}")

    return build


def _hour(rng, heldout):
    if rng.random() < 0.2:
        pool = (["it is night time", "the payment is made at night", "the payment is made during the night",
                 "it is nighttime"], ["it happens overnight"],
                ["it is the middle of the night"])
        return Atom(Compare("txn.hour", "<", 6), _pick(rng, pool, heldout), "night")
    # Only < and >= here: "after 10 PM" is ambiguous for a whole-hour field.
    op = rng.choice(("<", ">="))
    value = rng.choice((1, 2, 4, 5, 6, 8, 9, 12, 18, 20, 21, 22, 23))
    text = _pick(rng, HOUR[op], heldout).format(v=_time_text(rng, value, heldout))
    return Atom(Compare("txn.hour", op, value), text, "early_hours" if op == "<" else "late_hours")


def _bool(field: str):
    positive, negative, pos_slug, neg_slug = BOOLS[field]

    def build(rng, heldout):
        if rng.random() < 0.5:
            return Atom(BoolField(field), _pick(rng, positive, heldout), pos_slug)
        return Atom(
            Not(BoolField(field)),
            _pick(rng, negative, heldout),
            neg_slug,
            unless_text=_pick(rng, positive, heldout),
        )

    return build


def _enum(field: str, names: dict[str, str], phrases: dict[str, Pool], slug: str):
    def build(rng, heldout):
        kind = rng.choices(("eq", "ne", "in", "not_in"), weights=(5, 2, 2, 1))[0]
        if kind in ("eq", "ne"):
            value = rng.choice(tuple(names))
            word = names[value]
            text = _pick(rng, phrases[kind], heldout).format(v=word, a=_article(word))
            op = "==" if kind == "eq" else "!="
            return Atom(Compare(field, op, value), text, f"{value.lower()}_{slug}")
        count = 2 if kind == "not_in" else rng.randint(2, 3)
        values = tuple(rng.sample(tuple(names), count))
        words = [names[v] for v in values]
        if kind == "not_in":
            text = _pick(rng, phrases[kind], heldout).format(x=words[0], y=words[1])
            return Atom(Not(InList(field, values)), text, f"other_{slug}")
        listed = ", ".join(words[:-1]) + " or " + words[-1]
        text = _pick(rng, phrases[kind], heldout).format(v=listed)
        return Atom(InList(field, values), text, f"listed_{slug}")

    return build


_BUILDERS = {
    "txn.amount": _amount,
    "txn.method": _enum("txn.method", METHOD_NAMES, METHOD_PHRASES, "method"),
    "txn.hour": _hour,
    "txn.is_international": _bool("txn.is_international"),
    "txn.count_1h": _count(
        "txn.count_1h",
        ("transactions in the last hour", "payments in the last hour", "transactions over the last hour",
         "transactions per hour"),
        "payments in the past 60 minutes", "hourly transfers",
        (2, 3, 5, 10, 15, 20, 50), "txns",
    ),
    "device.age_days": _age("device.age_days", "device", 7),
    "device.is_rooted": _bool("device.is_rooted"),
    "merchant.category": _enum("merchant.category", CATEGORY_NAMES, CATEGORY_PHRASES, "merchant"),
    "merchant.whitelisted": _bool("merchant.whitelisted"),
    "user.account_age_days": _age("user.account_age_days", "account", 30),
    "user.failed_attempts_24h": _count(
        "user.failed_attempts_24h",
        ("failed attempts in the last 24 hours", "unsuccessful attempts in the last 24 hours",
         "failed logins in the last 24 hours", "rejected attempts today",
         "wrong PIN entries in the last 24 hours", "failed OTP attempts today"),
        "failed tries in the past day", "daily declined authentications", (2, 3, 5, 10), "failures",
    ),
    "user.kyc_verified": _bool("user.kyc_verified"),
}


def _condition(rng: random.Random, atoms: list[Atom], heldout: bool) -> tuple[Expr, str, str]:
    """Combine atoms into (expression, English, shape name)."""
    k = len(atoms)
    if k == 1:
        return atoms[0].expr, atoms[0].text, "single"

    exprs = [a.expr for a in atoms]
    texts = [a.text for a in atoms]
    if k == 2:
        shape = rng.choices(("and", "or"), weights=(8, 2))[0]
    elif k == 3:
        shape = rng.choices(("and", "or", "and_of_or", "or_of_and"), weights=(6, 1, 2, 1))[0]
    else:
        shape = rng.choices(("and", "and_of_or"), weights=(7, 3))[0]

    if shape == "and":
        if atoms[-1].unless_text and rng.random() < 0.5:
            head = _join(rng, texts[:-1], "and", heldout)
            return And(tuple(exprs)), f"{head} {_pick(rng, UNLESS, heldout)} {atoms[-1].unless_text}", "and"
        return And(tuple(exprs)), _join(rng, texts, "and", heldout), "and"
    if shape == "or":
        return Or(tuple(exprs)), _join(rng, texts, "or", heldout), "or"
    if shape == "and_of_or":
        head = _join(rng, texts[:-2], "and", heldout)
        group = Or((exprs[-2], exprs[-1]))
        if heldout == "rl":
            text = f"{head}, with {texts[-2]} or {texts[-1]}"
        elif heldout:
            text = f"{head}, provided that {texts[-2]} or {texts[-1]}"
        else:
            text = rng.choice((f"{head} and either {texts[-2]} or {texts[-1]}",
                               f"{head}, as long as either {texts[-2]} or {texts[-1]}"))
        return And((*exprs[:-2], group)), text, "and_of_or"
    group = And((exprs[0], exprs[1]))
    if heldout == "rl":
        text = f"{texts[0]} combined with {texts[1]}, or {texts[2]}"
    elif heldout:
        text = f"{texts[0]} along with {texts[1]}, or {texts[2]}"
    else:
        text = rng.choice((f"both {texts[0]} and {texts[1]}, or {texts[2]}",
                           f"either both {texts[0]} and {texts[1]}, or {texts[2]}"))
    return Or((group, exprs[2])), text, "or_of_and"


def _sentence(rng: random.Random, action: str, condition: str, heldout: bool) -> str:
    verb = _pick(rng, VERBS[action], heldout).format(o=_pick(rng, OBJECTS, heldout))
    form = _pick(rng, FORMS, heldout)
    text = form.format(V=verb[0].upper() + verb[1:], v=verb, C=condition)
    return text[0].upper() + text[1:]


def make_example(rng: random.Random, heldout: bool | str = False, max_clauses: int = 4) -> Example:
    weights = (2, 4, 3, 2)[:max_clauses]
    k = rng.choices(range(1, max_clauses + 1), weights=weights)[0]
    fields = rng.sample(tuple(_BUILDERS), k)
    atoms = [_BUILDERS[f](rng, heldout) for f in fields]
    expr, condition, shape = _condition(rng, atoms, heldout)
    action = rng.choices(("BLOCK", "REVIEW", "ALLOW"), weights=(5, 3, 2))[0]
    reason = "_".join(a.slug for a in atoms[:2])
    rule = Rule(expr, action, reason)
    parse_checked(format_rule(rule))  # A generator bug must fail loudly, not poison the data.
    return Example(_sentence(rng, action, condition, heldout), rule, k, shape)
