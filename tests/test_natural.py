"""A wrong label here would teach the model something false, so values are read back independently."""

import hashlib
import random
import re

import pytest

from payrules import equivalent, parse_checked
from payrules.natural import (
    _AMOUNTS,
    _DAYS,
    WordingPool,
    amount_text,
    days_text,
    hour_text,
    make_natural_example,
    sample_atom,
)
from payrules.nodes import And, BoolField, Compare, Not, Or, Rule, format_expr, format_rule

WORD_VALUES = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "twelve": 12, "fifteen": 15, "twenty": 20, "twenty-five": 25, "thirty": 30, "fifty": 50, "seventy-five": 75,
    "a hundred": 100, "a": 1,
}


def read_amount(text: str) -> float:
    """A separate reader, written from how people say these amounts, not from the generator."""
    cleaned = text.replace("₹", "").replace("Rs.", "").replace("INR", "").replace("Rs", "")
    cleaned = cleaned.replace("rupees", "").replace(" rs", "").replace(",", "").strip()
    if cleaned.endswith("crore") or cleaned.endswith(" cr"):
        head = cleaned.split()[0]
        return float(WORD_VALUES.get(head, head)) * 10_000_000
    match = re.fullmatch(r"([\d.]+|[a-z-]+(?: [a-z]+)??) ?(lakh|L|thousand|k|K)?", cleaned)
    assert match, text
    head, unit = match.groups()
    number = float(WORD_VALUES[head]) if head in WORD_VALUES else float(head)
    return number * {"lakh": 100_000, "L": 100_000, "thousand": 1000, "k": 1000, "K": 1000, None: 1}[unit]


def read_days(text: str) -> int:
    number, *unit = text.replace("d", " d", 1).split() if re.fullmatch(r"\d+d", text) else text.split()
    value = WORD_VALUES[number] if number in WORD_VALUES else int(number)
    return value * {"d": 1, "day": 1, "days": 1, "week": 7, "weeks": 7, "month": 30, "months": 30,
                    "year": 365, "years": 365}[unit[0]]


def read_hour(text: str) -> int:
    if text == "midnight":
        return 0
    if text in ("noon", "12 noon"):
        return 12
    match = re.fullmatch(r"(\d{2})(?::00|00 hrs)", text)
    if match:
        return int(match.group(1))
    match = re.fullmatch(r"(\d{1,2})(?::00)? ?(am|pm|AM|PM)", text)
    if match:
        hour, part = int(match.group(1)), match.group(2).lower()
        return hour % 12 + (12 if part == "pm" else 0)
    match = re.fullmatch(r"(\d{1,2}) (in the morning|in the afternoon|in the evening|at night)", text)
    assert match, text
    hour, part = int(match.group(1)), match.group(2)
    return hour % 12 + (0 if part == "in the morning" else 12)


@pytest.mark.parametrize("n", _AMOUNTS)
def test_every_amount_form_reads_back_exactly(n):
    rng = random.Random(n)
    for _ in range(200):
        text = amount_text(rng, n)
        assert read_amount(text) == n, text


@pytest.mark.parametrize("n", _DAYS)
def test_every_day_form_reads_back_exactly(n):
    rng = random.Random(n)
    for _ in range(100):
        text = days_text(rng, n)
        assert read_days(text) == n, text


@pytest.mark.parametrize("hour", range(24))
def test_every_hour_form_reads_back_exactly(hour):
    rng = random.Random(hour)
    for _ in range(100):
        text = hour_text(rng, hour)
        assert read_hour(text) == hour, text


CONCEPT_KEYS = sorted({
    *(f"{p}.{o}" for p in ("amount", "device_age", "account_age", "velocity", "failed") for o in ("gt", "gte", "lt", "lte")),
    "device_age.new", "account_age.new", "hour.before", "hour.from", "hour.night",
    *(f"{p}.{v}" for p in ("rooted", "whitelisted", "kyc", "international") for v in ("yes", "no")),
    *(f"{p}.{k}" for p in ("method", "category") for k in ("is", "is_not", "in", "not_in")),
})


def small_pool() -> WordingPool:
    concepts = {}
    for key in CONCEPT_KEYS:
        token = key.replace(".", "_")
        slot = "{" + {"amount": "v", "device_age": "v", "account_age": "v", "velocity": "n", "failed": "n",
                      "hour": "t", "method": "m", "category": "c"}.get(key.split(".")[0], "") + "}"
        slot = "" if slot == "{}" or key in ("device_age.new", "account_age.new", "hour.night") else slot
        concepts[key] = {
            "pre": [f"{token}pre"],
            "post": [f"{token}post{slot}"],
            "clause": [f"{token}clause{slot}"],
            "item": [f"{token}item{slot}"],
        }
    concepts["hour.between"] = {"pre": [], "post": ["btw {a} {b}"], "clause": ["btw {a} {b}"], "item": ["btw {a} {b}"]}
    frames = {action: {
        "np": [f"{action} {{np}}."], "clause": [f"{action} if {{c}}."], "np_clause": [f"{action} {{np}} if {{c}}."],
        "np_unless": [f"{action} {{np}} unless {{c}}."], "list": [f"{action}: {{items}}."],
    } for action in ("BLOCK", "ALLOW", "REVIEW")}
    return WordingPool(["payments"], ["anything"], frames, concepts)


def test_generated_rules_are_valid_and_deterministic():
    pool = small_pool()
    first = [make_natural_example(random.Random(i), pool) for i in range(1500)]
    second = [make_natural_example(random.Random(i), pool) for i in range(1500)]
    assert [e.prompt for e in first] == [e.prompt for e in second]
    shapes = {e.shape for e in first}
    assert shapes == {"single", "and", "or", "and_of_or", "or_of_and"}
    for example in first:
        assert "{" not in example.prompt, example.prompt


def test_action_word_in_frame_matches_the_rule():
    pool = small_pool()
    for seed in range(800):
        example = make_natural_example(random.Random(seed), pool)
        assert example.rule.action in example.prompt.upper()


def test_unmentioned_concepts_never_enter_the_rule():
    """Each atom's marker must be in the text: nothing in the rule is left unsaid."""
    pool = small_pool()
    for seed in range(800):
        example = make_natural_example(random.Random(seed), pool)
        rule_fields = set(re.findall(r"(?:txn|device|merchant|user)\.\w+", str(example.rule)))
        assert len(rule_fields) == example.clauses
        markers = {
            "txn.amount": "amount_", "device.age_days": "device_age_", "user.account_age_days": "account_age_",
            "txn.count_1h": "velocity_", "user.failed_attempts_24h": "failed_", "device.is_rooted": "rooted_",
            "merchant.whitelisted": "whitelisted_", "user.kyc_verified": "kyc_",
            "txn.is_international": "international_", "txn.method": "method_", "merchant.category": "category_",
        }
        for field in rule_fields - {"txn.hour"}:
            assert markers[field] in example.prompt.lower(), (field, example.prompt)


@pytest.mark.parametrize("field", ["txn.amount", "device.age_days", "user.account_age_days", "txn.count_1h",
                                   "user.failed_attempts_24h", "txn.hour", "device.is_rooted",
                                   "merchant.whitelisted", "user.kyc_verified", "txn.is_international",
                                   "txn.method", "merchant.category"])
def test_opposite_key_really_is_the_exact_negation(field):
    rng = random.Random(1)
    pool = small_pool()
    seen = 0
    for _ in range(300):
        atom = sample_atom(rng, field)
        if atom.opposite is None:
            continue
        seen += 1
        family, other = atom.opposite.split(".")
        rebuilt = _rebuild(field, other, atom)
        rule_a = Rule(atom.expr, "BLOCK", "r")
        rule_b = Rule(Not(rebuilt), "BLOCK", "r")
        assert equivalent(rule_a, rule_b).equivalent, (atom.key, atom.opposite)
        assert atom.opposite in pool.concepts
    assert seen > 0


def _rebuild(field, other, atom):
    """The expression the opposite key describes, derived from its name alone."""
    expr = atom.expr
    negated = expr.operand if isinstance(expr, Not) else None
    if isinstance(expr, Compare):
        value = expr.value
        if other in ("gt", "gte", "lt", "lte"):
            return Compare(field, {"gt": ">", "gte": ">=", "lt": "<", "lte": "<="}[other], value)
        if other == "from":
            return Compare(field, ">=", value)
        if other == "before":
            return Compare(field, "<", value)
        if other == "is":
            return Compare(field, "==", value)
        if other == "is_not":
            return Compare(field, "!=", value)
    if other == "yes":
        return negated
    if other == "no":
        return Not(expr)
    if other == "not_in":
        return Not(expr)
    if other == "in":
        return negated
    raise AssertionError((field, other, atom))


def test_hour_window_keeps_its_two_ends():
    rng = random.Random(0)
    windows = [a for a in (sample_atom(rng, "txn.hour") for _ in range(400)) if isinstance(a.expr, And)]
    assert windows
    for atom in windows:
        low, high = atom.expr.operands
        assert low.op == ">=" and high.op == "<" and low.value < high.value
        assert read_hour(atom.fill["a"]) == low.value and read_hour(atom.fill["b"]) == high.value


def test_groups_stay_grouped():
    pool = small_pool()
    seen = {"and_of_or": 0, "or_of_and": 0}
    for seed in range(1500):
        example = make_natural_example(random.Random(seed), pool)
        condition = example.rule.condition
        if example.shape == "and_of_or":
            assert isinstance(condition, And) and isinstance(condition.operands[-1], Or)
            seen["and_of_or"] += 1
        if example.shape == "or_of_and":
            assert isinstance(condition, Or) and isinstance(condition.operands[0], And)
            seen["or_of_and"] += 1
        parse_checked(f'WHEN {str(_fmt(condition))} THEN BLOCK reason="r"')
    assert all(seen.values())


def _fmt(condition):
    from payrules.nodes import format_expr
    return format_expr(condition)


@pytest.mark.parametrize("name", ["sft_natural", "sft_formal", "sft_ops", "rl_natural"])
def test_real_pool_generates_valid_text_without_leftover_placeholders(name):
    pool = WordingPool.load(name)
    assert set(CONCEPT_KEYS) <= set(pool.concepts)
    rng = random.Random(0)
    for _ in range(3000):
        example = make_natural_example(rng, pool)
        assert "{" not in example.prompt and "}" not in example.prompt, example.prompt


@pytest.mark.parametrize("name", ["sft_natural", "sft_formal", "sft_ops"])
def test_real_pools_share_no_wording_so_rl_has_something_new_to_teach(name):
    sft, rl = WordingPool.load(name), WordingPool.load("rl_natural")
    for key in CONCEPT_KEYS:
        for slot in ("post", "clause"):
            assert not set(sft.concepts[key][slot]) & set(rl.concepts[key][slot]), (key, slot)
    for action, frames in sft.frames.items():
        for kind, entries in frames.items():
            assert not set(entries) & set(rl.frames[action][kind]), (action, kind)


def test_pools_without_actor_wording_still_generate_the_older_data():
    """The v3 evaluation splits must stay reproducible, so the wider grammar cannot touch their draws."""
    pool = WordingPool.load("rl_natural")
    assert pool.actor is None
    digest = hashlib.sha256()
    for seed in range(2000):
        example = make_natural_example(random.Random(seed), pool)
        digest.update(f"{example.prompt}\t{format_rule(example.rule)}\n".encode("utf-8"))
    assert digest.hexdigest() == "9873393e21b373c38f091bd0d298327c4e266a45bf00e0d92f8f6e3a0f266c7f"


PEOPLE = ("kyc", "account_age", "device_age", "rooted", "failed", "velocity")
EXACT_KEYS = ("amount.eq", "failed.none")
PLACEHOLDER = {"amount": "v", "device_age": "v", "account_age": "v", "velocity": "n", "failed": "n", "hour": "t",
               "method": "m", "category": "c"}


def _slot(key: str) -> str:
    if key in ("device_age.new", "account_age.new", "hour.night", "failed.none"):
        return ""
    if key == "hour.between":
        return "{a}{b}"
    name = PLACEHOLDER.get(key.split(".")[0])
    return "{" + name + "}" if name else ""


def actor_pool() -> WordingPool:
    """Each piece carries its own key, so a test can read which meaning the text actually states."""
    base = small_pool()
    keys = [*CONCEPT_KEYS, *EXACT_KEYS, "hour.between"]
    concepts = dict(base.concepts)
    for key in EXACT_KEYS:
        token = key.replace(".", "_")
        concepts[key] = {slot: [f"{token}{slot}{_slot(key)}"] for slot in ("pre", "post", "clause", "item")}
    who = {k: [f"[who:{k}]{_slot(k)}"] for k in keys if k.split(".")[0] in PEOPLE}
    actor = {
        "people": ["folks"],
        "generic_does": ["transact"],
        "frames": {a: [f"{a} {{who}} {{does}}"] for a in ("BLOCK", "ALLOW", "REVIEW")},
        "only": ["BLOCK only {who} {does}"],
        "everyone": {a: [f"{a} everyone {{does}}"] for a in ("BLOCK", "ALLOW", "REVIEW")},
        "who": who,
        "with": {k: [f"[with:{k}]{_slot(k)}"] for k in who},
        "does": {k: [f"[does:{k}]{_slot(k)}"] for k in keys if k not in who},
    }
    return WordingPool(base.nouns, base.heads, base.frames, concepts, actor)


def _negated(atom):
    return atom.operand if isinstance(atom, Not) else Not(atom)


def test_only_sentences_block_everyone_who_is_not_named():
    """"Only verified users can pay abroad" means BLOCK when the user is not verified and pays abroad."""
    pool = actor_pool()
    seen = 0
    for seed in range(6000):
        example = make_natural_example(random.Random(seed), pool)
        if " only " not in f" {example.prompt.lower()} ":
            continue
        seen += 1
        assert example.rule.action == "BLOCK"
        stated = re.search(r"\[who:(\w+)\.(\w+)\]", example.prompt).groups()
        field = {"kyc": "user.kyc_verified", "rooted": "device.is_rooted", "account_age": "user.account_age_days",
                 "device_age": "device.age_days", "failed": "user.failed_attempts_24h",
                 "velocity": "txn.count_1h"}[stated[0]]
        operands = example.rule.condition.operands
        atom = next(a for a in operands if field in format_expr(a))
        if stated[1] == "yes":
            assert atom == Not(BoolField(field))
        elif stated[1] == "no":
            assert atom == BoolField(field)
        else:
            assert atom.op == {"gt": "<=", "gte": "<", "lt": ">=", "lte": ">"}[stated[1]], example.prompt
    assert seen > 50


def test_actor_sentences_state_every_condition_in_the_rule():
    pool = actor_pool()
    seen = 0
    for seed in range(6000):
        example = make_natural_example(random.Random(seed), pool)
        if "[who:" not in example.prompt and "[with:" not in example.prompt and "[does:" not in example.prompt:
            continue
        seen += 1
        fields = set(re.findall(r"(?:txn|device|merchant|user)\.\w+", format_rule(example.rule)))
        assert len(fields) == example.clauses, example.prompt
        assert example.rule.action in example.prompt.upper()
    assert seen > 500


def test_people_conditions_never_become_activities():
    pool = actor_pool()
    assert not set(pool.actor["does"]) & set(pool.actor["with"])
    assert set(pool.actor["does"]) | set(pool.actor["with"]) >= {*CONCEPT_KEYS, *EXACT_KEYS, "hour.between"}


def test_exact_values_and_no_failures_are_equality_rules():
    rng = random.Random(0)
    amounts = [a for a in (sample_atom(rng, "txn.amount", True) for _ in range(1000)) if a.key == "amount.eq"]
    assert amounts and all(a.expr.op == "==" and read_amount(a.fill["v"]) == a.expr.value for a in amounts)
    failures = [a for a in (sample_atom(rng, "user.failed_attempts_24h", True) for _ in range(500)) if a.key == "failed.none"]
    assert failures and all(a.expr == Compare("user.failed_attempts_24h", "==", 0) for a in failures)


def test_the_older_grammar_never_produces_exact_values():
    rng = random.Random(0)
    assert all(sample_atom(rng, "txn.amount").key != "amount.eq" for _ in range(2000))


@pytest.mark.parametrize("n", _DAYS)
def test_spelled_out_periods_agree_with_themselves(n):
    """"six months (180 days)" must state the same number twice, and that number must be n."""
    rng = random.Random(n)
    for _ in range(100):
        text = days_text(rng, n, spell_out=True)
        for part in re.split(r" \(|\)", text):
            if part:
                assert read_days(part) == n, text


def test_a_group_said_first_is_still_grouped():
    pool = actor_pool()
    seen = 0
    for seed in range(4000):
        example = make_natural_example(random.Random(seed), pool)
        if example.shape != "and_of_or":
            continue
        condition = example.rule.condition
        groups = [a for a in condition.operands if isinstance(a, Or)]
        assert isinstance(condition, And) and len(groups) == 1 and len(groups[0].operands) == 2
        seen += isinstance(condition.operands[0], Or)
    assert seen > 10


def test_or_rules_stay_or_rules_in_the_wider_grammar():
    pool = actor_pool()
    seen = 0
    for seed in range(4000):
        example = make_natural_example(random.Random(seed), pool)
        if example.shape == "or":
            seen += 1
            assert isinstance(example.rule.condition, Or)
    assert seen > 50


@pytest.mark.parametrize("name", ["sft_natural", "sft_formal", "sft_ops"])
def test_the_sft_pools_generate_the_wider_grammar_without_leftover_placeholders(name):
    pool = WordingPool.load(name)
    assert pool.actor is not None
    rng = random.Random(0)
    fields_used = set()
    for _ in range(6000):
        example = make_natural_example(rng, pool)
        assert "{" not in example.prompt and "}" not in example.prompt, example.prompt
        parse_checked(format_rule(example.rule))
        fields_used |= set(re.findall(r"(?:txn|device|merchant|user)\.\w+", format_rule(example.rule)))
    assert len(fields_used) == 12
