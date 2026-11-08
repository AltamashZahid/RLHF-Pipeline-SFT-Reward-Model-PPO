"""Training phrasings must not borrow wording from the held-out phrasings.

If they did, "accuracy on new phrasings" would partly measure memorised wording. The words
below were shared before the training phrasings were expanded (articles, field nouns, values);
any other word that training text shares with test or RL text is a leak.
"""

import random
import re

from payrules import nlgen
from payrules.nlgen import make_example

ALLOWED_SHARED = frozenset("""
a above account am amount an and are at banking below card comes days device electronics failed
food for from gaming grocery hours if in is it k kyc lakh least less merchant month more most
neither net night nor not older on or over payment payments pm review rooted rs rupees than the
there time to transaction travel two under up upi utilities verified wallet week weeks when
whitelist year
""".split())


def _vocabulary(style, count=8000) -> set[str]:
    rng = random.Random(0)
    words: set[str] = set()
    for _ in range(count):
        words |= set(re.findall(r"[a-z]+", make_example(rng, heldout=style).prompt.lower()))
    return words


def test_training_text_shares_no_new_words_with_heldout_text():
    train = _vocabulary(False)
    for style in (True, "rl"):
        leaked = (train & _vocabulary(style)) - ALLOWED_SHARED
        assert not leaked, (style, sorted(leaked))


def _pools():
    for table in (nlgen.AMOUNT, nlgen.AGE, nlgen.COUNT, nlgen.HOUR, nlgen.METHOD_PHRASES,
                  nlgen.CATEGORY_PHRASES, nlgen.VERBS):
        yield from table.values()
    for positive, negative, _, _ in nlgen.BOOLS.values():
        yield positive
        yield negative
    yield from (nlgen.OBJECTS, nlgen.FORMS, nlgen.UNLESS)


def test_no_training_template_is_also_a_heldout_template():
    for train, test, rl in _pools():
        assert not set(train) & (set(test) | set(rl)), train
