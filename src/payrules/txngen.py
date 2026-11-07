"""Synthetic transactions with planted fraud patterns.

These patterns are invented for testing. Results measured on this data say how a rule
does against these patterns, not how it would do on real payment traffic.
"""

import random

from .schema import CATEGORIES, METHODS

LABEL = "label.is_fraud"  # Not in the schema, so rules cannot read it.


def _clamp_amount(value: float) -> float:
    return round(min(max(value, 1.0), 10_000_000.0), 2)


def _legit(rng: random.Random) -> dict:
    return {
        "txn.amount": _clamp_amount(rng.lognormvariate(6.5, 1.3)),
        "txn.method": rng.choices(METHODS, weights=(70, 15, 5, 10))[0],
        "txn.hour": rng.choices(range(24), weights=[1] * 6 + [4] * 16 + [2] * 2)[0],
        "txn.is_international": rng.random() < 0.02,
        "txn.count_1h": min(int(rng.expovariate(0.8)), 1000),
        "device.age_days": min(int(rng.expovariate(1 / 300)), 10_000),
        "device.is_rooted": rng.random() < 0.01,
        "merchant.category": rng.choice(CATEGORIES),
        "merchant.whitelisted": rng.random() < 0.3,
        "user.account_age_days": min(int(rng.expovariate(1 / 700)), 10_000),
        "user.failed_attempts_24h": min(int(rng.expovariate(2.0)), 100),
        "user.kyc_verified": rng.random() < 0.9,
        LABEL: False,
    }


def _account_takeover(txn: dict, rng: random.Random) -> None:
    txn["device.age_days"] = rng.randint(0, 3)
    txn["user.failed_attempts_24h"] = rng.randint(3, 15)
    txn["txn.amount"] = _clamp_amount(rng.uniform(20_000, 200_000))
    txn["merchant.whitelisted"] = False


def _card_testing(txn: dict, rng: random.Random) -> None:
    txn["txn.method"] = "CARD"
    txn["txn.count_1h"] = rng.randint(15, 80)
    txn["txn.amount"] = _clamp_amount(rng.uniform(1, 50))
    txn["merchant.category"] = rng.choice(("GAMING", "ELECTRONICS"))


def _mule_account(txn: dict, rng: random.Random) -> None:
    txn["user.account_age_days"] = rng.randint(0, 14)
    txn["user.kyc_verified"] = False
    txn["txn.method"] = "UPI"
    txn["txn.amount"] = _clamp_amount(rng.uniform(40_000, 100_000))
    txn["txn.hour"] = rng.randint(0, 5)


_PATTERNS = (_account_takeover, _card_testing, _mule_account)


def generate_transactions(n: int, seed: int = 0, fraud_rate: float = 0.02) -> list[dict]:
    """Same n, seed and fraud_rate always produce the same transactions."""
    rng = random.Random(seed)
    txns = []
    for _ in range(n):
        txn = _legit(rng)
        if rng.random() < fraud_rate:
            rng.choice(_PATTERNS)(txn, rng)
            txn[LABEL] = True
        txns.append(txn)
    return txns
