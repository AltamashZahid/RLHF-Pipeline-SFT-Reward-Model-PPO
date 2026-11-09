from payrules.schema import SCHEMA, FieldType
from payrules.txngen import LABEL, generate_transactions


def test_same_seed_gives_same_data():
    assert generate_transactions(200, seed=9) == generate_transactions(200, seed=9)
    assert generate_transactions(200, seed=9) != generate_transactions(200, seed=10)


def test_every_value_is_inside_the_schema():
    for txn in generate_transactions(5000, seed=2, fraud_rate=0.3):
        assert set(txn) == set(SCHEMA) | {LABEL}
        for name, spec in SCHEMA.items():
            value = txn[name]
            if spec.type is FieldType.BOOL:
                assert isinstance(value, bool)
            elif spec.type is FieldType.ENUM:
                assert value in spec.values
            else:
                assert spec.min <= value <= spec.max, (name, value)
                if spec.type is FieldType.INT:
                    assert isinstance(value, int)


def test_fraud_rate_is_roughly_as_requested():
    txns = generate_transactions(20000, seed=4, fraud_rate=0.05)
    rate = sum(t[LABEL] for t in txns) / len(txns)
    assert 0.04 < rate < 0.06


def test_the_newer_fields_do_not_change_older_values():
    """The older results used these exact transactions, so the extra stream must stay separate."""
    import hashlib

    older = ("txn.amount", "txn.method", "txn.hour", "txn.is_international", "txn.count_1h", "device.age_days",
             "device.is_rooted", "merchant.category", "merchant.whitelisted", "user.account_age_days",
             "user.failed_attempts_24h", "user.kyc_verified", LABEL)
    digest = hashlib.sha256()
    for txn in generate_transactions(500, seed=7):
        digest.update(repr([txn[name] for name in older]).encode())
    assert digest.hexdigest() == "c93cc45a3ac288db9ca51f115aa164f178d75da3e6adc1b664ef253386cab240"
