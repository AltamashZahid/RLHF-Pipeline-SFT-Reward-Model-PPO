"""Measures per-transaction decision latency for a ruleset on this machine."""

import argparse
import statistics
import time

from payrules import RuleEngine, parse_checked
from payrules.txngen import generate_transactions

TEMPLATES = (
    "WHEN txn.method == UPI AND txn.amount > {a} AND device.age_days < 7 "
    'AND NOT merchant.whitelisted THEN BLOCK reason="high_value_new_device"',
    "WHEN txn.count_1h > {c} AND txn.amount < 50 AND merchant.category IN [GAMING, ELECTRONICS] "
    'THEN REVIEW reason="card_testing"',
    "WHEN (user.account_age_days < 14 AND NOT user.kyc_verified) OR user.failed_attempts_24h > {f} "
    'THEN BLOCK reason="risky_account"',
)


def build_rules(count: int):
    rules = []
    for i in range(count):
        template = TEMPLATES[i % len(TEMPLATES)]
        rules.append(parse_checked(template.format(a=50_000 + i * 100, c=15 + i, f=5 + i % 90)))
    return rules


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rules", type=int, default=100)
    parser.add_argument("--txns", type=int, default=100_000)
    args = parser.parse_args()

    engine = RuleEngine(build_rules(args.rules))
    txns = generate_transactions(args.txns, seed=1)

    samples = []
    for txn in txns:
        start = time.perf_counter_ns()
        engine.decide(txn)
        samples.append(time.perf_counter_ns() - start)

    samples.sort()
    pick = lambda q: samples[int(q * (len(samples) - 1))] / 1000
    print(f"{args.rules} rules, {args.txns} transactions, one thread")
    print(f"mean {statistics.fmean(samples) / 1000:.1f} us")
    print(f"p50  {pick(0.50):.1f} us")
    print(f"p99  {pick(0.99):.1f} us")
    print(f"max  {samples[-1] / 1000:.1f} us")


if __name__ == "__main__":
    main()
