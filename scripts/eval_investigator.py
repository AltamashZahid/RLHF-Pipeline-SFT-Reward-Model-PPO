"""Measures the investigation agent on transactions it has never seen.

The fraud patterns are planted by txngen, so high numbers here show the agent works, not that
it would catch real fraud. The novel-pattern section is the honest stress test: fraud shaped
unlike anything in its history. The escalation threshold is chosen on a validation stream.
"""

import argparse
import json
import random
import statistics
import time
from pathlib import Path

from payrules import RuleEngine, parse_checked
from payrules.investigate import History, Investigator, backtest
from payrules.nodes import format_rule
from payrules.txngen import LABEL, generate_transactions

# Deliberately broad starting rules, the kind a team writes first. Fixed before any results.
STARTING_RULES = (
    'WHEN txn.amount > 20000 AND NOT merchant.whitelisted THEN BLOCK reason="large_unlisted"',
    'WHEN txn.count_1h > 5 THEN REVIEW reason="high_velocity"',
    'WHEN user.account_age_days < 30 THEN REVIEW reason="new_account"',
)


def novel_fraud(txn: dict, rng: random.Random) -> dict:
    """A pattern absent from every history: mid-sized wallet payments from rooted devices."""
    return {
        **txn,
        "device.is_rooted": True,
        "txn.method": "WALLET",
        "txn.amount": round(rng.uniform(21_000, 30_000), 2),
        "merchant.whitelisted": False,
        "merchant.category": "TRAVEL",
        LABEL: True,
    }


def novel_fraud_second(txn: dict, rng: random.Random) -> dict:
    """Added after the unfamiliar-case check was designed, so that check is tested fresh."""
    return {
        **txn,
        "txn.method": "NETBANKING",
        "txn.is_international": True,
        "txn.hour": rng.randint(1, 4),
        "txn.amount": round(rng.uniform(25_000, 45_000), 2),
        "merchant.whitelisted": False,
        "merchant.category": "ELECTRONICS",
        "user.kyc_verified": True,
        LABEL: True,
    }


def novel_fraud_third(txn: dict, rng: random.Random) -> dict:
    """Added after feature scaling was changed, so that change is tested on a fresh pattern."""
    return {
        **txn,
        "txn.method": "CARD",
        "txn.amount": round(rng.uniform(21_000, 60_000), 2),
        "merchant.whitelisted": False,
        "merchant.category": "UTILITIES",
        "user.account_age_days": rng.randint(1500, 3000),
        "user.failed_attempts_24h": rng.randint(1, 2),
        "device.age_days": rng.randint(4, 10),
        LABEL: True,
    }


def triage(investigator, alerts) -> dict:
    fraud = sum(t[LABEL] for t in alerts)
    escalated = caught = 0
    for txn in alerts:
        if investigator.investigate(txn).verdict != "clear":
            escalated += 1
            caught += txn[LABEL]
    return {
        "alerts": len(alerts),
        "fraud_in_alerts": fraud,
        "alert_precision_before": fraud / max(len(alerts), 1),
        "escalated": escalated,
        "alert_precision_after": caught / max(escalated, 1),
        "fraud_caught": caught / max(fraud, 1),
        "analyst_workload_cut": 1 - escalated / max(len(alerts), 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history", type=int, default=100_000)
    parser.add_argument("--stream", type=int, default=50_000)
    parser.add_argument("--out", type=Path, default=Path("runs/investigator.json"))
    args = parser.parse_args()

    history = History(generate_transactions(args.history, seed=101))
    engine = RuleEngine([parse_checked(text) for text in STARTING_RULES])

    def alerts_from(seed: int) -> list[dict]:
        txns = generate_transactions(args.stream, seed=seed)
        return [t for t in txns if engine.decide(t).rule_index is not None]

    # Threshold: the highest that still catches at least 99% of fraud on the validation stream.
    probe = Investigator(engine, history, threshold=0.0)
    validation = alerts_from(202)
    peers = [(t, *probe.history.peers(t)) for t in validation]
    rates = sorted(rate for t, rate, _ in peers if t[LABEL])
    threshold = max(rates[int(0.01 * len(rates))], 0.02)
    # Unfamiliar: farther from its peers than 99% of the alerts seen in validation.
    distances = sorted(distance for _, _, distance in peers)
    unfamiliar = distances[int(0.99 * len(distances))]
    investigator = Investigator(engine, history, threshold=threshold, unfamiliar_distance=unfamiliar)

    test_alerts = alerts_from(303)
    results: dict = {"threshold": threshold, "unfamiliar_distance": unfamiliar, "triage_test": triage(investigator, test_alerts)}

    stream = generate_transactions(args.stream, seed=303)
    test_history = History(stream)
    rng = random.Random(404)
    novel = [novel_fraud(t, rng) for t in stream[:300]]
    novel = [t for t in novel if engine.decide(t).rule_index is not None]
    results["triage_novel_pattern"] = triage(investigator, novel)
    results["novel_pattern_without_unfamiliar_check"] = triage(
        Investigator(engine, history, threshold=threshold), novel
    )
    second = [novel_fraud_second(t, rng) for t in stream[300:600]]
    second = [t for t in second if engine.decide(t).rule_index is not None]
    results["triage_second_novel_pattern"] = triage(investigator, second)
    third = [novel_fraud_third(t, rng) for t in stream[600:900]]
    third = [t for t in third if engine.decide(t).rule_index is not None]
    results["triage_third_novel_pattern"] = triage(investigator, third)

    rules = []
    for index, rule in enumerate(engine.rules):
        proposal = investigator.investigate(
            next(t for t in test_alerts if engine.decide(t).rule_index == index)
        ).case.proposal
        before = backtest(test_history, rule.condition)
        entry = {"rule": format_rule(rule), "before": vars(before), "proposal": None}
        if proposal:
            after = backtest(test_history, proposal.rule.condition)
            entry.update(proposal=format_rule(proposal.rule), after=vars(after))
        rules.append(entry)
    results["rules_on_unseen_stream"] = rules

    timings = []
    for txn in test_alerts[:500]:
        started = time.perf_counter()
        investigator.investigate(txn)
        timings.append((time.perf_counter() - started) * 1000)
    timings.sort()
    results["investigation_ms"] = {
        "median": statistics.median(timings),
        "p99": timings[int(0.99 * len(timings))],
        "history_size": args.history,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=1), encoding="utf-8")
    print(json.dumps(results, indent=1))
    print("\n--- example report ---")
    print(investigator.investigate(next(t for t in test_alerts if t[LABEL])).render())


if __name__ == "__main__":
    main()
