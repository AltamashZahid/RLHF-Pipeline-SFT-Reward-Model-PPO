"""Compares payment routers on the simulator. Settings are tuned and scored on separate seeds."""

import argparse
import json
import random
import statistics
from pathlib import Path

from payrules.routing import (
    EpsilonGreedy,
    ExploreThenCommit,
    GreedyAfterExploration,
    PooledThompson,
    RoundRobin,
    Thompson,
    UCB,
    make_scenario,
    simulate,
)

SCENARIOS = ("stationary", "drift", "outage", "by_method")
DISCOUNTS = (0.99, 0.995, 0.998, 0.999)
PRIOR_STRENGTHS = (5.0, 20.0, 80.0)
BASELINE = "greedy_after_exploration"  # The stronger of the two non-bandit routers.


def run(factory, scenario_name, seeds, steps):
    results = []
    for seed in seeds:
        scenario = make_scenario(scenario_name, seed, steps)
        results.append(simulate(factory(scenario.gateways, seed), scenario, seed))
    return results


def mean_regret(factory, seeds, steps):
    return statistics.mean(r.regret_per_1000 for name in SCENARIOS for r in run(factory, name, seeds, steps))


def paired_interval(a, b, rounds=5000, seed=0):
    """95% interval of the mean of b - a, resampling seeds."""
    rng = random.Random(seed)
    differences = [y - x for x, y in zip(a, b)]
    means = sorted(statistics.fmean(rng.choices(differences, k=len(differences))) for _ in range(rounds))
    return means[int(0.025 * rounds)], means[int(0.975 * rounds)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=30_000)
    parser.add_argument("--tune-seeds", type=int, default=8)
    parser.add_argument("--test-seeds", type=int, default=30)
    parser.add_argument("--out", type=Path, default=Path("runs/routing.json"))
    args = parser.parse_args()

    tune = range(1000, 1000 + args.tune_seeds)
    test = range(3000, 3000 + args.test_seeds)  # 2000+ was looked at while this was designed.

    def build(kind, discount, contextual=True):
        def factory(gateways, seed):
            return kind(gateways, seed, contextual=contextual, discount=discount)
        return factory

    tuned = {}
    for label, kind in (("epsilon_greedy", EpsilonGreedy), ("ucb", UCB), ("thompson", Thompson)):
        best = min(DISCOUNTS, key=lambda d: mean_regret(build(kind, d), tune, args.steps))
        tuned[label] = best
        print(f"tuned {label}: discount {best}", flush=True)

    def pooled(discount, strength):
        def factory(gateways, seed):
            return PooledThompson(gateways, seed, discount=discount, prior_strength=strength)
        return factory

    pooled_discount = min(DISCOUNTS, key=lambda d: mean_regret(pooled(d, 20.0), tune, args.steps))
    pooled_strength = min(PRIOR_STRENGTHS, key=lambda m: mean_regret(pooled(pooled_discount, m), tune, args.steps))
    tuned["pooled_thompson"] = {"discount": pooled_discount, "prior_strength": pooled_strength}
    print(f"tuned pooled_thompson: {tuned['pooled_thompson']}", flush=True)

    routers = {
        "round_robin": lambda g, s: RoundRobin(g, s),
        "fixed_routing_table": lambda g, s: ExploreThenCommit(g, s, contextual=True),
        BASELINE: lambda g, s: GreedyAfterExploration(g, s, contextual=True),
        "thompson_no_forgetting": build(Thompson, 1.0),
        "thompson_no_method_context": build(Thompson, tuned["thompson"], contextual=False),
        "epsilon_greedy": build(EpsilonGreedy, tuned["epsilon_greedy"]),
        "ucb": build(UCB, tuned["ucb"]),
        "thompson": build(Thompson, tuned["thompson"]),
        "pooled_thompson": pooled(pooled_discount, pooled_strength),
    }

    table: dict = {"tuned_discounts": tuned, "scenarios": {}}
    for scenario in SCENARIOS:
        rows = {name: run(factory, scenario, test, args.steps) for name, factory in routers.items()}
        baseline = [r.success_rate for r in rows[BASELINE]]
        entry = {}
        for name, results in rows.items():
            rates = [r.success_rate for r in results]
            low, high = paired_interval(baseline, rates)
            reroutes = [r.outage_steps_to_reroute for r in results]
            entry[name] = {
                "success_rate": statistics.fmean(rates),
                "points_vs_baseline": 100 * statistics.fmean(y - x for x, y in zip(baseline, rates)),
                "interval_points": [100 * low, 100 * high],
                "regret_per_1000": statistics.fmean(r.regret_per_1000 for r in results),
            }
            if results[0].outage_steps_to_reroute is not None or scenario == "outage":
                done = [x for x in reroutes if x is not None]
                entry[name]["median_payments_to_reroute"] = statistics.median(done) if done else None
                entry[name]["never_rerouted_seeds"] = len(reroutes) - len(done)
                entry[name]["outage_share_on_failed_gateway"] = statistics.fmean(
                    r.outage_share_on_failed_gateway for r in results
                )
        table["scenarios"][scenario] = entry
        print(f"\n== {scenario} ==")
        for name, row in entry.items():
            extra = ""
            if "median_payments_to_reroute" in row:
                extra = (f"  on failed gateway {row['outage_share_on_failed_gateway']:.1%}, "
                         f"reroute {row['median_payments_to_reroute']} (never: {row['never_rerouted_seeds']})")
            print(f"{name:28s} success {row['success_rate']:.4f}  regret/1000 {row['regret_per_1000']:6.1f}  "
                  f"vs baseline {row['points_vs_baseline']:+.2f} "
                  f"[{row['interval_points'][0]:+.2f}, {row['interval_points'][1]:+.2f}]{extra}", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(table, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
