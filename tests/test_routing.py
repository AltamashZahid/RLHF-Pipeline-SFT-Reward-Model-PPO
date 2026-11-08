import numpy as np
import pytest

from payrules.routing import (
    ExploreThenCommit,
    GreedyAfterExploration,
    PooledThompson,
    RoundRobin,
    Scenario,
    Thompson,
    UCB,
    _reroute_time,
    make_scenario,
    simulate,
)


def fixed(rates, steps=4000):
    table = np.tile(np.array(rates), (steps, 4, 1))
    return Scenario("fixed", table)


def test_scenarios_are_deterministic_and_valid():
    for name in ("stationary", "drift", "outage", "by_method"):
        a, b = make_scenario(name, 1, 500), make_scenario(name, 1, 500)
        assert np.array_equal(a.rates, b.rates)
        assert a.rates.min() >= 0 and a.rates.max() <= 1


def test_unknown_scenario_is_rejected():
    with pytest.raises(ValueError):
        make_scenario("nope", 1, 100)


def test_outage_only_hits_one_gateway_inside_its_window():
    scenario = make_scenario("outage", 2, 3000)
    gateway, start, end = scenario.outage
    assert (scenario.rates[start:end, :, gateway] == 0.2).all()
    assert scenario.rates[start - 1, 0, gateway] > 0.2 and scenario.rates[end, 0, gateway] > 0.2


@pytest.mark.parametrize("policy", [UCB(3, 0), Thompson(3, 0), ExploreThenCommit(3, 0, explore_each=50)])
def test_learning_routers_beat_round_robin_on_a_clear_gap(policy):
    scenario = fixed([0.95, 0.5, 0.5])
    assert simulate(policy, scenario, 0).success_rate > simulate(RoundRobin(3, 0), scenario, 0).success_rate + 0.1


def test_oracle_bounds_the_regret():
    result = simulate(RoundRobin(2, 0), fixed([0.9, 0.5]), 0)
    assert result.oracle_rate == pytest.approx(0.9)
    assert result.regret_per_1000 == pytest.approx(1000 * (result.oracle_rate - result.success_rate))


def test_forgetting_lets_thompson_survive_an_outage_that_stationary_thompson_rides_longer():
    scenario = make_scenario("outage", 3, 30_000)
    forgetful = simulate(Thompson(5, 3, discount=0.99), scenario, 3)
    stubborn = simulate(Thompson(5, 3, discount=1.0), scenario, 3)
    assert forgetful.outage_steps_to_reroute < stubborn.outage_steps_to_reroute


def test_reroute_time():
    assert _reroute_time([0] * 300) == 200
    assert _reroute_time([1] * 300) is None
    assert 300 <= _reroute_time([1] * 300 + [0] * 500) <= 500


def test_fixed_table_never_changes_after_exploring_but_greedy_does():
    fixed_table = ExploreThenCommit(2, 0, explore_each=10)
    greedy = GreedyAfterExploration(2, 0, explore_each=10)
    for policy in (fixed_table, greedy):
        for gateway, wins in ((0, 9), (1, 5)):
            for i in range(10):
                policy.update(0, gateway, i < wins)
        assert policy.choose(0) == 0
        for _ in range(200):
            policy.update(0, 0, False)
    assert fixed_table.choose(0) == 0
    assert greedy.choose(0) == 1


def test_pooled_thompson_shares_an_outage_across_methods():
    policy = PooledThompson(2, 0, discount=1.0, prior_strength=80.0)
    for _ in range(500):
        policy.update(0, 0, False)
        policy.update(0, 1, True)
    picks = [policy.choose(3) for _ in range(200)]  # Method 3 has no evidence of its own.
    assert picks.count(1) > 190


def test_pooled_thompson_lets_a_methods_own_record_win():
    policy = PooledThompson(2, 0, discount=1.0, prior_strength=5.0)
    for _ in range(500):
        policy.update(0, 0, True)
        policy.update(0, 1, False)
        policy.update(1, 0, False)
        policy.update(1, 1, True)
    assert [policy.choose(0) for _ in range(100)].count(0) > 95
    assert [policy.choose(1) for _ in range(100)].count(1) > 95


def test_outage_share_is_reported():
    scenario = make_scenario("outage", 1, 2000)
    result = simulate(RoundRobin(5, 0), scenario, 0)
    assert result.outage_share_on_failed_gateway == pytest.approx(0.2, abs=0.01)
