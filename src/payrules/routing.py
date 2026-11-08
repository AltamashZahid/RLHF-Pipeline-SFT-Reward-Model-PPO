"""Payment routing as a bandit problem, on a simulator.

Each payment goes to one gateway and either succeeds or fails. Gateway success rates drift and
gateways have outages, so a router has to keep learning. The rates here are invented: results
compare routers under these conditions and say nothing about real gateway traffic.
"""

import math
import random
from dataclasses import dataclass

import numpy as np

METHODS = ("UPI", "CARD", "NETBANKING", "WALLET")


@dataclass(frozen=True)
class Scenario:
    """rates[step, method, gateway] is the true success probability. Routers never see it."""

    name: str
    rates: np.ndarray
    outage: tuple[int, int, int] | None = None  # gateway, first step, last step

    @property
    def steps(self) -> int:
        return self.rates.shape[0]

    @property
    def gateways(self) -> int:
        return self.rates.shape[2]


def _base_rates(rng: np.random.Generator, gateways: int, spread_by_method: bool) -> np.ndarray:
    shared = rng.uniform(0.78, 0.95, size=gateways)
    if not spread_by_method:
        return np.tile(shared, (len(METHODS), 1))
    return np.clip(shared + rng.normal(0, 0.06, size=(len(METHODS), gateways)), 0.5, 0.98)


def make_scenario(name: str, seed: int, steps: int = 30_000, gateways: int = 5) -> Scenario:
    rng = np.random.default_rng(seed)
    base = _base_rates(rng, gateways, spread_by_method=name == "by_method")
    rates = np.tile(base, (steps, 1, 1))
    outage = None
    if name == "drift":
        phase = rng.uniform(0, 2 * math.pi, size=gateways)
        period = rng.uniform(0.3, 1.0, size=gateways) * steps
        wave = 0.08 * np.sin(2 * math.pi * np.arange(steps)[:, None] / period + phase)
        rates = np.clip(rates + wave[:, None, :], 0.05, 0.99)
    elif name == "outage":
        best = int(base[0].argmax())
        start, end = steps // 3, steps // 3 + steps // 10
        rates[start:end, :, best] = 0.2
        outage = (best, start, end)
    elif name not in ("stationary", "by_method"):
        raise ValueError(f"unknown scenario {name}")
    return Scenario(name, rates, outage)


class Policy:
    """Keeps separate statistics per payment method when `contextual` is set."""

    def __init__(self, gateways: int, seed: int, contextual: bool = False):
        self.gateways, self.contextual = gateways, contextual
        self.rng = random.Random(seed)
        contexts = len(METHODS) if contextual else 1
        self.wins = [[0.0] * gateways for _ in range(contexts)]
        self.plays = [[0.0] * gateways for _ in range(contexts)]

    def _slot(self, method: int) -> int:
        return method if self.contextual else 0

    def choose(self, method: int) -> int:
        raise NotImplementedError

    def update(self, method: int, gateway: int, success: bool) -> None:
        slot = self._slot(method)
        self.plays[slot][gateway] += 1
        self.wins[slot][gateway] += success


class RoundRobin(Policy):
    def __init__(self, gateways, seed, contextual=False):
        super().__init__(gateways, seed, contextual)
        self.turn = 0

    def choose(self, method):
        self.turn = (self.turn + 1) % self.gateways
        return self.turn


class GreedyAfterExploration(Policy):
    """Measures every gateway for a while, then always uses the best running average."""

    def __init__(self, gateways, seed, contextual=False, explore_each=300):
        super().__init__(gateways, seed, contextual)
        self.explore_each = explore_each

    def choose(self, method):
        slot = self._slot(method)
        plays, wins = self.plays[slot], self.wins[slot]
        for gateway, count in enumerate(plays):
            if count < self.explore_each:
                return gateway
        return max(range(self.gateways), key=lambda g: wins[g] / plays[g])


class ExploreThenCommit(GreedyAfterExploration):
    """A fixed routing table: the same measurement, then the evidence stops changing."""

    def update(self, method, gateway, success):
        if self.plays[self._slot(method)][gateway] < self.explore_each:
            super().update(method, gateway, success)


class Discounted(Policy):
    """Old evidence fades by `discount` per update, which is what lets a router notice change."""

    def __init__(self, gateways, seed, contextual=False, discount=1.0):
        super().__init__(gateways, seed, contextual)
        self.discount = discount

    def update(self, method, gateway, success):
        slot = self._slot(method)
        plays, wins = self.plays[slot], self.wins[slot]
        if self.discount < 1.0:
            for g in range(self.gateways):
                plays[g] *= self.discount
                wins[g] *= self.discount
        plays[gateway] += 1
        wins[gateway] += success


class EpsilonGreedy(Discounted):
    def __init__(self, gateways, seed, contextual=False, discount=1.0, epsilon=0.05):
        super().__init__(gateways, seed, contextual, discount)
        self.epsilon = epsilon

    def choose(self, method):
        if self.rng.random() < self.epsilon:
            return self.rng.randrange(self.gateways)
        slot = self._slot(method)
        plays, wins = self.plays[slot], self.wins[slot]
        return max(range(self.gateways), key=lambda g: (wins[g] + 1) / (plays[g] + 2))


class UCB(Discounted):
    def __init__(self, gateways, seed, contextual=False, discount=1.0, width=0.5):
        super().__init__(gateways, seed, contextual, discount)
        self.width = width

    def choose(self, method):
        slot = self._slot(method)
        plays, wins = self.plays[slot], self.wins[slot]
        total = sum(plays) + 1
        best, best_score = 0, -1.0
        for gateway in range(self.gateways):
            if plays[gateway] < 1e-9:
                return gateway
            score = wins[gateway] / plays[gateway] + self.width * math.sqrt(math.log(total) / plays[gateway])
            if score > best_score:
                best, best_score = gateway, score
        return best


class Thompson(Discounted):
    def choose(self, method):
        slot = self._slot(method)
        plays, wins = self.plays[slot], self.wins[slot]
        draw = self.rng.betavariate
        samples = [draw(wins[g] + 1, plays[g] - wins[g] + 1) for g in range(self.gateways)]
        return max(range(self.gateways), key=samples.__getitem__)


class PooledThompson(Policy):
    """Per-method evidence on top of a prior from all methods.

    A gateway's pooled rate is worth `prior_strength` observations. With little evidence for a
    method the pool decides, so an outage seen on any method is acted on by all of them; with
    plenty of evidence the method's own record takes over. All evidence fades every payment.
    """

    def __init__(self, gateways, seed, discount=1.0, prior_strength=20.0):
        super().__init__(gateways, seed, contextual=True)
        self.discount, self.prior_strength = discount, prior_strength
        self.pool_wins, self.pool_plays = [0.0] * gateways, [0.0] * gateways

    def choose(self, method):
        plays, wins = self.plays[method], self.wins[method]
        samples = []
        for g in range(self.gateways):
            pooled = (self.pool_wins[g] + 1) / (self.pool_plays[g] + 2)
            alpha = wins[g] + self.prior_strength * pooled + 1
            beta = plays[g] - wins[g] + self.prior_strength * (1 - pooled) + 1
            samples.append(self.rng.betavariate(alpha, beta))
        return max(range(self.gateways), key=samples.__getitem__)

    def update(self, method, gateway, success):
        d = self.discount
        for slot in range(len(self.plays)):
            plays, wins = self.plays[slot], self.wins[slot]
            for g in range(self.gateways):
                plays[g] *= d
                wins[g] *= d
        for g in range(self.gateways):
            self.pool_plays[g] *= d
            self.pool_wins[g] *= d
        self.plays[method][gateway] += 1
        self.wins[method][gateway] += success
        self.pool_plays[gateway] += 1
        self.pool_wins[gateway] += success


@dataclass(frozen=True)
class Result:
    success_rate: float
    oracle_rate: float
    regret_per_1000: float
    outage_steps_to_reroute: int | None
    outage_share_on_failed_gateway: float | None = None


def simulate(policy: Policy, scenario: Scenario, seed: int) -> Result:
    """Same seed, same payments: routers compared on one seed face identical luck."""
    rng = np.random.default_rng(seed)
    methods = rng.integers(0, len(METHODS), size=scenario.steps)
    luck = rng.random(size=scenario.steps)  # One draw per payment, shared across gateways.
    rates = scenario.rates.tolist()
    oracle = float(scenario.rates[np.arange(scenario.steps), methods].max(axis=1).mean())
    successes = 0
    outage_traffic: list[int] = []
    for step in range(scenario.steps):
        method = int(methods[step])
        gateway = policy.choose(method)
        success = luck[step] < rates[step][method][gateway]
        successes += success
        policy.update(method, gateway, bool(success))
        if scenario.outage and scenario.outage[1] <= step < scenario.outage[2]:
            outage_traffic.append(gateway == scenario.outage[0])
    rate = successes / scenario.steps
    return Result(
        rate,
        oracle,
        1000 * (oracle - rate),
        _reroute_time(outage_traffic) if scenario.outage else None,
        sum(outage_traffic) / len(outage_traffic) if scenario.outage else None,
    )


def _reroute_time(on_failed_gateway: list[int], window: int = 200, share: float = 0.1) -> int | None:
    """Payments until under 10% of a 200-payment window still goes to the failed gateway."""
    total = sum(on_failed_gateway[:window])
    if total <= share * window:
        return window
    for end in range(window, len(on_failed_gateway)):
        total += on_failed_gateway[end] - on_failed_gateway[end - window]
        if total <= share * window:
            return end + 1
    return None
