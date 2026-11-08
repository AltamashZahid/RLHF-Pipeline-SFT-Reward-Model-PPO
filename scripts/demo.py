"""End-to-end: plain English to a checked rule, then decisions on traffic, then an investigation.

usage: python scripts/demo.py "Block UPI payments above 50k from rooted phones"

Transactions are synthetic, so the backtest shows the machinery working, not real fraud results.
"""

import argparse
import json

import torch

from payrules import RuleEngine
from payrules.compiler import compile_one
from payrules.investigate import History, Investigator, backtest
from payrules.llm import decode_response, generate_rules, load_lora_model, load_tokenizer, sample_responses
from payrules.nodes import format_rule
from payrules.reward import RewardModel
from payrules.txngen import LABEL, generate_transactions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("request")
    parser.add_argument("--policy", default="checkpoints/v4/ppo")
    parser.add_argument("--reward-model", default="checkpoints/v4/reward")
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--strategy", default="majority_then_reward")
    parser.add_argument("--minimum-agreement", type=float, default=0.75)
    parser.add_argument("--history", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    tokenizer = load_tokenizer()
    policy = load_lora_model(args.policy).eval()
    greedy = generate_rules(policy, tokenizer, [args.request])[0]
    samples = [decode_response(tokenizer, r)
               for r in sample_responses(policy, tokenizer, [args.request] * args.samples)]
    del policy
    torch.cuda.empty_cache()
    reward = RewardModel.load(args.reward_model, tokenizer).eval()

    def score(requests, texts):
        return reward.score(requests, texts, batch_size=8).tolist()

    compiled = compile_one(args.request, greedy, samples, score, args.strategy)
    print(f"request:   {args.request}")
    if compiled.answer is None:
        print(f"rule:      none of the {compiled.total} candidates was a valid rule. Send this to a person.")
        return
    print(f"rule:      {format_rule(compiled.answer.rule)}")
    print(f"agreement: {compiled.agreement:.0%} of {compiled.valid} valid candidates behave this way")
    if compiled.needs_review(args.minimum_agreement):
        print(f"status:    NEEDS HUMAN REVIEW (agreement below {args.minimum_agreement:.0%}); not deploying it.")
        return
    print("status:    passed checks; running on synthetic traffic")

    rule = compiled.answer.rule
    history = History(generate_transactions(args.history, seed=101))
    stats = backtest(history, rule.condition)
    print(f"backtest:  {stats.matches} matches, precision {stats.precision:.1%}, recall {stats.recall:.1%}")
    engine = RuleEngine([rule])
    flagged = next((t for t in generate_transactions(args.history, seed=202)
                    if engine.decide(t).rule_index is not None), None)
    if flagged is None:
        print("no synthetic transaction matched this rule")
        return
    print("\n--- investigation of the first flagged transaction ---")
    print(json.dumps({k: v for k, v in flagged.items() if k != LABEL}))
    print(Investigator(engine, history).investigate(flagged).render())


if __name__ == "__main__":
    main()
