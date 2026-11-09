"""PPO on the SFT policy, written directly in PyTorch.

Each response token gets a penalty for moving away from the frozen SFT model (a KL estimate),
and the final token also receives the reward for the finished rule. The verifier always logs
true accuracy, but with --reward learned its result never reaches the training signal.
"""

import argparse
import json
import random
import time
from pathlib import Path

import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence

from payrules.llm import (
    decode_response,
    evaluate,
    format_prompt,
    load_lora_model,
    load_tokenizer,
    read_jsonl,
    sample_responses,
)
from payrules.pairs import score_candidate
from payrules.parser import parse
from payrules.reward import RewardModel


class ValueHead(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(width, 256), nn.GELU(), nn.Linear(256, 1))

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.net(hidden).squeeze(-1)


def build_batch(prompt_ids, responses, pad_id, device):
    """Right-padded sequences, and a mask of the positions that predict a response token."""
    ids, action_mask = [], []
    for prompt, response in zip(prompt_ids, responses):
        ids.append(torch.tensor(prompt + response))
        action_mask.append(torch.tensor([False] * (len(prompt) - 1) + [True] * len(response)))
    return (
        pad_sequence(ids, batch_first=True, padding_value=pad_id).to(device),
        pad_sequence([torch.ones(len(s), dtype=torch.long) for s in ids], batch_first=True).to(device),
        pad_sequence(action_mask, batch_first=True, padding_value=False).to(device),
    )


def forward_policy(model, ids, mask, want_hidden=True):
    with torch.autocast("cuda", dtype=torch.float16):
        output = model(input_ids=ids, attention_mask=mask, output_hidden_states=want_hidden)
    log_probs = torch.log_softmax(output.logits[:, :-1].float(), dim=-1)
    taken = log_probs.gather(-1, ids[:, 1:].unsqueeze(-1)).squeeze(-1)
    hidden = output.hidden_states[-1][:, :-1].float() if want_hidden else None
    return taken, hidden


def advantages_and_returns(rewards, values, action_mask, gamma, lam):
    """GAE along each response. Values and rewards are aligned to action positions."""
    advantages = torch.zeros_like(rewards)
    for row in range(rewards.shape[0]):
        positions = action_mask[row].nonzero().squeeze(-1).tolist()
        running = 0.0
        next_value = 0.0
        for position in reversed(positions):
            delta = rewards[row, position] + gamma * next_value - values[row, position]
            running = delta + gamma * lam * running
            advantages[row, position] = running
            next_value = values[row, position]
    return advantages, advantages + values


def verifier_reward(score) -> float:
    if score.semantic_match:
        return 1.0
    if score.type_checks:
        return 0.3
    return 0.1 if score.parses else 0.0


def correlation(a: list[float], b: list[float]) -> float | None:
    x, y = torch.tensor(a, dtype=torch.float32), torch.tensor(b, dtype=torch.float32)
    if x.std() == 0 or y.std() == 0:
        return None
    return torch.corrcoef(torch.stack([x, y]))[0, 1].item()


class RunningStd:
    def __init__(self):
        self.count, self.mean, self.m2 = 0, 0.0, 0.0

    def update(self, values: list[float]) -> None:
        for value in values:
            self.count += 1
            delta = value - self.mean
            self.mean += delta / self.count
            self.m2 += delta * (value - self.mean)

    @property
    def std(self) -> float:
        return max((self.m2 / max(self.count - 1, 1)) ** 0.5, 1e-3)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sft", type=Path, default=Path("checkpoints/sft"))
    parser.add_argument("--init", type=Path, default=None, help="Continue from this policy; the KL reference stays SFT")
    parser.add_argument("--reward-model", type=Path, default=Path("checkpoints/reward"))
    parser.add_argument("--reward", choices=("learned", "verifier"), default="learned")
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--out", type=Path, default=Path("checkpoints/ppo_learned"))
    parser.add_argument("--log", type=Path, default=Path("runs/ppo_learned.jsonl"))
    parser.add_argument("--iterations", type=int, default=300)
    parser.add_argument("--rollout", type=int, default=32)
    parser.add_argument("--minibatch", type=int, default=4)
    parser.add_argument("--ppo-epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--value-lr", type=float, default=3e-4)
    parser.add_argument("--clip", type=float, default=0.2)
    parser.add_argument("--kl-beta", type=float, default=0.05)
    parser.add_argument("--target-kl", type=float, default=0.03)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--lam", type=float, default=0.95)
    parser.add_argument("--eval-every", type=int, default=25)
    parser.add_argument("--eval-rows", type=int, default=300)
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    tokenizer = load_tokenizer()
    policy = load_lora_model(args.init or args.sft, trainable=True)
    reference = load_lora_model(args.sft).eval()
    for parameter in reference.parameters():
        parameter.requires_grad_(False)
    reward_model = None
    if args.reward == "learned":
        reward_model = RewardModel.load(args.reward_model, tokenizer).eval()
        for parameter in reward_model.parameters():
            parameter.requires_grad_(False)

    value_head = ValueHead(policy.config.hidden_size).to(policy.device)
    policy_params = [p for p in policy.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW([
        {"params": policy_params, "lr": args.lr},
        {"params": value_head.parameters(), "lr": args.value_lr},
    ], weight_decay=0.0)
    scaler = torch.amp.GradScaler("cuda")
    all_params = policy_params + list(value_head.parameters())

    train_rows = read_jsonl(args.data / "sft_rl_train.jsonl")
    val_rows = read_jsonl(args.data / "sft_rl_val.jsonl")[: args.eval_rows]
    reward_scale = RunningStd()
    rng = random.Random(args.seed)

    args.log.parent.mkdir(parents=True, exist_ok=True)
    torch.cuda.reset_peak_memory_stats()
    started = time.time()

    def checkpoint_eval(iteration, log):
        metrics = evaluate(policy, tokenizer, val_rows)
        metrics.pop("predictions")
        record = {"event": "eval", "iteration": iteration, **metrics}
        print(json.dumps(record), flush=True)
        log.write(json.dumps(record) + "\n")
        log.flush()
        return metrics["semantic_match"]

    with args.log.open("w", encoding="utf-8") as log:
        best = checkpoint_eval(0, log)
        print(f"SFT starting point: val semantic accuracy {best:.3f}", flush=True)
        policy.save_pretrained(args.out)
        torch.save(value_head.state_dict(), args.out / "value_head.pt")

        for iteration in range(1, args.iterations + 1):
            rows = rng.sample(train_rows, args.rollout)
            requests = [r["prompt"] for r in rows]
            responses = sample_responses(policy, tokenizer, requests)
            texts = [decode_response(tokenizer, r) for r in responses]
            scores = [score_candidate(t, parse(r["rule"])) for t, r in zip(texts, rows)]
            matches = [float(s.semantic_match) for s in scores]
            if args.reward == "learned":
                terminal = reward_model.score(requests, texts).tolist()
            else:
                terminal = [verifier_reward(s) for s in scores]
            reward_scale.update(terminal)
            scaled = [t / reward_scale.std for t in terminal]

            prompt_ids = [tokenizer(format_prompt(q))["input_ids"] for q in requests]
            ids, mask, action_mask = build_batch(prompt_ids, responses, tokenizer.pad_token_id, policy.device)
            old_logp, old_values, ref_logp = [], [], []
            policy.eval()
            with torch.no_grad():  # Not inference_mode: these tensors feed the PPO loss below.
                for start in range(0, len(rows), args.minibatch):
                    chunk = slice(start, start + args.minibatch)
                    taken, hidden = forward_policy(policy, ids[chunk], mask[chunk])
                    old_logp.append(taken)
                    old_values.append(value_head(hidden))
                    ref_taken, _ = forward_policy(reference, ids[chunk], mask[chunk], want_hidden=False)
                    ref_logp.append(ref_taken)
            old_logp, old_values, ref_logp = map(torch.cat, (old_logp, old_values, ref_logp))

            token_kl = (old_logp - ref_logp) * action_mask
            rewards = -args.kl_beta * token_kl
            for row, position in enumerate(action_mask.int().cumsum(1).argmax(1).tolist()):
                rewards[row, position] += scaled[row]
            adv, returns = advantages_and_returns(rewards, old_values, action_mask, args.gamma, args.lam)
            values_in_mask = adv[action_mask]
            adv = (adv - values_in_mask.mean()) / (values_in_mask.std() + 1e-8) * action_mask

            policy.train()
            stop = False
            epochs_run, policy_loss_value, value_loss_value, approx_kl = 0, 0.0, 0.0, 0.0
            for _ in range(args.ppo_epochs):
                order = torch.randperm(len(rows)).tolist()
                for start in range(0, len(rows), args.minibatch):
                    index = torch.tensor(order[start : start + args.minibatch], device=ids.device)
                    taken, hidden = forward_policy(policy, ids[index], mask[index])
                    values = value_head(hidden.detach())
                    action = action_mask[index]
                    log_ratio = (taken - old_logp[index])[action]
                    ratio = log_ratio.exp()
                    batch_adv = adv[index][action]
                    surrogate = torch.min(
                        ratio * batch_adv,
                        ratio.clamp(1 - args.clip, 1 + args.clip) * batch_adv,
                    )
                    policy_loss = -surrogate.mean()
                    value_loss = (values[action] - returns[index][action]).pow(2).mean()
                    loss = policy_loss + 0.5 * value_loss
                    optimizer.zero_grad(set_to_none=True)
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(all_params, 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    approx_kl = (-log_ratio).mean().item()
                    policy_loss_value, value_loss_value = policy_loss.item(), value_loss.item()
                    if abs(approx_kl) > args.target_kl:
                        stop = True
                        break
                epochs_run += 1
                if stop:
                    break

            # Sampling from non-finite weights kills the CUDA context, which loses the evidence.
            bad_params = [p for p in policy_params if not torch.isfinite(p).all()]
            if bad_params:
                record = {"event": "stopped", "iteration": iteration, "non_finite_tensors": len(bad_params)}
                print(json.dumps(record), flush=True)
                log.write(json.dumps(record) + "\n")
                log.flush()
                break

            record = {
                "event": "iter",
                "iteration": iteration,
                "reward_mean": sum(terminal) / len(terminal),
                "sample_match_rate": sum(matches) / len(matches),
                "reward_vs_truth_corr": correlation(terminal, matches),
                "kl_to_sft_per_token": (token_kl.sum() / action_mask.sum()).item(),
                "approx_kl": approx_kl,
                "policy_loss": policy_loss_value,
                "value_loss": value_loss_value,
                "epochs_run": epochs_run,
                "peak_vram_mib": round(torch.cuda.max_memory_reserved() / 2**20),
                "elapsed_s": round(time.time() - started),
            }
            log.write(json.dumps(record) + "\n")
            log.flush()
            if iteration % 5 == 0 or iteration == 1:
                print(json.dumps(record), flush=True)

            if iteration % args.eval_every == 0:
                accuracy = checkpoint_eval(iteration, log)
                if accuracy > best:
                    best = accuracy
                    policy.save_pretrained(args.out)
                    torch.save(value_head.state_dict(), args.out / "value_head.pt")

    print(f"best val semantic accuracy {best:.3f}; peak VRAM "
          f"{torch.cuda.max_memory_reserved() / 2**20:.0f} MiB", flush=True)


if __name__ == "__main__":
    main()
