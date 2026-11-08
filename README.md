# payrules

payrules turns plain-English payment-risk rules into rules a typed engine can run. It uses a
135M-parameter model trained with the full RLHF recipe: supervised fine-tuning (SFT), a
Bradley-Terry reward model, and PPO written from scratch in PyTorch. Everything trains on a laptop
RTX 2050 with 4 GB of VRAM.

```text
Request: Given that the payment goes via UPI, with the payment is overseas or the number of daily
         declined authentications is 10 or higher, queue the purchase for review.
Rule:    WHEN txn.method == UPI AND (txn.is_international OR user.failed_attempts_24h >= 10)
         THEN REVIEW reason="upi_method_international"
```

Every rule is parsed and type-checked before it runs. A rule that reads a field that doesn't
exist, or compares a boolean with a number, is rejected. A deterministic engine then executes the
rules and explains each decision.

## Results

All test sets were scored once, after tuning finished. A prediction counts as correct only if
the verifier proves it makes the same decision as the correct rule on every possible
transaction. Intervals are 95% paired bootstrap intervals over rules.

| Test set | Rules | SFT | PPO | PPO − SFT (points) |
|---|---|---|---|---|
| Unseen rules, phrased like the PPO training requests | 500 | 45.6% | 65.6% | +20.0 (+16.4 to +23.6) |
| Unseen rules, in a phrasing style no stage trained on | 500 | 36.2% | 36.8% | +0.6 (−1.6 to +2.6) |
| Hand-written rules | 60 | 13.3% | 15.0% | +1.7 (−3.3 to +8.3) |

- **Fewer false blocks.** Each rule was run against 2,000 synthetic transactions. PPO's rules
  wrongly blocked fewer decisions on both 500-rule test sets. In the PPO phrasing it was 49.4 per
  1,000 against 76.8 for SFT (difference −27.4, interval −43.0 to −12.7). In the unseen style it
  was 32.5 against 45.9 (−13.4, interval −23.9 to −4.6). Missed blocks did not change
  significantly on either set. So even where exact accuracy did not move, PPO's mistakes did less
  damage.
- **Reward model.** It trained on 7,840 pairs. On fresh rules in the PPO phrasing, its pair
  accuracy is 99.4% on 500 synthetic pairs. On the SFT model's real mistakes there, it is 99.6%
  (272 pairs). In the unseen style it reaches 96.4% on synthetic pairs, but only 61.8% on real
  mistakes (319 pairs).
- **Engine.** p99 decision latency is 79–161 µs per transaction against 100 rules (median
  130 µs over four runs), on one CPU core ([bench_engine.py](scripts/bench_engine.py)).

## What the experiments showed

**1. The first model memorised templates.** The first round trained on a narrow set of
phrasings. SFT reached 16.3% on validation requests worded differently, and PPO 31.7%. The fix
was to rewrite the training phrasings with 2–4× more variety, checked by automated leakage tests
so that no test wording slipped in. Accuracy on the unseen phrasing style rose from 7.0% to
32.9% (+25.9 points, interval +21.4 to +30.4). This is measured on the 398 test rules that
neither round trained on.

**2. PPO's gains stayed inside the phrasing it trained on.** PPO lifted accuracy on fresh rules
in its own phrasing from 45.6% to 65.6%, but added nothing measurable on the unseen style. The
reward model shares the blind spot: on real mistakes in the new style it is right only 62% of
the time. PPO can only reinforce answers the policy actually samples and the reward model
recognises, and neither had seen the new wording. In the first round, SFT still produced invalid
rules 64% of the time, and there PPO did help on the new style (+5.5 points, interval +2.5 to
+8.5). It did so by fixing format errors, a skill that carries across phrasings.

**3. An audit metric that pointed the wrong way.** Across requests, the reward model separated
the PPO model's correct and wrong outputs with an AUC of only 0.70, which suggested retraining
it. But Bradley-Terry training only fixes score differences within one request, and PPO's value
head absorbs each request's baseline. Within a request, the reward model ranked the gold rule
above every sampled mistake in the audits: 191 from SFT and 100 from PPO. Retraining it would
not have helped. [audit_reward.py](scripts/audit_reward.py) now reports both measures.

**4. What's left is an exploration problem.** Over 6 samples, at least 91% of validation
requests get all-correct or all-wrong answers from the PPO model. Its remaining mistakes are
consistent misreadings: "local" read as international, "pending" read as verified, "no more
than" written as `<`. PPO cannot reinforce a correct answer that it never samples.

## How it works

| Stage | What it does | Time on an RTX 2050 |
|---|---|---|
| Data ([build_dataset.py](scripts/build_dataset.py)) | Rules are random 1–4 clause combinations over 12 typed fields. Each request is worded in one of three phrasing styles (SFT, RL and test) that share no templates and only basic words. | < 1 min |
| SFT ([train_sft.py](scripts/train_sft.py)) | LoRA (r=32) on SmolLM2-135M-Instruct, 5,000 examples, 3 epochs, fp16 autocast | 13 min |
| Pair mining ([mine_pairs.py](scripts/mine_pairs.py)) | Samples the SFT model twice on each of 3,000 RL requests. Each wrong sample becomes a (correct rule, mistake) pair. | 10 min |
| Reward model ([train_reward.py](scripts/train_reward.py)) | Bradley-Terry loss over 7,840 pairs: synthetic corruptions plus the mined mistakes. A scalar head reads the final token of the SFT backbone. | 35 min |
| Audit ([audit_reward.py](scripts/audit_reward.py)) | Shortcut checks: gold vs. one-edit mutants with the reason text blanked, the right rule vs. another request's rule, and ranking of real samples within a request | 4 min |
| PPO ([train_ppo.py](scripts/train_ppo.py)) | 300 iterations of 32 rollouts. Per-token KL penalty to the frozen SFT model, GAE, value head, running-std reward scaling, early stop on target KL. Keeps the checkpoint with the best validation accuracy. | 3.1 h |

**Labels.** Preference labels come from an exact verifier, not human raters, but the policy
never sees the verifier. It is trained only on the learned reward model's scores, exactly as if
the pairs had been labelled by people. The verifier's result is a proof, not a sample: a
comparison can only change its outcome at a threshold, so testing one value from each region,
for every field, covers every possible transaction. (It falls back to sampling above 200,000
combinations, but every verdict on the test sets was exhaustive.)

**Leakage guards.** [test_leakage.py](tests/test_leakage.py) fails if training text shares any
word with RL or test text beyond a frozen list of 71 (articles, field nouns, values), or if any
template appears in more than one style. Every split has its own seeded generator, and the
evaluation splits are drawn first, so editing the training phrasings cannot change them.

**Memory.** Peak GPU memory is 6.2 GB for SFT (the Windows driver serves the excess from shared
system memory), 2.3 GB for the reward model and 3.6 GB for PPO.

## Reproduce

```bash
python -m venv .venv
.venv/Scripts/pip install torch --index-url https://download.pytorch.org/whl/cu126
.venv/Scripts/pip install -e ".[train,dev]"
PYTHON=.venv/Scripts/python.exe bash scripts/run_pipeline.sh    # about 4.5 hours

# Once, after all tuning:
.venv/Scripts/python.exe scripts/evaluate_final.py --models sft=checkpoints/sft ppo=checkpoints/ppo_learned
.venv/Scripts/python.exe scripts/evaluate_final.py --models sft=checkpoints/sft ppo=checkpoints/ppo_learned \
    --new-phrasing data/sft_test_rl_phrasing.jsonl --set-name rl_phrasing \
    --pairs data/pairs_test_rl_phrasing.jsonl --out runs/final_eval_rl_phrasing.json
```

Tested with Python 3.13, torch 2.14.1+cu126, transformers 5.22.2 and peft 0.21.2. The pipeline
uses one seed.

## Limitations

- Requests come from templates. The 60 hand-written rules are the only natural text, and both
  models do poorly on them (13–15%), mostly by dropping a clause or picking the wrong field.
- Labels come from a verifier, which is only possible because rule equivalence can be decided
  exactly. Human preferences would be noisier.
- Validation results (rl_val) pick PPO checkpoints, so the test tables above are the numbers to
  quote.
- At 135M parameters, the model cannot work out that an unseen word such as "local" means
  domestic.

## Layout

```text
src/payrules/   schema, parser (Lark), type checker, engine, verifier,
                request generator (nlgen), reward-pair builder, model helpers
scripts/        one script per pipeline stage, plus evaluate_final.py and run_pipeline.sh
tests/          parser, type checker, engine, verifier, data and leakage tests
eval/           hand-written test rules
```
