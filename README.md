# payrules

payrules turns plain-English payment-risk rules into rules a typed engine can run. It uses a
360M-parameter model and the full RLHF recipe: supervised fine-tuning (SFT), a Bradley-Terry reward
model, and PPO written from scratch in PyTorch. Everything trains on a laptop RTX 2050 with 4 GB of
VRAM. An investigation agent, a payment router and a DSL miner sit around it.

```text
Request:   Block payments above 50000 made over a VPN
Rule:      WHEN txn.amount > 50000 AND txn.is_vpn THEN BLOCK reason="high_amount_vpn"
Agreement: 100% of 9 valid candidates behave this way
```

Every rule is parsed and type-checked before it runs. A rule that reads a field that doesn't
exist, or compares a boolean with a number, is rejected. A deterministic engine then executes the
rules and explains each decision. Rules can use 18 fields, including VPN or proxy use, high-risk
country, prepaid card, card/IP country mismatch, disposable email and cards tried per IP.

## Results

A prediction counts as correct only if the verifier proves it makes the same decision as the correct
rule on every possible transaction. Intervals are 95% paired bootstrap intervals over rules.

v4 is the last 12-field model and v5 is the current 18-field model. **The chosen policy is v5 PPO at
iteration 50.** It was picked on validation before these tests were opened. v5 reads more kinds of rule,
but it is not more accurate on the rules v4 could already read.

| Test set | Rules | v4 SFT | v5 SFT | v5 PPO, iter 50 (chosen) | v5 PPO, iter 150 |
|---|---|---|---|---|---|
| Unseen rules in RL wording, some using the new fields | 500 | 55.4% | 82.8% | **86.8%** | 88.4% |
| Natural-language rules, LLM-written, 12 fields only | 200 | 82.0% | 78.5% | **80.0%** | 75.0% |
| Unseen rules in RL wording, 12 fields only | 500 | 90.4% | 86.8% | **86.8%** | 90.8% |
| Rules from public payment docs, new fields | 20 | 0 of 20 | 16 of 20 | **15 of 20** | 16 of 20 |
| Rules from public payment docs, older fields | 12 | 9 of 12 | 11 of 12 | **10 of 12** | 9 of 12 |

Chosen policy against v4 SFT, in points:

| Test set | Difference | Rules gained / lost |
|---|---|---|
| RL wording, some using the new fields | +31.4 (+27.0 to +36.0) | 165 / 8 |
| Natural-language rules | −2.0 (−7.0 to +2.5) | 10 / 14 |
| RL wording, 12 fields only | −3.6 (−6.4 to −0.8) | 16 / 34 |

- **The new fields work.** v4 could not express them. In the first row, v5 SFT gets 143 of the 202 rules
  that use a new field (70.8%).
- **Rules on the older fields did not improve.** The drop is significant on one 500-rule test and not
  on the natural test. None of the 15 natural rules that v5 SFT lost against v4 uses a new field, so the
  model is not confusing old fields with new ones. The cause isn't tested. A likely one is that the new
  fields took training rows away from the older ones.
- **The 12 public older-field rules don't settle this.** With so few rules, a gain of 1 or 2 could be
  chance, and the larger tests point the other way.
- **PPO helped on its own wording, not on natural text.** These are against v5 SFT:

  | Test set | Iter 50 (points) | Iter 150 (points) |
  |---|---|---|
  | RL wording, some new fields | +4.0 (+1.4 to +6.8) | +5.6 (+2.8 to +8.4) |
  | RL wording, 12 fields only | 0.0 (−2.2 to +2.2) | +4.0 (+1.4 to +6.6) |
  | Natural-language rules | +1.5 (−1.5 to +5.0) | −3.5 (−6.5 to −0.5) |

  On natural validation, iter 50 scored 94.4% and iter 150 scored 90.4%, against 94.0% for SFT, which is
  why iter 50 was chosen. On public rules PPO did no better than SFT: 25 of 32 for both PPO checkpoints,
  27 of 32 for SFT.
- **PPO raised false blocks on RL wording.** Against 2,000 synthetic transactions, false blocks per 1,000
  decisions on the 12-field RL test went from 7.96 for SFT to 15.49 at iter 50 (+1.76 to +14.65) and 14.12
  at iter 150 (+1.24 to +12.44). On the natural test they barely moved: 2.18 to 2.28 (−0.04 to +0.38) and
  2.29 (−0.03 to +0.38). The RL-wording rise is a real cost of using PPO.
- **Compiler.** Sampling 8 candidates and taking the most common behaviour scored 80.0% on the natural
  test, the same as a single answer. Sending rules below 75% agreement to a person holds back 20 of 200,
  catches 14 of the 40 wrong answers and delays 6 right ones. The other 180 rules are 85.6% correct,
  against 89.9% for v4. Strategy and threshold were chosen on v4 validation sets.
- **Engine.** p99 decision latency is 77–95 µs per transaction against 100 rules (median 79 µs over
  four runs), on one CPU core ([bench_engine.py](scripts/bench_engine.py)).

**How far to trust these tests.**

- **All older tests.** They have now been scored several times. Nothing was tuned on them after scoring,
  and iter 50 was chosen on validation first, but they are no longer untouched.
- **Natural test.** v5 is a regression check here, not a clean score, because I read 8 of v4's errors
  on it before building v5.
- **Public rules.** [human_rules.jsonl](eval/human_rules.jsonl) was frozen before scoring, but the test
  isn't blind. I chose the rules and wrote their answers, and I wrote the new fields' training wording
  after seeing them, removing only the closest matches. The scores are probably optimistic.
  - 22 rows are labelled verbatim and 10 trimmed. One verbatim row (human-10) had ", review" added, so 21
    are strictly verbatim. Quotes outside Stripe came through a summarising fetch tool.
  - Some rules are code-like syntax, not sentences.
  - The compiler on the chosen policy got 26 of 32. Review held back 4, and 25 of the other 28 were right.
- **Earlier public rules.** On [online_rules.jsonl](eval/online_rules.jsonl), v4 SFT got 15 of 20, v5 SFT
  and PPO iter 150 got 16, and PPO iter 50 and its compiler got 14. Those rules had already been studied,
  so this is not clean either.
- **New fields in the simulator.** They carry no fraud signal, so results on them show only that a rule
  is compiled correctly, not that it catches fraud.

**The scores to expect are the test scores.** For v4, validation said 96.0% on natural wording and the
test said 82.0%. For v5 PPO iter 50, validation said 94.4% and the test said 80.0%. Validation rules come
from the same generator as the training data, while the test's rules were written independently and
include shapes training rarely or never covers. That explanation fits the errors but has not been
tested. v5 SFT's accuracy falls with rule size on the natural test: 85.7% for one clause, 84.6% for two,
76.9% for three, 66.7% for four and 0 of 2 for five.

### Where the natural test goes wrong

Of v5 SFT's 43 mistakes, 25 use the right fields with a wrong value or structure, 7 drop a condition, 6
invent one, 4 have the wrong action and 1 fails the type check. Typical cases:

- **Grouping:** `(device new OR account new) AND amount` loses its brackets, so AND binds first.
- **Reversed intent:** "Don't hold small UPI payments, anything under ₹300 should go straight through"
  becomes a BLOCK.
- **Wrong comparison:** "exactly 100000 or more" becomes `amount == 100000`.
- **Dropped first conditions:** "Travel payments above ₹1,50,000" loses the merchant category.

These are real risks for a payments system, which is why rules should pass review and a backtest
before they run.

## What the experiments showed

**1. Wording diversity beat everything else.** The first model memorised templates: 16.3% on
differently worded validation requests. Better phrasing templates helped, but the old model still
got only 13.3% of hand-written rules right. Three new wording pools, 1,500 free-form sentences and a
wider grammar (actors, "only X can Y", exact amounts, spelled-out periods) took it to 76.7% on the
same rules, and v4 reached 93.3% (v5: 91.7%). Those 60 rules guided the grammar work, so those scores
are optimistic. On the natural test, v4 SFT beat the first SFT by 62.5 points (19.5% to 82.0%, +55.5 to
+69.5). v4 also has more parameters (360M against 135M), so that gain is not from wording alone.

**2. PPO's gains stay in the wording it trains on.** In an early round, PPO took SFT from 45.6% to 65.6%
on 500 unseen rules in its own phrasing (+20.0, +16.4 to +23.6; 102 gained, 2 lost), but only 36.2% to
36.8% on an unseen phrasing style (+0.6, −1.6 to +2.6). v5 shows the same split on a much stronger model.
Longer training gains on RL wording and loses on natural wording, as the Results table shows. The v4 run
gained nothing significant in either (−0.5 on natural, +0.4 on RL wording).

**3. A better reward model made PPO work, but only so far.** The same v5 SFT and PPO settings gave
different results with different reward models. A shortened reward model (1,300 steps, 7,318 pairs, on an
old 12-field backbone) gave no PPO gain in 50 iterations: its two checks (86.3%, 85.7%) never beat the
86.7% start, so it kept the SFT weights. The full one (9,276 pairs, on a backbone trained on v5 data) gave
88.0% at iteration 50 and 90.7% at the end of the resumed run. The two runs also differ in random
sampling, so this is suggestive, not proof.

How often the reward model ranks the right rule above a wrong one on the tests:

| Wrong rules used | Full v4 reward model | Full v5 reward model |
|---|---|---|
| Synthetic corruptions, RL wording | 98.0% (500 pairs) | 98.6% (500 pairs, new test) |
| SFT's real mistakes, RL wording | 97.9% (48 pairs) | 88.4% (86 pairs, new test) |
| SFT's real mistakes, natural test | 36.1% (36 pairs) | 44.2% (43 pairs) |

Each column's real mistakes come from that version's own SFT, so the pairs differ. On natural wording the
reward model is still below a coin flip on small samples, which is the likely reason PPO doesn't carry
over to natural text, though that hasn't been tested. A compiler that picks purely by reward score scored
80.0% on the natural test, level with the other strategies, up from 74.5% with the shortened model.

**4. An audit metric pointed the wrong way.** Across requests, an earlier reward model separated the
PPO model's right and wrong outputs with an AUC of only 0.70, suggesting a retrain. But Bradley-Terry
training only fixes score differences within one request, and PPO's value head absorbs each request's
baseline. Within a request it ranked the gold rule above every sampled mistake.
[audit_reward.py](scripts/audit_reward.py) reports both.

**5. Remaining errors are mostly exploration and wording.** PPO can only reinforce answers the
policy actually samples. In an earlier round, at least 91% of validation requests got all-correct or
all-wrong answers across 6 samples, so there was little for it to choose between.

## The parts around the model

All of this runs on synthetic or simulated data, so it shows the machinery working, not results on
real payments.

**Investigation agent** ([investigate.py](src/payrules/investigate.py)). A small tool loop explains
a flagged decision, finds what would change it, compares past cases, backtests the rule and proposes
a narrower one. It uses no LLM.

- On 50,000 unseen transactions with planted fraud, it escalated 1,002 of 3,615 alerts and caught 975 of
  978 fraud cases, cutting review work by 72.3%. Median time was 9.0 ms a case, p99 15.6 ms.
- On three novel fraud patterns it had no rule for, it caught 10%, 18% and 4%. Before the six new
  columns were added, it caught 17%, 66% and 3%. They carry no signal, and the unfamiliarity check
  measures distance across every column, which may be why it got worse. That hasn't been tested.
- Rule refinement raised precision from 44.5% to 99.1% and moved recall from 33.3% to 32.9%.
- A "clear" verdict should only lower a case's priority. It should never release a payment.

**Payment routing** ([routing.py](src/payrules/routing.py)). A Thompson-sampling bandit with
forgetting and shared information across payment contexts, on an invented simulator: 30 test seeds,
30,000 payments, settings tuned on separate seeds. Points are against a greedy baseline.

| Scenario | Pooled Thompson (points, interval) |
|---|---|
| Stationary | +0.65 (+0.35 to +0.92) |
| Drifting success rates | +2.56 (+1.97 to +3.16) |
| Gateway outage | +2.61 (+1.84 to +3.44) |
| Success varies by payment method | −0.29 (−0.85 to +0.27) |

During outages, 1.2% of traffic stayed on the failed gateway, against 13.1% for greedy and 84.9% for
fixed routing. No router won everywhere: Thompson without forgetting did better on the stationary and
by-method scenarios, and a no-context version did better on drift.

**DSL mining** ([dslmine.py](src/payrules/dslmine.py)). Mines typed operations from repeated syntax
trees in one half of a real Python codebase, then measures coverage on packages it never read.
Copy-paste detection of the same size is the baseline.

| Operations | Mined DSL | Copy-paste |
|---|---|---|
| 10 | 6.5% | 3.4% |
| 50 | 15.7% | 7.3% |
| 200 | 25.8% | 11.8% |
| 1,000 | 35.0% | 19.8% |

Unlimited copy-paste reaches 38.1% but needs 17,262 entries. Every rewrite passes an exact syntax-tree
round trip. It finds structural repetition only, not meaning.

## How it works

These are the stages behind the chosen v5 policy, with measured times.

| Stage | What it does | Time on an RTX 2050 |
|---|---|---|
| Data ([build_dataset_v5.py](scripts/build_dataset_v5.py)) | 16,000 training rows over 18 fields: three LLM-written wording pools (80%), older templates (20%), and free-form sentences. RL and test wording come from a separate pool. Each new field appears about 1,500 times and each older field about 3,000. | < 1 min |
| SFT ([train_sft.py](scripts/train_sft.py)) | LoRA (r=32) on SmolLM2-360M-Instruct, 3 epochs, fp16 autocast, lr 3e-4, epoch picked on 650 validation rows | 66 min |
| 135M backbone ([train_sft.py](scripts/train_sft.py)) | The same SFT on the smaller model, for the reward model to build on. Validation 82.9%. | 36 min |
| Pair mining ([mine_pairs.py](scripts/mine_pairs.py)) | Samples the policy 4 times on each of 4,000 RL requests. Wrong samples become pairs against the gold rule and against right samples. | 30 min |
| Reward model ([train_reward.py](scripts/train_reward.py)) | Bradley-Terry loss on 9,276 pairs, on the 135M backbone to save memory. Validation pair accuracy 94.3%. | 19 min |
| Audit ([audit_reward.py](scripts/audit_reward.py)) | Shortcut checks and ranking of real samples within a request: right over wrong 100% on RL wording (11 requests). | 1.5 min |
| PPO ([train_ppo.py](scripts/train_ppo.py)) | 50 iterations of 32 rollouts. Per-token KL penalty to frozen SFT, GAE, value head, running-std reward scaling, early stop on target KL. Keeps the best validation checkpoint. | 12 min |
| Compiler ([compiler.py](src/payrules/compiler.py)) | Samples candidates, drops invalid ones, groups them by proven-identical behaviour, and flags low agreement for review. | about 3 s per rule |

**Labels.** Preference labels come from an exact verifier, not human raters, but the policy never
sees the verifier. It trains only on the learned reward model's scores. The verifier's result is a
proof, not a sample: a comparison can only change outcome at a threshold, so testing one value from
each region, for every field, covers every possible transaction. (It falls back to sampling above
200,000 combinations.)

**Leakage guards.** [test_leakage.py](tests/test_leakage.py) and
[test_natural.py](tests/test_natural.py) fail if training pools share wording with the RL pool.
Rules from all frozen and validation sets are excluded from training. Every split has its own seeded
generator and evaluation splits are drawn first, so editing training wording cannot change them. The
hashes of the natural test and both public rule sets are in `eval/*.sha256` and were checked before
scoring.

**Memory.** Peak GPU memory is 5,936 MiB for SFT (the Windows driver serves the excess from shared
system memory), 2,792 MiB for the reward model and up to 4,320 MiB for PPO.

**A crashed PPO run.** The 150-iteration run crashed just after iteration 60 with a CUDA assert during
sampling, which fits NaN or inf probabilities, but the weights at that moment were lost. So "iter 150" is
really two runs: 50 iterations, then 100 more from that checkpoint, with a new seed and a fresh value head,
optimizer and reward scale. The resumed run finished normally, so the cause is unknown.
[train_ppo.py](scripts/train_ppo.py) now stops cleanly if the weights go non-finite. Iter 50, the chosen
policy, was saved before the crash and its weights are all finite.

**A collapsed run.** One 360M run at the default learning rate collapsed to 0% parsing after epoch 1,
probably from fp16 overflow, though the log lines that would prove it were not kept. It reran stably at
lr 3e-4, and [train_sft.py](scripts/train_sft.py) now stops after 20 non-finite losses.

## Reproduce

```bash
python -m venv .venv
.venv/Scripts/pip install torch --index-url https://download.pytorch.org/whl/cu126
.venv/Scripts/pip install -e ".[train,dev]"
PYTHON=.venv/Scripts/python.exe bash scripts/run_pipeline.sh    # about 2 h 45 min

.venv/Scripts/python.exe scripts/demo.py "Block payments above 50000 made over a VPN"
```

[run_pipeline.sh](scripts/run_pipeline.sh) has not been rerun as one script from a clean clone, though
each of its stages ran with these settings. Expect close but not identical numbers. The pipeline uses one
seed.

Tested with Python 3.13, torch 2.14.1+cu126, transformers 5.22.2 and peft 0.21.2.

## Limitations

- **The natural test is LLM-written, not human.** Its writer was asked to be unambiguous, and comes
  from the same model family as the validation writer and one training pool. Real analysts would
  likely be harder. It has 200 rules, so its intervals are wide.
- **The public-rule test is small and not blind.** It has 32 rules, chosen and answered by the
  author of this pipeline.
- **Conventions are built in.** "After 10pm" means `hour >= 22` (whole hours), a "new device" is under
  7 days, a "new account" is under 30 days, and "night" is before 6am.
- **Free-form training rows were spot-checked, not double-annotated:** about 110 of 1,750 read by hand,
  all correct.
- **Labels come from a verifier**, which is only possible because rule equivalence can be decided
  exactly. Human preferences would be noisier.
- **The 80% is not deployment-ready.** About 1 in 7 rules that pass the agreement check are still
  wrong, so a person should review every rule before it blocks a payment.

## Layout

```text
src/payrules/   schema, parser (Lark), type checker, engine, verifier, wording pools (natural,
                phrasings/), compiler, investigation agent, router, DSL miner, model helpers
scripts/        one script per stage, the eval_* and evaluate_final.py scorers, demo.py, run_pipeline.sh
tests/          parser, type checker, engine, verifier, data, leakage, compiler, agent, router, miner
eval/           natural_test and human_rules (frozen tests), online_rules (public rules, already studied),
                natural_val (validation), handwritten (development set)
```
