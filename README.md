# payrules

payrules turns plain-English payment-risk rules into rules a typed engine can run. It uses a
360M-parameter model trained with the full RLHF recipe: supervised fine-tuning (SFT), a
Bradley-Terry reward model, and PPO written from scratch in PyTorch. Everything trains on a laptop
RTX 2050 with 4 GB of VRAM. An investigation agent, a payment router and a DSL miner sit around it.

```text
Request:   Block UPI payments above 50000 from rooted phones
Rule:      WHEN txn.method == UPI AND txn.amount > 50000 AND device.is_rooted
           THEN BLOCK reason="upi_method_high_amount"
Agreement: 89% of 9 valid candidates behave this way
```

Every rule is parsed and type-checked before it runs. A rule that reads a field that doesn't
exist, or compares a boolean with a number, is rejected. A deterministic engine then executes the
rules and explains each decision.

## Results

Each test set was scored once, after tuning finished. A prediction counts as correct only if the
verifier proves it makes the same decision as the correct rule on every possible transaction.
Intervals are 95% paired bootstrap intervals over rules.

| Test set | Rules | Old model (v2): SFT / PPO | Final SFT | Final PPO | PPO − SFT (points) |
|---|---|---|---|---|---|
| Natural-language rules, written separately and frozen before training | 200 | 19.5% / 22.5% | **82.0%** | 81.5% | −0.5 (−3.5 to +2.5) |
| Unseen rules, phrased like the PPO training requests | 500 | see below | 90.4% | 90.8% | +0.4 (−1.2 to +2.0) |

- **The wording work is what moved accuracy.** On the natural test, final SFT beats old SFT by
  62.5 points (+55.5 to +69.5): it got 129 more rules right and 4 fewer. The final model also has
  more parameters (360M against 135M), so the gain is not from wording alone. On earlier
  validation sets, wider wording alone took the 135M model from 13.3% to 76.7% on the development
  rules, and moving to 360M then added 1.6 points.
- **PPO added nothing measurable to accuracy this round.** It gained 4 rules and lost 5 on the
  natural test, and gained 10 and lost 8 on the other. See finding 2.
- **False blocks.** Against 2,000 synthetic transactions, PPO's false blocks per 1,000 decisions
  fell from 5.99 to 1.78 on the natural test (−4.21, interval −12.65 to +0.01, so not significant)
  and from 5.60 to 4.72 on the other (−0.88, interval −2.54 to −0.004, a very small effect).
- **Compiler.** Sampling 8 candidates and taking the most common behaviour scored 82.5% on the
  natural test, 1.0 point above PPO's single greedy answer (−1.5 to +3.5, not significant). The
  useful part is the review signal: sending rules below 75% agreement to a person holds back 22 of
  200 rules, catches 17 of the 35 wrong answers and delays 5 right ones. The other 178 rules are
  89.9% correct. Strategy and threshold were chosen on validation sets.
- **Engine.** p99 decision latency is 79–161 µs per transaction against 100 rules (median 130 µs
  over four runs), on one CPU core ([bench_engine.py](scripts/bench_engine.py)).

**The scores to expect are the test scores.** Validation said 96.0% on natural wording and the test
said 82.0%. Validation picked the checkpoint and was written from rules drawn by the same generator
as the training data. The test's rules were written independently and include shapes training
rarely or never covers, such as five-clause rules (0 of 2 right). That explanation fits the errors
but has not been tested. PPO's accuracy falls with rule size on the natural test: 90.5% for one
clause, 92.3% for two, 76.9% for three, 66.7% for four.

### Where the natural test goes wrong

Of PPO's 37 mistakes, 17 use the right fields with a wrong value or structure, 9 invent a condition,
7 drop one, 2 have the wrong action, 1 uses a wrong field and 1 fails the type check. Typical cases:

- **Grouping:** `(device new OR account new) AND amount` loses its brackets, so AND binds first.
- **Inclusive ranges:** "1am to 4am inclusive" becomes `hour < 4`.
- **Extra conditions:** "even in the night" becomes a required `hour < 6`.
- **Dropped first conditions:** "Travel bookings above 5 lakh" loses the merchant category.

These are real risks for a payments system, which is why rules should pass review and a backtest
before they run.

## What the experiments showed

**1. Wording diversity beat everything else.** The first model memorised templates: 16.3% on
differently worded validation requests. Better phrasing templates helped, but the old model still
got only 13.3% of hand-written rules right. Three new wording pools, 1,500 free-form sentences and a
wider grammar (actors, "only X can Y", exact amounts, spelled-out periods) took it to 76.7% on the
same rules, and the final model reaches 93.3%. Those 60 rules guided the grammar work, so that score
is optimistic.

**2. PPO helps a weak policy, not a strong one.** In the earlier round, PPO took SFT from 45.6% to
65.6% on 500 unseen rules in its own phrasing (+20.0, +16.4 to +23.6; 102 gained, 2 lost), but only
36.2% to 36.8% on an unseen phrasing style (+0.6, −1.6 to +2.6). Now SFT already gets about 90% in
PPO's phrasing and samples the right answer most of the time, so there is little to reinforce. The
PPO run moved validation from 88.3% to 89.3% (3 more rules of 300, best of 7 checks), and that did
not hold on the test.

**3. The reward model shares the policy's blind spot.** On the wording it trained on, it ranks the
right rule above synthetic corruptions 98.0% of the time (500 pairs) and above SFT's real mistakes
97.9% (48 pairs). On the natural test it picks the right rule over SFT's real mistakes only 36.1% of
the time (36 pairs). That is a small sample, so treat it as a warning, not a precise figure. A
compiler that picks purely by reward score also did worst there, at 78.0%. It had already been
rejected on validation (90.4% on natural wording against 96.8%).

**4. An audit metric pointed the wrong way.** Across requests, the old reward model separated the
PPO model's right and wrong outputs with an AUC of only 0.70, suggesting a retrain. But Bradley-Terry
training only fixes score differences within one request, and PPO's value head absorbs each request's
baseline. Within a request it ranked the gold rule above every sampled mistake.
[audit_reward.py](scripts/audit_reward.py) reports both.

**5. Remaining errors are mostly exploration and wording.** PPO can only reinforce answers the
policy actually samples. In the earlier round, at least 91% of validation requests got all-correct
or all-wrong answers across 6 samples, so there was little for it to choose between.

## The parts around the model

All of this runs on synthetic or simulated data, so it shows the machinery working, not results on
real payments.

**Investigation agent** ([investigate.py](src/payrules/investigate.py)). A small tool loop explains
a flagged decision, finds what would change it, compares past cases, backtests the rule and proposes
a narrower one. It uses no LLM.

- On 50,000 unseen transactions with planted fraud, it escalated 991 of 3,615 alerts and caught all
  978 fraud cases, cutting review work by 72.6%. Median time was 15.3 ms a case, p99 33.8 ms.
- On three novel fraud patterns it had no rule for, it caught 17%, 66% and 3%.
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

| Stage | What it does | Time on an RTX 2050 |
|---|---|---|
| Data ([build_dataset_v4.py](scripts/build_dataset_v4.py)) | 16,000 training rows: three LLM-written wording pools (80%), older templates (20%), and 1,500 free-form sentences. RL and test wording come from a separate pool. | < 1 min |
| SFT ([train_sft.py](scripts/train_sft.py)) | LoRA (r=32) on SmolLM2-360M-Instruct, 3 epochs, fp16 autocast, lr 3e-4, epoch picked on natural validation | 61 min |
| Pair mining ([mine_pairs.py](scripts/mine_pairs.py)) | Samples the policy on each of 4,000 RL requests. Wrong samples become pairs against the gold rule and against right samples. | 31 min |
| Reward model ([train_reward.py](scripts/train_reward.py)) | Bradley-Terry loss over 10,231 pairs, on a 135M backbone to save memory. Validation pair accuracy 93.3%. | 43 min |
| Audit ([audit_reward.py](scripts/audit_reward.py)) | Shortcut checks and ranking of real samples within a request: right over wrong 88.9% on RL wording (9 requests), 81.3% on natural wording (16). | 4 min |
| PPO ([train_ppo.py](scripts/train_ppo.py)) | 150 iterations of 32 rollouts. Per-token KL penalty to frozen SFT, GAE, value head, running-std reward scaling, early stop on target KL. Keeps the best validation checkpoint. | 1.8 h |
| Compiler ([compiler.py](src/payrules/compiler.py)) | Samples candidates, drops invalid ones, groups them by proven-identical behaviour, and flags low agreement for review. | about 3 s per rule |

**Labels.** Preference labels come from an exact verifier, not human raters, but the policy never
sees the verifier. It trains only on the learned reward model's scores. The verifier's result is a
proof, not a sample: a comparison can only change outcome at a threshold, so testing one value from
each region, for every field, covers every possible transaction. (It falls back to sampling above
200,000 combinations.)

**Leakage guards.** [test_leakage.py](tests/test_leakage.py) and
[test_natural.py](tests/test_natural.py) fail if training pools share wording with the RL pool.
Rules from the natural test and validation sets are excluded from training. Every split has its own
seeded generator and evaluation splits are drawn first, so editing training wording cannot change
them. The natural test's hash is in [natural_test.sha256](eval/natural_test.sha256) and was checked
before scoring.

**Memory.** Peak GPU memory is 6.0 GB for SFT (the Windows driver serves the excess from shared system
memory), 2.8 GB for the reward model and 4.2 GB for PPO.

**A collapsed run.** One 360M run at the default learning rate collapsed to 0% parsing after epoch 1,
probably from fp16 overflow, though the log lines that would prove it were not kept. It reran stably at
lr 3e-4, and [train_sft.py](scripts/train_sft.py) now stops after 20 non-finite losses.

## Reproduce

```bash
python -m venv .venv
.venv/Scripts/pip install torch --index-url https://download.pytorch.org/whl/cu126
.venv/Scripts/pip install -e ".[train,dev]"
PYTHON=.venv/Scripts/python.exe bash scripts/run_pipeline.sh    # about 5 hours

.venv/Scripts/python.exe scripts/demo.py "Block UPI payments above 50000 from rooted phones"
```

[run_pipeline.sh](scripts/run_pipeline.sh) has not been rerun from a clean clone. It also differs
slightly from the measured run: there, the reward model's backbone and some extra mined mistakes came
from models trained on an earlier dataset whose build script no longer exists. Expect close but not
identical numbers. The pipeline uses one seed.

Tested with Python 3.13, torch 2.14.1+cu126, transformers 5.22.2 and peft 0.21.2.

## Limitations

- **The natural test is LLM-written, not human.** Its writer was asked to be unambiguous, and comes
  from the same model family as the validation writer and one training pool. Real analysts would
  likely be harder. It has 200 rules, so its intervals are wide.
- **Conventions are built in.** "After 10pm" means `hour >= 22` (whole hours), a "new device" is under
  7 days, a "new account" is under 30 days, and "night" is before 6am.
- **Free-form training rows were spot-checked, not double-annotated:** about 110 of 1,750 read by hand,
  all correct.
- **Labels come from a verifier**, which is only possible because rule equivalence can be decided
  exactly. Human preferences would be noisier.
- **The 82% is not deployment-ready.** About 1 in 10 rules that pass the agreement check are still
  wrong, so a person should review every rule before it blocks a payment.

## Layout

```text
src/payrules/   schema, parser (Lark), type checker, engine, verifier, wording pools (natural,
                phrasings/), compiler, investigation agent, router, DSL miner, model helpers
scripts/        one script per stage, the eval_* and evaluate_final.py scorers, demo.py, run_pipeline.sh
tests/          parser, type checker, engine, verifier, data, leakage, compiler, agent, router, miner
eval/           natural_test (frozen test), natural_val (validation), handwritten (development set)
```
