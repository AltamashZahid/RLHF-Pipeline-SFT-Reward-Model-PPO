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

v4 is the last 12-field model and v5 is the current 18-field model. v5 reads more kinds of rule, but it
is not more accurate on the rules v4 could already read.

| Test set | Rules | v4 SFT | v5 SFT | v5 − v4 (points) |
|---|---|---|---|---|
| Unseen rules in RL wording, some using the new fields | 500 | 55.4% | **82.8%** | +27.4 (+22.8 to +31.8) |
| Natural-language rules, LLM-written, 12 fields only | 200 | 82.0% | **78.5%** | −3.5 (−8.5 to +1.0) |
| Unseen rules in RL wording, 12 fields only | 500 | 90.4% | **86.8%** | −3.6 (−6.2 to −1.0) |
| Rules from public payment docs, new fields | 20 | 0 of 20 | **16 of 20** | see below |
| Rules from public payment docs, older fields | 12 | 9 of 12 | **11 of 12** | +2 rules (0 to +5) |

- **The new fields work.** v4 could not express them. In the first row, v5 gets 143 of the 202 rules
  that use a new field (70.8%).
- **Rules on the older fields got slightly worse.** The drop is significant on one 500-rule test and
  not on the natural test (8 rules gained, 15 lost). None of the 15 lost natural rules uses a new
  field, so the model is not confusing old fields with new ones. The cause isn't tested. A likely one
  is that the new fields took training rows away from the older ones.
- **The 12 public older-field rules don't settle this.** v5 gained 2 of them and lost none, but with 12
  rules that could be chance, and the larger tests point the other way.
- **False blocks did not change significantly.** Against 2,000 synthetic transactions, false blocks
  per 1,000 decisions went from 5.99 to 2.18 on the natural test (−13.94 to +2.38) and from 5.60 to 7.96
  on the older RL test (−2.10 to +7.65).
- **PPO changed nothing in v5.** The run was shortened to 50 iterations. Its two checks (86.3%, 85.7%)
  never beat the SFT starting point (86.7%), so it kept the SFT weights, and the saved PPO checkpoint
  is identical to SFT. In the full 150-iteration v4 run, PPO also made no significant difference:
  −0.5 points (−3.5 to +2.5) on the natural test and +0.4 (−1.2 to +2.0) on RL wording.
- **Compiler.** Sampling 8 candidates and taking the most common behaviour scored 78.5% on the natural
  test, the same as a single answer. Sending rules below 75% agreement to a person holds back 23 of 200,
  catches 16 of the 43 wrong answers and delays 7 right ones. The other 177 rules are 84.7% correct,
  against 89.9% for v4. Strategy and threshold were chosen on v4 validation sets.
- **Engine.** p99 decision latency is 77–95 µs per transaction against 100 rules (median 79 µs over
  four runs), on one CPU core ([bench_engine.py](scripts/bench_engine.py)).

**How far to trust these tests.**

- **Natural test.** v5 is a regression check here, not a clean score, because I read 8 of v4's errors
  on it before building v5.
- **Public rules.** [human_rules.jsonl](eval/human_rules.jsonl) was frozen before scoring, but the test
  isn't blind. I chose the rules and wrote their answers, and I wrote the new fields' training wording
  after seeing them, removing only the closest matches. The scores are probably optimistic.
  - 22 rows are labelled verbatim and 10 trimmed. One verbatim row (human-10) had ", review" added, so 21
    are strictly verbatim. Quotes outside Stripe came through a summarising fetch tool.
  - Some rules are code-like syntax, not sentences.
  - The compiler got 26 of 32. Review held back 2, and 25 of the other 30 were right.
- **Earlier public rules.** Scores on [online_rules.jsonl](eval/online_rules.jsonl) went from 15 of 20
  to 16 of 20. Those rules had already been studied, so this is not clean either.
- **New fields in the simulator.** They carry no fraud signal, so results on them show only that a rule
  is compiled correctly, not that it catches fraud.

**The scores to expect are the test scores.** For v4, validation said 96.0% on natural wording and the
test said 82.0%. For v5, validation said 94.0% and the test said 78.5%. Validation rules come from the
same generator as the training data, while the test's rules were written independently and include
shapes training rarely or never covers. That explanation fits the errors but has not been tested. v5's
accuracy falls with rule size on the natural test: 85.7% for one clause, 84.6% for two, 76.9% for three,
66.7% for four and 0 of 2 for five.

### Where the natural test goes wrong

Of v5's 43 mistakes, 25 use the right fields with a wrong value or structure, 7 drop a condition, 6
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

**2. PPO helps a weak policy, not a strong one.** In an earlier round, PPO took SFT from 45.6% to 65.6%
on 500 unseen rules in its own phrasing (+20.0, +16.4 to +23.6; 102 gained, 2 lost), but only 36.2% to
36.8% on an unseen phrasing style (+0.6, −1.6 to +2.6). Since then, SFT has reached 80–90% in PPO's
phrasing and samples the right answer most of the time, so there is little to reinforce. Neither later
run produced a significant gain. The short v5 run does not settle whether a longer one would help.

**3. The reward model shares the policy's blind spot.** Each figure below is how often it ranks the
right rule above a wrong one.

| Wrong rules used | Full v4 reward model | Shortened v5 reward model |
|---|---|---|
| Synthetic corruptions, RL wording | 98.0% (500 pairs) | 94.8% (500 pairs, new test) |
| SFT's real mistakes, RL wording | 97.9% (48 pairs) | 93.7% (223 pairs, new test) |
| Same 48 pairs as the v4 column | 97.9% | 70.8% |
| v4 SFT's real mistakes, natural test | 36.1% (36 pairs) | 16.7% (same 36 pairs) |

These are small samples, so treat them as warnings. The v5 model had a much shorter run on fewer mined
mistakes, which may explain the drop, but that hasn't been tested. A compiler that picks purely by
reward score did worst on the natural test in both rounds (v4 78.0%, v5 74.5%).

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

These are the v5 stages. Mining, the reward model and PPO were shortened to meet a deadline, and the
last column shows their measured times.

| Stage | What it does | Time on an RTX 2050 |
|---|---|---|
| Data ([build_dataset_v5.py](scripts/build_dataset_v5.py)) | 16,000 training rows over 18 fields: three LLM-written wording pools (80%), older templates (20%), and free-form sentences. RL and test wording come from a separate pool. Each new field appears about 1,500 times and each older field about 3,000. | < 1 min |
| SFT ([train_sft.py](scripts/train_sft.py)) | LoRA (r=32) on SmolLM2-360M-Instruct, 3 epochs, fp16 autocast, lr 3e-4, epoch picked on 650 validation rows | 66 min |
| Pair mining ([mine_pairs.py](scripts/mine_pairs.py)) | Samples the policy twice on each of 4,000 RL requests. Wrong samples become pairs against the gold rule and against right samples. | 15 min |
| Reward model ([train_reward.py](scripts/train_reward.py)) | Bradley-Terry loss on 7,318 pairs for 1,300 steps, on a 135M backbone to save memory. Validation pair accuracy 92.3%. | 11 min |
| Audit ([audit_reward.py](scripts/audit_reward.py)) | Shortcut checks and ranking of real samples within a request: right over wrong 90.9% on RL wording (11 requests). | 1.5 min |
| PPO ([train_ppo.py](scripts/train_ppo.py)) | 50 iterations of 32 rollouts. Per-token KL penalty to frozen SFT, GAE, value head, running-std reward scaling, early stop on target KL. Keeps the best validation checkpoint. | 13 min |
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
system memory), 2,796 MiB for the reward model and 3,826 MiB for PPO.

**A collapsed run.** One 360M run at the default learning rate collapsed to 0% parsing after epoch 1,
probably from fp16 overflow, though the log lines that would prove it were not kept. It reran stably at
lr 3e-4, and [train_sft.py](scripts/train_sft.py) now stops after 20 non-finite losses.

## Reproduce

```bash
python -m venv .venv
.venv/Scripts/pip install torch --index-url https://download.pytorch.org/whl/cu126
.venv/Scripts/pip install -e ".[train,dev]"
PYTHON=.venv/Scripts/python.exe bash scripts/run_pipeline.sh    # about 2 hours, plus a 135M backbone

.venv/Scripts/python.exe scripts/demo.py "Block payments above 50000 made over a VPN"
```

[run_pipeline.sh](scripts/run_pipeline.sh) has not been rerun from a clean clone. It also differs
slightly from the measured run: there, the reward model's backbone came from a model trained on an
earlier 12-field dataset whose build script no longer exists, so that backbone never saw the new
fields' wording. Expect close but not identical numbers. The pipeline uses one seed.

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
- **The 78.5% is not deployment-ready.** About 1 in 7 rules that pass the agreement check are still
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
