#!/usr/bin/env bash
# Runs every training stage in order: about 5-6 hours on a 4 GB RTX 2050.
# The final test-set evaluation is deliberately not here: run scripts/evaluate_final.py once,
# by hand, after all tuning is finished.
set -euo pipefail
PY=${PYTHON:-python}

$PY scripts/build_dataset.py
$PY -m pytest -q
$PY scripts/train_sft.py
$PY scripts/mine_pairs.py --rounds 2
$PY scripts/train_reward.py
$PY scripts/audit_reward.py | tee runs/audit_reward.txt
$PY scripts/train_ppo.py
$PY scripts/audit_reward.py --sft checkpoints/ppo_learned | tee runs/audit_reward_ppo.txt
