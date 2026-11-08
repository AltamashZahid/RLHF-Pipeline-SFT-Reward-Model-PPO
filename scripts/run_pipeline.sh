#!/usr/bin/env bash
# Runs every training stage in order: about 5 hours on a 4 GB RTX 2050.
# The final test-set evaluation is deliberately not here: run scripts/evaluate_final.py once,
# by hand, after all tuning is finished.
set -euo pipefail
PY=${PYTHON:-python}

PYTHONPATH=scripts $PY scripts/build_dataset_v4.py
$PY -m pytest -q
PAYRULES_BASE_MODEL=HuggingFaceTB/SmolLM2-360M-instruct $PY scripts/train_sft.py --data data/v4 \
  --out checkpoints/v4/sft360 --log runs/v4_sft360.jsonl --lr 3e-4 --select-on eval/natural_val.jsonl
# The measured run took this 135M backbone, and extra mined mistakes from a weaker 360M policy, from
# an earlier dataset. That dataset's build script no longer exists, so a rerun differs slightly here.
$PY scripts/train_sft.py --data data/v4 --out checkpoints/v4/sft135 --log runs/v4_sft135.jsonl
$PY scripts/mine_pairs.py --checkpoint checkpoints/v4/sft360 --data data/v4 --tag mined --rounds 4 --sample-pairs
$PY scripts/train_reward.py --sft checkpoints/v4/sft135 --data data/v4 --mined mined \
  --out checkpoints/v4/reward --log runs/v4_reward.jsonl
$PY scripts/audit_reward.py --reward-model checkpoints/v4/reward --sft checkpoints/v4/sft360 \
  --rows data/v4/sft_rl_val.jsonl | tee runs/audit_reward.txt
$PY scripts/train_ppo.py --sft checkpoints/v4/sft360 --reward-model checkpoints/v4/reward --data data/v4 \
  --out checkpoints/v4/ppo --log runs/v4_ppo.jsonl --iterations 150
