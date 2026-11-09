#!/usr/bin/env bash
# Runs every training stage in order: about 2 h 45 min on a 4 GB RTX 2050.
# The final test-set evaluation is deliberately not here: run scripts/evaluate_final.py and
# scripts/eval_online.py by hand after all tuning is finished.
set -euo pipefail
PY=${PYTHON:-python}

PYTHONPATH=scripts $PY scripts/build_dataset_v5.py
$PY -m pytest -q
PAYRULES_BASE_MODEL=HuggingFaceTB/SmolLM2-360M-instruct $PY scripts/train_sft.py --data data/v5 \
  --out checkpoints/v5/sft360 --log runs/v5_sft360.jsonl --lr 3e-4 --select-on data/v5/select.jsonl --eval-rows 650
$PY scripts/train_sft.py --data data/v5 --out checkpoints/v5/sft135 --log runs/v5_sft135.jsonl \
  --select-on data/v5/select.jsonl --eval-rows 650
$PY scripts/mine_pairs.py --checkpoint checkpoints/v5/sft360 --data data/v5 --tag mined4 --rounds 4 --sample-pairs
$PY scripts/train_reward.py --sft checkpoints/v5/sft135 --data data/v5 --mined mined4 \
  --out checkpoints/v5_full/reward --log runs/v5_full_reward.jsonl
$PY scripts/audit_reward.py --reward-model checkpoints/v5_full/reward --sft checkpoints/v5/sft360 \
  --rows data/v5/sft_rl_val.jsonl | tee runs/audit_reward.txt
# 50 iterations gives the chosen policy. Running longer raised scores on RL wording and lowered them on
# natural wording, and it keeps the best checkpoint by RL-wording validation only.
$PY scripts/train_ppo.py --sft checkpoints/v5/sft360 --reward-model checkpoints/v5_full/reward --data data/v5 \
  --out checkpoints/v5_full/ppo --log runs/v5_full_ppo.jsonl --iterations 50
