#!/usr/bin/env bash
# Runs every training stage in order. The measured stages took about 1 h 46 min on a 4 GB RTX 2050; the
# 135M backbone below was not part of that time.
# The final test-set evaluation is deliberately not here: run scripts/evaluate_final.py and
# scripts/eval_online.py once, by hand, after all tuning is finished.
set -euo pipefail
PY=${PYTHON:-python}

PYTHONPATH=scripts $PY scripts/build_dataset_v5.py
$PY -m pytest -q
PAYRULES_BASE_MODEL=HuggingFaceTB/SmolLM2-360M-instruct $PY scripts/train_sft.py --data data/v5 \
  --out checkpoints/v5/sft360 --log runs/v5_sft360.jsonl --lr 3e-4 --select-on data/v5/select.jsonl --eval-rows 650
# The measured run took this 135M backbone from an earlier dataset whose build script no longer exists,
# so a rerun differs slightly here.
$PY scripts/train_sft.py --data data/v5 --out checkpoints/v5/sft135 --log runs/v5_sft135.jsonl
# Mining, the reward model and PPO were shortened to meet a deadline. The earlier full run used
# --rounds 4, no step limit and --iterations 150.
$PY scripts/mine_pairs.py --checkpoint checkpoints/v5/sft360 --data data/v5 --tag mined --rounds 2 --sample-pairs
$PY scripts/train_reward.py --sft checkpoints/v5/sft135 --data data/v5 --mined mined --max-steps 1300 \
  --out checkpoints/v5/reward --log runs/v5_reward.jsonl
$PY scripts/audit_reward.py --reward-model checkpoints/v5/reward --sft checkpoints/v5/sft360 \
  --rows data/v5/sft_rl_val.jsonl | tee runs/audit_reward.txt
$PY scripts/train_ppo.py --sft checkpoints/v5/sft360 --reward-model checkpoints/v5/reward --data data/v5 \
  --out checkpoints/v5/ppo --log runs/v5_ppo.jsonl --iterations 50
