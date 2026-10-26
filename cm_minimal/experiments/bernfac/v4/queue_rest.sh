#!/bin/zsh
# W2 tasks 3-4, started after the count queue finishes (one heavy process at a time)
PY=/Users/erdem/Documents/github/connectionMiner/outputs/connectionminer_ready_inputs/.venv/bin/python
cd /Users/erdem/Documents/github/connectionMiner/cm_minimal/experiments/bernfac/v4
L=/Users/erdem/Documents/github/connectionMiner/outputs/bernfac_20261020/v4/logs
until grep -q QUEUE_DONE $L/queue_counts.out 2>/dev/null; do sleep 60; done
$PY run_offsets_v4.py 80 100 cold >> $L/queue_rest.out 2>&1
$PY run_offsets_v4.py 120 30 cold >> $L/queue_rest.out 2>&1
$PY run_neural_v4.py pg_affine_d20_w64_lv1_r1 >> $L/queue_rest.out 2>&1
$PY run_neural_v4.py pg_affine_d40_w64_lv1_r1 >> $L/queue_rest.out 2>&1
$PY run_neural_v4.py pg_affine_d40_w64_lv10_r1 >> $L/queue_rest.out 2>&1
$PY run_offsets_v4.py 120 100 cold >> $L/queue_rest.out 2>&1
echo QUEUE_DONE >> $L/queue_rest.out
