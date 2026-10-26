#!/bin/zsh
# sequential count queue (W2 tasks 1-2)
PY=/Users/erdem/Documents/github/connectionMiner/outputs/connectionminer_ready_inputs/.venv/bin/python
cd /Users/erdem/Documents/github/connectionMiner/cm_minimal/experiments/bernfac/v4
L=/Users/erdem/Documents/github/connectionMiner/outputs/bernfac_20261020/v4/logs
$PY run_counts_v4.py pad poisson 120 10 80 old >> $L/queue_counts.out 2>&1
$PY run_counts_v4.py pad poisson 160 10 120 v4 80 >> $L/queue_counts.out 2>&1
$PY run_counts_v4.py pad nb 160 10 120 old >> $L/queue_counts.out 2>&1
$PY run_counts_v4.py pad poisson 120 30 80 old >> $L/queue_counts.out 2>&1
echo QUEUE_DONE >> $L/queue_counts.out
