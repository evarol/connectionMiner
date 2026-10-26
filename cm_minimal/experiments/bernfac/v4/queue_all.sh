#!/bin/zsh
# W2 master queue (replaces queue_counts.sh / queue_rest.sh after the first Poisson fit); one heavy process at a time
PY=/Users/erdem/Documents/github/connectionMiner/outputs/connectionminer_ready_inputs/.venv/bin/python
cd /Users/erdem/Documents/github/connectionMiner/cm_minimal/experiments/bernfac/v4
L=/Users/erdem/Documents/github/connectionMiner/outputs/bernfac_20261020/v4/logs
while kill -0 55586 2>/dev/null; do sleep 30; done
echo "== poisson k120 lam10 finished $(date)" >> $L/queue_all.out
for job in "run_counts_v4.py pad nb 160 10 120 old" \
           "run_counts_v4.py pad poisson 160 10 120 v4 80" \
           "run_offsets_v4.py 80 100 cold" \
           "run_offsets_v4.py 120 30 cold" \
           "run_neural_v4.py pg_affine_d20_w64_lv1_r1" \
           "run_neural_v4.py pg_affine_d40_w64_lv1_r1" \
           "run_neural_v4.py pg_affine_d40_w64_lv10_r1" \
           "run_counts_v4.py pad poisson 120 30 80 old" \
           "run_offsets_v4.py 120 100 cold"; do
  [[ -e $L/STOP ]] && { echo "STOP file: skipping $job" >> $L/queue_all.out; continue; }
  echo "== start $job $(date)" >> $L/queue_all.out
  $PY ${=job} >> $L/queue_all.out 2>&1
  echo "== end $job rc=$? $(date)" >> $L/queue_all.out
done
echo QUEUE_DONE >> $L/queue_all.out
