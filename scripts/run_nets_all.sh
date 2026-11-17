#!/bin/sh
# FBCNet + ATCNet: finish the exam grid (3 seeds x xsess/xrun/xsubj). Seed 33 xsess/xrun already done (results/nets_s33.log).
# Order: seed 33 xsubj first, then seeds 34 and 35 on the two quick exams, then xsubj for 34 and 35.
cd "$(dirname "$0")/.." || exit 1
PY=./.venv/bin/python
run() {  # seed protocols models
  EEG_SEED=$1 $PY -I scripts/eval_baselines.py "$2" "$3" "nets_s$1_$2" > "results/nets_s$1_$2.log" 2>&1
  echo "done seed=$1 protocols=$2 $(date +%H:%M)" >> results/run_nets_all.progress
}
: > results/run_nets_all.progress
run 33 xsubj fbcnet,atcnet
run 34 xsess,xrun fbcnet,atcnet
run 35 xsess,xrun fbcnet,atcnet
run 34 xsubj fbcnet,atcnet
run 35 xsubj fbcnet,atcnet
echo ALL_DONE >> results/run_nets_all.progress
