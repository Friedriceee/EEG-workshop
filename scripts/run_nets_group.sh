#!/bin/sh
# FBCNet + ATCNet with the NEW validation (EEG_VAL=group): hold out whole subjects/sessions/runs to pick the best epoch.
# Waits for run_nets_all.sh to finish first so the two grids never compete for the GPU.
cd "$(dirname "$0")/.." || exit 1
PY=./.venv/bin/python
until grep -q ALL_DONE results/run_nets_all.progress 2>/dev/null; do sleep 60; done
run() {  # seed protocols models
  EEG_VAL=group EEG_SEED=$1 $PY -I scripts/eval_baselines.py "$2" "$3" "nets_grp_s$1_$2" > "results/nets_grp_s$1_$2.log" 2>&1
  echo "done seed=$1 protocols=$2 $(date +%H:%M)" >> results/run_nets_group.progress
}
: > results/run_nets_group.progress
for s in 33 34 35; do run $s xsess,xrun fbcnet,atcnet; done
for s in 33 34 35; do run $s xsubj fbcnet,atcnet; done
echo ALL_DONE >> results/run_nets_group.progress
