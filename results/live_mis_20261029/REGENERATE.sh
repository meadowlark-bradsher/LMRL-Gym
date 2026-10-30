#!/usr/bin/env bash
# Regenerates the CD-MIS-1 grid as registered in llm_rl_scripts/car_dealer_live/PREREGISTRATION_MIS.md
# (tag prereg-mis-20261029): seven true opponent processes x nine arms x 144 cells x K=5, then the merge,
# analysis, verdicts and rendered note. No model calls; ~2.5 h serial, ~30 min with the processes in
# parallel. Run from the repo root.
set -euo pipefail
cd "$(dirname "$0")/../.."
OUT=results/live_mis_20261029
for p in matched threshold stochastic anchoring withholding history evasive; do
  PYTHONPATH=. python3 -u -m llm_rl_scripts.car_dealer_live.run_mis --process $p --horizon 4 --ask-cost 0 \
    --arms A_prior,B_naive,B_temp0.5,B_temp0.75,B_mix0.05,B_mix0.2,C_naive,C_mix0.2,O_true --repeats 5 --seed 41 \
    --out $OUT/$p &
done
wait
PYTHONPATH=. python3 -m llm_rl_scripts.car_dealer_live.merge_mis --dir $OUT
PYTHONPATH=. python3 -m llm_rl_scripts.car_dealer_live.analyze_mis --dir $OUT --B 5000 --seed 41
PYTHONPATH=. python3 -m llm_rl_scripts.car_dealer_live.verdicts_mis --analysis $OUT/analysis.json
PYTHONPATH=. python3 -m llm_rl_scripts.car_dealer_live.render_mis_note --dir $OUT --verdicts $OUT/verdicts.json --out llm_rl_scripts/car_dealer_live/MIS_RESULTS.md
