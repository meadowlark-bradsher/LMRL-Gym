#!/usr/bin/env bash
# Regenerates the CD-MIS-2 grid as registered in llm_rl_scripts/car_dealer_live/PREREGISTRATION_MIS2.md
# (tag prereg-mis2-20261030): six evasive flip-rate levels x eight arms and six behavioural severity
# levels x five arms, 144 cells x K=5 each, then the merge, analysis, verdicts and rendered note.
# No model calls; the twelve processes run in parallel (~5-10 min on 12 cores). Run from the repo root.
set -euo pipefail
export PYTHONHASHSEED=0   # belt and braces: the code orders every likelihood table, so results do not depend on it
cd "$(dirname "$0")/../.."
OUT=results/live_mis2_20261030
for p in evasive@0 evasive@0.1 evasive@0.2 evasive@0.3 evasive@0.4 evasive@0.5; do
  PYTHONPATH=. python3 -u -m llm_rl_scripts.car_dealer_live.run_mis --process $p --horizon 4 --ask-cost 0 --design CD-MIS-2 \
    --arms A_prior,B_naive,C_naive,C_mix0.2,C_ans0.1,C_ans0.2,C_ans0.3,O_true --repeats 5 --seed 41 --out $OUT/$p > $OUT/$p.log 2>&1 &
done
for p in history@0.1 history@0.2 history@0.4 stochastic@0.15 stochastic@0.3 stochastic@0.5; do
  PYTHONPATH=. python3 -u -m llm_rl_scripts.car_dealer_live.run_mis --process $p --horizon 4 --ask-cost 0 --design CD-MIS-2 \
    --arms A_prior,B_naive,B_mix0.2,C_naive,O_true --repeats 5 --seed 41 --out $OUT/$p > $OUT/$p.log 2>&1 &
done
wait
PYTHONPATH=. python3 -m llm_rl_scripts.car_dealer_live.merge_mis --dir $OUT
PYTHONPATH=. python3 -m llm_rl_scripts.car_dealer_live.analyze_mis --dir $OUT --B 5000 --seed 41
PYTHONPATH=. python3 -m llm_rl_scripts.car_dealer_live.verdicts_mis2 --analysis $OUT/analysis.json --mis1 results/live_mis_20261029/analysis.json
PYTHONPATH=. python3 -m llm_rl_scripts.car_dealer_live.render_mis2_note --dir $OUT --verdicts $OUT/verdicts.json --out llm_rl_scripts/car_dealer_live/MIS2_RESULTS.md
