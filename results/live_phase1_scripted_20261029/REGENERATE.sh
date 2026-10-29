#!/usr/bin/env bash
# Regenerates the Phase 1 scripted baselines (four regimes), the costlessness check, the
# analyses and the rendered note. Run from the repo root. No model calls are made
# (the Jev arms are not run here; they are gated by the pre-registration).
set -euo pipefail
cd "$(dirname "$0")/../.."
OUT=results/live_phase1_scripted_20261029
PYTHONPATH=. python3 -u -m llm_rl_scripts.car_dealer_live.costlessness_check --horizon 4 --ask-cost 0 --sizes "2x2,3x2,2x3,3x3,4x9" --out $OUT/costlessness_h4_cq0.json
for spec in "4 0 h4_cq0" "2 0 h2_cq0" "3 0 h3_cq0" "4 3/100 h4_cq3"; do
  set -- $spec
  PYTHONPATH=. python3 -u -m llm_rl_scripts.car_dealer_live.run_phase1 --horizon $1 --ask-cost $2 \
      --arms exact_ask,exact_noask,greedy_eig,random --repeats 5 --seed 0 --out $OUT/$3
  PYTHONPATH=. python3 -m llm_rl_scripts.car_dealer_live.analyze_phase1 --dir $OUT/$3
done
PYTHONPATH=. python3 -m llm_rl_scripts.car_dealer_live.render_phase1_note \
    --dirs $OUT/h4_cq0,$OUT/h2_cq0,$OUT/h3_cq0,$OUT/h4_cq3 --out llm_rl_scripts/car_dealer_live/PHASE1_SCRIPTED_RESULTS.md --costlessness $OUT/costlessness_h4_cq0.json
