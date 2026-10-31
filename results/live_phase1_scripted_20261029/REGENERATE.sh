#!/usr/bin/env bash
# Regenerates the PRE-REGISTRATION Phase 1 scripted baselines (four regimes) on the two-candidate
# heuristic-only menu (SAFE + AMBITIOUS + two questions; the costlessness check also sizes the menu
# with BALANCED as a third candidate), the analyses and a rendered note. Run from the repo root. No
# model calls are made. These are the baselines that motivated the registered menu; the registered
# baselines live in results/live_phase1_scripted3_20261029/ and its REGENERATE.sh renders the
# committed PHASE1_SCRIPTED_RESULTS.md (run that one last). --heuristic-menu / --n-offers 2 pin the
# pre-2026-10-29 menu rule, which stopped being the runner's default when the registered menu landed.
set -euo pipefail
export PYTHONHASHSEED=0   # belt and braces: the code orders every likelihood table, so results do not depend on it
cd "$(dirname "$0")/../.."
OUT=results/live_phase1_scripted_20261029
PYTHONPATH=. python3 -u -m llm_rl_scripts.car_dealer_live.costlessness_check --horizon 4 --ask-cost 0 --sizes "2x2,3x2,2x3,3x3,4x9" --heuristic-menu --out $OUT/costlessness_h4_cq0.json
for spec in "4 0 h4_cq0" "2 0 h2_cq0" "3 0 h3_cq0" "4 3/100 h4_cq3"; do
  set -- $spec
  PYTHONPATH=. python3 -u -m llm_rl_scripts.car_dealer_live.run_phase1 --horizon $1 --ask-cost $2 --n-offers 2 --heuristic-menu \
      --arms exact_ask,exact_noask,greedy_eig,random --repeats 5 --seed 0 --out $OUT/$3
  PYTHONPATH=. python3 -m llm_rl_scripts.car_dealer_live.analyze_phase1 --dir $OUT/$3 --B 2000 --seed 0   # the first run's bootstrap (the default became 5000 later)
done
PYTHONPATH=. python3 -m llm_rl_scripts.car_dealer_live.render_phase1_note \
    --dirs $OUT/h4_cq0,$OUT/h2_cq0,$OUT/h3_cq0,$OUT/h4_cq3 --out llm_rl_scripts/car_dealer_live/PHASE1_SCRIPTED_RESULTS.md --costlessness $OUT/costlessness_h4_cq0.json
