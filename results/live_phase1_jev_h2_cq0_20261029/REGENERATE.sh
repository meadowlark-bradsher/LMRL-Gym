#!/usr/bin/env bash
# Regenerates this Jev result dir as registered in llm_rl_scripts/car_dealer_live/PREREGISTRATION_LIVE.md
# (tag prereg-live-20261029). Needs TYPESAFE_API_KEY; jev-latest is stochastic server-side, so a re-run
# reproduces the protocol and the analysis, not the exact decisions. Run from the repo root.
set -euo pipefail
cd "$(dirname "$0")/../.."
PYTHONPATH=. python3 -u -m llm_rl_scripts.car_dealer_live.run_phase1 --horizon 2 --ask-cost 0 --n-offers 3 \
    --arms exact_ask,exact_noask,greedy_eig,random,jev_ask_plain,jev_noask_plain --repeats 5 --seed 22 --jev-backend http --out results/live_phase1_jev_h2_cq0_20261029
PYTHONPATH=. python3 -m llm_rl_scripts.car_dealer_live.analyze_phase1 --dir results/live_phase1_jev_h2_cq0_20261029 --B 5000 --seed 21
