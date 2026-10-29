#!/usr/bin/env bash
# Exact V*_ask / V*_noask / perfect-information value by horizon for the registered kernel, buyer focal.
# H = 5 takes ~3 min; run from the repo root. No model calls.
set -euo pipefail
cd "$(dirname "$0")/../.."
PYTHONPATH=. python3 -m llm_rl_scripts.car_dealer_live.horizon_growth --horizons 1,2,3,4,5 --out results/live_phase0_horizon_growth_20261029/growth.json
