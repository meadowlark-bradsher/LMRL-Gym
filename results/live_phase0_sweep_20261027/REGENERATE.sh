#!/usr/bin/env bash
# Regenerates this result dir and the rendered Phase 0 note from scratch (exact oracle; ~1.7 h on a laptop).
# Run from the repo root. No model calls are made.
set -euo pipefail
cd "$(dirname "$0")/../.."
PYTHONPATH=. python3 -u -m llm_rl_scripts.car_dealer_live.sweep_ask_value --out results/live_phase0_sweep_20261027/sweep.json
PYTHONPATH=. python3 -m llm_rl_scripts.car_dealer_live.render_phase0_note --sweep results/live_phase0_sweep_20261027/sweep.json --out llm_rl_scripts/car_dealer_live/PHASE0_RESULTS.md
