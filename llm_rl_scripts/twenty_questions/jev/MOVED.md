# Moved: this work now lives in `meadowlark-bradsher/elimination`

The 20Q elimination work in this directory moved, with its history, to the private repo
**`meadowlark-bradsher/elimination`** (package `elimination.twentyq`; the Jev client is
`elimination.clients.jev`). Its framing there is general elimination: question choice by
expected information gain, hard elimination behind the P0a gate, a Pitman-Yor novelty
detector, and escalation to an expert.

- **Archive tag:** `twentyq-archive-20261121` (commit `55713751`). This directory is the frozen
  record at that tag. **No new study starts here.** New studies (FLYW-8, FLYW-9 and later) start
  in the new repo.
- **Moved:** the history of this directory and `LLM_RL/jev/`, filtered to those paths. The new
  repo's `main` starts at `25d9b5d`; the change that makes it run standalone is
  `meadowlark-bradsher/elimination#1` (commit `a3e052d`, Phase 1).
- **Stays here:** `results/` (the committed runs), `matrix_jev.json`, the object list in
  `../env/data.py`, `EXTRACTION_PLAN.md`, and the 11 `prereg-*` tags. The new repo carries no
  LMRL-Gym content: it reads the objects and the matrix from a local LMRL-Gym checkout
  (`$LMRL_GYM_PATH`) and checks them against SHA-256s recorded at the tag.
- **Checked:** gates G1–G4 passed for the move (license scan over every blob in the new
  history; byte-identical regression against this code at the tag; the ST-3 end-game and FLYW-1
  cost analyses re-score exactly; no upstream import). The output is in the Phase 1 PR.
- `LLM_RL/jev/` is not frozen: Wordle and Car Dealer still use its client.

The plan is `EXTRACTION_PLAN.md` in this directory.
