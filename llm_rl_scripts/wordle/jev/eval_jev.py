"""Benchmark Jev's Choice policy on Wordle against the exact-EIG optimum.

Jev picks each guess from the BED posterior (the words still consistent with the
feedback) via the Choice primitive -- no RL training. The posterior and the
exact-EIG optimum are maintained in pure code (bed.py), reusing the repo's own
game semantics, so the comparison is state-matched: at every turn Jev faces,
we know exactly what the optimal guess from that same posterior is.

Two kinds of result:

  * Selection quality (the core): on every turn Jev actually chose, how often did
    it pick an exact-EIG-optimal word (match@1), and how many bits of expected
    information did it leave on the table (regret = EIG_opt - EIG_jev)?
  * Whole game: solve rate and average guesses for the Jev arm vs a pure
    greedy-EIG optimal arm on the same hidden words.

On vocabularies with <= 255 words (e.g. wordle_official_200.txt) Jev chooses on
every turn, so both views are fully paired. On larger vocabularies Jev plays a
fixed model-free opener until the posterior drops under the Choice cap (see
policy.py), so the whole-game number is "Jev-from-shortlist" end to end while the
per-turn number still isolates Jev's own picks.

    # Offline (no key, validates the loop + scoring); 'optimal' mock should
    # match the optimal arm and score ~0 regret:
    conda run --no-capture-output -n LLM_RL python -u \
        -m llm_rl_scripts.wordle.jev.eval_jev --self-test --episodes 20 \
        --mock-strategy optimal

    # Live (needs TYPESAFE_API_KEY):
    conda run --no-capture-output -n LLM_RL python -u \
        -m llm_rl_scripts.wordle.jev.eval_jev --episodes 50 \
        --vocab llm_rl_scripts/wordle/vocab/wordle_official_200.txt
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path
from statistics import mean
from typing import Dict, List, Optional

from LLM_RL.jev.client import HttpJevBackend, JevClient, MockJevBackend
from llm_rl_scripts.wordle.env.game import N_TRIES, Vocabulary
from llm_rl_scripts.wordle.jev import bed
from llm_rl_scripts.wordle.jev.policy import JevWordlePolicy

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_VOCAB = "llm_rl_scripts/wordle/vocab/wordle_official_200.txt"


def resolve_vocab(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else (REPO_ROOT / p)


def load_vocab(path: str) -> Vocabulary:
    full = resolve_vocab(path)
    if not full.exists():
        raise SystemExit("vocab not found: {}".format(full))
    return Vocabulary.from_file(str(full), fill_cache=True)


# --- offline mock choosers (chooser(state, qid, options) -> chosen word) -------

def _mock_chooser(strategy: str, rng: random.Random):
    if strategy == "first":
        return lambda state, qid, options: options[0]
    if strategy == "random":
        return lambda state, qid, options: rng.choice(options)
    if strategy == "optimal":
        # options ARE the posterior shortlist, so the exact optimum is
        # computable right here -- a mock that should score zero regret and let
        # the harness validate its own scoring against the optimal arm.
        return lambda state, qid, options: bed.optimal_guesses(options)[0][0]
    raise SystemExit("unknown mock strategy: {}".format(strategy))


# --- arms ----------------------------------------------------------------------

def play_optimal(target: str, full_vocab: Vocabulary, rng: random.Random,
                 max_guesses: int) -> Dict:
    """Pure greedy-EIG optimal arm (restricted to the posterior, ties broken by
    rng) -- the repo's OptimalPolicy semantics, no Choice cap."""
    game = bed.FixedTargetWordle.new(target, full_vocab, max_guesses=max_guesses)
    while not game.done:
        best_words, _ = bed.optimal_guesses(game.posterior)
        game.step(rng.choice(best_words))
    return {"solved": game.solved,
            "guesses": len(game.guesses) if game.solved else None}


def play_jev(target: str, full_vocab: Vocabulary, policy: JevWordlePolicy,
             max_guesses: int) -> Dict:
    """The Jev arm. Records every turn Jev itself chose, with the optimal pick on
    that same posterior for scoring."""
    game = bed.FixedTargetWordle.new(target, full_vocab, max_guesses=max_guesses)
    turns: List[Dict] = []
    while not game.done:
        posterior = game.posterior
        decision = policy.choose(game)
        turn = {"source": decision.source,
                "shortlist_size": decision.shortlist_size,
                "word": decision.word,
                "confidence": decision.confidence}
        if decision.source == "jev" and decision.shortlist_size > 1:
            best_words, eig_opt = bed.optimal_guesses(posterior)
            eig_jev = bed.expected_info_gain(decision.word, posterior)
            turn.update({"match": decision.word in best_words,
                         "eig_jev": eig_jev,
                         "eig_opt": eig_opt,
                         "regret": eig_opt - eig_jev})
        turns.append(turn)
        game.step(decision.word)
    return {"solved": game.solved,
            "guesses": len(game.guesses) if game.solved else None,
            "turns": turns}


# --- reporting -----------------------------------------------------------------

def summarize(games: List[Dict]) -> Dict:
    n = len(games)
    solved = [g for g in games if g["solved"]]
    scored_turns = [t for g in games for t in g.get("turns", [])
                    if "match" in t]
    out = {
        "episodes": n,
        "solve_rate": len(solved) / n if n else 0.0,
        "avg_guesses_solved": mean(g["guesses"] for g in solved) if solved else None,
        "jev_decisions_scored": len(scored_turns),
    }
    if scored_turns:
        out["match_at_1"] = mean(1.0 if t["match"] else 0.0 for t in scored_turns)
        out["mean_regret_bits"] = mean(t["regret"] for t in scored_turns)
        out["max_regret_bits"] = max(t["regret"] for t in scored_turns)
    return out


def print_report(tag: str, jev_sum: Dict, opt_sum: Dict):
    def line(name, s):
        ag = s["avg_guesses_solved"]
        print("  {:9s} solve={:.3f}  avg_guesses={}  (n={})".format(
            name, s["solve_rate"],
            "{:.3f}".format(ag) if ag is not None else "  n/a",
            s["episodes"]))

    print("\n== Wordle x Jev  [{}] ==".format(tag))
    line("jev", jev_sum)
    line("optimal", opt_sum)
    if "match_at_1" in jev_sum:
        print("  selection (Jev turns only, n={}): match@1={:.3f}  "
              "mean_regret={:.3f} bits  max_regret={:.3f} bits".format(
                  jev_sum["jev_decisions_scored"], jev_sum["match_at_1"],
                  jev_sum["mean_regret_bits"], jev_sum["max_regret_bits"]))
    else:
        print("  (no scored Jev turns -- posterior never dropped under the "
              "Choice cap for any episode)")


def main():
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except (AttributeError, ValueError):
        pass

    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--vocab", default=DEFAULT_VOCAB)
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--model", default="jev-latest")
    p.add_argument("--backend", choices=["http", "mock"], default="http")
    p.add_argument("--max-options", type=int, default=255,
                   help="Choice cap; Jev plays a fixed opener above it")
    p.add_argument("--max-guesses", type=int, default=N_TRIES,
                   help="guess budget (default {}); lower it so chance-level "
                        "selection starts costing solve rate".format(N_TRIES))
    p.add_argument("--self-test", action="store_true",
                   help="offline: force --backend mock, no key or network")
    p.add_argument("--mock-strategy", choices=["first", "optimal", "random"],
                   default="optimal", help="offline chooser for the mock backend")
    p.add_argument("--out", default=None, help="write full per-episode JSON here")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    if args.self_test:
        args.backend = "mock"

    vocab = load_vocab(args.vocab)
    pool = sorted(vocab.all_vocab)
    rng = random.Random(args.seed)
    targets = [rng.choice(pool) for _ in range(args.episodes)]

    print("Wordle x Jev  vocab={} (|V|={})  episodes={}  seed={}  cap={}  "
          "max_guesses={}".format(
              os.path.basename(args.vocab), len(pool), args.episodes, args.seed,
              args.max_options, args.max_guesses))
    if len(pool) > args.max_options:
        print("  note: |V| > cap, so early turns use a fixed model-free opener; "
              "whole-game numbers are Jev-from-shortlist end to end.")

    if args.backend == "mock":
        backend = MockJevBackend(chooser=_mock_chooser(args.mock_strategy,
                                                       random.Random(args.seed + 1)))
        print("  backend=mock strategy={}".format(args.mock_strategy))
    else:
        backend = HttpJevBackend()
        print("  backend=http model={}".format(args.model))

    client = JevClient(backend=backend, model=args.model)
    policy = JevWordlePolicy(client, max_options=args.max_options)

    # Optimal arm gets its own rng so Jev's calls don't perturb its tie-breaks.
    opt_rng = random.Random(args.seed + 2)

    jev_games, opt_games = [], []
    for i, target in enumerate(targets):
        jg = play_jev(target, vocab, policy, args.max_guesses)
        og = play_optimal(target, vocab, opt_rng, args.max_guesses)
        jg["target"], og["target"] = target, target
        jev_games.append(jg)
        opt_games.append(og)
        if args.verbose:
            jgc = jg["guesses"]
            print("  [{:3d}] {}  jev={}({})  opt={}({})".format(
                i, target,
                "OK" if jg["solved"] else "XX", jgc if jgc else "-",
                "OK" if og["solved"] else "XX", og["guesses"] or "-"))

    jev_sum = summarize(jev_games)
    opt_sum = summarize(opt_games)
    print_report(args.mock_strategy if args.backend == "mock" else args.model,
                 jev_sum, opt_sum)

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump({"args": vars(args), "jev": jev_sum, "optimal": opt_sum,
                       "jev_games": jev_games, "optimal_games": opt_games},
                      f, indent=2)
        print("\nwrote {}".format(out_path))


if __name__ == "__main__":
    main()
