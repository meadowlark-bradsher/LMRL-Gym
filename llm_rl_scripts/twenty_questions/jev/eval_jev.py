"""Run BED Twenty Questions with Jev as the Noul likelihood (and, optionally, the
J/J oracle or the question selector).

Offline self-test (no API key, no network):
    PYTHONPATH=. python -m llm_rl_scripts.twenty_questions.jev.eval_jev --self-test

Build the one-time Jev Noul matrix (live, ~158 nouls calls):
    PYTHONPATH=. python -m llm_rl_scripts.twenty_questions.jev.eval_jev \
        --build-matrix --matrix llm_rl_scripts/twenty_questions/jev/matrix_jev.json

Play episodes (live oracle = Jev; deterministic oracle = matrix threshold):
    PYTHONPATH=. python -m llm_rl_scripts.twenty_questions.jev.eval_jev \
        --matrix <path> --oracle matrix --episodes 50
"""

from __future__ import annotations

import argparse
import json
import os
import random
import statistics
from dataclasses import asdict
from typing import Dict, List, Optional, Sequence

from . import bed
from .bed import (ATTRIBUTE, Hypothesis, build_question_bank, load_hypotheses,
                  make_likelihood, matrix_oracle, run_episode)
from . import matrix as matrix_mod


# --------------------------------------------------------------------------- #
# Hypothesis selection                                                         #
# --------------------------------------------------------------------------- #

def select_hypotheses(categories: Optional[List[str]], max_objects: Optional[int],
                      seed: int) -> List[Hypothesis]:
    hyps = load_hypotheses()
    if categories:
        wanted = set(categories)
        hyps = [h for h in hyps if h.category in wanted]
        if not hyps:
            raise SystemExit("no objects in categories {!r}".format(categories))
    if max_objects is not None and max_objects < len(hyps):
        rng = random.Random(seed)
        hyps = rng.sample(hyps, max_objects)
        hyps.sort(key=lambda h: (h.category, h.name))
    return hyps


# --------------------------------------------------------------------------- #
# Oracle construction                                                          #
# --------------------------------------------------------------------------- #

def make_oracle(kind: str, attribute_matrix, client=None, threshold: float = 0.5):
    if kind == "matrix":
        return matrix_oracle(attribute_matrix, threshold=threshold)
    if kind == "jev":
        from .policy import JevOracle
        if client is None:
            raise SystemExit("--oracle jev needs a live client")
        return JevOracle(client, closed_form_known=True)
    raise SystemExit("unknown oracle {!r}".format(kind))


# --------------------------------------------------------------------------- #
# Running                                                                      #
# --------------------------------------------------------------------------- #

def run_episodes(hyps: Sequence[Hypothesis], attribute_matrix, oracle_kind: str,
                 episodes: int, max_questions: int, confidence: float, eps: float,
                 seed: int, client=None, selector=None,
                 jsonl_path: Optional[str] = None, census: bool = False,
                 no_repeat: bool = False) -> dict:
    bank = build_question_bank(hyps)
    like = make_likelihood(attribute_matrix)
    oracle = make_oracle(oracle_kind, attribute_matrix, client=client, threshold=0.5)

    if census:
        targets = list(hyps)
    else:
        rng = random.Random(seed)
        targets = [rng.choice(hyps) for _ in range(episodes)]

    results: List[bed.EpisodeResult] = []
    jfile = open(jsonl_path, "w") if jsonl_path else None
    try:
        for i, true_h in enumerate(targets):
            res = run_episode(
                true_h, hyps, bank, like, oracle, max_questions=max_questions,
                confidence=confidence, eps=eps, selector=selector,
                no_repeat=no_repeat,
            )
            results.append(res)
            if jfile:
                rec = {
                    "episode": i, "true_name": res.true_name,
                    "true_category": res.true_category, "won": res.won,
                    "n_questions": res.n_questions, "reward": res.reward,
                    "final_top": res.final_top,
                    "steps": [asdict(s) for s in res.steps],
                }
                jfile.write(json.dumps(rec) + "\n")
            if (i + 1) % 10 == 0:
                print("  {}/{} episodes".format(i + 1, episodes), flush=True)
    finally:
        if jfile:
            jfile.close()

    return summarize(results, oracle_kind, oracle, selector)


def summarize(results: List[bed.EpisodeResult], oracle_kind: str, oracle,
              selector) -> dict:
    n = len(results)
    wins = [r for r in results if r.won]
    solve_rate = len(wins) / n if n else 0.0
    avg_q_all = statistics.mean(r.n_questions for r in results) if n else 0.0
    avg_q_won = statistics.mean(r.n_questions for r in wins) if wins else None
    avg_reward = statistics.mean(r.reward for r in results) if n else 0.0

    by_cat: Dict[str, List[bed.EpisodeResult]] = {}
    for r in results:
        by_cat.setdefault(r.true_category, []).append(r)
    per_cat = {
        c: {
            "n": len(rs),
            "solve_rate": sum(1 for r in rs if r.won) / len(rs),
            "avg_questions": statistics.mean(r.n_questions for r in rs),
        }
        for c, rs in sorted(by_cat.items())
    }

    summary = {
        "n_episodes": n,
        "oracle": oracle_kind,
        "solve_rate": solve_rate,
        "avg_questions_all": avg_q_all,
        "avg_questions_won": avg_q_won,
        "avg_reward": avg_reward,
        "per_category": per_cat,
    }
    if oracle_kind == "jev" and hasattr(oracle, "calls"):
        summary["oracle_api_calls"] = oracle.calls
    if selector is not None and getattr(selector, "picks", None):
        picks = selector.picks
        summary["selector"] = {
            "n_picks": len(picks),
            "match_at_1": sum(1 for p in picks if p["match_top"]) / len(picks),
            "mean_regret_bits": statistics.mean(p["regret_bits"] for p in picks),
        }
    return summary


# --------------------------------------------------------------------------- #
# Self-test                                                                    #
# --------------------------------------------------------------------------- #

def self_test() -> None:
    print("=== twenty_questions.jev self-test (offline, no API) ===")
    hyps = load_hypotheses()
    assert len(hyps) == 158, len(hyps)
    assert len({h.name for h in hyps}) == 158
    bank = build_question_bank(hyps)
    assert len(bank[bed.CATEGORY]) == 17
    assert len(bank[bed.ATTRIBUTE]) == len(bed.ATTRIBUTE_BANK)
    assert len(bank[bed.GUESS]) == 158
    by_name = {h.name: h for h in hyps}

    # EIG sanity on the uniform posterior: a balanced category question beats a
    # single guess (a guess only splits off one object).
    mtx = matrix_mod.synthetic_matrix(hyps)
    like = make_likelihood(mtx)
    post = bed.uniform_posterior(hyps)
    cat_q = next(q for q in bank[bed.CATEGORY] if q.key == "Animals")
    guess_q = bank[bed.GUESS][0]
    eig_cat = bed.expected_info_gain(cat_q, post, by_name, like)
    eig_guess = bed.expected_info_gain(guess_q, post, by_name, like)
    print("  EIG(category=Animals)={:.4f}  EIG(guess)={:.4f}".format(eig_cat, eig_guess))
    assert eig_cat > eig_guess > 0.0

    # Posterior update is a proper distribution and moves mass correctly.
    post2 = bed.posterior_update(post, cat_q, True, by_name, like)
    assert abs(sum(post2.values()) - 1.0) < 1e-9
    animals = [h.name for h in hyps if h.category == "Animals"]
    assert sum(post2[a] for a in animals) > 0.8, sum(post2[a] for a in animals)

    # Full games on the synthetic matrix with the self-consistent matrix oracle
    # must solve, and in far fewer than 20 questions.
    oracle = matrix_oracle(mtx)
    solved, qs = 0, []
    for h in hyps:
        res = run_episode(h, hyps, bank, like, oracle, max_questions=20)
        solved += res.won
        qs.append(res.n_questions)
        if res.won:
            assert res.final_top[0] == h.name
    print("  matrix-oracle full sweep: solved {}/{}, avg_q={:.2f}, max_q={}".format(
        solved, len(hyps), statistics.mean(qs), max(qs)))
    assert solved == len(hyps), "BED failed to solve some objects"
    assert statistics.mean(qs) < 12.0

    # Mock-backed likelihood path through the client (exercises nouls plumbing).
    from LLM_RL.jev.client import JevClient, MockJevBackend

    def noul_fn(state, qid, instr):
        # "...the question "<text>" ... is YES." -> crude keyword truth.
        return 0.9 if ("alive" in instr and "Cat" in state) else 0.1

    client = JevClient(MockJevBackend(noul_fn=noul_fn))
    small = [h for h in hyps if h.category in ("Animals", "Tools")][:6]
    sub_mtx = matrix_mod.build_attribute_matrix(
        client, small, attribute_bank=bed.ATTRIBUTE_BANK[:3], batch_size=30, log=False)
    assert set(sub_mtx) == {aid for aid, _ in bed.ATTRIBUTE_BANK[:3]}
    assert all(len(row) == len(small) for row in sub_mtx.values())
    print("  nouls plumbing ok: built {}x{} sub-matrix via MockJevBackend".format(
        len(sub_mtx), len(small)))

    # End-to-end summary shape on a small sample.
    summ = run_episodes(hyps, mtx, "matrix", episodes=20, max_questions=20,
                        confidence=0.9, eps=0.05, seed=0)
    assert summ["solve_rate"] == 1.0, summ
    print("  run_episodes(20) solve_rate={:.3f} avg_q_all={:.2f} avg_q_won={:.2f}".format(
        summ["solve_rate"], summ["avg_questions_all"], summ["avg_questions_won"]))
    print("ALL SELF-TESTS PASSED")


# --------------------------------------------------------------------------- #
# CLI                                                                          #
# --------------------------------------------------------------------------- #

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--build-matrix", action="store_true",
                    help="build the Jev Noul matrix and save to --matrix")
    ap.add_argument("--matrix", default="llm_rl_scripts/twenty_questions/jev/matrix_jev.json")
    ap.add_argument("--oracle", choices=["matrix", "jev"], default="matrix")
    ap.add_argument("--episodes", type=int, default=50)
    ap.add_argument("--census", action="store_true",
                    help="play every object once instead of sampling --episodes")
    ap.add_argument("--max-questions", type=int, default=20)
    ap.add_argument("--confidence", type=float, default=0.9)
    ap.add_argument("--eps", type=float, default=0.05)
    ap.add_argument("--selector", choices=["none", "jev"], default="none",
                    help="who picks the question: none=EIG argmax, jev=Jev Choice")
    ap.add_argument("--selector-history", action="store_true",
                    help="give the Jev selector the running Q&A transcript")
    ap.add_argument("--selector-norepeat", action="store_true",
                    help="drop already-asked questions from the selector shortlist")
    ap.add_argument("--categories", default=None,
                    help="comma-separated category names to restrict the game")
    ap.add_argument("--max-objects", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=30)
    ap.add_argument("--jsonl", default=None)
    ap.add_argument("--summary", default=None, help="write the summary JSON here")
    args = ap.parse_args()

    if args.self_test:
        self_test()
        return

    categories = [c.strip() for c in args.categories.split(",")] if args.categories else None
    hyps = select_hypotheses(categories, args.max_objects, args.seed)
    print("hypotheses: {} objects, {} categories".format(
        len(hyps), len({h.name for h in hyps}) and len({h.category for h in hyps})))

    client = None
    need_client = args.build_matrix or args.oracle == "jev" or args.selector == "jev"
    if need_client:
        from LLM_RL.jev.client import JevClient, HttpJevBackend
        client = JevClient(HttpJevBackend())

    if args.build_matrix:
        print("building Jev Noul matrix ({} objects x {} attributes)...".format(
            len(hyps), len(bed.ATTRIBUTE_BANK)))
        mtx = matrix_mod.build_attribute_matrix(
            client, hyps, batch_size=args.batch_size, log=True)
        payload = matrix_mod.matrix_payload(mtx, hyps, client.model)
        matrix_mod.save_matrix(args.matrix, payload)
        print("saved matrix -> {}".format(args.matrix))
        return

    payload = matrix_mod.load_matrix(args.matrix)
    attribute_matrix = payload["matrix"]
    # Keep only objects present in the matrix.
    have = set(payload["objects"])
    missing = [h.name for h in hyps if h.name not in have]
    if missing:
        raise SystemExit("matrix is missing objects: {}".format(missing[:5]))

    selector = None
    if args.selector == "jev":
        from .policy import JevSelector
        selector = JevSelector(client, history=args.selector_history)

    summ = run_episodes(
        hyps, attribute_matrix, args.oracle, episodes=args.episodes,
        max_questions=args.max_questions, confidence=args.confidence, eps=args.eps,
        seed=args.seed, client=client, selector=selector, jsonl_path=args.jsonl,
        census=args.census, no_repeat=args.selector_norepeat,
    )
    summ["config"] = {
        "oracle": args.oracle, "selector": args.selector,
        "selector_history": args.selector_history,
        "selector_norepeat": args.selector_norepeat,
        "confidence": args.confidence, "eps": args.eps,
        "max_questions": args.max_questions, "census": args.census,
        "n_objects": len(hyps),
    }
    print(json.dumps(summ, indent=2))
    if args.summary:
        os.makedirs(os.path.dirname(os.path.abspath(args.summary)), exist_ok=True)
        with open(args.summary, "w") as f:
            json.dump(summ, f, indent=2)
        print("saved summary -> {}".format(args.summary))


if __name__ == "__main__":
    main()
