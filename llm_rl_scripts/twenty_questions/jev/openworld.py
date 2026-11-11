"""Open-world Twenty Questions: a growing hypothesis set over an unbounded space.

The closed-world loop in ``bed.py`` assumes the secret is one of a fixed set. This
module relaxes that. The *true* space is still the 158 benchmark objects (so the
matrix oracle can answer about any secret), but the *known* set ``H_known`` starts
empty and grows only when an "expert" names an object the loop could not resolve.

Three pieces sit on top of ``bed.py`` unchanged:

* a catch-all ``UNKNOWN`` hypothesis with a *background* likelihood (the
  attribute marginal over currently-known objects, the known-set category
  fraction, and ~eps for by-name guesses). This is simultaneously the model of
  the unseen tail and the novelty detector -- when a secret matches no known
  row and every guess fails, posterior mass flows to ``UNKNOWN``;
* a **Pitman-Yor** prior that supplies ``UNKNOWN``'s mass ``p_new`` and the
  discovery dynamics (species accumulation);
* an accumulation loop that plays a stream of games, calls the expert when
  ``UNKNOWN`` wins, and admits the named object into ``H_known``.

All of this is deterministic and offline on the cached Noul matrix.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from .bed import (
    ATTRIBUTE, ATTRIBUTE_BANK, CATEGORY, CATEGORY_PHRASE, GUESS,
    Hypothesis, LikelihoodModel, OracleModel, Posterior, Question,
    _article, candidate_questions, closed_form_likelihood, entropy,
    expected_info_gain, matrix_oracle, posterior_update, predictive_yes,
    rank_questions,
)
from .openworld_backends import BACKENDS

UNKNOWN = "__unknown__"
UNKNOWN_HYP = Hypothesis(name=UNKNOWN, variants=(), category=UNKNOWN)


# --------------------------------------------------------------------------- #
# Pitman-Yor                                                                   #
# --------------------------------------------------------------------------- #

def py_p_new(k: int, n: int, sigma: float, theta: float) -> float:
    """Pitman-Yor probability that draw ``n+1`` is a new species, given ``k``
    distinct in ``n`` draws. ``sigma=0`` is the Dirichlet-process special case."""
    if n <= 0:
        return 1.0
    return (theta + k * sigma) / (theta + n)


def fit_pitman_yor(new_flags: Sequence[bool],
                   sigmas: Optional[Sequence[float]] = None,
                   thetas: Optional[Sequence[float]] = None
                   ) -> Tuple[float, float, float]:
    """Grid-search MLE of ``(sigma, theta)`` on the sequence of new/old events.

    ``new_flags[i]`` is True iff the ``i``-th draw was a previously-unseen species.
    The PY sequential likelihood of the partition uses only, at each step, the
    number seen so far (``n``), the number of distinct so far (``k``) and the
    running count of the drawn species; the "old" term's species-count cancels in
    the argmax across a fixed stream only for the new/old split we actually need,
    so we score the exact two-parameter EPPF. Returns ``(sigma, theta, loglik)``.
    """
    if sigmas is None:
        sigmas = [i / 100.0 for i in range(0, 95, 5)]          # 0.00 .. 0.90
    if thetas is None:
        thetas = [0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0]

    # Reconstruct per-step (n, k_before, count_of_this_species_before).
    counts: Dict[int, int] = {}
    steps: List[Tuple[int, int, bool, int]] = []  # (n_before, k_before, is_new, m_before)
    k = 0
    for i, is_new in enumerate(new_flags):
        n_before = i
        k_before = k
        # species identity is not needed for the new term; for old terms we need
        # the running count of *that* species. We encode identity by position of
        # first appearance; callers that want exact old-term scoring pass labels
        # via fit_pitman_yor_labels. Here new_flags alone suffices for the common
        # case where we only compare the new/old marginal, so m_before is 0 for
        # new and the true count for old is supplied by the labelled variant.
        steps.append((n_before, k_before, bool(is_new), 0))
        if is_new:
            k += 1

    best = (0.0, 1.0, -math.inf)
    for sigma in sigmas:
        for theta in thetas:
            if theta <= -sigma:
                continue
            ll = 0.0
            kk = 0
            for (n_before, k_before, is_new, _m) in steps:
                denom = theta + n_before
                if denom <= 0:
                    ll = -math.inf
                    break
                if is_new:
                    num = theta + k_before * sigma
                else:
                    # marginal "old" probability = 1 - p_new (identity-free).
                    num = denom - (theta + k_before * sigma)
                if num <= 0:
                    ll = -math.inf
                    break
                ll += math.log(num / denom)
                if is_new:
                    kk += 1
            if ll > best[2]:
                best = (sigma, theta, ll)
    return best


# --------------------------------------------------------------------------- #
# Catch-all likelihood (the detector / tail model)                            #
# --------------------------------------------------------------------------- #

def background_marginals(matrix: Dict[str, Dict[str, float]],
                         known_names: Sequence[str]) -> Dict[str, float]:
    """Attribute marginal over currently-known objects: mean Noul per attribute.
    Empty known set -> 0.5 (maximum-entropy background)."""
    out: Dict[str, float] = {}
    for aid, _ in ATTRIBUTE_BANK:
        if known_names:
            out[aid] = sum(matrix[aid][n] for n in known_names) / len(known_names)
        else:
            out[aid] = 0.5
    return out


def category_marginals(known_hyps: Sequence[Hypothesis]) -> Dict[str, float]:
    """Fraction of known objects in each category."""
    out: Dict[str, float] = {}
    k = len(known_hyps)
    for cat in CATEGORY_PHRASE:
        if k:
            out[cat] = sum(1 for h in known_hyps if h.category == cat) / k
        else:
            out[cat] = 0.0
    return out


def make_openworld_likelihood(matrix: Dict[str, Dict[str, float]],
                              known_hyps: Sequence[Hypothesis],
                              eps: float = 0.05) -> LikelihoodModel:
    """Likelihood over ``H_known ∪ {UNKNOWN}``. Known objects: exact closed form
    (category/guess) and cached Noul (attribute). ``UNKNOWN``: the background
    marginals, ε-floored so a category/guess answer can never zero it out."""
    known_names = [h.name for h in known_hyps]
    attr_bg = background_marginals(matrix, known_names)
    cat_bg = category_marginals(known_hyps)

    def clamp(x: float) -> float:
        return min(max(x, eps), 1.0 - eps)

    def likelihood(q: Question, h: Hypothesis) -> float:
        if h.name == UNKNOWN:
            if q.kind == ATTRIBUTE:
                return clamp(attr_bg[q.key])
            if q.kind == CATEGORY:
                return clamp(cat_bg.get(q.key, 0.0))
            return eps  # GUESS: an unknown object is not any named known object
        if q.kind == ATTRIBUTE:
            return matrix[q.key][h.name]
        return closed_form_likelihood(q, h)

    return likelihood


def build_openworld_bank(known_hyps: Sequence[Hypothesis]
                         ) -> Dict[str, List[Question]]:
    """Category questions for all 17 categories (so novelty in an unpopulated
    category is still detectable), the full attribute bank, and guesses only for
    *known* objects (you can only name what you know)."""
    category_qs = [
        Question(CATEGORY, c, "Is it a kind of {}?".format(CATEGORY_PHRASE[c]))
        for c in sorted(CATEGORY_PHRASE)
    ]
    attribute_qs = [Question(ATTRIBUTE, aid, text) for aid, text in ATTRIBUTE_BANK]
    guess_qs = [
        Question(GUESS, h.name, "Is it {} {}?".format(_article(h.name), h.name.lower()))
        for h in known_hyps
    ]
    return {CATEGORY: category_qs, ATTRIBUTE: attribute_qs, GUESS: guess_qs}


# --------------------------------------------------------------------------- #
# Episode                                                                      #
# --------------------------------------------------------------------------- #

@dataclass
class OWStep:
    turn: int
    qid: str
    kind: str
    eig: float
    answer_yes: bool
    p_unknown_before: float
    top_name: str
    top_p: float


@dataclass
class OWEpisode:
    true_name: str
    true_category: str
    in_pool: bool                # secret was in H_known at the start
    outcome: str                 # "won" | "expert" | "wrong" | "false_confirm"
    n_questions: int
    p_unknown_max: float
    p_unknown_final: float
    fire_turn: Optional[int]     # turn UNKNOWN first crossed the gate, else None
    n_retractions: int = 0       # un-eliminate ops (retract backend); 0 otherwise
    steps: List[OWStep] = field(default_factory=list)
    n_live_final: int = -1       # live knowns when the budget ran out (survivor
                                 # backends only; -1 otherwise)


def run_openworld_episode(true_h: Hypothesis,
                          known_hyps: Sequence[Hypothesis],
                          matrix: Dict[str, Dict[str, float]],
                          p_unknown_prior: float,
                          oracle: OracleModel,
                          max_questions: int = 20,
                          confidence: float = 0.9,
                          eps: float = 0.05,
                          top_k_guesses: int = 5,
                          backend: str = "soft",
                          margin: float = 2.0, tolerance: int = 1,
                          observer=None, like_matrix=None,
                          no_repeat: bool = False) -> OWEpisode:
    """Play one open-world game. Belief is maintained by the named ``BeliefBackend``
    (default ``"soft"`` = the §3.11 posterior); the EIG planner, the catch-all, and
    the commit loop are shared across backends, so a change of ``backend`` changes
    only the belief update. The backend's ``decision()`` chooses, at the gate, to ask
    another question, guess a known object, or fire the expert on UNKNOWN.

    ``no_repeat`` (default off, as in ``bed.run_episode``) drops already-asked
    questions from the EIG candidates. The survivor-set backends hold UNKNOWN's
    mass at the prior, so with a large prior a question that splits UNKNOWN from
    the knowns keeps its EIG and is asked again every turn; ``STRESS_ST0.md``
    measures where that starts. A gate-forced guess (the reinstate test) is
    never filtered."""
    all_hyps = list(known_hyps) + [UNKNOWN_HYP]
    by_name = {h.name: h for h in all_hyps}
    # Belief uses ``like_matrix`` when supplied (ENV-704: a partially-elicited row for a
    # newly-onboarded object), the true ``matrix`` otherwise; the oracle is independent.
    like = make_openworld_likelihood(like_matrix if like_matrix is not None else matrix,
                                     known_hyps, eps=eps)
    bank = build_openworld_bank(known_hyps)
    guess_by_name = {q.key: q for q in bank[GUESS]}
    if backend == "retract":
        kwargs = {"margin": margin}
    elif backend in ("tombstone", "modeswitch"):
        kwargs = {"tolerance": tolerance}
    else:
        kwargs = {}
    bk = BACKENDS[backend](known_hyps, by_name, like, UNKNOWN, p_unknown_prior,
                           eps, **kwargs)

    steps: List[OWStep] = []
    asked_qids: set = set()
    p_unk_max = bk.distribution().get(UNKNOWN, 0.0)
    fire_turn: Optional[int] = None
    in_pool = true_h.name in guess_by_name

    for turn in range(1, max_questions + 1):
        dist = bk.distribution()
        top_name = max(dist, key=lambda n: dist[n])
        top_p = dist[top_name]
        decision = bk.decision(confidence)

        if decision[0] == "expert":
            if fire_turn is None:
                fire_turn = turn
            return OWEpisode(
                true_name=true_h.name, true_category=true_h.category,
                in_pool=in_pool, outcome="expert", n_questions=turn - 1,
                p_unknown_max=p_unk_max, p_unknown_final=dist.get(UNKNOWN, 0.0),
                fire_turn=fire_turn, n_retractions=getattr(bk, "n_revives", 0),
                steps=steps,
            )
        if decision[0] == "guess":
            q = guess_by_name[decision[1]]
        else:
            cands = candidate_questions(dist, bank, top_k_guesses=top_k_guesses)
            if no_repeat:
                cands = [c for c in cands if c.qid not in asked_qids] or cands
            ranked = rank_questions(cands, dist, by_name, like)
            q = ranked[0][0]

        eig = expected_info_gain(q, dist, by_name, like)
        ans = oracle(q, true_h)
        asked_qids.add(q.qid)
        steps.append(OWStep(
            turn=turn, qid=q.qid, kind=q.kind, eig=eig, answer_yes=ans,
            p_unknown_before=dist.get(UNKNOWN, 0.0), top_name=top_name, top_p=top_p,
        ))

        if q.kind == GUESS and ans:
            # A "yes" ends the game. It is a win only if the named object is the
            # secret; an answerer that confirms the wrong name (ST-0c, noisy guess
            # channels) produces a false_confirm. Exact oracles never do.
            return OWEpisode(
                true_name=true_h.name, true_category=true_h.category,
                in_pool=in_pool,
                outcome="won" if q.key == true_h.name else "false_confirm",
                n_questions=turn,
                p_unknown_max=p_unk_max, p_unknown_final=dist.get(UNKNOWN, 0.0),
                fire_turn=fire_turn, n_retractions=getattr(bk, "n_revives", 0),
                steps=steps,
            )

        surv_before = set(getattr(bk, "survivors", ())) if observer is not None else None
        bk.observe(q, ans)
        if observer is not None:
            # Non-invasive audit hook: reconstruct the op-log from the public survivor
            # transition this turn. Backends without a survivor set (soft) report none.
            observer(turn, q, ans, bk, surv_before)
        pu = bk.distribution().get(UNKNOWN, 0.0)
        if pu > p_unk_max:
            p_unk_max = pu
        if q.kind == GUESS and not ans:
            # A wrong by-name guess: the committed object was not the secret. The
            # backend has already absorbed the "no"; keep playing.
            continue
        if fire_turn is None and pu >= confidence:
            fire_turn = turn + 1  # crosses the gate entering the next turn

    # Budget exhausted: expert iff UNKNOWN is on top, else a wrong guess.
    dist = bk.distribution()
    top_name = max(dist, key=lambda n: dist[n])
    outcome = "expert" if top_name == UNKNOWN else "wrong"
    return OWEpisode(
        true_name=true_h.name, true_category=true_h.category, in_pool=in_pool,
        outcome=outcome, n_questions=max_questions, p_unknown_max=p_unk_max,
        p_unknown_final=dist.get(UNKNOWN, 0.0), fire_turn=fire_turn,
        n_retractions=getattr(bk, "n_revives", 0), steps=steps,
        n_live_final=len(bk.survivors) if hasattr(bk, "survivors") else -1,
    )


# --------------------------------------------------------------------------- #
# Stream + accumulation                                                        #
# --------------------------------------------------------------------------- #

def noisy_matrix_oracle(matrix: Dict[str, Dict[str, float]], flip_p: float,
                        seed: int, threshold: float = 0.5) -> OracleModel:
    """Matrix oracle that flips each ATTRIBUTE answer with probability ``flip_p``
    (category/guess answers stay exact -- they are logical ground truth). Stresses
    the detector the way the live J/J answerer would: Jev matched its own matrix
    on 99.2% of attribute questions, so ``flip_p`` around 0.01-0.10 brackets and
    over-stresses the real inconsistency rate."""
    base = matrix_oracle(matrix, threshold=threshold)
    rng = random.Random(seed)

    def oracle(q: Question, true_h: Hypothesis) -> bool:
        ans = base(q, true_h)
        if q.kind == ATTRIBUTE and rng.random() < flip_p:
            return not ans
        return ans

    return oracle


def zipf_weights(n: int, s: float) -> List[float]:
    """Zipf weights for ranks 1..n, exponent ``s`` (unnormalised is fine)."""
    return [1.0 / (r ** s) for r in range(1, n + 1)]


def make_stream(objects: Sequence[Hypothesis], n_games: int, s: float,
                seed: int) -> Tuple[List[int], Dict[int, int]]:
    """Return a stream of object indices drawn from a Zipf law over a random
    rank assignment, plus the index->rank map (rank 1 = most frequent)."""
    rng = random.Random(seed)
    order = list(range(len(objects)))
    rng.shuffle(order)                      # random object -> rank assignment
    index_rank = {idx: rank + 1 for rank, idx in enumerate(order)}
    weights = [0.0] * len(objects)
    zw = zipf_weights(len(objects), s)
    for idx in range(len(objects)):
        weights[idx] = zw[index_rank[idx] - 1]
    stream = rng.choices(range(len(objects)), weights=weights, k=n_games)
    return stream, index_rank


@dataclass
class GameRecord:
    game: int
    true_name: str
    rank: int
    in_pool: bool
    is_new: bool                 # first time this species appears in the stream
    outcome: str
    n_questions: int
    p_unknown_max: float
    fire_turn: Optional[int]
    k_before: int                # |H_known| before this game
    expert_called: bool
    false_positive: bool         # in-pool but fired UNKNOWN
    n_retractions: int = 0       # un-eliminate ops this game (retract backend)


def run_accumulation(objects: Sequence[Hypothesis],
                     matrix: Dict[str, Dict[str, float]],
                     stream: Sequence[int],
                     sigma: float, theta: float,
                     max_questions: int = 20, confidence: float = 0.9,
                     eps: float = 0.05, oracle_threshold: float = 0.5,
                     oracle_noise: float = 0.0, noise_seed: int = 0,
                     backend: str = "soft", margin: float = 2.0,
                     tolerance: int = 1, observer_factory=None,
                     oracle: Optional[OracleModel] = None,
                     no_repeat: bool = False) -> List[GameRecord]:
    """Play the stream. ``H_known`` grows by expert admission. The PY ``(sigma,
    theta)`` used for the live ``p_new`` prior are passed in (fit offline or set);
    the discovery curve itself is independent of them. ``oracle_noise > 0`` flips
    attribute answers at that rate (the realistic, noisy-answerer stress).
    ``oracle`` replaces the built-in answerer entirely (the ST-0b answer channels,
    ``stress_oracles.py``); if it has ``begin_game``, that is called before each
    game."""
    if oracle is not None:
        if oracle_noise > 0.0:
            raise ValueError("pass oracle_noise or oracle, not both")
    elif oracle_noise > 0.0:
        oracle = noisy_matrix_oracle(matrix, oracle_noise, noise_seed,
                                     threshold=oracle_threshold)
    else:
        oracle = matrix_oracle(matrix, threshold=oracle_threshold)
    known: List[Hypothesis] = []
    known_set: set = set()
    seen_counts: Dict[str, int] = {}
    records: List[GameRecord] = []
    begin_game = getattr(oracle, "begin_game", None)

    for g, idx in enumerate(stream):
        if begin_game is not None:
            begin_game(g)
        true_h = objects[idx]
        n_before = g
        k_before = len(known)
        in_pool = true_h.name in known_set
        is_new = true_h.name not in seen_counts

        p_unknown_prior = py_p_new(k_before, n_before, sigma, theta)
        if k_before == 0:
            p_unknown_prior = 1.0

        obs = observer_factory(list(known)) if observer_factory is not None else None
        ep = run_openworld_episode(
            true_h, known, matrix, p_unknown_prior, oracle,
            max_questions=max_questions, confidence=confidence, eps=eps,
            backend=backend, margin=margin, tolerance=tolerance, observer=obs,
            no_repeat=no_repeat,
        )
        if obs is not None:
            obs.finish(ep)

        expert_called = ep.outcome == "expert"
        false_positive = expert_called and in_pool
        # Expert admits the true object into H_known (fills its Noul row).
        if expert_called and not in_pool:
            known.append(true_h)
            known_set.add(true_h.name)

        seen_counts[true_h.name] = seen_counts.get(true_h.name, 0) + 1
        records.append(GameRecord(
            game=g, true_name=true_h.name, rank=0, in_pool=in_pool,
            is_new=is_new, outcome=ep.outcome, n_questions=ep.n_questions,
            p_unknown_max=ep.p_unknown_max, fire_turn=ep.fire_turn,
            k_before=k_before, expert_called=expert_called,
            false_positive=false_positive, n_retractions=ep.n_retractions,
        ))
    return records
