"""Bayesian-experimental-design engine for Twenty Questions (pure code).

The hypothesis space is the LMRL-Gym Twenty Questions object set (``DEFAULT_OBJECT_DICT``
in ``llm_rl_scripts.twenty_questions.env.data`` -- 158 objects across 17
categories). The benchmark asks a guesser to name the hidden object within 20
yes/no questions; here the belief/planning loop is externalised into this module
and the only learned quantity is the per-(object, question) yes-probability.

That probability is the likelihood ``p(answer=yes | hypothesis, question)``. For
*category* and *guess* questions it is closed-form (derivable from the object
dictionary). For natural-language *attribute* questions it has no closed form --
it needs a world model -- and that is exactly the slot Jev's Noul fills (see
``matrix.py``). The same probability feeds both the EIG acquisition and the
posterior filter, mirroring the BED-LLM design.

This module imports nothing from Jev and nothing heavy (no nltk, no JaxSeq): it
takes the likelihood as an injected callable, so the whole loop runs offline.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from llm_rl_scripts.twenty_questions.env.data import DEFAULT_OBJECT_DICT


# --------------------------------------------------------------------------- #
# Hypotheses                                                                   #
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class Hypothesis:
    """One candidate object. ``name`` is the canonical surface form (unique);
    ``variants`` keeps the alternate spellings from the benchmark; ``category``
    is the object dictionary key it came from."""

    name: str
    variants: Tuple[str, ...]
    category: str


def load_hypotheses() -> List[Hypothesis]:
    """All 158 objects from the benchmark dictionary, in dictionary order."""
    hyps: List[Hypothesis] = []
    for category, entries in DEFAULT_OBJECT_DICT.items():
        for entry in entries:
            variants = tuple(entry.split(";"))
            hyps.append(Hypothesis(name=variants[0], variants=variants, category=category))
    return hyps


# Natural phrase for each category, used in "Is it a kind of {phrase}?".
CATEGORY_PHRASE: Dict[str, str] = {
    "Sports": "piece of sports equipment",
    "Animals": "animal",
    "Fruits": "fruit",
    "Vehicles": "vehicle",
    "Clothes": "piece of clothing",
    "Electronics": "electronic device",
    "Musical Instruments": "musical instrument",
    "Furniture": "piece of furniture",
    "Office Supplies": "office or school supply",
    "Vegetables": "vegetable",
    "Art": "art supply",
    "Kitchen Tools": "kitchen tool or utensil",
    "Nature": "natural thing found in nature",
    "Toys": "toy",
    "Jewelry": "piece of jewelry",
    "Garden Supplies": "gardening tool or supply",
    "Tools": "hand tool",
}


# Natural-language yes/no attribute questions. These have no closed-form answer
# -- Jev's Noul supplies the likelihood for each (object, attribute) pair.
ATTRIBUTE_BANK: List[Tuple[str, str]] = [
    ("alive", "Is it alive?"),
    ("living_part", "Is it a living thing or a part of one?"),
    ("manmade", "Is it man-made (manufactured by people)?"),
    ("edible", "Is it edible?"),
    ("food", "Is it a type of food?"),
    ("in_kitchen", "Is it usually found in a kitchen?"),
    ("electronic", "Is it an electronic device?"),
    ("uses_electricity", "Does it use electricity?"),
    ("has_motor", "Does it have a motor or an engine?"),
    ("transport", "Is it used for transportation?"),
    ("wearable", "Is it something you wear?"),
    ("fabric", "Is it made mostly of fabric or cloth?"),
    ("instrument", "Is it a musical instrument?"),
    ("furniture", "Is it a piece of furniture?"),
    ("indoors", "Is it typically found indoors in a house?"),
    ("outdoors_nature", "Is it found outdoors in nature?"),
    ("tool", "Is it a tool?"),
    ("build_repair", "Is it used for building or repairing things?"),
    ("garden", "Is it used in a garden?"),
    ("toy_play", "Is it a toy or used for play?"),
    ("write_draw", "Is it used for writing or drawing?"),
    ("office_class", "Is it found in an office or a classroom?"),
    ("decoration", "Is it worn as decoration or jewelry?"),
    ("sports_games", "Is it used in sports or games?"),
    ("animal", "Is it an animal?"),
    ("plant", "Is it a plant or part of a plant?"),
    ("handheld", "Can you comfortably hold it in one hand?"),
    ("smaller_microwave", "Is it smaller than a microwave oven?"),
    ("larger_person", "Is it larger than an adult person?"),
    ("metal", "Is it made mostly of metal?"),
    ("wood", "Is it made mostly of wood?"),
    ("plastic", "Is it made mostly of plastic?"),
    ("wheels", "Does it have wheels?"),
    ("round", "Is it round or spherical?"),
    ("legs", "Does it have legs?"),
    ("can_fly", "Can it fly?"),
    ("in_water", "Does it live in or travel on water?"),
    ("pet", "Is it commonly kept as a pet?"),
    ("sharp_cut", "Is it sharp or used for cutting?"),
    ("in_bathroom", "Is it usually found in a bathroom?"),
    ("eat_serve", "Is it used for eating or serving food?"),
    ("battery", "Does it run on a battery?"),
    ("makes_sound", "Does it make sound?"),
    ("container", "Is it a container that holds things?"),
]


# --------------------------------------------------------------------------- #
# Questions                                                                    #
# --------------------------------------------------------------------------- #

CATEGORY = "category"
ATTRIBUTE = "attribute"
GUESS = "guess"


@dataclass(frozen=True)
class Question:
    kind: str          # CATEGORY | ATTRIBUTE | GUESS
    key: str           # category name | attribute id | hypothesis name
    text: str          # the natural-language yes/no question

    @property
    def qid(self) -> str:
        return "{}:{}".format(self.kind, self.key)


def _article(name: str) -> str:
    return "an" if name[:1].lower() in "aeiou" else "a"


def build_question_bank(hypotheses: Sequence[Hypothesis]) -> Dict[str, List[Question]]:
    """Return the question bank grouped by kind."""
    cats = sorted({h.category for h in hypotheses})
    category_qs = [
        Question(CATEGORY, c, "Is it a kind of {}?".format(CATEGORY_PHRASE[c]))
        for c in cats
    ]
    attribute_qs = [Question(ATTRIBUTE, aid, text) for aid, text in ATTRIBUTE_BANK]
    guess_qs = [
        Question(GUESS, h.name, "Is it {} {}?".format(_article(h.name), h.name.lower()))
        for h in hypotheses
    ]
    return {CATEGORY: category_qs, ATTRIBUTE: attribute_qs, GUESS: guess_qs}


# --------------------------------------------------------------------------- #
# Likelihood                                                                   #
# --------------------------------------------------------------------------- #

# A LikelihoodModel maps (Question, Hypothesis) -> p(answer == "yes").
LikelihoodModel = Callable[[Question, Hypothesis], float]


def closed_form_likelihood(q: Question, h: Hypothesis) -> float:
    """Exact p(yes) for the questions that have one."""
    if q.kind == CATEGORY:
        return 1.0 if h.category == q.key else 0.0
    if q.kind == GUESS:
        return 1.0 if h.name == q.key else 0.0
    raise ValueError("no closed form for attribute question {!r}".format(q.qid))


def make_likelihood(attribute_matrix: Dict[str, Dict[str, float]]) -> LikelihoodModel:
    """Likelihood that is closed-form for category/guess and reads the Noul
    ``attribute_matrix[attr_id][hypothesis_name]`` for attributes."""

    def likelihood(q: Question, h: Hypothesis) -> float:
        if q.kind == ATTRIBUTE:
            return attribute_matrix[q.key][h.name]
        return closed_form_likelihood(q, h)

    return likelihood


# --------------------------------------------------------------------------- #
# Belief state + information gain                                              #
# --------------------------------------------------------------------------- #

Posterior = Dict[str, float]  # hypothesis name -> probability


def uniform_posterior(hypotheses: Sequence[Hypothesis]) -> Posterior:
    p = 1.0 / len(hypotheses)
    return {h.name: p for h in hypotheses}


def entropy(post: Posterior) -> float:
    """Shannon entropy of the posterior, in bits."""
    h = 0.0
    for prob in post.values():
        if prob > 0.0:
            h -= prob * math.log2(prob)
    return h


def predictive_yes(q: Question, post: Posterior, by_name: Dict[str, Hypothesis],
                   like: LikelihoodModel) -> float:
    """p(answer == yes) = sum_h post(h) * p(yes | h, q)."""
    return sum(post[name] * like(q, by_name[name]) for name in post)


def expected_info_gain(q: Question, post: Posterior, by_name: Dict[str, Hypothesis],
                       like: LikelihoodModel) -> float:
    """EIG(q) = H(post) - E_a[H(post | a)], in bits."""
    p_yes = predictive_yes(q, post, by_name, like)
    p_no = 1.0 - p_yes
    if p_yes <= 0.0 or p_yes >= 1.0:
        return 0.0
    # Posterior entropies after each answer, computed in one pass.
    h_yes = 0.0
    h_no = 0.0
    for name, prior in post.items():
        ly = like(q, by_name[name])
        # p(h | yes) propto prior * ly ; p(h | no) propto prior * (1 - ly)
        py = prior * ly / p_yes
        pn = prior * (1.0 - ly) / p_no
        if py > 0.0:
            h_yes -= py * math.log2(py)
        if pn > 0.0:
            h_no -= pn * math.log2(pn)
    return entropy(post) - (p_yes * h_yes + p_no * h_no)


def posterior_update(post: Posterior, q: Question, answer_yes: bool,
                     by_name: Dict[str, Hypothesis], like: LikelihoodModel,
                     eps: float = 0.05) -> Posterior:
    """Bayesian filter. Noisy *attribute* likelihoods (Jev's Noul) are clamped to
    ``[eps, 1 - eps]`` so a single confidently-wrong Noul cannot irreversibly
    eliminate the true hypothesis; closed-form category/guess likelihoods are
    ground truth (and the oracle agrees with them by construction), so they are
    applied exactly."""
    soft = q.kind == ATTRIBUTE
    new: Posterior = {}
    total = 0.0
    for name, prior in post.items():
        ly = like(q, by_name[name])
        if soft:
            ly = min(max(ly, eps), 1.0 - eps)
        w = ly if answer_yes else (1.0 - ly)
        new[name] = prior * w
        total += new[name]
    if total <= 0.0:
        # Degenerate (should not happen with the eps floor); fall back to uniform.
        n = len(post)
        return {name: 1.0 / n for name in post}
    for name in new:
        new[name] /= total
    return new


# --------------------------------------------------------------------------- #
# Question selection                                                           #
# --------------------------------------------------------------------------- #

def rank_questions(candidates: Sequence[Question], post: Posterior,
                   by_name: Dict[str, Hypothesis], like: LikelihoodModel
                   ) -> List[Tuple[Question, float]]:
    """Candidates sorted by EIG descending, deterministic tie-break on qid."""
    scored = [(q, expected_info_gain(q, post, by_name, like)) for q in candidates]
    scored.sort(key=lambda t: (-t[1], t[0].qid))
    return scored


def optimal_question(candidates: Sequence[Question], post: Posterior,
                     by_name: Dict[str, Hypothesis], like: LikelihoodModel
                     ) -> Tuple[Question, float]:
    ranked = rank_questions(candidates, post, by_name, like)
    return ranked[0]


# A selector picks a question given the EIG ranking and the running transcript.
# ``None`` means "take the argmax" (BED-the-planner). A Jev selector picks via
# Choice.
Selector = Callable[[List[Tuple[Question, float]], Posterior, List[Tuple[str, bool]]], Question]


def candidate_questions(post: Posterior, bank: Dict[str, List[Question]],
                        top_k_guesses: int = 5, support_eps: float = 1e-9
                        ) -> List[Question]:
    """Questions worth scoring this turn: all category + attribute questions,
    plus guesses for the current top-k hypotheses (guesses only earn high EIG
    once the posterior concentrates). If the support has shrunk to <= top_k,
    include a guess for every surviving hypothesis."""
    support = [name for name, p in post.items() if p > support_eps]
    ordered = sorted(support, key=lambda n: -post[n])
    guess_names = ordered if len(ordered) <= top_k_guesses else ordered[:top_k_guesses]
    guess_by_name = {q.key: q for q in bank[GUESS]}
    guesses = [guess_by_name[n] for n in guess_names if n in guess_by_name]
    return list(bank[CATEGORY]) + list(bank[ATTRIBUTE]) + guesses


# --------------------------------------------------------------------------- #
# Oracle                                                                       #
# --------------------------------------------------------------------------- #

# An OracleModel answers one question about the true hypothesis: returns True for
# "Yes." and False for "No.".
OracleModel = Callable[[Question, Hypothesis], bool]


def matrix_oracle(attribute_matrix: Dict[str, Dict[str, float]], threshold: float = 0.5
                  ) -> OracleModel:
    """Deterministic oracle: category/guess by closed form, attributes by
    thresholding the same Noul matrix the likelihood uses. Self-consistent, so
    the filter never eliminates the truth -- isolates the planning loop."""

    def oracle(q: Question, true_h: Hypothesis) -> bool:
        if q.kind == ATTRIBUTE:
            return attribute_matrix[q.key][true_h.name] >= threshold
        return closed_form_likelihood(q, true_h) >= 0.5

    return oracle


# --------------------------------------------------------------------------- #
# Episode                                                                      #
# --------------------------------------------------------------------------- #

@dataclass
class Step:
    turn: int
    qid: str
    question: str
    kind: str
    eig: float
    p_yes_pred: float
    answer_yes: bool
    support_before: int
    entropy_before: float
    true_hyp_prob_before: float
    sel_match_top: Optional[bool] = None   # selector picked the EIG argmax?
    sel_regret_bits: Optional[float] = None  # EIG(argmax) - EIG(chosen)


@dataclass
class EpisodeResult:
    true_name: str
    true_category: str
    won: bool
    n_questions: int
    reward: float
    steps: List[Step] = field(default_factory=list)
    final_top: Optional[Tuple[str, float]] = None


def run_episode(true_h: Hypothesis, hypotheses: Sequence[Hypothesis],
                bank: Dict[str, List[Question]], like: LikelihoodModel,
                oracle: OracleModel, max_questions: int = 20,
                confidence: float = 0.9, eps: float = 0.05,
                top_k_guesses: int = 5, no_repeat: bool = False,
                selector: Optional[Selector] = None) -> EpisodeResult:
    """Play one game of twenty questions with the BED loop.

    ``selector=None`` takes the EIG argmax (BED-the-planner, consuming Jev's
    likelihood). A selector consumes the EIG ranking and returns a question (a
    Jev Choice over candidate questions, say). When the posterior puts at least
    ``confidence`` on one hypothesis the loop commits to guessing it.
    """
    by_name = {h.name: h for h in hypotheses}
    guess_by_name = {q.key: q for q in bank[GUESS]}
    post = uniform_posterior(hypotheses)
    steps: List[Step] = []
    transcript: List[Tuple[str, bool]] = []  # (question text, answer_yes)
    asked_qids: set = set()

    for turn in range(1, max_questions + 1):
        support = sum(1 for p in post.values() if p > 1e-9)
        ent = entropy(post)
        top_name = max(post, key=lambda n: post[n])
        top_p = post[top_name]

        # Commit to a guess when confident or when only one hypothesis remains.
        used_selector = False
        if top_p >= confidence or support == 1:
            q = guess_by_name[top_name]
        else:
            cands = candidate_questions(post, bank, top_k_guesses=top_k_guesses)
            if no_repeat:
                cands = [c for c in cands if c.qid not in asked_qids] or cands
            ranked = rank_questions(cands, post, by_name, like)
            if selector is None:
                q = ranked[0][0]
            else:
                q = selector(ranked, post, transcript)
                used_selector = True

        eig = expected_info_gain(q, post, by_name, like)
        p_yes = predictive_yes(q, post, by_name, like)
        ans = oracle(q, true_h)
        sel_match = sel_regret = None
        if used_selector and getattr(selector, "picks", None):
            last = selector.picks[-1]
            sel_match = last.get("match_top")
            sel_regret = last.get("regret_bits")
        steps.append(Step(
            turn=turn, qid=q.qid, question=q.text, kind=q.kind, eig=eig,
            p_yes_pred=p_yes, answer_yes=ans, support_before=support,
            entropy_before=ent, true_hyp_prob_before=post[true_h.name],
            sel_match_top=sel_match, sel_regret_bits=sel_regret,
        ))

        transcript.append((q.text, ans))
        asked_qids.add(q.qid)

        if q.kind == GUESS and ans:
            return EpisodeResult(
                true_name=true_h.name, true_category=true_h.category, won=True,
                n_questions=turn, reward=-(turn - 1.0), steps=steps,
                final_top=(q.key, post[q.key]),
            )
        post = posterior_update(post, q, ans, by_name, like, eps=eps)

    top_name = max(post, key=lambda n: post[n])
    return EpisodeResult(
        true_name=true_h.name, true_category=true_h.category, won=False,
        n_questions=max_questions, reward=-float(max_questions), steps=steps,
        final_top=(top_name, post[top_name]),
    )
