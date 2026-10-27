"""Jev as the live answerer (oracle) and, optionally, the question selector.

The likelihood that drives BED comes from the cached Noul matrix (``matrix.py``).
This module is the other place Jev touches the loop:

* ``JevOracle`` -- the J/J answerer. Given the hidden object, it answers one
  yes/no question with a Choice over {yes, no} (a crisp answer, like the
  benchmark's T5 oracle). Category and guess questions can be answered by closed
  form instead, so only attribute answers cost an API call.
* ``JevSelector`` -- the Wordle-parallel role: pick which candidate question to
  ask via a Choice over the shortlist, graded against the EIG argmax.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from LLM_RL.jev.client import JevClient
from .bed import (ATTRIBUTE, CATEGORY, GUESS, Hypothesis, Posterior, Question,
                  closed_form_likelihood)

MAX_SELECTOR_OPTIONS = 255


class JevOracle:
    """Live J/J answerer. ``closed_form_known=True`` answers category/guess
    questions by lookup (the oracle knows the object) and spends an API call only
    on attribute questions; ``False`` asks Jev every question (fully pure J/J)."""

    def __init__(self, client: JevClient, closed_form_known: bool = True):
        self.client = client
        self.closed_form_known = closed_form_known
        self.calls = 0

    def __call__(self, q: Question, true_h: Hypothesis) -> bool:
        if self.closed_form_known and q.kind in (CATEGORY, GUESS):
            return closed_form_likelihood(q, true_h) >= 0.5
        state = (
            "A game of Twenty Questions. You are the oracle and you know the "
            "hidden object is: {} (a kind of {}). Answer the question "
            "truthfully.".format(true_h.name, true_h.category)
        )
        self.calls += 1
        ans = self.client.choice(
            state,
            "Answer this yes/no question about the object: {}".format(q.text),
            {"yes": "the truthful answer is yes", "no": "the truthful answer is no"},
            question_id="answer",
        )
        return ans.choice == "yes"


class JevSelector:
    """Pick a candidate question via Jev Choice. Records, for each pick, whether
    it matched the EIG argmax and the EIG regret in bits, so the selector role is
    graded exactly like Wordle.

    ``history`` puts the running transcript of asked questions and their answers
    into the Choice state (the fairness control: without it the selector cannot
    know what it already asked). The runner can also drop already-asked questions
    from the shortlist (``--selector-norepeat``), which removes the repetition
    failure mechanically and isolates residual ranking quality.
    """

    def __init__(self, client: JevClient, max_options: int = 48, history: bool = False):
        self.client = client
        self.max_options = max_options
        self.history = history
        self.picks: List[dict] = []

    def __call__(self, ranked: List[Tuple[Question, float]], post: Posterior,
                 transcript: Optional[List[Tuple[str, bool]]] = None) -> Question:
        shortlist = ranked[: self.max_options]
        if len(shortlist) == 1:
            best_q, best_eig = shortlist[0]
            self.picks.append({"qid": best_q.qid, "match_top": True, "regret_bits": 0.0})
            return best_q
        options = {q.qid: q.text for q, _ in shortlist}
        by_qid = {q.qid: q for q, _ in shortlist}
        eig_by_qid = {q.qid: e for q, e in shortlist}
        best_q, best_eig = shortlist[0]
        state = (
            "A game of Twenty Questions. Pick the single yes/no question that is "
            "most informative for identifying the hidden object right now."
        )
        if self.history and transcript:
            lines = "\n".join("- {} {}".format(q, "Yes." if a else "No.")
                              for q, a in transcript)
            state += "\n\nQuestions asked so far and their answers:\n" + lines + (
                "\n\nDo not repeat a question that has already been answered.")
        ans = self.client.choice(state, "Choose the most informative question.",
                                 options, question_id="question")
        chosen = by_qid.get(ans.choice, best_q)
        self.picks.append({
            "qid": chosen.qid,
            "match_top": chosen.qid == best_q.qid,
            "regret_bits": max(0.0, best_eig - eig_by_qid.get(chosen.qid, best_eig)),
            "confidence": ans.confidence,
        })
        return chosen
