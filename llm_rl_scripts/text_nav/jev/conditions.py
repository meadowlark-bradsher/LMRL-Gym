"""The condition matrix for the Text-Nav x Jev experiments.

One :class:`Condition` is a fully-specified prompt recipe: which persistent
context (if any) frames every state, whether the true objective is disclosed,
whether episode history is carried forward, and what the perishable target is
named. The runner turns a condition plus a seed into an episode, so every
experimental cell is produced by the same code path (design 3.1, 9).

See ``EXPERIMENT_DESIGN.md`` sections 3 (Experiment A) and 6 (Experiment B) and
Appendix A.1 (why G1 injects an authored goal string rather than the env's own
``objective``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

from llm_rl_scripts.text_nav.jev.policy import NARRATIVE_CONTEXT

# A single instruction used for every condition, so the instruction text itself
# never varies with a factor. It refers to context/goal/history generically; a
# part the condition omits from the state simply isn't there, and the sentence
# still reads correctly. This keeps the G/H/N/L factors carried entirely by
# which parts of the state are present, not by a change of wording.
UNIFIED_INSTRUCTIONS = (
    "Consider your situation, any goal you have been given, the history so far, "
    "and what you can see right now. From the commands available now, choose the "
    "single command to take next."
)

# --- Factor levels ----------------------------------------------------------

# G1 (explicit goal). An authored string, deliberately NOT the env's own
# game.objective: that string says "put the X on the table in the living room
# into the fridge" and omits the closing step, and tying our "explicit goal"
# factor to the env's phrasing would confound disclosure with wording
# (Appendix A.1). This states the true reward condition in neutral terms.
EXPLICIT_GOAL = (
    "Your goal is to locate the perishable food, put it inside the "
    "refrigerator, and close the refrigerator."
)

# N0 (neutral narrative) for Experiment B: same structural slot as the
# perishability narrative but with none of its salience cues (no spoiling, no
# calories, no waste). Holds prompt shape fixed while removing the content under
# test.
NEUTRAL_CONTEXT = (
    "You live alone and have just gotten home after a long shift at work. "
    "You are moving through the rooms of your home this evening."
)

# N1 (perishability narrative) is the established narrative from policy.py.
PERISHABILITY_CONTEXT = NARRATIVE_CONTEXT

# Target food labels. L0 is the original/generic label; L1 is the salient one.
FOOD_GENERIC = "stale food"
FOOD_MILK = "milk"


@dataclass(frozen=True)
class Condition:
    """A fully-specified prompt recipe for one experimental cell."""

    name: str
    context: Optional[str]      # persistent framing, or None
    goal_text: Optional[str]    # authored explicit goal (G1), or None (G0)
    history: bool               # carry episode transcript forward (H1) or not
    food_name: str
    experiment: str             # "A" or "B"
    factors: Dict[str, str]     # e.g. {"G": "G1", "H": "H1"}

    def describe(self) -> str:
        bits = [self.experiment, self.name]
        bits += ["{}={}".format(k, v) for k, v in sorted(self.factors.items())]
        bits.append("food={}".format(self.food_name))
        bits.append("hist={}".format("on" if self.history else "off"))
        return " ".join(bits)


# --- Experiment A: goal (G) x history (H), narrative always on, food = milk --

EXPERIMENT_A: Dict[str, Condition] = {
    "A1": Condition("A1", PERISHABILITY_CONTEXT, None, False, FOOD_MILK, "A",
                    {"G": "G0", "H": "H0"}),
    "A2": Condition("A2", PERISHABILITY_CONTEXT, EXPLICIT_GOAL, False, FOOD_MILK, "A",
                    {"G": "G1", "H": "H0"}),
    "A3": Condition("A3", PERISHABILITY_CONTEXT, None, True, FOOD_MILK, "A",
                    {"G": "G0", "H": "H1"}),
    "A4": Condition("A4", PERISHABILITY_CONTEXT, EXPLICIT_GOAL, True, FOOD_MILK, "A",
                    {"G": "G1", "H": "H1"}),
}

# --- Experiment B: narrative (N) x food label (L), reactive, no goal ---------

EXPERIMENT_B: Dict[str, Condition] = {
    "B1": Condition("B1", NEUTRAL_CONTEXT, None, False, FOOD_GENERIC, "B",
                    {"N": "N0", "L": "L0"}),
    "B2": Condition("B2", NEUTRAL_CONTEXT, None, False, FOOD_MILK, "B",
                    {"N": "N0", "L": "L1"}),
    "B3": Condition("B3", PERISHABILITY_CONTEXT, None, False, FOOD_GENERIC, "B",
                    {"N": "N1", "L": "L0"}),
    "B4": Condition("B4", PERISHABILITY_CONTEXT, None, False, FOOD_MILK, "B",
                    {"N": "N1", "L": "L1"}),
}

ALL_CONDITIONS: Dict[str, Condition] = {**EXPERIMENT_A, **EXPERIMENT_B}


def get_conditions(names: Optional[List[str]] = None, experiment: Optional[str] = None
                   ) -> List[Condition]:
    """Resolve a list of condition names (or a whole experiment) to objects."""
    if names:
        missing = [n for n in names if n not in ALL_CONDITIONS]
        if missing:
            raise KeyError("unknown condition(s): {}".format(", ".join(missing)))
        return [ALL_CONDITIONS[n] for n in names]
    if experiment:
        table = {"A": EXPERIMENT_A, "B": EXPERIMENT_B}.get(experiment.upper())
        if table is None:
            raise KeyError("unknown experiment: {!r}".format(experiment))
        return list(table.values())
    raise ValueError("pass either names=... or experiment=...")
