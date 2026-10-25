"""A tiny offline stand-in for TextNavEnv.

The real environment needs TextWorld (and runs only in the LLM_RL conda env).
This fake mirrors just the surface the Jev runner touches -- ``_reset()``,
``_step(command)`` and a ``state`` object exposing ``.feedback`` plus the
``admissible_commands`` / ``score`` / ``done`` items -- over a trivial fixed
house. It exists so the policy loop can be exercised end to end with no
TextWorld, no API key and no network. It is not a benchmark environment and its
scores mean nothing beyond "the loop works".
"""

from __future__ import annotations

from typing import Dict, List

# room -> {command: next_room}. "bedroom" is the goal.
_HOUSE: Dict[str, Dict[str, str]] = {
    "kitchen": {"go north": "hallway", "go east": "pantry"},
    "pantry": {"go west": "kitchen"},
    "hallway": {"go south": "kitchen", "go north": "bedroom", "open door": "study"},
    "study": {"go south": "hallway"},
    "bedroom": {},
}
_GOAL = "bedroom"


class _FakeState:
    # Mirrors the subset of TextWorld's GameState schema the runner reads:
    # there is no "done" key on the real state -- done is won or lost.
    def __init__(self, feedback: str, admissible: List[str], score: float, won: bool):
        self.feedback = feedback
        self._data = {
            "admissible_commands": admissible,
            "score": score,
            "won": won,
            "lost": False,
            "max_score": 1,
        }

    def __getitem__(self, key):
        return self._data[key]


class FakeTextNavEnv:
    """Mimics TextNavEnv's _reset/_step/state surface over a fixed house."""

    def __init__(self, start: str = "kitchen", max_score: float = 1.0):
        self.start = start
        self.max_score = max_score
        self.room = start
        self.score = 0.0
        self.done = False
        self.state = None
        self._reset()

    def _describe(self) -> str:
        if self.room == _GOAL:
            return "You are in the bedroom. You have reached the goal."
        exits = ", ".join(sorted(_HOUSE[self.room].keys()))
        return "You are in the {}. Available exits: {}.".format(self.room, exits)

    def _make_state(self) -> _FakeState:
        if self.done:
            admissible: List[str] = []
        else:
            admissible = list(_HOUSE[self.room].keys())
        return _FakeState(self._describe(), admissible, self.score, won=self.done)

    def _reset(self, seed=None):
        self.room = self.start
        self.score = 0.0
        self.done = False
        self.state = self._make_state()

    def _step(self, command: str):
        command = command.strip()
        transitions = _HOUSE.get(self.room, {})
        if command in transitions:
            self.room = transitions[command]
            if self.room == _GOAL:
                self.score = self.max_score
                self.done = True
        # Unknown commands are no-ops here; the real env simply won't offer them.
        self.state = self._make_state()
