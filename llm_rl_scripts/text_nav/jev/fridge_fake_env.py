"""An offline fridge-scenario stand-in that emits TextWorld-style facts.

Unlike :mod:`fake_env` (navigation only), this fake models the actual Text-Nav
task shape -- a perishable target and a distractor on a table, a fridge in
another room, and the eat/open/insert/close actions -- and exposes a
``state["facts"]`` list of proposition-like objects with the same ``.name`` /
``.arguments`` surface TextWorld uses. That lets the full instrumentation path
(``snapshot_from_facts`` -> ``MilestoneTracker`` -> ``Recorder``) and the
condition/history/goal wiring be tested end to end with no TextWorld, no API key
and no network. It is NOT a benchmark environment; its reward only means "the
milestone and logging machinery works".
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass(frozen=True)
class _Var:
    name: str
    type: str


@dataclass(frozen=True)
class _Prop:
    name: str
    arguments: tuple


class _State:
    def __init__(self, feedback, admissible, score, won, lost, facts):
        self.feedback = feedback
        self._data = {
            "admissible_commands": admissible,
            "score": score,
            "won": won,
            "lost": lost,
            "max_score": 1,
            "facts": facts,
            "objective": "",  # present but empty; the runner never reads it
        }

    def __getitem__(self, key):
        return self._data[key]


class FridgeFakeEnv:
    """A minimal two-room fridge task that emits ground-truth facts.

    Winning sequence: take the target, go to the kitchen, open the fridge,
    insert the target, close the fridge. Eating the target fails the episode.
    """

    def __init__(self, target_name: str = "milk", seed: Optional[int] = None):
        self.target = target_name
        self.distractor = "fresh food"
        self.seed = seed
        self._reset()

    # -- world model ---------------------------------------------------------

    def _reset(self, seed: Optional[int] = None):
        self.room = "living room"          # player start
        self.carrying: List[str] = []
        self.on_table: List[str] = [self.target, self.distractor]
        self.fridge_closed = True
        self.in_fridge: List[str] = []
        self.eaten = False
        self.score = 0.0
        self.won = False
        self.lost = False
        self.state = self._make_state()

    def _admissible(self) -> List[str]:
        if self.won or self.lost:
            return []
        cmds: List[str] = []
        if self.room == "living room":
            cmds.append("go west")  # to kitchen
            for f in self.on_table:
                cmds.append("take {} from table".format(f))
        if self.room == "kitchen":
            cmds.append("go east")  # to living room
            if self.fridge_closed:
                cmds.append("open fridge")
            else:
                cmds.append("close fridge")
                for f in list(self.carrying):
                    cmds.append("insert {} into fridge".format(f))
        for f in list(self.carrying):
            cmds.append("eat {}".format(f))
            cmds.append("drop {}".format(f))
        return cmds

    def _facts(self) -> List[_Prop]:
        P = _Var("P", "P")
        I = _Var("I", "I")
        fridge = _Var("fridge", "c")
        facts: List[_Prop] = [
            _Prop("at", (P, _Var(self.room, "r"))),
            _Prop("at", (fridge, _Var("kitchen", "r"))),
            _Prop("closed" if self.fridge_closed else "open", (fridge,)),
        ]
        for f in self.carrying:
            facts.append(_Prop("in", (_Var(f, "f"), I)))
        for f in self.on_table:
            facts.append(_Prop("on", (_Var(f, "f"), _Var("table", "s"))))
        for f in self.in_fridge:
            facts.append(_Prop("in", (_Var(f, "f"), fridge)))
        if self.eaten:
            facts.append(_Prop("eaten", (_Var(self.target, "f"),)))
        return facts

    def _describe(self) -> str:
        if self.won:
            return "The fridge is closed with the {} inside. Done.".format(self.target)
        if self.lost:
            return "You ate the {}. That was the one thing to keep.".format(self.target)
        where = "You are in the {}.".format(self.room)
        if self.room == "living room" and self.on_table:
            where += " On the table: {}.".format(", ".join(self.on_table))
        if self.room == "kitchen":
            where += " There is a fridge here ({}).".format(
                "closed" if self.fridge_closed else "open")
        if self.carrying:
            where += " You are carrying: {}.".format(", ".join(self.carrying))
        return where

    def _make_state(self) -> _State:
        return _State(self._describe(), self._admissible(), self.score,
                      self.won, self.lost, self._facts())

    # -- transition ----------------------------------------------------------

    def _step(self, command: str):
        command = command.strip()
        if command == "go west" and self.room == "living room":
            self.room = "kitchen"
        elif command == "go east" and self.room == "kitchen":
            self.room = "living room"
        elif command.startswith("take ") and " from table" in command:
            food = command[len("take "):-len(" from table")].strip()
            if food in self.on_table:
                self.on_table.remove(food)
                self.carrying.append(food)
        elif command == "open fridge" and self.room == "kitchen":
            self.fridge_closed = False
        elif command == "close fridge" and self.room == "kitchen":
            self.fridge_closed = True
            if self.target in self.in_fridge:
                self.score = 1.0
                self.won = True
        elif command.startswith("insert ") and " into fridge" in command:
            food = command[len("insert "):-len(" into fridge")].strip()
            if self.room == "kitchen" and not self.fridge_closed and food in self.carrying:
                self.carrying.remove(food)
                self.in_fridge.append(food)
        elif command.startswith("drop "):
            food = command[len("drop "):].strip()
            if food in self.carrying:
                self.carrying.remove(food)
                if self.room == "living room":
                    self.on_table.append(food)  # else it just leaves the hand
        elif command.startswith("eat "):
            food = command[len("eat "):].strip()
            if food in self.carrying:
                self.carrying.remove(food)
                if food == self.target:
                    self.eaten = True
                    self.lost = True
        self.state = self._make_state()
