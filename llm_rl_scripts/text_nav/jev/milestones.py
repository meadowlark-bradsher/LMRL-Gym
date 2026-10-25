"""Milestone and world-state instrumentation for Text-Nav x Jev experiments.

The policy prompt only ever sees observations and admissible commands. For
*scoring* we want ground-truth progress, so the runner reads the environment's
world facts (``EnvInfos(facts=True)``) and reduces them to a small, typed
``WorldSnapshot``. Milestones are then computed from snapshots.

Design note (EXPERIMENT_DESIGN.md Appendix A.3): nothing in this module may be
placed into the policy state. It exists solely to label trajectories after the
fact.

Two layers, deliberately separated so the logic is testable offline:

* ``WorldSnapshot`` + ``MilestoneTracker`` are pure data — unit-tested with
  hand-built snapshots, no TextWorld required.
* ``snapshot_from_facts`` adapts TextWorld's ``Proposition`` list into a
  ``WorldSnapshot``. It is tolerant about proposition spellings and is verified
  against the real environment in the conda check, not offline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple

# Ordered milestones. The index in this list is the milestone's ordinal; the
# tracker reports the greatest *correctly ordered* milestone reached.
MILESTONES: Tuple[str, ...] = (
    "M1_acquire",    # target perishable is in the player's inventory
    "M2_transport",  # player is co-located with the fridge while carrying target
    "M3_prepare",    # fridge is open while the player carries the target
    "M4_store",      # target is inside the fridge
    "M5_complete",   # target in fridge AND fridge closed AND env reports won
)


@dataclass(frozen=True)
class WorldSnapshot:
    """A reduced, policy-independent view of the world at one step.

    Every field is derived from ground-truth facts, never from the prompt.
    ``None`` means "could not be determined from the facts available" and is
    treated as false for milestone purposes but is distinguishable in logs.
    """

    carrying_target: Optional[bool] = None
    player_with_fridge: Optional[bool] = None  # player in the fridge's room
    fridge_open: Optional[bool] = None
    target_in_fridge: Optional[bool] = None
    target_eaten: Optional[bool] = None
    won: bool = False
    # Freeform extras for diagnostics (e.g. player_room, inventory list).
    extra: Dict[str, object] = field(default_factory=dict)

    def achieved(self) -> Dict[str, bool]:
        """Which milestone *conditions* hold in this snapshot (momentary)."""
        carrying = bool(self.carrying_target)
        return {
            "M1_acquire": carrying,
            "M2_transport": carrying and bool(self.player_with_fridge),
            "M3_prepare": carrying and bool(self.fridge_open),
            "M4_store": bool(self.target_in_fridge),
            "M5_complete": bool(self.target_in_fridge)
            and (self.fridge_open is False)
            and bool(self.won),
        }

    def to_json(self) -> Dict[str, object]:
        return {
            "carrying_target": self.carrying_target,
            "player_with_fridge": self.player_with_fridge,
            "fridge_open": self.fridge_open,
            "target_in_fridge": self.target_in_fridge,
            "target_eaten": self.target_eaten,
            "won": self.won,
            "extra": self.extra,
        }


class MilestoneTracker:
    """Accumulates milestone progress across an episode.

    Call ``update(snapshot)`` once per step (including the initial reset state).
    The tracker records (a) which milestones were *ever* achieved and (b) the
    greatest *correctly ordered* milestone reached: the largest k such that
    M1..Mk each held at some step, each no earlier than the previous one was
    first reached.

    Initial-state awareness (design 4.2): the reset snapshot is recorded so a
    milestone already true at t=0 can be flagged. Ordered progress still
    requires the agent to establish M1 (carrying the target) itself, so a lucky
    start room cannot inflate the ordered count.
    """

    def __init__(self) -> None:
        self.ever: Dict[str, bool] = {m: False for m in MILESTONES}
        self.first_step: Dict[str, int] = {}
        self.ordered_progress: int = 0  # 0..len(MILESTONES)
        self.initial: Optional[WorldSnapshot] = None
        self.initial_true: Dict[str, bool] = {m: False for m in MILESTONES}
        self._step: int = -1

    def update(self, snapshot: WorldSnapshot) -> None:
        self._step += 1
        achieved = snapshot.achieved()
        if self._step == 0:
            self.initial = snapshot
            self.initial_true = dict(achieved)
        for m, ok in achieved.items():
            if ok and not self.ever[m]:
                self.ever[m] = True
                self.first_step[m] = self._step
        # Advance the ordered counter greedily: the next milestone in sequence
        # must hold *now* to count, which keeps the ordering honest for the
        # momentary conditions (M2/M3) without requiring all to co-occur.
        while self.ordered_progress < len(MILESTONES) and achieved[
            MILESTONES[self.ordered_progress]
        ]:
            self.ordered_progress += 1

    @property
    def greatest_ordered(self) -> int:
        return self.ordered_progress

    @property
    def greatest_ordered_name(self) -> Optional[str]:
        return MILESTONES[self.ordered_progress - 1] if self.ordered_progress else None

    def to_json(self) -> Dict[str, object]:
        return {
            "ever": dict(self.ever),
            "first_step": dict(self.first_step),
            "greatest_ordered": self.greatest_ordered,
            "greatest_ordered_name": self.greatest_ordered_name,
            "initial_true": dict(self.initial_true),
        }


# --- TextWorld fact adapter -------------------------------------------------
#
# TextWorld facts are ``Proposition`` objects: ``.name`` (str) and
# ``.arguments`` (tuple of Variable, each with ``.name`` and ``.type``). We
# match tolerantly by predicate name and by the known entity names (the target
# food and "fridge"), so small theme/spelling differences don't break scoring.
# Verified against the live env in the conda check.

_INVENTORY_TYPES = {"I"}  # TextWorld's player-inventory holder type


def _arg_names(prop) -> List[str]:
    return [getattr(a, "name", str(a)) for a in getattr(prop, "arguments", ())]


def _arg_types(prop) -> List[str]:
    return [getattr(a, "type", "") for a in getattr(prop, "arguments", ())]


def snapshot_from_facts(
    facts: Iterable,
    target_name: str,
    fridge_name: str = "fridge",
    won: bool = False,
    player_type: str = "P",
) -> WorldSnapshot:
    """Reduce a TextWorld facts iterable to a :class:`WorldSnapshot`.

    ``facts`` is whatever ``state["facts"]`` yields (a list of Proposition).
    Unknown or absent predicates leave the corresponding field ``None``.
    """
    facts = list(facts or [])

    player_room: Optional[str] = None
    fridge_room: Optional[str] = None
    carrying_target: Optional[bool] = None
    target_in_fridge: Optional[bool] = None
    fridge_open: Optional[bool] = None
    target_eaten: Optional[bool] = None
    inventory: List[str] = []

    # Default the booleans we can affirm/deny from presence of facts. We only
    # flip to False once we've seen the relevant predicate family at all, so a
    # missing fact family stays None (unknown) rather than a silent False.
    saw_at = saw_in = saw_openclosed = saw_eaten = False

    for prop in facts:
        name = getattr(prop, "name", "")
        argn = _arg_names(prop)
        argt = _arg_types(prop)

        if name == "at" and len(argn) == 2:
            saw_at = True
            holder_type = argt[0]
            if holder_type == player_type:
                player_room = argn[1]
            if argn[0] == fridge_name:
                fridge_room = argn[1]
        elif name == "in" and len(argn) == 2:
            saw_in = True
            obj, cont = argn[0], argn[1]
            cont_type = argt[1]
            if obj == target_name and cont_type in _INVENTORY_TYPES:
                carrying_target = True
            if cont_type in _INVENTORY_TYPES:
                inventory.append(obj)
            if obj == target_name and cont == fridge_name:
                target_in_fridge = True
        elif name == "open" and fridge_name in argn:
            saw_openclosed = True
            fridge_open = True
        elif name == "closed" and fridge_name in argn:
            saw_openclosed = True
            fridge_open = False
        elif name == "eaten" and target_name in argn:
            saw_eaten = True
            target_eaten = True

    if saw_in and carrying_target is None:
        carrying_target = False
    if saw_in and target_in_fridge is None:
        target_in_fridge = False
    if saw_eaten is False:
        target_eaten = False if saw_in else None  # no eaten fact => not eaten

    player_with_fridge: Optional[bool] = None
    if saw_at and player_room is not None and fridge_room is not None:
        player_with_fridge = player_room == fridge_room

    return WorldSnapshot(
        carrying_target=carrying_target,
        player_with_fridge=player_with_fridge,
        fridge_open=fridge_open,
        target_in_fridge=target_in_fridge,
        target_eaten=target_eaten,
        won=bool(won),
        extra={
            "player_room": player_room,
            "fridge_room": fridge_room,
            "inventory": inventory,
        },
    )
