"""A belief-space exploration+planning agent — Experiment D, Mode A.

This is the principled baseline the design (D.3) calls for: an agent that does
what Jev structurally cannot — maintain a map (belief), explore to reduce
uncertainty, and plan to the goal once it is located. It makes NO Jev calls.

In this POMDP the observation is deterministic and (verified in Exp D Mode B)
*bijective* with the room, so an information-greedy policy reduces to frontier
exploration: an unvisited room's observation is maximally informative, a revisited
one yields nothing. The agent therefore explores unexplored exits until it has
located the target food and the fridge, then executes the plan
(fetch -> fridge -> open -> insert -> close) along shortest known paths.

Fairness / information used: the agent uses only what the observation exposes —
room identity (equivalently the observation signature, which is 1:1 with the room)
and the admissible command list — plus the stated goal and target identity (the
G1/explicit-goal setting). It never reads hidden object-location or adjacency
facts; it discovers the map by moving. Those facts are used only outside the agent
for milestone scoring.
"""

from __future__ import annotations

import collections
from typing import Dict, List, Optional


class BeliefSpaceExplorer:
    """Stateful per-episode agent. Call reset() then choose() each step."""

    def __init__(self, target_name: str = "milk"):
        self.target = target_name
        self.reset()

    def reset(self):
        # directed map: room -> {direction: dest_room or None(frontier/unexplored)}
        self.graph: Dict[str, Dict[str, Optional[str]]] = {}
        self.food_room: Optional[str] = None
        self.fridge_room: Optional[str] = None
        self.inserted = False
        self._prev: Optional[tuple] = None  # (room, direction) of the last move

    # -- map maintenance -----------------------------------------------------

    def _note_room(self, room: str, go_dirs: List[str]):
        node = self.graph.setdefault(room, {})
        for d in go_dirs:
            node.setdefault(d, None)  # a frontier until we traverse it
        # resolve the edge we just traversed to land here
        if self._prev is not None:
            pr, pd = self._prev
            self.graph.setdefault(pr, {})[pd] = room
            self._prev = None

    def _path_first_dir(self, start: str, goal: str) -> Optional[str]:
        """First move direction on a shortest known path start->goal, or None."""
        if start == goal:
            return None
        q = collections.deque([(start, None)])
        seen = {start}
        while q:
            room, first = q.popleft()
            for d, dst in self.graph.get(room, {}).items():
                if dst is None or dst in seen:
                    continue
                f = d if room == start else first
                if dst == goal:
                    return f
                seen.add(dst)
                q.append((dst, f))
        return None

    def _dir_to_nearest_frontier(self, start: str) -> Optional[str]:
        """First move toward the nearest room that still has an unexplored exit."""
        q = collections.deque([(start, None)])
        seen = {start}
        while q:
            room, first = q.popleft()
            if any(dst is None for dst in self.graph.get(room, {}).values()):
                if room != start:
                    return first
            for d, dst in self.graph.get(room, {}).items():
                if dst is None or dst in seen:
                    continue
                seen.add(dst)
                q.append((dst, d if room == start else first))
        return None

    # -- the policy ----------------------------------------------------------

    def choose(self, room: str, admissible: List[str]) -> str:
        adm = list(admissible)
        go_dirs = [c[len("go "):] for c in adm if c.startswith("go ")]
        self._note_room(room, go_dirs)

        take_t = next((c for c in adm if c.startswith("take ") and self.target in c
                       and "table" in c), None)
        insert_t = next((c for c in adm if c.startswith("insert") and self.target in c), None)
        carrying = any(c in adm for c in ("eat " + self.target, "drop " + self.target))
        fridge_here = ("open fridge" in adm or "close fridge" in adm or insert_t is not None)

        if take_t is not None and not carrying:
            self.food_room = room
        if fridge_here:
            self.fridge_room = room

        cmd = self._decide(room, adm, take_t, insert_t, carrying, fridge_here)
        if cmd.startswith("go "):
            self._prev = (room, cmd[len("go "):])
        if cmd == insert_t:
            self.inserted = True
        return cmd

    def _decide(self, room, adm, take_t, insert_t, carrying, fridge_here) -> str:
        # 1) finish at the fridge
        if carrying and fridge_here:
            if insert_t is not None:
                return insert_t
            if "open fridge" in adm:
                return "open fridge"
        if self.inserted and "close fridge" in adm:
            return "close fridge"
        # 2) acquire the target when it is right here
        if take_t is not None and not carrying:
            return take_t
        # 3) head to a known subgoal room
        goal_room = self.fridge_room if carrying else self.food_room
        if goal_room is not None and goal_room != room:
            d = self._path_first_dir(room, goal_room)
            if d is not None:
                return "go " + d
        # 4) explore: an unexplored exit here, else toward the nearest frontier
        here = self.graph.get(room, {})
        for d, dst in here.items():
            if dst is None and ("go " + d) in adm:
                return "go " + d
        d = self._dir_to_nearest_frontier(room)
        if d is not None and ("go " + d) in adm:
            return "go " + d
        # 5) fallback: any legal move, else first admissible
        for c in adm:
            if c.startswith("go "):
                return c
        return adm[0]
