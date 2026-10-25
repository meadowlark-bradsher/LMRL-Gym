"""A Jev-driven policy for the Text-Nav task.

The task's environment hands us the admissible commands at every turn, so the
policy's job is only to pick one. That maps exactly onto Jev's Choice primitive:
the admissible commands become the options and the running transcript becomes
the state. Nothing here is generated as free text, which is why Text-Nav is the
cleanest Jev fit in the suite.
"""

from __future__ import annotations

from typing import List, Optional

from LLM_RL.jev.client import ChoiceAnswer, JevClient

INSTRUCTIONS = (
    "You are navigating a house to reach the goal described in the latest "
    "observation. Given the transcript so far and the commands available now, "
    "choose the single command that best progresses toward the goal."
)

# Narrative-mode prompts. These deliberately name no object or action from the
# task (no "fridge", "food", "refrigerate", "put away"): the point is to test
# whether situational common sense alone -- perishable food left out, and a
# reason not to eat it -- evokes the latent "refrigerate it" behavior. The
# calorie detail also discourages the "eat food" action, which is the game's
# fail event.
NARRATIVE_CONTEXT = (
    "You live alone and have just gotten home after a long shift at work. "
    "You rushed out this morning and may have accidentally left something out "
    "that can spoil if it is not kept cold. "
    "You are watching your calorie intake and have already exceeded your "
    "calorie goal for the day, so you do not want to eat anything else tonight. "
    "You dislike letting things go to waste. "
    "Behave as this tired, sensible person naturally would while moving through "
    "their home and dealing with whatever has been left out."
)
NARRATIVE_INSTRUCTIONS = (
    "Given your situation and what you can see right now, choose the single "
    "command that this person would most naturally and sensibly do next."
)


class JevTextNavPolicy:
    """Chooses the next command from the environment's admissible commands."""

    def __init__(
        self,
        client: JevClient,
        instructions: str = INSTRUCTIONS,
        context: Optional[str] = None,
        history_chars: int = 4000,
    ):
        self.client = client
        self.instructions = instructions
        # A persistent narrative/persona pinned to every state. Used to test
        # whether situational context alone can evoke the task's latent
        # common-sense behavior without naming the actions.
        self.context = context
        # Cap how much transcript we send so the state stays bounded on long
        # episodes; the most recent context matters most for the next move.
        self.history_chars = history_chars
        self.last_answer: Optional[ChoiceAnswer] = None

    def build_state(
        self, transcript: str, observation: str, objective: Optional[str] = None
    ) -> str:
        transcript = transcript[-self.history_chars :]
        parts = []
        # Narrative context (if any) leads, so the situation frames every move.
        if self.context and self.context.strip():
            parts.append("Context:\n" + self.context.strip())
        # Pin the objective into every state: in this task it is stated only in
        # the first observation, so relying on the (capped) transcript to carry
        # it forward loses the goal on longer episodes.
        if objective and objective.strip():
            parts.append("Goal:\n" + objective.strip())
        if transcript.strip():
            parts.append("Transcript so far:\n" + transcript.strip())
        parts.append("Current observation:\n" + observation.strip())
        return "\n\n".join(parts)

    def choose(
        self,
        transcript: str,
        observation: str,
        admissible_commands: List[str],
        objective: Optional[str] = None,
    ) -> str:
        if not admissible_commands:
            raise ValueError("no admissible commands to choose from")
        # Deduplicate while preserving order; commands are the option names and
        # are self-describing, so their descriptions are None.
        options = {}
        for cmd in admissible_commands:
            cmd = cmd.strip()
            if cmd and cmd not in options:
                options[cmd] = None

        state = self.build_state(transcript, observation, objective)
        answer = self.client.choice(
            state=state,
            instructions=self.instructions,
            options=options,
            question_id="command",
        )
        self.last_answer = answer
        # The returned choice is one of the option names, i.e. a command string.
        return answer.choice
