"""A small, reusable client for the TypeSafe Jev "System One" API.

Jev answers typed questions about a state rather than generating text. For the
LMRL-Gym discrete-action tasks we only need the ``choice`` primitive: given a
state string and a map of option -> description, Jev returns the chosen option
plus a probability for every option and a confidence score.

See https://docs.typesafe.ai/primitives/choice.md . The endpoint is
``POST https://api.typesafe.ai/v1/systemone`` with a ``Bearer`` token read from
the ``TYPESAFE_API_KEY`` environment variable.

The client is transport-agnostic: ``HttpJevBackend`` talks to the real API with
nothing but the standard library, and ``MockJevBackend`` lets the whole policy
loop be exercised offline with no key and no network.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Callable, Dict, List, Mapping, Optional

# The Choice primitive accepts at most 255 options per question.
MAX_CHOICE_OPTIONS = 255

DEFAULT_ENDPOINT = "https://api.typesafe.ai/v1/systemone"
DEFAULT_MODEL = "jev-latest"


class JevError(RuntimeError):
    """Raised when a Jev request cannot be built or its response parsed."""


@dataclass(frozen=True)
class ChoiceAnswer:
    """The parsed answer to a single Choice question."""

    choice: str
    probabilities: Dict[str, float]
    confidence: Optional[float]
    raw: Optional[dict] = None


class JevBackend:
    """Transport interface. Implementations turn a request body into a dict."""

    def system_one(self, body: dict) -> dict:  # pragma: no cover - interface
        raise NotImplementedError


class HttpJevBackend(JevBackend):
    """Talks to the real Jev endpoint using only the standard library."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        endpoint: str = DEFAULT_ENDPOINT,
        timeout: float = 60.0,
    ):
        self.api_key = api_key or os.environ.get("TYPESAFE_API_KEY")
        if not self.api_key:
            raise JevError(
                "No API key: pass api_key=... or set TYPESAFE_API_KEY. "
                "Create one at https://console.typesafe.ai/keys"
            )
        self.endpoint = endpoint
        self.timeout = timeout

    def system_one(self, body: dict) -> dict:
        data = json.dumps(body).encode("utf-8")
        req = urllib.request.Request(
            self.endpoint,
            data=data,
            method="POST",
            headers={
                "Authorization": "Bearer {}".format(self.api_key),
                "Content-Type": "application/json",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", "replace")
            raise JevError("Jev HTTP {}: {}".format(e.code, detail)) from e
        except urllib.error.URLError as e:
            raise JevError("Jev request failed: {}".format(e.reason)) from e


class MockJevBackend(JevBackend):
    """Offline backend. Delegates option choice to a pluggable function.

    ``chooser(state, question_id, options)`` returns the chosen option name.
    The default picks the first option, which keeps dry-runs deterministic.
    """

    def __init__(
        self,
        chooser: Optional[Callable[[str, str, List[str]], str]] = None,
    ):
        self.chooser = chooser or (lambda state, qid, options: options[0])
        self.calls: List[dict] = []

    def system_one(self, body: dict) -> dict:
        self.calls.append(body)
        answers = {}
        state = body.get("state", "")
        for qid, q in body.get("questions", {}).items():
            if q.get("type") != "choice":
                raise JevError("MockJevBackend only handles choice questions")
            options = list(q.get("criteria", {}).keys())
            if not options:
                raise JevError("choice question {!r} has no options".format(qid))
            chosen = self.chooser(state, qid, options)
            if chosen not in options:
                raise JevError(
                    "chooser returned {!r}, not one of the options".format(chosen)
                )
            probs = {opt: (1.0 if opt == chosen else 0.0) for opt in options}
            answers[qid] = {
                "type": "choice",
                "choice": chosen,
                "confidence": 1.0,
                "probabilities": probs,
            }
        return {"model": "mock", "answers": answers}


class JevClient:
    """High-level client exposing the primitives LMRL-Gym policies need."""

    def __init__(self, backend: JevBackend, model: str = DEFAULT_MODEL):
        self.backend = backend
        self.model = model

    def choice(
        self,
        state: str,
        instructions: str,
        options: Mapping[str, Optional[str]],
        question_id: str = "action",
    ) -> ChoiceAnswer:
        """Ask one Choice question and return the parsed answer.

        ``options`` maps each option name to a description (or ``None`` when the
        name speaks for itself). Option names are what the model returns and
        what the caller maps back to a concrete action.
        """
        if not options:
            raise JevError("choice requires at least one option")
        if len(options) > MAX_CHOICE_OPTIONS:
            raise JevError(
                "choice accepts at most {} options, got {}; shortlist first "
                "(e.g. BED-LLM posterior filtering for Wordle)".format(
                    MAX_CHOICE_OPTIONS, len(options)
                )
            )

        body = {
            "model": self.model,
            "state": state,
            "questions": {
                question_id: {
                    "type": "choice",
                    "instructions": instructions,
                    "criteria": dict(options),
                }
            },
        }
        resp = self.backend.system_one(body)
        try:
            answer = resp["answers"][question_id]
            choice = answer["choice"]
        except (KeyError, TypeError) as e:
            raise JevError("unexpected Jev response: {!r}".format(resp)) from e

        return ChoiceAnswer(
            choice=choice,
            probabilities=answer.get("probabilities", {}),
            confidence=answer.get("confidence"),
            raw=answer,
        )
