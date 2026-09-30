"""Experimental, read-only client for the native SystemOne decision API.

This module selects among caller-supplied labels. It never imports or executes
Cascade skills, changes the orchestrator, or turns a probability into motion
permission. Choice and Noul use the native https://docs.typesafe.ai/api wire
format. This deliberately small subset accepts string instructions and string
Choice descriptions; Score and structured rubrics are not implemented.
"""

from __future__ import annotations

import hashlib
from http.client import HTTPException
import json
import math
import socket
import time
from dataclasses import dataclass
from typing import Any, Mapping
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener


class DecisionError(RuntimeError):
    """An explicit transport or protocol failure; callers must not act on it."""


@dataclass(frozen=True)
class ChoiceQuestion:
    instructions: str
    criteria: Mapping[str, str | None]

    def to_wire(self) -> dict:
        if not isinstance(self.instructions, str) or not self.instructions.strip():
            raise ValueError("Choice instructions must be a nonempty string")
        if not 2 <= len(self.criteria) <= 255:
            raise ValueError("Choice requires 2 to 255 options")
        if any(not isinstance(k, str) or not k.strip() for k in self.criteria):
            raise ValueError("Choice option names must be nonempty strings")
        if any(v is not None and not isinstance(v, str) for v in self.criteria.values()):
            raise ValueError("Choice descriptions must be strings or null")
        return {"type": "choice", "instructions": self.instructions,
                "criteria": dict(self.criteria)}


@dataclass(frozen=True)
class NoulQuestion:
    instructions: str

    def to_wire(self) -> dict:
        if not isinstance(self.instructions, str) or not self.instructions.strip():
            raise ValueError("Noul instructions must be a nonempty string")
        return {"type": "noul", "instructions": self.instructions}


Question = ChoiceQuestion | NoulQuestion


# Compatible servers can serialize probabilities rounded to four decimal
# places. Each option then contributes at most half a 0.0001 rounding unit to
# the total error. Keep the returned values; never renormalize them silently.
_PROBABILITY_ROUNDING_UNIT = 0.0001


@dataclass(frozen=True)
class ChoiceAnswer:
    choice: str
    probabilities: Mapping[str, float]
    confidence: float


@dataclass(frozen=True)
class NoulAnswer:
    noul: float


@dataclass(frozen=True)
class DecisionResponse:
    requested_model: str
    reported_model: str
    answers: Mapping[str, ChoiceAnswer | NoulAnswer]
    latency_ms: float
    request_sha256: str
    response_sha256: str
    usage: Mapping[str, int] | None


class _NoRedirect(HTTPRedirectHandler):
    # In particular, never forward an Authorization header to a redirect target.
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False,
                      separators=(",", ":")).encode("utf-8")


def _probability(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise DecisionError(f"{field} must be a number, not a boolean or string")
    # Check the range first: math.isfinite() converts Python integers to float
    # and can overflow on a perfectly valid JSON integer such as 10**400.
    if not 0 <= value <= 1 or not math.isfinite(value):
        raise DecisionError(f"{field} must be finite and between 0 and 1")
    return float(value)


def _unique_object(pairs: list[tuple[str, Any]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise DecisionError("Response contains a duplicate JSON key")
        result[key] = value
    return result


def _invalid_constant(value: str):
    raise DecisionError("Response contains a non-finite JSON number")


def validate_response(payload: Any, questions: Mapping[str, Question]) -> tuple:
    """Validate without normalizing probabilities or inventing missing fields."""
    if not isinstance(payload, dict):
        raise DecisionError("Response must be a JSON object")
    model = payload.get("model")
    if not isinstance(model, str) or not model.strip():
        raise DecisionError("Response must identify the evaluated model")
    answers = payload.get("answers")
    if not isinstance(answers, dict) or set(answers) != set(questions):
        raise DecisionError("Answer IDs must match the requested question IDs exactly")
    parsed = {}
    for key, question in questions.items():
        answer = answers[key]
        if not isinstance(answer, dict):
            raise DecisionError(f"Answer {key} must be an object")
        if isinstance(question, NoulQuestion):
            if answer.get("type") != "noul":
                raise DecisionError(f"Answer {key} has the wrong type")
            parsed[key] = NoulAnswer(_probability(answer.get("noul"), f"{key}.noul"))
            continue
        if answer.get("type") != "choice":
            raise DecisionError(f"Answer {key} has the wrong type")
        raw_probs = answer.get("probabilities")
        if not isinstance(raw_probs, dict) or set(raw_probs) != set(question.criteria):
            raise DecisionError(f"Answer {key} must give a probability for every option")
        probs = {k: _probability(v, f"{key}.probabilities") for k, v in raw_probs.items()}
        rounding_tolerance = len(probs) * _PROBABILITY_ROUNDING_UNIT / 2 + 1e-12
        if not math.isclose(math.fsum(probs.values()), 1.0, rel_tol=0,
                            abs_tol=rounding_tolerance):
            raise DecisionError(
                f"Answer {key} probabilities must sum to 1 within four-decimal rounding error"
            )
        choice = answer.get("choice")
        if not isinstance(choice, str) or choice not in probs:
            raise DecisionError(f"Answer {key} chose an unlisted option")
        if probs[choice] < max(probs.values()) - 1e-6:
            raise DecisionError(f"Answer {key} choice is not a highest-probability option")
        parsed[key] = ChoiceAnswer(
            choice, probs, _probability(answer.get("confidence"), f"{key}.confidence")
        )
    usage = payload.get("usage")
    if usage is not None:
        if not isinstance(usage, dict) or any(
            isinstance(v, bool) or not isinstance(v, int) or v < 0 for v in usage.values()
        ):
            raise DecisionError("Token usage must contain nonnegative integers")
    return model, parsed, usage


class SystemOneClient:
    """Native POST /v1/systemone, one attempt per call and no redirects.

    Credentials are explicitly supplied and only accepted for api.typesafe.ai
    over HTTPS. Local Kev servers receive no key, even if the environment has
    TYPESAFE_API_KEY. A local compatible server is not the official Jev model.
    """

    def __init__(self, base_url: str, model: str, *, timeout_s: float = 30.0,
                 api_key: str | None = None):
        parsed = urlsplit(base_url)
        try:
            port = parsed.port
        except ValueError as exc:
            raise ValueError("Invalid endpoint port") from exc
        if (not parsed.hostname or parsed.username is not None or parsed.password is not None
                or parsed.path not in ("", "/") or parsed.query or parsed.fragment):
            raise ValueError("base_url must be an origin without a path or credentials")
        local = parsed.hostname in {"127.0.0.1", "localhost", "::1"}
        official = (parsed.scheme == "https" and parsed.hostname == "api.typesafe.ai"
                    and port in (None, 443))
        if parsed.scheme != "https" and not (local and parsed.scheme == "http"):
            raise ValueError("Remote endpoints require HTTPS; HTTP is only allowed on loopback")
        if api_key is not None and not official:
            raise ValueError("TypeSafe credentials may only go to https://api.typesafe.ai")
        if api_key is not None and (not isinstance(api_key, str) or not api_key.strip()
                                    or "\n" in api_key or "\r" in api_key):
            raise ValueError("API key must be a nonempty single-line string")
        if not isinstance(model, str) or not model.strip():
            raise ValueError("model must explicitly identify the requested model")
        if isinstance(timeout_s, bool) or not math.isfinite(timeout_s) or timeout_s <= 0:
            raise ValueError("timeout_s must be positive and finite")
        self.endpoint = base_url.rstrip("/") + "/v1/systemone"
        self.model = model
        self.timeout_s = float(timeout_s)
        self._api_key = api_key
        self._opener = build_opener(_NoRedirect())

    def evaluate(self, state: str | dict | list,
                 questions: Mapping[str, Question]) -> DecisionResponse:
        if not isinstance(state, (str, dict, list)):
            raise ValueError("state must be a string, object, or array")
        if not questions or any(not isinstance(k, str) or not k.strip() for k in questions):
            raise ValueError("questions must have nonempty string IDs")
        if any(not isinstance(q, (ChoiceQuestion, NoulQuestion)) for q in questions.values()):
            raise ValueError("Only ChoiceQuestion and NoulQuestion are supported")
        request_body = canonical_json({"model": self.model, "state": state,
                                       "questions": {k: q.to_wire() for k, q in questions.items()}})
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        if self._api_key is not None:
            headers["Authorization"] = "Bearer " + self._api_key
        request = Request(self.endpoint, data=request_body, headers=headers, method="POST")
        start = time.monotonic()
        try:
            with self._opener.open(request, timeout=self.timeout_s) as response:
                if response.status != 200:
                    raise DecisionError(f"SystemOne returned HTTP {response.status}; no retry")
                raw = response.read(4 * 1024 * 1024 + 1)
                if len(raw) > 4 * 1024 * 1024:
                    raise DecisionError("SystemOne response exceeds 4 MiB")
        except HTTPError as exc:
            # Do not log the response body: it can echo request data or credentials.
            raise DecisionError(f"SystemOne returned HTTP {exc.code}; no retry") from None
        except (URLError, TimeoutError, socket.timeout, OSError, HTTPException):
            raise DecisionError("SystemOne connection failed or timed out; no retry") from None
        elapsed = (time.monotonic() - start) * 1000
        try:
            body = json.loads(raw, object_pairs_hook=_unique_object, parse_constant=_invalid_constant)
        except (ValueError, UnicodeError):
            raise DecisionError("SystemOne response is not valid JSON") from None
        reported, answers, usage = validate_response(body, questions)
        return DecisionResponse(self.model, reported, answers, elapsed,
                                hashlib.sha256(request_body).hexdigest(),
                                hashlib.sha256(raw).hexdigest(), usage)
