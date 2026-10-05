"""Bounded model views of already-recorded tool results; never new verdicts."""
from __future__ import annotations

import hashlib
import json


OUTPUT_BYTES = 32768
_FALLBACK = {"ok": False, "error": "tool receipt exceeds speech transport bound; inspect runtime trace"}
# Only known bulky observation attachments can be omitted. Unknown structures,
# flags, errors, verdicts, metrics and bindings never undergo generic truncation.
_ATTACHMENTS = (
    (("image_before_jpeg_b64",), str),
    (("image_jpeg_b64",), str),
    (("measured", "samples"), list),
    (("postcondition", "evidence"), (dict, list)),
)

# Agent episodes re-send every tool message on every planner turn. They also
# omit the full state snapshots bracketing a motion; the receipt's measured
# scalars, verdict and metrics stay exact, and get_base_state reads state.
AGENT_OUTPUT_BYTES = 8192
_AGENT_FALLBACK = {"ok": False, "error": "tool receipt exceeds agent context bound; inspect runtime trace"}
_AGENT_ATTACHMENTS = _ATTACHMENTS + (
    (("measured", "before"), dict),
    (("measured", "after"), dict),
    (("distance_baseline",), dict),
)


def _encode(value):
    return json.dumps(value, allow_nan=False)


def _digest(value):
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def _bounded_output(result, *, recorded, limit, attachments, fallback, label):
    original = _encode(result)
    if len(original.encode()) <= limit:
        return original
    if not recorded or not isinstance(result, dict):
        return _encode(fallback)
    view = dict(result)
    omitted = []
    for path, allowed_type in attachments:
        parent = view
        for key in path[:-1]:
            if not isinstance(parent.get(key), dict):
                break
            parent[key] = dict(parent[key])
            parent = parent[key]
        else:
            value = parent.get(path[-1])
            if not isinstance(value, allowed_type):
                continue
            del parent[path[-1]]
            omitted.append({"path": "/" + "/".join(path), "json_sha256": _digest(value),
                            "json_bytes": len(_encode(value).encode()), "items": len(value)})
    if not omitted:
        return _encode(fallback)
    output = _encode({"result": view, label: {
        "details_omitted": True, "full_result_sha256": _digest(result),
        "full_result_json_bytes": len(original.encode()), "omitted": omitted,
        "note": "This is a partial view of the recorded tool receipt. Retained values are unchanged. "
                "Omitted evidence is in the runtime trace; its digest is not a physical verdict."}})
    return output if len(output.encode()) <= limit else _encode(fallback)


def speech_tool_output(result, *, recorded=False):
    """Keep small results exact; omit named attachments from oversized receipts.

    The caller has already recorded the full result through RobotRuntime. This
    projection is display/model context only: it never enters a verifier, changes
    a task ledger, or extends command authority. If the remaining result is too
    large, preserve the existing explicit transport-error fallback.
    """
    return _bounded_output(result, recorded=recorded, limit=OUTPUT_BYTES, attachments=_ATTACHMENTS,
                           fallback=_FALLBACK, label="speech_transport")


def agent_tool_output(result, *, recorded=False):
    """The same contract for an agent planner's tool message, with a tighter bound.

    Only the caller's model context changes: the runtime trace keeps the complete
    receipt and every verdict, metric, binding and error is retained exactly or
    the explicit fallback is returned.
    """
    return _bounded_output(result, recorded=recorded, limit=AGENT_OUTPUT_BYTES,
                           attachments=_AGENT_ATTACHMENTS, fallback=_AGENT_FALLBACK, label="agent_view")
