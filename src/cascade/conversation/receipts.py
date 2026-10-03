"""Bounded speech views of already-recorded tool results; never new verdicts."""
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


def _encode(value):
    return json.dumps(value, allow_nan=False)


def _digest(value):
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def speech_tool_output(result, *, recorded=False):
    """Keep small results exact; omit named attachments from oversized receipts.

    The caller has already recorded the full result through RobotRuntime. This
    projection is display/model context only: it never enters a verifier, changes
    a task ledger, or extends command authority. If the remaining result is too
    large, preserve the existing explicit transport-error fallback.
    """
    original = _encode(result)
    if len(original.encode()) <= OUTPUT_BYTES:
        return original
    if not recorded or not isinstance(result, dict):
        return _encode(_FALLBACK)
    view = dict(result)
    omitted = []
    for path, allowed_type in _ATTACHMENTS:
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
        return _encode(_FALLBACK)
    output = _encode({"result": view, "speech_transport": {
        "details_omitted": True, "full_result_sha256": _digest(result),
        "full_result_json_bytes": len(original.encode()), "omitted": omitted,
        "note": "This is a partial view of the recorded tool receipt. Retained values are unchanged. "
                "Omitted evidence is in the runtime trace; its digest is not a physical verdict."}})
    return output if len(output.encode()) <= OUTPUT_BYTES else _encode(_FALLBACK)
