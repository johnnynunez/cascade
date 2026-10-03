"""Receipts for observed software teardown; never physical rest assertions."""
from __future__ import annotations

import copy


def teardown_step(name, operation):
    """Keep an owner's error while allowing subsequent owners to close."""
    try:
        result = operation()
        if result is None:  # existing synchronous close()/disconnect() convention
            return {"stage": name, "ok": True, "complete": True}
        if isinstance(result, dict):
            pending = any(result.get(key) for key in
                          ("pending", "pending_threads", "pending_domains", "pending_providers"))
            ok = result.get("ok") is True and result.get("complete", True) is True and not pending
            return {"stage": name, "ok": ok, "complete": ok, "result": copy.deepcopy(result)}
        return {"stage": name, "ok": False, "complete": False,
                "errors": [{"type": "InvalidTeardownResult", "message": type(result).__name__}]}
    except Exception as error:
        chain, seen = [], set()
        while error is not None and id(error) not in seen:
            seen.add(id(error))
            chain.append({"type": type(error).__name__, "message": str(error)})
            error = error.__cause__ or error.__context__
        return {"stage": name, "ok": False, "complete": False, "errors": chain}


def teardown_receipt(stages, *, pending_threads=()):
    complete = all(stage.get("ok") is True and stage.get("complete") is True for stage in stages)
    complete = complete and not pending_threads
    return {"schema": 1, "ok": bool(complete), "complete": bool(complete),
            "scope": "software_teardown", "physical_rest_verified": False,
            "stages": copy.deepcopy(stages), "pending_threads": list(pending_threads)}
