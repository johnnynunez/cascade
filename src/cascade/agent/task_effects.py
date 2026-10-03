"""Task-scoped completion obligations, separate from actor result dictionaries.

Only explicitly registered effects enter this ledger. A later read, successful
call, world reset, or completion claim cannot erase an earlier unknown effect.
Starting a new task is a trusted host operation, never an agent tool.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import threading
import uuid

from .effects import CONFIRMED, Postcondition, REFUTED, UNVERIFIED


@dataclass(frozen=True)
class EffectRecord:
    task_id: str
    action_id: int
    actor: str
    skill: str
    kind: str
    status: str = "pending"
    reason: str = "effect call is still active"


class TaskEffects:
    def __init__(self):
        self._lock = threading.Lock()
        self._task_id = uuid.uuid4().hex
        self._records: list[EffectRecord] = []

    def begin_task(self) -> str:
        with self._lock:
            if any(row.status == "pending" for row in self._records):
                raise ValueError("cannot begin a new task while an effect call is active")
            self._task_id = uuid.uuid4().hex
            self._records.clear()
            return self._task_id

    def begin(self, *, actor: str, skill: str, kind: str) -> EffectRecord:
        with self._lock:
            row = EffectRecord(self._task_id, len(self._records), actor, skill, kind)
            self._records.append(row)
            return row

    def finish(self, row: EffectRecord, result: dict | None, pc: Postcondition | None):
        # pc is supplied by the local checker path, NOT result['postcondition'].
        matched = isinstance(pc, Postcondition) and pc.skill == row.skill and pc.kind == row.kind
        if matched and pc.status == REFUTED:
            status, reason = REFUTED, pc.evidence
        elif not isinstance(result, dict):
            status, reason = "failed", "effect call returned no valid result mapping"
        elif result.get("ok") is not True:
            status, reason = "failed", str(result.get("error", "effect call did not complete successfully"))
        elif result.get("execution_ok") is False or result.get("delivery_uncertain") is True:
            status, reason = UNVERIFIED, "execution failed or command delivery is uncertain"
        elif matched and pc.status == CONFIRMED:
            status, reason = CONFIRMED, pc.evidence
        else:
            status = UNVERIFIED
            reason = (pc.evidence if isinstance(pc, Postcondition) else "independent effect verdict unavailable")
        with self._lock:
            if row.task_id != self._task_id or self._records[row.action_id] != row:
                raise ValueError("effect completion belongs to another task or was already recorded")
            self._records[row.action_id] = replace(row, status=status, reason=reason)

    def unverified_actions(self) -> list[str]:
        with self._lock:
            return [f"{row.actor}:{row.skill}#{row.action_id} {row.status}: {row.reason}"
                    for row in self._records if row.status != CONFIRMED]

    def snapshot(self) -> dict:
        with self._lock:
            return {"task_id": self._task_id, "effects": [asdict(row) for row in self._records]}
