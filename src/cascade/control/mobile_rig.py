"""Ordered, named mobile bases; no arm surrogate and no navigation planner."""
from __future__ import annotations

from types import MappingProxyType

from .mobile_base import identifier


class MobileRig:
    def __init__(self, bases: list, names: list[str]):
        if not bases or len(bases) != len(names):
            raise ValueError("MobileRig requires nonempty, equally sized bases and names")
        order = tuple(identifier(name, "base name") for name in names)
        if len(set(order)) != len(order):
            raise ValueError("duplicate base names")
        self.bases = MappingProxyType(dict(zip(order, bases)))
        self._order = order

    @property
    def primary(self):
        return self.bases[self._order[0]]

    @property
    def names(self) -> list[str]:
        return list(self._order)

    def get(self, name=None):
        if name is None:
            return self.primary
        if name not in self.bases:
            raise KeyError(f"no base {name!r}; available: {self.names}")
        return self.bases[name]

    def __iter__(self):
        return iter(self.bases.values())

    def __len__(self):
        return len(self.bases)

    def connect(self) -> None:
        attempted = []
        try:
            for base in self:
                attempted.append(base)
                base.connect()
        except Exception as error:
            cleanup_errors = []
            for base in reversed(attempted):
                try:
                    base.disconnect()
                except Exception as cleanup:
                    cleanup_errors.append(str(cleanup))
            if cleanup_errors:
                raise RuntimeError(f"connect failed: {error}; cleanup: {cleanup_errors}") from error
            raise

    def _fanout(self, operation, **kwargs) -> dict:
        results = {}
        for name, base in self.bases.items():
            try:
                result = getattr(base, operation)(**kwargs)
                results[name] = {"ok": True} if result is None else result
            except Exception as error:
                results[name] = {"ok": False, "error": str(error)}
        return {"ok": all(isinstance(r, dict) and r.get("ok") is True for r in results.values()),
                "bases": results}

    def stop(self, *, latch=True) -> dict:
        if type(latch) is not bool:
            raise ValueError("latch must be boolean")
        return self._fanout("stop", latch=latch)

    def disconnect(self) -> dict:
        return self._fanout("disconnect")

    def close(self) -> dict:
        return self.disconnect()
