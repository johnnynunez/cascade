"""Passive resource catalog and declared controller-writer conflict detection."""
from __future__ import annotations

from types import MappingProxyType

from .contracts import ResourceDescriptor


class ResourceCatalog:
    """Validate a composition, without claiming a bus or inspecting a driver.

    A controller can expose several logical resources to one writer. Disjoint
    joint subsets do not permit two writers to a whole-body command endpoint.
    Cross-process leases and runtime cancellation remain backend obligations.
    """

    def __init__(self, resources):
        items = tuple(resources)
        if any(not isinstance(item, ResourceDescriptor) for item in items):
            raise TypeError("catalog requires ResourceDescriptor entries")
        if len({item.resource_id for item in items}) != len(items):
            raise ValueError("duplicate resource IDs")
        self._resources = MappingProxyType({item.resource_id: item for item in items})
        self.validate()

    def validate(self) -> None:
        owners = {}
        for item in self._resources.values():
            if item.controller_id is None:
                continue
            owner = owners.setdefault(item.controller_id, item.writer_id)
            if owner != item.writer_id:
                raise ValueError(f"controller {item.controller_id!r} has incompatible writers: "
                                 f"{owner!r}, {item.writer_id!r}")

    def require(self, resource_id: str, capability: str | None = None) -> ResourceDescriptor:
        try:
            resource = self._resources[resource_id]
        except KeyError:
            raise ValueError(f"unknown resource: {resource_id!r}") from None
        if capability is not None and capability not in resource.capabilities:
            raise ValueError(f"resource {resource_id!r} lacks capability {capability!r}")
        return resource

    def describe(self) -> list[dict]:
        return [resource.as_dict() for resource in self._resources.values()]

    def as_dict(self) -> dict:
        return {"resources": self.describe()}

    def __iter__(self):
        return iter(self._resources.values())

    def __len__(self):
        return len(self._resources)
