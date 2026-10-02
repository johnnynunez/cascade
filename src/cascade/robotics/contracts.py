"""Immutable, hardware-free contracts for composed robots and their tools.

Descriptions do not discover hardware, acquire actuators, or establish physical
admission. In particular, never introspect a lazy driver to populate this data.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
import math
import re
from types import MappingProxyType
from typing import Any, Protocol


def identifier(value: str, label: str = "identifier", *, limit: int = 128) -> str:
    if not isinstance(value, str) or not 0 < len(value) <= limit or not re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9_.:/@+-]*", value
    ):
        raise ValueError(f"invalid {label}")
    return value


def freeze_json(value: Any, *, _depth: int = 0) -> Any:
    """Copy finite JSON into immutable containers, without invoking encoders."""
    if _depth > 32:
        raise ValueError("contract nesting exceeds 32 levels")
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("contract numbers must be finite")
        return value
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise ValueError("contract object keys must be strings")
        return MappingProxyType({key: freeze_json(item, _depth=_depth + 1)
                                 for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(freeze_json(item, _depth=_depth + 1) for item in value)
    raise ValueError(f"unsupported contract value: {type(value).__name__}")


def plain_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: plain_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [plain_json(item) for item in value]
    return value


def _identifiers(values, label):
    if isinstance(values, str):
        raise ValueError(f"{label} must be a collection")
    result = tuple(identifier(value, label) for value in values)
    if len(set(result)) != len(result):
        raise ValueError(f"duplicate {label}")
    return result


def _local_schema(schema):
    """Schemas are data, not a means to fetch a remote resource at dispatch."""
    value = plain_json(freeze_json(schema))

    def visit(node):
        if isinstance(node, dict):
            for key, item in node.items():
                if key in {"$ref", "$dynamicRef"} and (
                    not isinstance(item, str) or not item.startswith("#")
                ):
                    raise ValueError("tool schemas may only reference local definitions")
                visit(item)
        elif isinstance(node, list):
            for item in node:
                visit(item)

    visit(value)
    from jsonschema import Draft202012Validator, SchemaError

    try:
        Draft202012Validator.check_schema(value)
    except SchemaError as exc:
        raise ValueError(f"invalid tool schema: {exc.message}") from exc
    return Draft202012Validator(value)


@dataclass(frozen=True)
class ResourceDescriptor:
    resource_id: str
    kind: str
    robot_id: str
    capabilities: tuple[str, ...] = ()
    controller_id: str | None = None
    writer_id: str | None = None
    synthetic: bool = False
    admission: str = "unvalidated"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        for key in ("resource_id", "kind", "robot_id", "admission"):
            identifier(getattr(self, key), key)
        if type(self.synthetic) is not bool:
            raise ValueError("synthetic must be a boolean")
        if (self.controller_id is None) != (self.writer_id is None):
            raise ValueError("controller_id and writer_id must be declared together")
        for key in ("controller_id", "writer_id"):
            if getattr(self, key) is not None:
                identifier(getattr(self, key), key, limit=512)
        object.__setattr__(self, "capabilities", _identifiers(self.capabilities, "capability"))
        if not isinstance(self.metadata, Mapping):
            raise ValueError("resource metadata must be an object")
        object.__setattr__(self, "metadata", freeze_json(self.metadata))

    def as_dict(self) -> dict:
        return {"resource_id": self.resource_id, "kind": self.kind,
                "robot_id": self.robot_id, "capabilities": list(self.capabilities),
                "controller_id": self.controller_id, "writer_id": self.writer_id,
                "synthetic": self.synthetic, "admission": self.admission,
                "metadata": plain_json(self.metadata)}


@dataclass(frozen=True)
class ToolDescriptor:
    name: str
    description: str
    parameters: Mapping[str, Any]
    domain: str
    local_name: str
    effect: str = "read"
    requires: tuple[str, ...] = ()
    writes: tuple[str, ...] = ()
    result_schema: Mapping[str, Any] | None = None

    def __post_init__(self):
        if not isinstance(self.name, str) or not re.fullmatch(r"[A-Za-z0-9_.-]{1,64}", self.name):
            raise ValueError("invalid tool name")
        identifier(self.domain, "domain")
        identifier(self.local_name, "local tool name")
        if not isinstance(self.description, str):
            raise ValueError("tool description must be text")
        if self.effect not in {"read", "motion", "control", "stop"}:
            raise ValueError("unsupported tool effect")
        if not isinstance(self.parameters, Mapping) or self.parameters.get("type") != "object":
            raise ValueError("tool parameters must declare an object schema")
        object.__setattr__(self, "parameters", freeze_json(self.parameters))
        for key in ("requires", "writes"):
            object.__setattr__(self, key, _identifiers(getattr(self, key), key))
        if self.effect == "read" and self.writes:
            raise ValueError("read-only tools cannot claim actuator writes")
        if self.effect == "motion" and not self.writes:
            raise ValueError("motion tools must declare actuator writes")
        if self.result_schema is not None:
            if not isinstance(self.result_schema, Mapping):
                raise ValueError("result schema must be an object")
            object.__setattr__(self, "result_schema", freeze_json(self.result_schema))

    def as_spec(self) -> dict:
        return {"name": self.name, "description": self.description,
                "parameters": plain_json(self.parameters)}

    def as_dict(self) -> dict:
        return {**self.as_spec(), "domain": self.domain, "local_name": self.local_name,
                "effect": self.effect, "requires": list(self.requires),
                "writes": list(self.writes), "result_schema": plain_json(self.result_schema)}

    def check_schemas(self) -> None:
        _local_schema(self.parameters)
        if self.result_schema is not None:
            _local_schema(self.result_schema)

    def validate_arguments(self, arguments: dict) -> None:
        from jsonschema import ValidationError

        value = plain_json(freeze_json(arguments))
        # Undeclared keyword arguments should fail before a lazy driver opens.
        schema = plain_json(self.parameters)
        schema.setdefault("additionalProperties", False)
        try:
            _local_schema(schema).validate(value)
        except ValidationError as exc:
            raise ValueError(f"invalid arguments for {self.name}: {exc.message}") from exc

    def validate_result(self, result: dict) -> None:
        from jsonschema import ValidationError

        if self.result_schema is not None:
            try:
                _local_schema(self.result_schema).validate(plain_json(freeze_json(result)))
            except ValidationError as exc:
                raise ValueError(f"invalid result for {self.name}: {exc.message}") from exc


class RobotDomain(Protocol):
    """A domain owns its existing execution, verification and teardown semantics."""

    domain_id: str
    tool_specs: list[dict]
    motion_skills: frozenset[str]
    resources: tuple[ResourceDescriptor, ...]

    def execute(self, name: str, args: dict) -> dict: ...
    def stop(self) -> dict: ...
    def reset_stop(self) -> dict: ...
    def close(self) -> dict: ...
