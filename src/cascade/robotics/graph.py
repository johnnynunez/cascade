"""Bounded skill graphs over the same runtime used by MCP.

This is an execution substrate inspired by graph-as-policy, not an implementation
of GaP's learning harness. Graphs cannot load code, reset stop permission, or
replace domain verifiers. Deadlines request priority stop; an uncooperative native
call still needs its backend/process deadline to return.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import json
import threading
import time

from .contracts import freeze_json, identifier, plain_json


def _digest(value):
    return hashlib.sha256(json.dumps(plain_json(value), sort_keys=True,
                                    separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _references(value):
    if isinstance(value, Mapping):
        if "$result" in value:
            ref = value["$result"]
            if set(value) != {"$result"} or not isinstance(ref, Mapping) or set(ref) != {"node", "path"}:
                raise ValueError("invalid result reference")
            identifier(ref["node"], "source node")
            if not isinstance(ref["path"], (list, tuple)) or any(
                not (isinstance(p, str) or type(p) is int and p >= 0) for p in ref["path"]
            ):
                raise ValueError("result path must contain object keys or nonnegative array indices")
            yield ref
        else:
            for item in value.values():
                yield from _references(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _references(item)


def _resolve(value, results):
    if isinstance(value, Mapping):
        if "$result" in value:
            ref = value["$result"]
            if ref["node"] not in results:
                raise ValueError("result source has not executed on this path")
            result = results[ref["node"]]
            for key in ref["path"]:
                if isinstance(key, str) and isinstance(result, Mapping):
                    result = result[key]
                elif type(key) is int and isinstance(result, (list, tuple)):
                    result = result[key]
                else:
                    raise ValueError("result path type mismatch")
            return plain_json(freeze_json(result))
        return {key: _resolve(item, results) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_resolve(item, results) for item in value]
    return value


@dataclass(frozen=True)
class SkillGraph:
    """Immutable, acyclic graph. Edges select outcomes, references carry data.

    Terminals are ``$success`` and ``$failure``. Every node must explicitly route
    every possible outcome: success/error for reads, and
    confirmed/refuted/unverified/error for motion. No hidden success fallback.
    """
    specification: Mapping

    def __post_init__(self):
        object.__setattr__(self, "specification", freeze_json(self.specification))

    @property
    def sha256(self):
        return _digest(self.specification)

    def validate(self, runtime):
        spec = self.specification
        if not isinstance(spec, Mapping) or set(spec) != {"version", "entry", "nodes", "max_steps", "timeout_s"}:
            raise ValueError("graph requires version, entry, nodes, max_steps and timeout_s")
        if type(spec["version"]) is not int or spec["version"] != 1:
            raise ValueError("unsupported graph version")
        if type(spec["max_steps"]) is not int or not 1 <= spec["max_steps"] <= 256:
            raise ValueError("max_steps must be in [1, 256]")
        if type(spec["timeout_s"]) not in (int, float) or not 0 < spec["timeout_s"] <= 3600:
            raise ValueError("timeout_s must be in (0, 3600]")
        nodes = spec["nodes"]
        if not isinstance(nodes, Mapping) or not 1 <= len(nodes) <= 256 or spec["entry"] not in nodes:
            raise ValueError("invalid graph nodes or entry")
        for name, node in nodes.items():
            identifier(name, "graph node")
            if not isinstance(node, Mapping) or set(node) != {"tool", "arguments", "edges"}:
                raise ValueError("invalid graph node contract")
            descriptor = runtime.tool_descriptors.get(node["tool"])
            if descriptor is None or descriptor.domain == "robot" or descriptor.effect not in {"read", "motion"}:
                raise ValueError("graph nodes must use registered domain read or motion tools")
            descriptor.check_schemas()
            for resource in (*descriptor.requires, *descriptor.writes):
                runtime.resources.require(resource)
            outcomes = {"confirmed", "refuted", "unverified", "error"} if descriptor.effect == "motion" else {"success", "error"}
            if not isinstance(node["edges"], Mapping) or set(node["edges"]) != outcomes:
                raise ValueError("every tool outcome must have an explicit edge")
            if any(target not in nodes and target not in {"$success", "$failure"} for target in node["edges"].values()):
                raise ValueError("edge points to an unknown node")
            if not isinstance(node["arguments"], Mapping):
                raise ValueError("node arguments must be an object")
            refs = list(_references(node["arguments"]))
            if any(ref["node"] not in nodes or ref["node"] == name for ref in refs):
                raise ValueError("reference points to an unknown or self node")
            if not refs:
                descriptor.validate_arguments(plain_json(node["arguments"]))
        # Reject cycles and unreachable nodes, and require result sources to
        # dominate their consumers. Branch-dependent stale values cannot leak.
        predecessors = {name: set() for name in nodes}
        visited, active, order = set(), set(), []

        def visit(name):
            if name.startswith("$"):
                return
            if name in active:
                raise ValueError("skill graph must be acyclic")
            if name in visited:
                return
            active.add(name)
            for target in set(nodes[name]["edges"].values()):
                if target in nodes:
                    predecessors[target].add(name)
                visit(target)
            active.remove(name)
            visited.add(name)
            order.append(name)

        visit(spec["entry"])
        if visited != set(nodes):
            raise ValueError("graph contains unreachable nodes")
        dominators = {}
        for name in reversed(order):
            parents = predecessors[name]
            before = set.intersection(*(dominators[p] for p in parents)) if parents else set()
            for ref in _references(nodes[name]["arguments"]):
                if ref["node"] not in before:
                    raise ValueError("result source must precede consumer on every path")
            dominators[name] = before | {name}
        return self


def _outcome(descriptor, result):
    if descriptor.effect != "motion":
        return "success" if result.get("ok") is True else "error"
    status = result.get("postcondition", {}).get("status")
    if status == "refuted":
        return "refuted"
    if result.get("execution_ok") is False or result.get("delivery_uncertain") is True:
        return "error"
    if result.get("ok") is not True:
        return "error"
    return "confirmed" if status == "confirmed" else "unverified"


def run_skill_graph(graph: SkillGraph, runtime, *, cancel_event=None):
    """Run a frozen graph, preserving original domain receipts and verdicts.

    Data bindings are validated against the target tool schema before dispatch.
    Legacy tools without result schemas are checked at the consumer boundary.
    A terminal success cannot erase any previous failed/unverified motion.
    """
    graph.validate(runtime)  # entire graph checked before any runtime IO
    spec = graph.specification
    catalog_hash = _digest({name: descriptor.as_dict() for name, descriptor in runtime.tool_descriptors.items()})
    start, finished = time.monotonic(), threading.Event()
    cancelled = cancel_event if cancel_event is not None else threading.Event()
    state, state_lock = {"reason": None, "stop_receipt": None}, threading.Lock()
    token = runtime.cancellation_token

    def invalidate(reason):
        with state_lock:
            if state["reason"] is not None:
                return
            state["reason"] = reason
        try:
            receipt = runtime.stop()
        except Exception as exc:
            receipt = {"ok": False, "error": str(exc)}
        with state_lock:
            state["stop_receipt"] = receipt

    def watchdog():
        while not finished.wait(.01):
            if cancelled.is_set():
                invalidate("cancelled")
                return
            if time.monotonic() - start >= spec["timeout_s"]:
                invalidate("deadline")
                return

    monitor = threading.Thread(target=watchdog, name="skill-graph-watchdog", daemon=True)
    results, trace, unresolved = {}, [], []
    current, terminal = spec["entry"], "$failure"
    monitor.start()
    try:
        for _ in range(spec["max_steps"]):
            if cancelled.is_set():
                invalidate("cancelled")
            if time.monotonic() - start >= spec["timeout_s"]:
                invalidate("deadline")
            if state["reason"] or runtime.stopped or runtime.cancellation_token != token:
                invalidate("runtime_invalidated")
                break
            node = spec["nodes"][current]
            descriptor = runtime.tool_descriptors[node["tool"]]
            args = {}
            try:
                args = _resolve(node["arguments"], results)
                descriptor.validate_arguments(args)
                # A replacement runtime catalog cannot silently change a graph
                # while its fixed input schemas and ownership are being used.
                if catalog_hash != _digest({n: d.as_dict() for n, d in runtime.tool_descriptors.items()}):
                    raise ValueError("runtime tool catalog changed during episode")
                result = runtime.execute(descriptor.name, args)
                if not isinstance(result, dict):
                    raise ValueError("tool returned a non-object result")
                descriptor.validate_result(result)
                result = plain_json(freeze_json(result))
                outcome = _outcome(descriptor, result)
            except Exception as exc:
                result, outcome = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}, "error"
            if descriptor.effect == "motion":
                synthetic = any(runtime.resources.require(r).synthetic for r in descriptor.writes)
                if outcome != "confirmed" or synthetic:
                    unresolved.append({"node": current, "outcome": outcome, "synthetic": synthetic})
            results[current] = result
            trace.append({"node": current, "tool": descriptor.name, "arguments": args,
                          "result": result, "outcome": outcome})
            if state["reason"] or runtime.stopped or runtime.cancellation_token != token:
                invalidate("runtime_invalidated")
                break
            target = node["edges"][outcome]
            if target.startswith("$"):
                terminal = target
                break
            current = target
        else:
            invalidate("step_limit")
    finally:
        finished.set()
        monitor.join(timeout=.6)
    if cancelled.is_set() and state["reason"] is None:
        invalidate("cancelled")
    if time.monotonic() - start >= spec["timeout_s"] and state["reason"] is None:
        invalidate("deadline")
    if runtime.stopped or runtime.cancellation_token != token:
        invalidate("runtime_invalidated")
    reason = state["reason"]
    return {"ok": terminal == "$success" and not unresolved and reason is None,
            "terminal": terminal, "interrupted": reason, "unresolved_motion": unresolved,
            "graph_sha256": graph.sha256, "catalog_sha256": catalog_hash,
            "elapsed_s": time.monotonic() - start, "steps": trace,
            "stop_receipt": state["stop_receipt"], "physical_admission": False}
