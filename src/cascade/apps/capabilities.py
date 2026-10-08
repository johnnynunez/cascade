"""Capability matrix: what THIS rig can do, derived from the built runtime.

Waddle's stack (/developers/waddle-stack) publishes a capability matrix and
offers the model only the tools whose preconditions the robot meets -- no
depth, no 3D tools. CASCADE did that by hand (`CASCADE_HIDE_TOOLS`). Now the
matrix is derived from PROBED state: the camera streams' depth source (what
`DepthProvider.ensure_depth` would really produce), the sidecar probes that
`runtime.backends()` already reports, the `ArmRig`, the mobile bases, the
postcondition verifier and the episodic memory. `apps/mcp_server.py` trims
its catalog by it (`TOOL_REQUIREMENTS` below), rejects a withheld tool if a
model calls it anyway, and `CASCADE_HIDE_TOOLS` stays the explicit operator
override on top. The matrix is printed next to the `backends:` banner line,
served by the dashboard `/state` and by the MCP `world_state` tool, so a
hidden tool is never silent.

Two rules keep this honest:

* UNPROBED is not UNAVAILABLE. Every capability is tri-state (`available`:
  True / False / None). Before the runtime exists nothing is probed and
  nothing is withheld; a partial runtime that lacks a camera rig reads as
  "unknown", not "RGB-only". A tool is withheld only when probed state
  POSITIVELY shows a precondition unmet.
* A FALLBACK is reported, never hidden. GraspGen-X down means the grasp
  tools run on the analytic OBB planner and the matrix says so; the
  occupancy bridge down means the clearance gate is off and the matrix says
  so. Only a tool that cannot run at all without the sidecar is withheld
  (no shipped tool is in that position today; the mechanism is pinned by
  `tests/test_mcp_server.py`).

The stop path (`emergency_stop`, `reset_stop`) can never depend on a probe.
"""

from __future__ import annotations

from typing import Any

#: some manipulation/fusing camera can lift a pixel to metres (sensor, mono
#: or table-plane depth) -- the precondition of every label/pixel -> 3D tool
CAP_DEPTH_3D = "depth_3d"
#: MEASURED depth (sensor or mono): object heights are real. Table-plane
#: depth puts every pixel ON the table (AGENTS.md: tall-object heights are
#: wrong on RGB-only cameras), so stacking cannot aim from it.
CAP_DEPTH_HEIGHTS = "depth_heights"
#: a real GraspGen-X server answered its startup probe (not the protocol
#: stub) and has not latched down since
CAP_LEARNED_GRASPS = "learned_grasps"
#: the occupancy bridge answered its probe: the clearance gate is live
CAP_OCCUPANCY = "occupancy"
#: two or more arms in the ArmRig
CAP_MULTI_ARM = "multi_arm"
#: a mobile base is part of this runtime
CAP_MOBILE_BASE = "mobile_base"
#: Pigey postcondition verifier attached (`verify_effects`)
CAP_VERIFIER = "verifier"
#: episodic memory with the Vesta frame harness
CAP_MEMORY = "memory"

CAPABILITIES = (
    CAP_DEPTH_3D, CAP_DEPTH_HEIGHTS, CAP_LEARNED_GRASPS, CAP_OCCUPANCY,
    CAP_MULTI_ARM, CAP_MOBILE_BASE, CAP_VERIFIER, CAP_MEMORY,
)

#: Hand-maintained like `_MOTION_SKILLS`: tool name -> capabilities it cannot
#: run without. A tool that merely DEGRADES without a capability (grasp tools
#: on OBB, `handover`/`throw` without a label, `reset_scene` on a real rig)
#: is deliberately absent: the fallback is reported in the matrix, not
#: hidden. `tests/test_mcp_server.py` pins every key to a served tool.
TOOL_REQUIREMENTS: dict[str, tuple[str, ...]] = {
    # label/pixel -> metres: nothing to aim without 3D grounding
    "localize_object": (CAP_DEPTH_3D,),
    "preview_grasp": (CAP_DEPTH_3D,),
    "grasp_object": (CAP_DEPTH_3D,),
    "pick_and_place": (CAP_DEPTH_3D,),
    "push_object": (CAP_DEPTH_3D,),
    "point_at": (CAP_DEPTH_3D,),
    "sort_by_color": (CAP_DEPTH_3D,),
    "turn_screw": (CAP_DEPTH_3D,),
    "grasp_at_pixel": (CAP_DEPTH_3D,),
    "probe_point": (CAP_DEPTH_3D,),
    "locate_pixel": (CAP_DEPTH_3D,),
    # stacking releases above the target's OBSERVED top: plane-cast depth
    # reads every top as the table
    "place_on_object": (CAP_DEPTH_3D, CAP_DEPTH_HEIGHTS),
    # one arm: nothing to enumerate (the `arm` parameter is dropped too)
    "list_arms": (CAP_MULTI_ARM,),
    # reads saved verdicts: none exist without the verifier
    "verify_last_action": (CAP_VERIFIER,),
    # the Vesta frame harness and the event digest live in episodic memory
    "task_memory": (CAP_MEMORY,),
    "recall_memory": (CAP_MEMORY,),
}

#: never withheld by any probe: the stop path is not a capability
_NEVER_WITHHELD = frozenset({"emergency_stop", "reset_stop"})

_3D_SOURCES = ("sensor", "mono", "plane")
_MEASURED_SOURCES = ("sensor", "mono")


def _entry(available: bool | None, detail: str, why: str | None = None) -> dict:
    """One matrix cell. `why` is the human reason a tool is withheld; it is
    only set when `available` is False."""
    return {"available": available, "detail": detail,
            "why": why if available is False else None}


def _camera_entries(runtime) -> dict[str, dict] | None:
    """Per camera: the depth source `ensure_depth` would produce for its
    latest frame (never grabs one), whether it is the manipulation camera
    and whether it fuses 3D beliefs. None when the runtime has no rig to
    probe (partial/unit-test runtimes) -- unknown, not RGB-only."""
    rig = getattr(runtime, "rig", None)
    streams = getattr(rig, "streams", None)
    if not isinstance(streams, dict) or not streams:
        return None
    primary = getattr(rig, "primary", None)
    # the watcher wiring knows each camera's DepthProvider and fuse flag
    providers: dict[Any, tuple[Any, bool | None]] = {}
    for cam in getattr(getattr(runtime, "watcher", None), "_cams", None) or ():
        stream = getattr(cam, "stream", None)
        if stream is not None:
            providers[id(stream)] = (getattr(cam, "depth", None),
                                     bool(getattr(cam, "fuse", True)))
    out: dict[str, dict] = {}
    for name, stream in streams.items():
        provider, fuses = providers.get(id(stream), (None, None))
        is_primary = stream is primary
        if is_primary:
            provider = provider if provider is not None else getattr(runtime, "depth", None)
            fuses = True
        frame = stream.latest() if hasattr(stream, "latest") else None
        declared = getattr(stream, "has_depth", None)
        declared = bool(declared) if isinstance(declared, bool) else None
        source: str | None
        if provider is not None and hasattr(provider, "depth_source_for"):
            source = provider.depth_source_for(frame, sensor=declared)
        elif frame is not None:
            source = frame.depth_source if frame.has_depth else "none"
        elif declared is not None:
            source = "sensor" if declared else "none"
        else:
            source = None
        entry = {"depth": source, "primary": is_primary, "fuses": fuses}
        if frame is None and source is not None:
            entry["note"] = "declared by the driver; no frame grabbed yet"
        out[str(name)] = entry
    return out


def _depth_capabilities(cameras: dict[str, dict] | None) -> tuple[dict, dict]:
    if cameras is None:
        return (_entry(None, "no camera rig to probe"),
                _entry(None, "no camera rig to probe"))
    # the cameras `_localize` can aim from: the manipulation camera and the
    # fusing secondaries (a side camera that only streams video never feeds
    # a grasp)
    usable = {n: c for n, c in cameras.items() if c["primary"] or c["fuses"]}
    summary = ", ".join(f"{n}: {c['depth'] or 'unknown'}" for n, c in cameras.items())
    sources = [c["depth"] for c in usable.values()]
    if any(s in _3D_SOURCES for s in sources):
        three_d = _entry(True, summary)
    elif all(s is None for s in sources):
        three_d = _entry(None, f"no frame probed yet ({summary})")
    else:
        three_d = _entry(
            False, summary,
            "no camera on this rig can lift a pixel to metres (no sensor, "
            f"mono or table-plane depth: {summary}); 3D grounding cannot aim",
        )
    if any(s in _MEASURED_SOURCES for s in sources):
        heights = _entry(True, summary)
    elif all(s is None for s in sources):
        heights = _entry(None, f"no frame probed yet ({summary})")
    else:
        heights = _entry(
            False, summary,
            f"no measured depth on this rig (depth: {summary}); plane-cast "
            "depth puts every pixel on the table, so object heights are unknown",
        )
    return three_d, heights


def _grasp_capability(runtime, backends: dict) -> dict:
    grasp = backends.get("grasp_planner")
    if grasp is None:
        return _entry(None, "grasp planner not reported")
    grasp = str(grasp)
    cfg = getattr(runtime, "cfg", None)
    gcfg = cfg.get("grasp") if cfg is not None and hasattr(cfg, "get") else None
    want = str(gcfg.get("backend", "obb")) if gcfg is not None and hasattr(gcfg, "get") else None
    if want == "hug":
        return _hug_capability(runtime, grasp)
    planner = getattr(runtime, "_graspgenx", None)
    status = getattr(planner, "status", None)
    down = bool(getattr(runtime, "_graspgenx_down", False))
    if status and not status.get("stub") and not down:
        return _entry(True, grasp)
    if down:
        return _entry(False, grasp, f"GraspGen-X is not answering ({grasp}); grasps run "
                                    "on the analytic OBB planner")
    if status and status.get("stub"):
        return _entry(False, grasp, f"only the GraspGen-X protocol stub answered ({grasp}): "
                                    "analytic, not learned")
    if want is not None and want != "graspgenx":
        return _entry(False, grasp, f"GraspGen-X is not configured (grasp.backend: {want}); "
                                    "grasps use the analytic OBB planner")
    if want is None and not grasp.startswith("graspgenx"):
        return _entry(False, grasp, f"GraspGen-X is not in the loop ({grasp}); grasps use "
                                    "the analytic OBB planner")
    # configured, probe deferred to the first grasp: not a fact either way
    return _entry(None, grasp)


def _hug_capability(runtime, grasp: str) -> dict:
    """`grasp.backend: hug`: learned (human-hand) grasps only when a real HUG
    server answered; the stub and an outage are named, never "OBB" alone."""
    status = getattr(getattr(runtime, "_hug", None), "status", None)
    if bool(getattr(runtime, "_hug_down", False)):
        return _entry(False, grasp, f"HUG is not answering ({grasp}); a required profile "
                                    "refuses grasps, an optional one uses the analytic "
                                    "OBB planner")
    if status and status.get("stub"):
        return _entry(False, grasp, f"only the HUG protocol stub answered ({grasp}): "
                                    "analytic, not learned")
    if status:
        return _entry(True, grasp)
    return _entry(None, grasp)


def _occupancy_capability(backends: dict) -> dict:
    live = backends.get("occupancy_live")
    detail = str(backends.get("occupancy", "not reported"))
    if live is None:
        return _entry(None, detail)
    if live:
        return _entry(True, detail)
    return _entry(False, detail,
                  f"the occupancy bridge is not answering ({detail}); the "
                  "clearance gate is off")


def capability_matrix(runtime) -> dict:
    """Evaluate every capability from a BUILT runtime. `runtime=None` is the
    unprobed matrix (nothing known, nothing withheld). Cheap, deterministic
    and read-only: it never grabs a frame, never touches a LazyArm, so the
    dashboard can call it on every poll."""
    if runtime is None:
        return {"probed": False, "cameras": {}, "arms": [], "bases": [],
                **{c: _entry(None, "runtime not built yet") for c in CAPABILITIES}}
    out: dict = {"probed": True}
    try:
        backends = dict(runtime.backends() or {})
    except Exception as e:  # noqa: BLE001 -- a report must never fail the call
        backends = {}
        out["backends_error"] = f"{type(e).__name__}: {e}"

    cameras = _camera_entries(runtime)
    out["cameras"] = cameras or {}
    out[CAP_DEPTH_3D], out[CAP_DEPTH_HEIGHTS] = _depth_capabilities(cameras)
    out[CAP_LEARNED_GRASPS] = _grasp_capability(runtime, backends)
    out[CAP_OCCUPANCY] = _occupancy_capability(backends)

    arm_rig = getattr(runtime, "arm_rig", None)
    if arm_rig is not None:
        names = [str(n) for n in getattr(arm_rig, "names", [])] or ["default"]
    else:
        names = ["default"]
    out["arms"] = names
    if len(names) >= 2:
        out[CAP_MULTI_ARM] = _entry(True, f"{len(names)} arms: {', '.join(names)}")
    else:
        out[CAP_MULTI_ARM] = _entry(
            False, f"1 arm: {names[0]}",
            f"single-arm rig ({names[0]}): nothing to enumerate; the `arm` "
            "parameter is dropped from the motion tools",
        )

    cfg = getattr(runtime, "cfg", None)
    bases = cfg.get("bases") if cfg is not None and hasattr(cfg, "get") else None
    base_names = [str(b.get("name", i)) for i, b in enumerate(bases or [])] if isinstance(bases, list) else []
    out["bases"] = base_names
    out[CAP_MOBILE_BASE] = (_entry(True, ", ".join(base_names)) if base_names else
                            _entry(False, "none", "no mobile base in this rig"))

    verifier = getattr(runtime, "effects", None)
    out[CAP_VERIFIER] = (_entry(True, "postcondition verifier attached") if verifier is not None
                         else _entry(False, "off",
                                     "the postcondition verifier is off (verify_effects: false); "
                                     "there are no verdicts to read"))

    memory = getattr(runtime, "memory", None)
    out[CAP_MEMORY] = (_entry(True, "episodic memory with frame harness")
                       if memory is not None and hasattr(memory, "memory_frames")
                       else _entry(False, "none", "no episodic memory in this runtime"))
    return out


def withheld_tools(matrix: dict, requirements: dict[str, tuple[str, ...]] | None = None) -> dict[str, str]:
    """tool -> reason, for every tool with a requirement the matrix shows as
    positively UNMET. Unknown (None) never withholds; an unprobed matrix
    withholds nothing; the stop path is never in here."""
    if not matrix.get("probed"):
        return {}
    reqs = TOOL_REQUIREMENTS if requirements is None else requirements
    out: dict[str, str] = {}
    for tool, caps in reqs.items():
        if tool in _NEVER_WITHHELD:
            continue
        unmet = [c for c in caps if (matrix.get(c) or {}).get("available") is False]
        if unmet:
            out[tool] = "; ".join(f"{c}: {matrix[c].get('why') or matrix[c].get('detail')}"
                                  for c in unmet)
    return out


def format_matrix(matrix: dict) -> str:
    """One banner line, same shape as `backends: grasp_planner=... | occupancy=...`."""
    if not matrix.get("probed"):
        return "unprobed (runtime not built yet)"

    def flag(cap: str, detail: str | None = None, *, always_detail: bool = False) -> str:
        cell = matrix.get(cap) or {}
        avail = cell.get("available")
        word = "yes" if avail else ("no" if avail is False else "unknown")
        detail = cell.get("detail") if detail is None else detail
        show = detail and (always_detail or avail is not True)
        return f"{cap}={word}" + (f" ({detail})" if show else "")

    cams = matrix.get("cameras") or {}
    depth = ", ".join(f"{n}: {c.get('depth') or 'unknown'}" for n, c in cams.items()) or "no cameras"
    parts = [
        flag(CAP_DEPTH_3D, depth, always_detail=True),
        flag(CAP_DEPTH_HEIGHTS, ""),
        flag(CAP_LEARNED_GRASPS),
        flag(CAP_OCCUPANCY),
        f"arms={len(matrix.get('arms') or [])}",
        f"bases={len(matrix.get('bases') or [])}",
        flag(CAP_VERIFIER, ""),
        flag(CAP_MEMORY, ""),
    ]
    return " | ".join(parts)
