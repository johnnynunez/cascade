"""Helpers shared by the base_pose Option 2 tests (spec
docs/superpowers/specs/2026-10-17-base-pose-runtime-frame-design.md, §6.1).

The equivariance oracle: rig A is an arm WITHOUT a base_pose; rig B is the same
arm mounted at a base_pose T with every camera moved by T, so the scene moves
with the arm. Both rigs run the same calls (agent coordinates mapped by T) and
must agree:

- arm commands within 1e-6 rad, jaw commands equal;
- every numeric 2- or 3-vector in the results classified: a point maps to T a
  (1.1 mm, after the results' rounding to the mm), an xy to the xy of T (x, y,
  0), a direction to R a; a frame-free value stays equal; a clock-dependent one
  (ages, frame ids, timings) is not compared. An unclassified vector fails with
  "classify result key <path>", so a new result field forces a choice;
- beliefs map to T a within 1e-6 m.

Fixtures are imported by name into each test module
(`from base_pose_oracle import make_runtime, make_pair_rigs`).
"""

from __future__ import annotations

import contextlib
import re

import numpy as np
import pytest

SHIPPED_RIGHT = [0.0, -0.22, 0.0, 0.0, 0.0, 0.0]
YAWED = [0.03, -0.22, 0.0, 0.0, 0.0, 0.4]
#: used with `safety.table_z: -0.03` and workspace min z -0.05 (Task 4)
RAISED = [0.0, -0.22, 0.03, 0.0, 0.0, 0.0]
TILTED = [0.0, -0.22, 0.0, 0.1, 0.0, 0.0]

#: Every skill the oracle tests run. Each task adds its own; Task 6's
#: derivation test reads the set.
ORACLE_SKILLS: frozenset[str] = frozenset({
    # Task 3: perception into the arm frame, reports back out
    "localize_object", "point_at", "preview_grasp", "get_observation", "list_objects", "describe_scene",
    "count_objects",
    # Task 4: the grasp path
    "grasp_object", "grasp_at_pixel",
    # Task 5: releases and pushes
    "place_at", "place_on_object", "push_object", "pick_and_place",
    # Task 6: the remaining world-target skills, the viewer frame, the probe (R10)
    "sort_by_color", "search_for_object", "snapshot_scene", "restore_scene", "turn_screw", "throw", "handover",
    "move_near_object", "move_relative", "undo_move", "rotate_gripper", "probe_point",
})

#: Result keys whose numeric 3-vector is a POINT: rig B's is T a. Task 5:
#: place_at's `at` / `tcp_at`, pick_and_place's `placed_at`, the retreat's
#: `tcp_at`, push's `new_position_estimate`, a postcondition's `final` pose.
#: Task 6: the spatial moves' `tcp_before` / `tcp_after`, the probe's `center`
#: (the belief it hit).
POINT_KEYS: frozenset[str] = frozenset({"position", "tcp_xyz", "object_xyz", "at", "tcp_at", "placed_at",
                                        "new_position_estimate", "final", "tcp_before", "tcp_after", "center"})
#: Result keys whose numeric 2-vector is an xy in the table frame: rig B's is
#: the xy of T (x, y, 0). Task 5: a failed pick's configured `target`. Task 6:
#: move_near_object's `center_xy_m`, sort_by_color's `zone`, the composites'
#: move records (`from`, `to`; restore_scene's `target`), search_for_object's
#: postcondition `reported` sighting.
XY_KEYS: frozenset[str] = frozenset({"position_xy_m", "target", "center_xy_m", "zone", "from", "to",
                                     "reported"})
#: Mappings whose every value is such an xy (sort_by_color's `zones`, by colour).
XY_MAP_KEYS: frozenset[str] = frozenset({"zones"})
#: Scalar distances in metres measured between two perceived poses (a
#: postcondition's errors and travel): frame-free, but read from perception,
#: so compared at the point tolerance instead of 1e-9.
DISTANCE_KEYS: frozenset[str] = frozenset({"target_err_m", "dest_err_m", "moved_m", "err_m",
                                           "offset_m", "distance_m"})   # Task 6: restore's / the probe's
#: Result keys whose numeric 3-vector is a DIRECTION: rig B's is R a. Task 6:
#: the probe's `delta_xyz_m` (gripper -> point, table frame).
DIRECTION_KEYS: frozenset[str] = frozenset({"delta_xyz_m"})
#: Result keys whose numeric vector does not depend on the frame: equal.
#: Task 6: `displacement_base_m` and `rotation_axis_base` (directions are the
#: arm base's, D8), the probe's `pixel` and `image_size`.
FRAME_FREE_KEYS: frozenset[str] = frozenset({"extent_m", "q_deg", "displacement_base_m", "rotation_axis_base",
                                             "pixel", "image_size"})
#: Clock- or capture-counter-dependent keys: never compared (frame counters,
#: ages, the visible/remembered state derived from an age, timings).
VOLATILE_KEYS: frozenset[str] = frozenset({
    "age_s", "frame_age_s", "frame_id", "state", "perception_timing", "observation_retry",
    "selection_s",    # the observed-finger choice time (wall clock)
    "duration_s", "timings",    # pick_and_place's wall-clock stage times
    "snapshot_age_s"})          # Task 6: restore_scene's snapshot age (wall clock)
#: Heights in the table frame: rig B's is a + T[2, 3] (a level mount; a
#: tilted one refuses every world-target skill). #400 review Minor 3: a
#: postcondition's measured `z` (the object's pose); Minor 4: move_near_object's
#: target `top_z_m`.
HEIGHT_KEYS: frozenset[str] = frozenset({"z", "top_z_m"})
#: Prose that quotes xy positions as "[x, y]": the numbers map like XY_KEYS,
#: the rest of the text stays equal. Task 6: the same prose may quote a point
#: "[x, y, z]" (search_for_object's resume note): it maps like POINT_KEYS.
#: #400 review Minor 3: a height "z=<n>" maps like HEIGHT_KEYS, and `error` is
#: prose like its `next_action` (a refusal that quotes the arm frame's numbers
#: is caught there too), as are the spatial moves' `interpretation` / `summary`.
TEXT_XY_KEYS: frozenset[str] = frozenset({"description", "next_action", "error", "interpretation", "summary"})

_POINT_ATOL = 1.1e-3
_CMD_ATOL = 1e-6
_BELIEF_ATOL = 1e-6
_XY_IN_TEXT = re.compile(r"\[(-?\d+(?:\.\d+)?), (-?\d+(?:\.\d+)?)\]")
#: A printed zero with a minus sign ("-0.000"): the float noise of a frame
#: round trip (T^-1 T) can flip the sign of an exact zero; the text is equal.
_SIGNED_ZERO = re.compile(r"-(?=0\.0*(?![0-9]*[1-9]))")
_XYZ_IN_TEXT = re.compile(r"\[(-?\d+(?:\.\d+)?), (-?\d+(?:\.\d+)?), (-?\d+(?:\.\d+)?)\]")
#: "z=<n>" (a height; "+0.000" after `_SIGNED_ZERO`), never "dz=" / "wz=".
_Z_IN_TEXT = re.compile(r"\bz=([-+]?\d+(?:\.\d+)?)")


def transform(pose) -> np.ndarray:
    """The 4x4 of a base_pose (None: the identity)."""
    from cascade.types import pose_to_transform

    return np.eye(4) if pose is None else pose_to_transform(pose)


def lift(T, p) -> np.ndarray:
    """Arm-frame point(s) (3,) or (N, 3) in the table frame: T p."""
    T = np.asarray(T, dtype=float)
    return np.asarray(p, dtype=float) @ T[:3, :3].T + T[:3, 3]


def bind_like_primary(rt, *names) -> None:
    """Register the runtime's own Kinematics and Cfg for hand-built rig arms.

    `execute(arm=<non-primary>)` refuses an arm with no bound kinematics /
    profile (spec D4: fail closed). Tests that build an ArmRig by hand (data
    doubles for the arms, one shared profile) register the primary's binding
    for their other arm names, which is exactly what those arms ran with
    before the per-call frame existed.
    """
    from cascade.skills.arm_frame import ArmBinding

    rt.register_arm_bindings({str(name): ArmBinding(rt.kin, rt.cfg, None) for name in names})


# ── configs ───────────────────────────────────────────────────────────────


def _data(x):
    return x._data if hasattr(x, "_data") else x


def _arm_entries(data):
    """Every arm entry of a demo config: `arm` and each `arms[*]` (de-duplicated)."""
    seen, out = set(), []
    for entry in [data.get("arm"), *(data.get("arms") or [])]:
        entry = None if entry is None else _data(entry)
        if isinstance(entry, dict) and id(entry) not in seen:
            seen.add(id(entry))
            out.append(entry)
    return out


def _camera_dicts(data):
    """Every camera dict a build can read: `camera`, `cameras[*]`, each arm's
    resolved `camera` / `cameras[*]`, and the MuJoCo loader's
    `mj_prop_from_camera` / `mj_cameras[*]` (top level and per arm).
    De-duplicated by identity, so a dict listed twice moves once."""
    seen, out = set(), []

    def add(cam):
        cam = None if cam is None else _data(cam)
        if isinstance(cam, dict) and isinstance(_data(cam.get("extrinsics")), dict) and id(cam) not in seen:
            seen.add(id(cam))
            out.append(cam)

    def walk(d):
        add(d.get("camera"))
        for cam in d.get("cameras") or []:
            add(cam)
        add(d.get("mj_prop_from_camera"))      # `true` in a profile; a camera dict once planted
        for cam in d.get("mj_cameras") or []:
            add(cam)

    walk(data)
    for entry in _arm_entries(data):
        walk(entry)
        if entry.get("resolved") is not None:
            walk(_data(entry["resolved"]))
    return out


def _set_base_pose(entry, pose) -> None:
    targets = [entry]
    resolved = entry.get("resolved")
    if resolved is not None and isinstance(_data(resolved).get("arm"), dict):
        targets.append(_data(resolved)["arm"])
    for target in targets:
        if pose is None:
            target.pop("base_pose", None)
        else:
            target["base_pose"] = list(pose)


def moved_cfg(arm, pose, *, camera="mock_small", shift_xy=(0.0, 0.0), move_cameras=True):
    """A demo config whose arm stands at `pose` (None: no base_pose).

    `arm` is one profile name; with `move_cameras` every camera's
    `extrinsics.T` becomes T @ T_profile (the scene moves with the arm), then
    every camera shifts by `shift_xy` in the table frame (both rigs alike). A
    list of profile names builds that rig: `pose` is then {name: pose or None}
    (unnamed arms keep their shipped base_pose) and no camera moves with an
    arm (only `shift_xy` applies).
    """
    from cascade.config import load_demo_config

    arms = [arm] if isinstance(arm, str) else list(arm)
    if len(arms) == 1:
        cfg = load_demo_config(arm=arms[0], camera=camera, llm="mock")
        poses = {arms[0]: pose}
        T = transform(pose) if move_cameras else np.eye(4)
    else:
        if move_cameras:
            raise ValueError("a rig's cameras cannot move with one arm: pass move_cameras=False")
        cfg = load_demo_config(arms=arms, camera=camera, llm="mock")
        poses = dict(pose or {})
        T = np.eye(4)
    for entry in _arm_entries(cfg._data):
        if entry.get("name") in poses:
            _set_base_pose(entry, poses[entry["name"]])
    for cam in _camera_dicts(cfg._data):
        ext = _data(cam["extrinsics"])
        if ext.get("T") is None:
            continue
        M = T @ np.asarray(ext["T"], dtype=float)
        M[0, 3] += float(shift_xy[0])
        M[1, 3] += float(shift_xy[1])
        ext["T"] = M.tolist()
    return cfg


def raised_table(cfg, *, table_z=-0.03, workspace_min_z=-0.05):
    """The arm's `safety.table_z` and workspace floor for a base 3 cm above the
    table (RAISED): every arm entry's resolved `safety` and the top level."""
    sections = [cfg._data.get("safety")]
    for entry in _arm_entries(cfg._data):
        resolved = entry.get("resolved")
        if resolved is not None:
            sections.append(_data(resolved).get("safety"))
    for safety in filter(None, (None if s is None else _data(s) for s in sections)):
        safety["table_z"] = float(table_z)
        workspace = _data(safety.get("workspace") or {})
        if "min" in workspace:
            workspace["min"] = [*list(workspace["min"])[:2], float(workspace_min_z)]
    return cfg


def map_args(skill, args, T):
    """Rig A's agent arguments as rig B's: place_at's (x, y, z) -> T p; every
    other skill's arguments are frame-free (labels; directions are relative to
    the arm's base, D8)."""
    args = dict(args)
    if skill == "place_at":
        q = lift(T, [float(args["x"]), float(args["y"]), float(args.get("z") or 0.0)])
        args["x"], args["y"] = float(q[0]), float(q[1])
        if args.get("z") is not None:
            args["z"] = float(q[2])
    return args


# ── runtimes ──────────────────────────────────────────────────────────────


def safe_arms(rt):
    if getattr(rt, "arm_rig", None) is not None:
        return list(rt.arm_rig.arms.values())
    return [rt.arm]


def raw_arm(safe):
    """The backend MockArm behind a SafeArm; None for an unmaterialized LazyArm."""
    raw = safe.raw
    return raw.__dict__.get("_arm", raw)


@pytest.fixture
def make_runtime(tmp_path, monkeypatch):
    """`make(cfg, *, lazy=False) -> rt` on the mock stack. Each runtime gets its
    own belief / grasp-memory / envelope / episodic stores (a shared store
    would carry one rig's table-frame beliefs into the next build), the OBB
    grasp backend, and `object_stop_frac = 0.5` on every raw MockArm (a mock
    jaw that closes fully reports an air grasp). Every runtime is shut down."""
    from cascade.apps.demo import build_runtime, shutdown_runtime

    monkeypatch.setenv("CASCADE_GRASP_BACKEND", "obb")
    built = []

    def make(cfg, *, lazy=False):
        run = tmp_path / f"rt{len(built)}"
        stores = tmp_path / f"rt{len(built)}_stores"
        stores.mkdir(parents=True)
        for env, name in (("CASCADE_BELIEFS_PATH", "beliefs.json"), ("CASCADE_GRASP_MEMORY_PATH", "grasp.json"),
                          ("CASCADE_ENVELOPE_PATH", "envelope.json"), ("CASCADE_EPISODIC_PATH", "episodic.json")):
            monkeypatch.setenv(env, str(stores / name))
        rt, arm = build_runtime(cfg, run, view=False, serve=False, lazy_arm=lazy)
        built.append((rt, arm))
        for safe in safe_arms(rt):
            raw = raw_arm(safe)
            if raw is not None and hasattr(raw, "object_stop_frac"):
                raw.object_stop_frac = 0.5
        return rt

    try:
        yield make
    finally:
        for rt, arm in reversed(built):
            shutdown_runtime(rt, arm)


@pytest.fixture
def make_pair_rigs(make_runtime):
    """`make(pose, *, arm="so101_right", shift_xy=(0, 0), edit=None) -> (rt_a, rt_b, T)`.

    Rig A: `moved_cfg(arm, None)`; rig B: `moved_cfg(arm, pose)` with its
    cameras moved by T; `edit(cfg)` (optional) changes both configs alike
    before the build. The watcher's belief fusion is held on both rigs for
    the whole test (`watcher.paused()`: its 3 Hz fusion at arbitrary moments
    would make belief ages and states differ), and both stores are seeded with
    one fusion pass over the same camera image (`_reobserve(1)`)."""
    with contextlib.ExitStack() as stack:

        def make(pose, *, arm="so101_right", shift_xy=(0.0, 0.0), edit=None):
            rigs = []
            for p in (None, pose):
                cfg = moved_cfg(arm, p, shift_xy=shift_xy)
                if edit is not None:
                    edit(cfg)
                rt = make_runtime(cfg)
                if rt.watcher is not None:
                    stack.enter_context(rt.watcher.paused())
                rigs.append(rt)
            for rt in rigs:
                rt._reobserve(1)
            return rigs[0], rigs[1], transform(pose)

        yield make


# ── running and comparing ─────────────────────────────────────────────────


class _Recorder:
    """Every arm command (MockArm.commands) and jaw command of one runtime."""

    def __init__(self, rt):
        self.raws = [r for r in (raw_arm(s) for s in safe_arms(rt)) if r is not None]
        self.jaws = []
        for raw in self.raws:
            original = raw.set_gripper

            def jaw(*a, _original=original, **k):
                self.jaws.append(tuple(float(x) for x in a if isinstance(x, (int, float))))
                return _original(*a, **k)

            raw.set_gripper = jaw

    def mark(self):
        return [len(r.commands) for r in self.raws], len(self.jaws)

    def since(self, mark):
        n0, j0 = mark
        cmds = [np.asarray(c, dtype=float) for r, n in zip(self.raws, n0) for c in r.commands[n:]]
        return cmds, list(self.jaws[j0:])


def recorder(rt) -> _Recorder:
    """The runtime's command recorder (installed on first use)."""
    rec = rt.__dict__.get("_oracle_recorder")
    if rec is None:
        rec = rt.__dict__["_oracle_recorder"] = _Recorder(rt)
    return rec


def _beliefs(rt):
    return [{"label": b.label, "position": np.asarray(b.position, dtype=float).copy(),
             "top_z": None if b.top_z is None else float(b.top_z), "color": b.color,
             "extent": None if b.extent is None else np.asarray(b.extent, dtype=float).copy()}
            for b in rt.beliefs.all()]


def run_pair(rt_a, rt_b, T, skill, args) -> dict:
    """One call on both rigs (rig B's agent coordinates mapped by T): keys
    cmds_a/b, jaws_a/b, result_a/b, beliefs_a/b."""
    out = {}
    for tag, rt, call in (("a", rt_a, dict(args)), ("b", rt_b, map_args(skill, args, T))):
        rec = recorder(rt)
        mark = rec.mark()
        out[f"result_{tag}"] = rt.execute(skill, call)
        out[f"cmds_{tag}"], out[f"jaws_{tag}"] = rec.since(mark)
        out[f"beliefs_{tag}"] = _beliefs(rt)
    return out


def _is_vector(v) -> bool:
    return (isinstance(v, (list, tuple)) and len(v) in (2, 3)
            and all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in v))


def _key_of(path) -> str | None:
    return next((p for p in reversed(path) if isinstance(p, str)), None)


def _compare(a, b, T, path, problems) -> None:
    key = _key_of(path)
    where = "/".join(str(p) for p in path) or "<result>"
    if key in VOLATILE_KEYS:
        return
    if isinstance(a, str) and isinstance(b, str):
        a, b = _SIGNED_ZERO.sub("+", a), _SIGNED_ZERO.sub("+", b)
    if isinstance(a, dict) and isinstance(b, dict):
        if set(a) != set(b):
            problems.append(f"{where}: keys differ {sorted(set(a) ^ set(b), key=str)}")
            return
        for k in sorted(a, key=str):
            # a mapping of xy values (`zones`): each value classified as an xy
            _compare(a[k], b[k], T, (*path, k, "target") if key in XY_MAP_KEYS else (*path, k), problems)
        return
    if _is_vector(a) and _is_vector(b) and len(a) == len(b):
        va, vb = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
        if key in POINT_KEYS and len(a) == 3:
            want = lift(T, va)
        elif key in XY_KEYS and len(a) == 2:
            want = lift(T, [va[0], va[1], 0.0])[:2]
        elif key in DIRECTION_KEYS and len(a) == 3:
            want = np.asarray(T)[:3, :3] @ va
        elif key in FRAME_FREE_KEYS:
            want = va
        else:
            problems.append(f"classify result key {where}")
            return
        if not np.allclose(vb, want, rtol=0.0, atol=_POINT_ATOL):
            problems.append(f"{where}: B {vb.tolist()} != T a {np.round(want, 4).tolist()} (a {va.tolist()})")
        return
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        if len(a) != len(b):
            problems.append(f"{where}: lengths differ {len(a)} != {len(b)}")
            return
        for i, (x, y) in enumerate(zip(a, b)):
            _compare(x, y, T, (*path, i), problems)
        return
    if key in TEXT_XY_KEYS and isinstance(a, str) and isinstance(b, str):
        pa3, pb3 = _XYZ_IN_TEXT.findall(a), _XYZ_IN_TEXT.findall(b)
        a2, b2 = _XYZ_IN_TEXT.sub("[x, y, z]", a), _XYZ_IN_TEXT.sub("[x, y, z]", b)
        za, zb = _Z_IN_TEXT.findall(a2), _Z_IN_TEXT.findall(b2)
        a2, b2 = _Z_IN_TEXT.sub("z=<z>", a2), _Z_IN_TEXT.sub("z=<z>", b2)
        xa, xb = _XY_IN_TEXT.findall(a2), _XY_IN_TEXT.findall(b2)
        if (_XY_IN_TEXT.sub("[x, y]", a2) != _XY_IN_TEXT.sub("[x, y]", b2) or len(xa) != len(xb)
                or len(pa3) != len(pb3) or len(za) != len(zb)):
            problems.append(f"{where}: text differs: {a!r} != {b!r}")
            return
        for ha, hb in zip(za, zb):
            if not np.isclose(float(hb), float(ha) + float(np.asarray(T)[2, 3]), rtol=0.0, atol=_POINT_ATOL):
                problems.append(f"{where}: z={hb} != z={ha} + {float(np.asarray(T)[2, 3]):g}")
        for pa, pb in zip(pa3, pb3):
            want = lift(T, [float(v) for v in pa])
            if not np.allclose([float(v) for v in pb], want, rtol=0.0, atol=_POINT_ATOL):
                problems.append(f"{where}: [{', '.join(pb)}] != T [{', '.join(pa)}]")
        for pa, pb in zip(xa, xb):
            want = lift(T, [float(pa[0]), float(pa[1]), 0.0])[:2]
            if not np.allclose([float(v) for v in pb], want, rtol=0.0, atol=_POINT_ATOL):
                problems.append(f"{where}: [{pb[0]}, {pb[1]}] != T [{pa[0]}, {pa[1]}]")
        return
    if (key in HEIGHT_KEYS and isinstance(a, (int, float)) and isinstance(b, (int, float))
            and not isinstance(a, bool) and not isinstance(b, bool)):
        want = float(a) + float(np.asarray(T)[2, 3])
        if not np.isclose(b, want, rtol=0.0, atol=_POINT_ATOL):
            problems.append(f"{where}: {b!r} != {a!r} + {float(np.asarray(T)[2, 3]):g} (a table-frame height)")
        return
    if key in DISTANCE_KEYS and isinstance(a, (int, float)) and isinstance(b, (int, float)):
        if not np.isclose(a, b, rtol=0.0, atol=_POINT_ATOL):
            problems.append(f"{where}: {a!r} != {b!r} (distance, {_POINT_ATOL} m)")
        return
    if isinstance(a, float) or isinstance(b, float):
        if not (isinstance(a, (int, float)) and isinstance(b, (int, float))
                and np.isclose(a, b, rtol=0.0, atol=1e-9)):
            problems.append(f"{where}: {a!r} != {b!r}")
        return
    if a != b:
        problems.append(f"{where}: {a!r} != {b!r}")


def assert_equivariant(run, T) -> None:
    """Rig B equals rig A transformed by T (module docstring)."""
    problems = []
    ca, cb = run["cmds_a"], run["cmds_b"]
    if len(ca) != len(cb):
        problems.append(f"arm commands: {len(ca)} in A, {len(cb)} in B")
    else:
        worst = max((float(np.max(np.abs(x - y))) for x, y in zip(ca, cb)), default=0.0)
        if worst > _CMD_ATOL:
            problems.append(f"arm commands differ by {worst:.3g} rad")
    if len(run["jaws_a"]) != len(run["jaws_b"]) or not all(
            np.allclose(x, y, rtol=0.0, atol=1e-9) for x, y in zip(run["jaws_a"], run["jaws_b"])):
        problems.append(f"jaw commands: {run['jaws_a']} != {run['jaws_b']}")
    _compare(run["result_a"], run["result_b"], T, (), problems)
    ba, bb = run["beliefs_a"], run["beliefs_b"]
    if [x["label"] for x in ba] != [x["label"] for x in bb]:
        problems.append(f"belief labels: {[x['label'] for x in ba]} != {[x['label'] for x in bb]}")
    else:
        for x, y in zip(ba, bb):
            if not np.allclose(y["position"], lift(T, x["position"]), rtol=0.0, atol=_BELIEF_ATOL):
                problems.append(f"belief {x['label']!r}: {y['position'].tolist()} != T {x['position'].tolist()}")
            if (x["top_z"] is None) != (y["top_z"] is None) or (
                    x["top_z"] is not None
                    and not np.isclose(y["top_z"], x["top_z"] + T[2, 3], rtol=0.0, atol=_BELIEF_ATOL)):
                problems.append(f"belief {x['label']!r} top_z: {y['top_z']} != {x['top_z']} + {T[2, 3]}")
            if x["color"] != y["color"]:
                problems.append(f"belief {x['label']!r} colour: {x['color']} != {y['color']}")
            if (x["extent"] is None) != (y["extent"] is None) or (
                    x["extent"] is not None and not np.allclose(x["extent"], y["extent"], rtol=0.0,
                                                                atol=_BELIEF_ATOL)):
                problems.append(f"belief {x['label']!r} extent: {x['extent']} != {y['extent']}")
    assert not problems, "not equivariant:\n  " + "\n  ".join(problems)


def lifted_tcps(rt, safe, n0, T) -> np.ndarray:
    """The TCP of every arm command `safe` received after the first `n0`, by
    that arm's own harness Kinematics, lifted into the table frame by T."""
    harness = safe.__dict__.get("harness")
    kin = getattr(harness, "kin", None)
    kin = rt.kin if kin is None else kin
    commands = raw_arm(safe).commands[n0:]
    tcps = np.array([kin.fk(np.asarray(q, dtype=float))[:3, 3] for q in commands]).reshape(-1, 3)
    return lift(T, tcps) if len(tcps) else tcps
