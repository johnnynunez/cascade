"""The workspace filter excludes every arm's base (spec 2026-10-17 base_pose runtime frame, §3.8, D6; R18).

The filter stays in the TABLE frame. On a rig whose arms declare a base_pose it
rejects as "robot" a point inside ANY arm's base cylinder, and as
"out_of_reach" only a point beyond EVERY arm's reach; each base carries that
arm's own resolved `workspace_filter` radius / height / reach. Without a
base_pose (or with the identity) `bases` is None and every decision is main's.
The watcher and the runtime share ONE filter object (R18).
"""

from __future__ import annotations

import numpy as np
import pytest

from cascade.config import load_demo_config
from cascade.perception.workspace import WorkspaceFilter

SHIPPED_RIGHT = [0.0, -0.22, 0.0, 0.0, 0.0, 0.0]
RAISED = [0.0, -0.22, 0.03, 0.0, 0.0, 0.0]
IDENTITY = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]


def _main_reject(f, center, extents=None, mask_frac=None):
    """main's `WorkspaceFilter.reject` (f5ad8ab), copied verbatim over `f`'s fields."""
    c = np.asarray(center, dtype=float).reshape(3)

    if mask_frac is not None and mask_frac > f.max_frame_frac:
        return "scene"

    if extents is not None:
        e = np.asarray(extents, dtype=float).reshape(3)
        if float(np.max(e)) > f.max_extent_m:
            return "oversize"
        if float(np.max(e)) < f.min_extent_m:
            return "speck"

    if float(np.hypot(c[0], c[1])) < f.base_radius_m and c[2] < f.base_height_m:
        return "robot"

    if c[2] < f.table_z_m:
        return "below_table"
    if c[2] < f.min_height_m:
        return "flat"
    if c[2] > f.max_z_m:
        return "too_high"
    if float(np.hypot(c[0], c[1])) > f.reach_m:
        return "out_of_reach"
    return None


#: main's dataclass defaults (f5ad8ab), literally
_MAIN_DEFAULTS = dict(base_radius_m=0.12, base_height_m=0.50, max_extent_m=0.35, min_extent_m=0.015,
                      table_z_m=-0.05, max_z_m=0.60, reach_m=1.00, max_frame_frac=0.45, min_height_m=0.010,
                      self_mask=True, self_mask_max_frac=0.5)


def _main_fields(cfg):
    """main's `WorkspaceFilter.from_config` (f5ad8ab), as a field dict."""
    if cfg is None or not hasattr(cfg, "get"):
        return dict(_MAIN_DEFAULTS)
    out = {}
    for key, default in _MAIN_DEFAULTS.items():
        v = cfg.get(key, default)
        out[key] = default if v is None else (v if isinstance(default, bool) else float(v))
    return out


def _cfg(arms, poses=None, camera="mock_small"):
    cfg = (load_demo_config(arm=arms[0], camera=camera, llm="mock") if len(arms) == 1
           else load_demo_config(arms=arms, camera=camera, llm="mock"))
    entries = [cfg._data["arm"], *(cfg._data.get("arms") or [])]
    for name, pose in (poses or {}).items():
        hit = [e for e in entries if e.get("name") == name]
        assert hit, name
        for entry in hit:
            if pose is None:
                entry.pop("base_pose", None)
            else:
                entry["base_pose"] = list(pose)
    return cfg


def _rig_filter(cfg):
    from cascade.apps import demo

    bases = demo._workspace_bases(cfg, demo._arm_cfgs(cfg))
    return WorkspaceFilter.from_config(cfg.get("workspace_filter"), bases=bases)


def _pair():
    return _rig_filter(_cfg(["so101_left", "so101_right"]))


# ── 1. no base_pose: main's decisions, every one ──────────────────────────


@pytest.mark.parametrize("arm", [None, "mock", "so101_mock", "so101_left", "rebot_rs", "isaac_kitchen"])
def test_no_bases_keeps_every_decision(arm):
    section = (None if arm is None                     # the dataclass defaults
               else load_demo_config(arm=arm, camera="mock", llm="mock").get("workspace_filter"))
    plain, explicit = WorkspaceFilter.from_config(section), WorkspaceFilter.from_config(section, bases=None)
    assert plain == explicit and plain.bases is None
    fields = {k: getattr(plain, k) for k in _MAIN_DEFAULTS}
    assert fields == _main_fields(section)                  # main's numbers, field by field
    rng = np.random.default_rng(20261018)
    reasons = {}
    span = 1.3 * plain.reach_m
    for _ in range(20_000):
        center = rng.uniform([-span, -span, -0.15], [span, span, 0.75])
        if rng.random() < 0.3:                           # hug the base cylinder's edge
            r = plain.base_radius_m * rng.uniform(0.9, 1.1)
            a = rng.uniform(0, 2 * np.pi)
            center[:2] = r * np.cos(a), r * np.sin(a)
        u = rng.random()
        extents = (None if u < 0.2 else rng.uniform(0.0, 0.02, size=3) if u < 0.3
                   else rng.uniform(0.0, 0.45, size=3))
        mask_frac = None if rng.random() < 0.2 else float(rng.uniform(0.0, 0.5))
        want = _main_reject(plain, center, extents, mask_frac)
        assert plain.reject(center, extents, mask_frac=mask_frac) == want, (center, extents, mask_frac)
        reasons[want] = reasons.get(want, 0) + 1
    # every branch of the rule was decided at least 50 times
    assert set(reasons) == {None, "scene", "oversize", "speck", "robot", "below_table", "flat", "too_high",
                            "out_of_reach"}, reasons
    assert min(reasons.values()) >= 50, reasons


@pytest.mark.parametrize("arms, poses", [
    (["so101_mock"], None),
    (["mock"], None),
    (["rebot_rs"], None),
    (["so101_left"], {"so101_left": IDENTITY}),                    # the identity is no base_pose (§4)
    (["so101_left", "so101_right"], {"so101_left": IDENTITY, "so101_right": IDENTITY}),
    (["so101_mock", "mock"], None),
], ids=["so101_mock", "mock", "rebot_rs", "so101_left_identity", "pair_both_identity", "mixed"])
def test_a_rig_without_a_base_pose_has_no_bases(arms, poses):
    from cascade.apps import demo

    cfg = _cfg(arms, poses)
    assert demo._workspace_bases(cfg, demo._arm_cfgs(cfg)) is None


# ── 2. a base_pose rig: each base, its own numbers ────────────────────────


def test_the_pair_bases_sit_at_each_base_pose_with_that_arms_numbers():
    from cascade.perception.workspace import WorkspaceBase

    assert _pair().bases == (
        WorkspaceBase(x=0.0, y=0.22, z=0.0, radius_m=0.08, height_m=0.50, reach_m=0.48),
        WorkspaceBase(x=0.0, y=-0.22, z=0.0, radius_m=0.08, height_m=0.50, reach_m=0.48),
    )


def test_each_arm_base_is_excluded_and_the_phantom_origin_is_not():
    wf = _pair()
    main = WorkspaceFilter.from_config(_cfg(["so101_left", "so101_right"]).get("workspace_filter"))
    assert wf.reject([0.0, 0.22, 0.05]) == "robot"
    assert wf.reject([0.0, -0.22, 0.05]) == "robot"
    assert wf.reject([0.0, 0.0, 0.05]) is None
    assert main.reject([0.0, 0.0, 0.05]) == "robot"                 # main: a phantom base at the origin
    assert main.reject([0.0, -0.22, 0.05]) is None                  # ... and none where the right arm stands
    # the cylinder's edge and top, around the RIGHT base
    assert wf.reject([0.0, -0.22 - 0.079, 0.05]) == "robot"
    assert wf.reject([0.0, -0.22 - 0.081, 0.05]) is None
    assert wf.reject([0.0, -0.22, 0.49]) == "robot"
    assert wf.reject([0.0, -0.22, 0.51]) is None


def test_a_raised_base_raises_its_cylinder():
    wf = _rig_filter(_cfg(["so101_left", "so101_right"], {"so101_right": RAISED}))
    assert [b.z for b in wf.bases] == [0.0, 0.03]
    assert wf.reject([0.0, -0.22, 0.52]) == "robot"                 # 0.52 < 0.03 + 0.50
    assert wf.reject([0.0, 0.22, 0.52]) is None                     # the left base is not raised


def test_out_of_reach_only_beyond_every_arm():
    wf = _pair()
    assert wf.reject([0.0, 0.62, 0.05]) is None                     # 0.40 from the left base
    assert wf.reject([0.0, -0.62, 0.05]) is None                    # 0.40 from the right base
    assert wf.reject([0.0, 0.75, 0.05]) == "out_of_reach"           # 0.53 from the nearest base
    assert wf.reject([0.0, -0.75, 0.05]) == "out_of_reach"
    # reach counts from the bases, not from the phantom origin: 0.45 m out on x
    # is 0.501 m from either base
    assert wf.reject([0.45, 0.0, 0.05]) == "out_of_reach"


def test_each_base_uses_its_own_arms_reach_and_radius():
    cfg = _cfg(["so101_left", "so101_right"])
    right_view = cfg._data["arms"][1]["resolved"]["workspace_filter"]
    right_view.update(reach_m=0.30, base_radius_m=0.10, base_height_m=0.20)
    wf = _rig_filter(cfg)
    left, right = wf.bases
    assert (left.reach_m, left.radius_m, left.height_m) == (0.48, 0.08, 0.50)
    assert (right.reach_m, right.radius_m, right.height_m) == (0.30, 0.10, 0.20)
    # 0.40 from the right base (beyond its 0.30) and 0.84 from the left: out
    assert wf.reject([0.0, -0.62, 0.05]) == "out_of_reach"
    # 0.09 from the right base: inside its 0.10 cylinder only
    assert wf.reject([0.0, -0.31, 0.05]) == "robot"
    assert wf.reject([0.0, -0.22, 0.25]) is None                    # above its 0.20 cylinder


def test_an_arm_without_a_base_pose_on_a_posed_rig_sits_at_the_origin():
    wf = _rig_filter(_cfg(["so101_left", "so101_right"], {"so101_left": None}))
    assert [(b.x, b.y, b.z) for b in wf.bases] == [(0.0, 0.0, 0.0), (0.0, -0.22, 0.0)]
    assert wf.reject([0.0, 0.0, 0.05]) == "robot"
    # R12's identity rig: left exact identity (at the origin), right at the shipped spacing
    wf = _rig_filter(_cfg(["so101_left", "so101_right"],
                          {"so101_left": IDENTITY, "so101_right": [0.0, -0.44, 0.0, 0.0, 0.0, 0.0]}))
    assert [(b.x, b.y) for b in wf.bases] == [(0.0, 0.0), (0.0, -0.44)]


def test_bases_is_never_empty():
    with pytest.raises(ValueError, match="bases"):
        WorkspaceFilter(bases=())


# ── 3. one filter for the watcher and the runtime (R18) ───────────────────


@pytest.fixture
def make_runtime(tmp_path, monkeypatch):
    from cascade.apps.demo import build_runtime, shutdown_runtime

    monkeypatch.setenv("CASCADE_GRASP_BACKEND", "obb")
    built = []

    def make(arms, poses=None):
        rt, arm = build_runtime(_cfg(arms, poses), tmp_path / f"run{len(built)}", view=False, serve=False)
        built.append((rt, arm))
        return rt

    try:
        yield make
    finally:
        for rt, arm in built:
            shutdown_runtime(rt, arm)


def test_the_watcher_and_the_runtime_share_the_bases(make_runtime):
    rt = make_runtime(["so101_left", "so101_right"])
    assert rt.watcher is not None
    assert rt.watcher._workspace is rt._workspace
    assert [(b.x, b.y) for b in rt._workspace.bases] == [(0.0, 0.22), (0.0, -0.22)]
    single = make_runtime(["so101_mock"])
    assert single.watcher._workspace is single._workspace
    assert single._workspace.bases is None
    assert single._workspace == WorkspaceFilter.from_config(single.cfg.get("workspace_filter"))
