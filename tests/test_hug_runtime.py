"""HUG as an opt-in grasp backend: config wiring, fail-visible paths, memory
re-ranking and the golden behaviour of every existing profile.

The rules this pins (ROADMAP follow-up #7, ledger B17):

  - opt-in only: `grasp.backend: hug` comes from the `isaac_kitchen_hug`
    profile or an explicit `CASCADE_GRASP_BACKEND=hug`; every other shipped
    profile keeps its GraspGen-X/OBB backend, and those backends are called
    exactly as before (no RGB-D frame is handed to them);
  - a REQUIRED learned backend fails visibly (startup and per grasp) and
    never substitutes OBB; an optional one reports its OBB fallback;
  - HUG emits no score: candidates go through the same `GraspOutcomeMemory`
    re-rank as every backend, and the selector/harness stay the authority.
"""

from __future__ import annotations

import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from conftest import REPO

from cascade.config import load_demo_config
from cascade.types import Grasp, SkillError

#: arm profiles that select HUG. Every other shipped profile must not.
_HUG_PROFILES = {"isaac_kitchen_hug"}


@pytest.fixture(autouse=True)
def _no_ambient_hug_endpoint(monkeypatch):
    monkeypatch.delenv("CASCADE_HUG_PORT", raising=False)
    monkeypatch.delenv("CASCADE_HUG_HOST", raising=False)
    monkeypatch.delenv("CASCADE_GRASP_BACKEND", raising=False)
    # Runtimes built here dial only this item's port block (43700-43799),
    # never the shared GraspGen-X/occupancy/bridge ports; nothing listens there.
    monkeypatch.setenv("CASCADE_GRASPGENX_PORT", "43701")
    monkeypatch.setenv("CASCADE_OCCUPANCY_PORT", "43702")
    monkeypatch.setenv("CASCADE_BRIDGE_PORT", "43703")


def _arm_profiles():
    from cascade.config import _load_profile_raw

    arms = REPO / "configs" / "arms"
    return [p.stem for p in sorted(arms.glob("*.yaml"))
            if _load_profile_raw("arms", p.stem, arms.parent).get("template") is not True]


# ── golden: existing profiles keep their backend ───────────────────────────────


def test_no_existing_profile_selects_hug_and_the_presenter_keeps_required_graspgenx():
    """Golden pin, true before and after HUG landed: HUG is opt-in."""
    for name in _arm_profiles():
        if name in _HUG_PROFILES:
            continue
        grasp = load_demo_config(arm=name).grasp
        assert grasp.get("backend") in {"graspgenx", "obb"}, name
    kitchen = load_demo_config(arm="isaac_kitchen_gpu").grasp
    assert kitchen.backend == "graspgenx" and kitchen.graspgenx.get("required") is True
    default = load_demo_config().grasp
    assert default.backend == "graspgenx" and not default.graspgenx.get("required", False)


@pytest.mark.parametrize("backend", ["obb", "graspgenx", "hug"])
def test_only_hug_is_handed_the_rgbd_frame(backend):
    """Golden pin of the call shape for obb/graspgenx (identical to main):
    only HUG consumes the RGB-D frame, so the existing backends are planned
    with exactly the arguments they always got."""
    from cascade.skills.runtime import SkillRuntime
    from test_hug_backend import _scene

    frame, fix, _ = _scene()
    cfg = load_demo_config()
    cfg._data["grasp"]["backend"] = backend
    calls = []
    grasp = Grasp(fix.position, np.eye(3), 0.05, np.array([0.0, 0.0, -1.0]), quality=0.5)
    fake = SimpleNamespace(
        cfg=cfg, _localize=lambda label, spatial_hint=None: (frame, fix),
        _plan_grasps=lambda f, **kw: calls.append(kw) or [grasp],
        grasp_memory=SimpleNamespace(prior=lambda *a: None),
        memory=SimpleNamespace(add=lambda *a: None))
    out = SkillRuntime.skill_preview_grasp(fake, "red cube")
    assert out["ok"] is True
    expected = {"label": "red cube", "_frame": frame} if backend == "hug" else {"label": "red cube"}
    assert calls == [expected]


@pytest.mark.parametrize("backend", ["obb", "graspgenx"])
def test_grasp_object_plans_the_existing_backends_without_the_frame(tmp_path, monkeypatch, backend):
    """Golden pin of grasp_object's planning call site (the bounded-search
    batch generator): obb/graspgenx get exactly main's arguments, no frame.
    GraspGen-X is optional here and dials only this item's block port
    (43701, nothing listening), never :5556 -- recorded at connect time."""
    pytest.importorskip("pinocchio")
    if backend == "graspgenx":
        pytest.importorskip("zmq")
    import cascade.grasping.graspgenx_backend as ggx
    from cascade.apps.demo import build_runtime, shutdown_runtime

    dialled, real_connect = [], ggx.GraspGenXClient._connect

    def spy_connect(client):
        dialled.append(client._port)
        return real_connect(client)

    monkeypatch.setattr(ggx.GraspGenXClient, "_connect", spy_connect)
    cfg = load_demo_config(camera="mock_small", arm="so101_mock", llm="mock")
    cfg._data["grasp"]["backend"] = backend
    cfg._data["grasp"]["graspgenx"]["required"] = False
    runtime, arm = build_runtime(cfg, tmp_path / "run")
    try:
        calls, real = [], runtime._plan_grasps

        def spy(fix, **kwargs):
            calls.append(sorted(kwargs))
            return real(fix, **kwargs)

        runtime._plan_grasps = spy
        arm.object_stop_frac = 0.5
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline and runtime.beliefs.find("red cube") is None:
            time.sleep(0.05)
        with runtime.watcher.paused():
            result = runtime.execute("grasp_object", {"label": "red cube"})
        assert result["ok"] is True, result
        assert calls and all("_frame" not in kw for kw in calls), calls
        assert set(dialled) == ({43701} if backend == "graspgenx" else set()), dialled
    finally:
        shutdown_runtime(runtime, arm)


# ── config wiring ──────────────────────────────────────────────────────────────


def test_launcher_env_can_select_hug_on_every_arm_and_still_rejects_typos(monkeypatch):
    monkeypatch.setenv("CASCADE_GRASP_BACKEND", "hug")
    cfg = load_demo_config(arms=["isaac_kitchen_gpu", "mock"])
    assert cfg.grasp.backend == "hug"
    assert all(arm["resolved"]["grasp"]["backend"] == "hug" for arm in cfg.arms)
    monkeypatch.setenv("CASCADE_GRASP_BACKEND", "hugs")
    with pytest.raises(ValueError, match="CASCADE_GRASP_BACKEND"):
        load_demo_config()


@pytest.mark.parametrize("flag, env", [
    (["--graspgenx", "none"], {}),
    (["--graspgenx", "local"], {}),
    (["--graspgenx", "external"], {}),
    ([], {"CASCADE_GRASP_BACKEND": "graspgenx"}),
    ([], {"CASCADE_GRASP_BACKEND": "obb"}),
])
def test_the_launcher_refuses_to_replace_the_hug_profiles_backend(flag, env):
    """`--graspgenx local|external|none` export CASCADE_GRASP_BACKEND, which
    would silently turn the HUG profile into GraspGen-X or OBB. The launcher
    refuses the contradiction before touching anything."""
    import os
    import subprocess

    clean = {k: v for k, v in os.environ.items()
             if k not in {"CASCADE_INSTALL_PROFILE", "CASCADE_GRASP_BACKEND"}}
    result = subprocess.run(
        ["bash", str(REPO / "scripts/launch.sh"), "--dry-run", "--arm", "isaac_kitchen_hug",
         *flag], env={**clean, **env}, capture_output=True, text=True, timeout=60)
    assert result.returncode == 2, result.stdout + result.stderr
    assert "isaac_kitchen_hug selects the HUG grasp backend" in result.stderr


def test_hug_defaults_live_in_demo_yaml_and_the_default_backend_is_unchanged():
    hug = load_demo_config().grasp.hug
    assert hug.port == 5558 and hug.host == "127.0.0.1"
    assert hug.required is True
    assert hug.num_samples >= 1 and hug.crop == "center"


def test_kitchen_hug_profile_is_the_kitchen_gpu_profile_with_a_required_hug_backend():
    """The opt-in profile changes the grasp backend and nothing else, so a
    live A/B against `isaac_kitchen_gpu` measures the backend."""
    import copy

    gpu = copy.deepcopy(load_demo_config(arm="isaac_kitchen_gpu").as_dict())
    hug = copy.deepcopy(load_demo_config(arm="isaac_kitchen_hug").as_dict())
    assert hug["grasp"]["backend"] == "hug"
    assert hug["grasp"]["hug"]["required"] is True
    assert hug["arm"]["resolved"]["grasp"]["backend"] == "hug"
    # the countertop's approach rule applies to every backend on this arm
    assert hug["grasp"]["hug"]["approach_z_max"] == gpu["grasp"]["graspgenx"]["approach_z_max"]
    for d in (gpu, hug):
        d.pop("arms")
        view = d["arm"].pop("resolved")
        for v in (d, view):
            v["grasp"].pop("backend")
            v["grasp"].pop("hug")
            v["arm"].pop("name", None)
    assert gpu == hug


# ── per-grasp: required fails visibly, optional falls back visibly ─────────────


class _FakeHug:
    def __init__(self, grasps, fail_first=False):
        self.grasps, self.fail_first = grasps, fail_first
        self.status, self.calls, self.kwargs = None, 0, None
        self.last_latency_s, self.last_counts = 0.2, {"returned": 4, "kept": len(grasps)}

    def probe(self, **kwargs):
        self.probe_kwargs = kwargs
        self.status = {"status": "ok", "stub": False, "learned": True}
        return self.status

    def plan(self, frame, fix, **kwargs):
        self.calls += 1
        self.frame, self.kwargs = frame, kwargs
        if self.fail_first and self.calls == 1:
            raise RuntimeError("temporary timeout")
        return list(self.grasps)

    def describe(self):
        return "hug (learned human hand -> cascade pinch, cuda:0)"


def _runtime(required, hug, memory=None, notes=None):
    cfg = load_demo_config()
    cfg._data["grasp"]["backend"] = "hug"
    cfg._data["grasp"]["hug"]["required"] = required
    return SimpleNamespace(
        cfg=cfg, _max_width=0.09, _tool_axis_order="down_open",
        _hug=hug, _hug_down=False,
        extrinsics=SimpleNamespace(cam_to_base=lambda: pytest.fail("frame carries T_base_cam")),
        memory=SimpleNamespace(add=lambda kind, text: (notes if notes is not None else []).append(text)),
        grasp_memory=memory or SimpleNamespace(prior=lambda *a: None))


def _framed_scene():
    from test_hug_backend import _scene

    frame, fix, T = _scene()
    frame.T_base_cam = T
    return frame, fix, T


def _learned(fix, approach=(0.0, 0.0, -1.0), quality=0.8):
    a = np.asarray(approach, float) / np.linalg.norm(approach)
    x = np.array([0.0, 1.0, 0.0])
    R = np.column_stack([a, x, np.cross(a, x)])
    return Grasp(np.r_[fix.points[:, :2].mean(axis=0), 0.03], R, 0.05, a, quality=quality,
                 label=fix.label)


def test_required_hug_fails_visibly_and_never_substitutes_obb():
    from cascade.skills.runtime import SkillRuntime

    frame, fix, _ = _framed_scene()
    hug = _FakeHug([_learned(fix)], fail_first=True)
    rt = _runtime(True, hug)
    with pytest.raises(SkillError, match="HUG required but unavailable"):
        SkillRuntime._plan_grasps(rt, fix, _frame=frame)
    assert rt.grasp_planner_used == "hug (unavailable)"
    assert rt._hug_down is True
    result = SkillRuntime._plan_grasps(rt, fix, _frame=frame)   # next command retries
    assert result == hug.grasps                                  # learned only, no OBB mixed in
    assert rt._hug_down is False and rt.grasp_planner_used == hug.describe()


def test_optional_hug_falls_back_to_a_labelled_obb_and_recovers():
    from cascade.skills.runtime import SkillRuntime

    frame, fix, _ = _framed_scene()
    notes = []
    hug = _FakeHug([_learned(fix, quality=2.0)], fail_first=True)
    rt = _runtime(False, hug, notes=notes)
    fallback = SkillRuntime._plan_grasps(rt, fix, _frame=frame)
    assert fallback and all(g is not hug.grasps[0] for g in fallback)
    assert rt.grasp_planner_used == "obb (hug down)"
    assert any("hug unavailable" in n and "OBB fallback" in n for n in notes)
    rt._hug_retry_after = 0.0
    mixed = SkillRuntime._plan_grasps(rt, fix, _frame=frame)
    assert mixed[0] is hug.grasps[0] and len(mixed) > 1     # learned + reported OBB
    assert rt.grasp_planner_used == hug.describe() and hug.calls == 2


def test_hug_gets_the_frame_its_transform_and_the_arms_jaw_convention():
    from cascade.skills.runtime import SkillRuntime

    frame, fix, T = _framed_scene()
    hug = _FakeHug([_learned(fix)])
    rt = _runtime(True, hug)
    rt._tool_axis_order = "open_down"
    SkillRuntime._plan_grasps(rt, fix, _frame=frame)
    assert hug.frame is frame
    np.testing.assert_array_equal(hug.kwargs["T_base_cam"], T)
    assert hug.kwargs["axis_order"] == "open_down"
    assert hug.kwargs["max_width_m"] == 0.09
    assert hug.kwargs["width_pad_m"] == rt.cfg.grasp.get("width_pad_m", 0.015)


def test_hug_without_the_localized_frame_is_refused_not_guessed():
    from cascade.skills.runtime import SkillRuntime

    _, fix, _ = _framed_scene()
    hug = _FakeHug([_learned(fix)])
    with pytest.raises(SkillError, match="frame"):
        SkillRuntime._plan_grasps(_runtime(True, hug), fix)
    assert hug.calls == 0


@pytest.mark.parametrize("required", [True, False])
def test_a_bounded_search_reraises_hug_errors_unlatched_like_graspgenx(required):
    """Inside grasp_object's bounded search the deadline and cancellation
    check reach the probe and the request, and an error propagates as-is:
    no OBB fallback, no outage latch, no SkillError rewrap -- the search
    decides (same rule as GraspGen-X)."""
    from cascade.skills.runtime import SkillRuntime

    frame, fix, _ = _framed_scene()
    hug = _FakeHug([_learned(fix)], fail_first=True)
    rt = _runtime(required, hug)
    deadline, check = time.monotonic() + 30.0, (lambda: None)
    with pytest.raises(RuntimeError, match="temporary timeout") as info:
        SkillRuntime._plan_grasps(rt, fix, _frame=frame, _deadline=deadline, _check=check,
                                  _prior_snapshot=(None,))
    assert info.type is RuntimeError          # not a SkillError rewrap
    assert rt._hug_down is False
    assert hug.probe_kwargs == {"deadline": deadline, "check": check}
    assert hug.kwargs["deadline"] == deadline and hug.kwargs["check"] is check
    out = SkillRuntime._plan_grasps(rt, fix, _frame=frame, _deadline=deadline, _check=check,
                                    _prior_snapshot=(None,))
    # optional mode appends OBB candidates; the HUG pinch is among the offers
    assert any(g.position is hug.grasps[0].position for g in out)
    assert ("deadline" in hug.kwargs) and rt._hug_down is False


def test_outside_a_bounded_search_no_deadline_reaches_hug():
    """The unbounded call shape stays the plain one (preview_grasp, tests)."""
    from cascade.skills.runtime import SkillRuntime

    frame, fix, _ = _framed_scene()
    hug = _FakeHug([_learned(fix)])
    SkillRuntime._plan_grasps(_runtime(True, hug), fix, _frame=frame)
    assert hug.probe_kwargs == {}
    assert "deadline" not in hug.kwargs and "check" not in hug.kwargs


def test_hug_candidates_are_reranked_by_grasp_outcome_memory(tmp_path):
    """No HUG score exists, so the order a learned prior imposes is OURS:
    the same `GraspOutcomeMemory` re-rank every backend gets."""
    from cascade.memory.grasp_memory import GraspOutcomeMemory
    from cascade.skills.runtime import SkillRuntime

    frame, fix, _ = _framed_scene()
    vertical = _learned(fix, approach=(0.0, 0.0, -1.0), quality=0.8)
    tilted = _learned(fix, approach=(0.7, 0.0, -0.714), quality=0.8)
    memory = GraspOutcomeMemory(path=tmp_path / "gm.json")
    rt = _runtime(True, _FakeHug([vertical, tilted]), memory=memory)
    assert SkillRuntime._plan_grasps(rt, fix, _frame=frame)[0] is vertical
    for _ in range(3):
        memory.record("red cube", fix, tilted, success=True)
    rt._hug = _FakeHug([vertical, tilted])
    assert SkillRuntime._plan_grasps(rt, fix, _frame=frame)[0] is tilted


# ── banner and capability matrix name HUG, never "analytic OBB" ───────────────


def test_banner_and_capability_matrix_describe_hug_honestly():
    from cascade.apps import capabilities
    from cascade.skills.runtime import SkillRuntime

    cfg = load_demo_config()
    cfg._data["grasp"]["backend"] = "hug"
    rt = SimpleNamespace(cfg=cfg, grasp_planner_used=None, _hug=None, _hug_down=False,
                         arm=SimpleNamespace(harness=SimpleNamespace(occupancy=None)))
    assert SkillRuntime.backends(rt)["grasp_planner"].startswith("hug (configured; not yet probed")
    cell = capabilities._grasp_capability(rt, SkillRuntime.backends(rt))
    assert cell["available"] is None
    rt._hug = SimpleNamespace(status={"status": "ok", "stub": False, "learned": True})
    rt.grasp_planner_used = "hug (learned human hand -> cascade pinch, cuda:0)"
    cell = capabilities._grasp_capability(rt, SkillRuntime.backends(rt))
    assert cell["available"] is True and "OBB" not in (cell["why"] or "")
    rt._hug = SimpleNamespace(status={"status": "ok", "stub": True})
    cell = capabilities._grasp_capability(rt, SkillRuntime.backends(rt))
    assert cell["available"] is False and "stub" in cell["why"]
    rt._hug, rt._hug_down = SimpleNamespace(status=None), True
    rt.grasp_planner_used = "obb (hug down)"
    cell = capabilities._grasp_capability(rt, SkillRuntime.backends(rt))
    assert cell["available"] is False and "HUG" in cell["why"]


def test_startup_probe_fails_visibly_for_required_hug_and_degrades_loudly_otherwise(capsys):
    from cascade.apps.demo import _probe_grasp_backend
    from owned_server import held_dead_port

    for required in (True, False):
        cfg = load_demo_config()
        cfg._data["grasp"]["backend"] = "hug"
        rt = SimpleNamespace(cfg=cfg, grasp_planner_used=None, _hug=None, _hug_down=False)
        # A dead server held for the probe (B70), not a port bound and released.
        with held_dead_port() as port:
            cfg._data["grasp"]["hug"].update(required=required, port=port, probe_timeout_ms=200)
            if required:
                with pytest.raises(RuntimeError, match="HUG required at startup"):
                    _probe_grasp_backend(rt)
                continue
            _probe_grasp_backend(rt)
        assert rt._hug_down is True and rt.grasp_planner_used == "obb (hug down)"
        assert "grasp.backend=hug" in capsys.readouterr().err


# ── end to end on the mock stack, through the selector and the harness ────────


def test_mock_stack_plans_and_executes_hug_pinches_through_the_harness(tmp_path, monkeypatch):
    """The whole path on CPU: mock RGB-D camera -> localize -> HUG stub hands
    -> pinch mapping -> memory re-rank -> select_grasp (width, IK, harness
    pre-vet) -> harness-approved motion on the mock arm. The stub is a
    protocol double, so this proves wiring, not grasp quality."""
    pytest.importorskip("pinocchio")
    pytest.importorskip("zmq")
    from cascade.apps.demo import build_runtime, shutdown_runtime

    port = _start_stub(tmp_path)
    cfg = load_demo_config(camera="mock_small", arm="so101_mock", llm="mock")
    cfg._data["grasp"]["backend"] = "hug"
    # `vertical`: the 5-DoF SO-101 cannot reach the stub's ~8 deg tilted
    # palm approach (pregrasp IK fails for every `hand` pinch, measured), so
    # this arm keeps HUG's contacts and approaches top-down.
    cfg._data["grasp"]["hug"].update(port=port, required=False, pinch_approach="vertical")
    import cascade.skills.runtime as runtime_module

    # Optional mode appends the analytic OBB candidates and the pre-existing
    # quality sort orders the mix (as for GraspGen-X); OBB's 0.95 outranks
    # the stub's off-centre pinches (0.945). Withhold the analytic ones here
    # so whatever executes can only be a HUG pinch.
    monkeypatch.setattr(runtime_module, "plan_grasps_from_fix", lambda *a, **k: [])
    selected, proposed = [], []
    real_select = runtime_module.select_grasp

    def spy_select(grasps, *args, **kwargs):
        out = real_select(grasps, *args, **kwargs)
        selected.append(out[0])
        return out

    monkeypatch.setattr(runtime_module, "select_grasp", spy_select)
    runtime, arm = build_runtime(cfg, tmp_path / "run")
    try:
        assert runtime.grasp_planner_used.startswith("hug-stub")
        real_plan = runtime._hug.plan

        def spy_plan(*args, **kwargs):
            proposed[:] = real_plan(*args, **kwargs)
            return list(proposed)

        monkeypatch.setattr(runtime._hug, "plan", spy_plan)
        arm.object_stop_frac = 0.5      # the mock jaws stop on an object: a held grasp
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline and runtime.beliefs.find("red cube") is None:
            time.sleep(0.05)
        with runtime.watcher.paused():
            preview = runtime.execute("preview_grasp", {"label": "red cube"})
            assert preview["ok"] is True, preview
            assert preview["planned_grasp"]["approach"] == "top-down"
            assert runtime.grasp_planner_used.startswith("hug-stub")
            result = runtime.execute("grasp_object", {"label": "red cube"})
        assert result["ok"] is True, result
        assert result["grip_verified"] is True
        assert runtime.grasp_planner_used.startswith("hug-stub")
        # Executed through select_grasp (width, IK, harness pre-vet): one of
        # this batch's HUG pinches. The selector may return the jaw-flip twin
        # and the runtime may lift z to the table floor in place; both keep
        # the candidate's own position array.
        (chosen,) = selected
        assert proposed and any(chosen.position is g.position for g in proposed)
        assert np.allclose(chosen.approach, [0.0, 0.0, -1.0])
    finally:
        shutdown_runtime(runtime, arm)
        _stop_stub()


_STUB = {}


def _start_stub(tmp_path) -> int:
    import re
    import subprocess
    import sys

    proc = subprocess.Popen([sys.executable, str(REPO / "scripts" / "serve_hug.py"),
                             "--stub", "--port", "0"],
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    _STUB["proc"] = proc
    line = proc.stdout.readline()
    match = re.search(r"tcp://127\.0\.0\.1:(\d+)", line or "")
    assert match, (line, proc.stderr.read()[:400] if proc.poll() is not None else "")
    return int(match.group(1))


def _stop_stub():
    proc = _STUB.pop("proc", None)
    if proc is not None:
        proc.kill()
        proc.wait(timeout=5)
