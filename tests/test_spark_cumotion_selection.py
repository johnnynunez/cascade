"""Real shell/dispatch boundaries with CPU doubles; no native admission."""

import json
from pathlib import Path
import re
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
LAUNCH = ROOT / "scripts/launch.sh"


def options(tmp_path, *args, inherited=None):
    script = tmp_path / "options.sh"
    script.write_text(
        LAUNCH.read_text().split("\nlog()  {")[0]
        + '\n"$BOUNDARY_PY" - "$ARM" <<\'PY\'\nimport json,os,sys\nprint(json.dumps({"arm":sys.argv[1], "renderer":os.environ.get("CASCADE_CAMERA_RENDERER"), "evidence":os.environ.get("CASCADE_KITCHEN_CAMERA_RENDERER"), "cuda":os.environ.get("CASCADE_REQUIRE_CUDA")}))\nPY\n'
    )
    env = {
        "PATH": "/usr/bin:/bin",
        "BOUNDARY_PY": sys.executable,
        "CASCADE_INSTALL_PROFILE": "spark",
        "CASCADE_LAUNCH_STATE": str(tmp_path / "state"),
        **(inherited or {}),
    }
    result = subprocess.run(
        ["bash", str(script), *args],
        env=env,
        text=True,
        capture_output=True,
        timeout=10,
    )
    assert not (tmp_path / "state").exists()
    return result


def test_explicit_cumotion_and_renderer_pass_real_spark_option_boundary(tmp_path):
    result = options(
        tmp_path, "--arm", "isaac_kitchen_cumotion", "--camera-renderer", "isaac"
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == dict(
        arm="isaac_kitchen_cumotion", renderer="isaac", evidence="isaac", cuda="1"
    )


def test_spark_default_is_unchanged_and_does_not_request_selection_evidence(tmp_path):
    result = options(tmp_path)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == dict(
        arm="isaac_kitchen_gpu", renderer=None, evidence=None, cuda="1"
    )


@pytest.mark.parametrize(
    "args,env",
    [
        (["--arm", "isaac_kitchen_cumotion"], {}),
        (["--arm", "isaac_kitchen_cumotion"], {"CASCADE_CAMERA_RENDERER": "isaac"}),
        (["--arm", "isaac_cumotion", "--camera-renderer", "isaac"], {}),
        (["--arm", "isaac_kitchen_cumotion", "--camera-renderer", "fake"], {}),
        (
            ["--arm", "isaac_kitchen_cumotion", "--camera-renderer", "isaac"],
            {"CASCADE_CAMERA_RENDERER": "ovrtx"},
        ),
        (["--arm", "isaac_kitchen_cumotion", "--camera-renderer", "ovrtx"], {}),
    ],
)
def test_invalid_or_incomplete_selection_rejects_before_startup(tmp_path, args, env):
    result = options(tmp_path, *args, inherited=env)
    assert result.returncode == 2, result.stdout + result.stderr


def test_explicit_ovrtx_requires_existing_interpreter_and_fresh_output(tmp_path):
    env = {
        "CASCADE_OVRTX_PYTHON": sys.executable,
        "CASCADE_OVRTX_OUTPUT": str(tmp_path / "fresh-renderer"),
    }
    result = options(
        tmp_path,
        "--arm",
        "isaac_kitchen_cumotion",
        "--camera-renderer",
        "ovrtx",
        inherited=env,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["evidence"] == "ovrtx"
    assert not (tmp_path / "fresh-renderer").exists()


def test_selection_reaches_real_mcp_registration_without_budget_change(
    monkeypatch, capsys
):
    blocks = re.findall(r"<<'PYEOF'[^\n]*\n(.*?)\nPYEOF", LAUNCH.read_text(), re.S)
    [registration] = [b for b in blocks if '"requestTimeoutMs"' in b]
    monkeypatch.setenv("CASCADE_KITCHEN_CAMERA_RENDERER", "isaac")
    monkeypatch.setenv("CASCADE_REQUIRE_CUDA", "1")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "-",
            sys.executable,
            "/fixture/repo",
            "isaac,isaac_side,isaac_proof",
            "isaac_kitchen_cumotion",
            "/fixture/detector",
            "",
            "isaac",
            "/fixture/state",
            "owner",
            "0",
            "none",
        ],
    )
    exec(compile(registration, str(LAUNCH), "exec"), {})
    config = json.loads(capsys.readouterr().out)
    assert config["env"]["CASCADE_KITCHEN_CAMERA_RENDERER"] == "isaac"
    assert config["env"]["CASCADE_REQUIRE_CUDA"] == "1"
    assert (
        config["requestTimeoutMs"] == 300000 and config["connectionTimeoutMs"] == 120000
    )
    assert config["args"][:2] == ["-m", "cascade.apps.mcp_server"]


# Existing synthetic prepared-planner/capture and real SkillRuntime fixtures.
from test_kitchen_cumotion_selection import subject as _subject
from test_motion_evidence import runtime as _runtime

subject = _subject
runtime = _runtime


def test_configuration_gate_is_pure_and_requires_exact_opt_in(subject):
    from cascade.planning.backend_evidence import selection_from_environment

    env = {"CASCADE_KITCHEN_CAMERA_RENDERER": "ovrtx", "CASCADE_REQUIRE_CUDA": "1"}
    assert selection_from_environment(subject.cfg, env) == "ovrtx"
    assert selection_from_environment(subject.cfg, {}) is None
    for changes in (
        {"CASCADE_REQUIRE_CUDA": "0"},
        {"CASCADE_KITCHEN_CAMERA_RENDERER": "fake"},
    ):
        with pytest.raises(ValueError):
            selection_from_environment(subject.cfg, {**env, **changes})
    subject.cfg.arm._data["name"] = "isaac_kitchen_gpu"
    with pytest.raises(ValueError):
        selection_from_environment(subject.cfg, env)
    assert subject.calls == []


@pytest.mark.parametrize(
    "fault", [None, "missing", "wrong_renderer", "model", "cancel", "raised"]
)
def test_real_dispatch_guards_stream_and_restores_scoped_hook(subject, runtime, fault):
    s = subject
    runtime.cfg = s.cfg
    runtime.arm = s.runtime.arm
    runtime.rig = s.runtime.rig
    runtime._kitchen_camera_renderer = "ovrtx"
    runtime.effects = None
    original = s.runtime.arm.raw.stream_profile
    if fault == "missing":
        s.frames[0].capture = None
    if fault == "wrong_renderer":
        s.frames[0].capture["renderer"] = "isaac"
    if fault == "model":
        s.native.model_sha256 = "d" * 64

    class Cancel(BaseException):
        pass

    def motion():
        if fault == "cancel":
            raise Cancel()
        if fault == "raised":
            raise RuntimeError("original synthetic failure")
        s.runtime.arm.raw.stream_profile(s.profile, planned_state=s.state)
        return {"ok": True}

    runtime.skill_move_home = motion
    if fault == "cancel":
        with pytest.raises(Cancel):
            runtime.execute("move_home", {})
    else:
        result = runtime.execute("move_home", {})
        assert result["ok"] is (fault is None)
        row = json.loads(runtime.trace._trace_path.read_text().splitlines()[-1])
        evidence = row["context"]["selected_backends"]
        assert evidence["physical_acceptance"] is False
        assert "pass" not in evidence
        if fault is None:
            assert evidence["curves"][0]["plan"]["execution_authorized"] is False
            assert evidence["curves"][0]["completed"] is True
    assert s.runtime.arm.raw.stream_profile == original
    assert len(s.calls) == (1 if fault is None else 0)


def test_read_only_call_does_not_inspect_or_construct_backend(runtime):
    runtime._kitchen_camera_renderer = "ovrtx"
    runtime.skill_get_observation = lambda: {"ok": True}
    assert runtime.execute("get_observation", {})["ok"] is True
    row = json.loads(runtime.trace._trace_path.read_text().splitlines()[-1])
    assert "selected_backends" not in row["context"]


def test_scope_rejects_history_overflow_before_an_extra_stream(subject):
    from cascade.planning.backend_evidence import ordinary_call

    s = subject
    s.runtime._kitchen_camera_renderer = "ovrtx"
    s.runtime.cfg = s.cfg
    original = s.runtime.arm.raw.stream_profile
    context = {}
    with pytest.raises(RuntimeError, match="evidence capacity"):
        with ordinary_call(s.runtime, "move_home", context):
            for _ in range(129):
                s.runtime.arm.raw.stream_profile(s.profile, planned_state=s.state)
    assert len(s.calls) == 128
    assert len(context["selected_backends"]["curves"]) == 128
    assert s.runtime.arm.raw.stream_profile == original


def test_lazy_arm_scope_does_not_materialize_and_restores_on_exception(subject):
    from cascade.control.lazy_arm import LazyArm
    from cascade.planning.backend_evidence import ordinary_call

    s = subject
    s.runtime.cfg = s.cfg
    s.runtime._kitchen_camera_renderer = "ovrtx"
    constructions = []

    def unexpected():
        constructions.append(True)
        raise AssertionError("catalog/guard must not construct an actuator")

    lazy = LazyArm(unexpected, n_joints=6, profile_type="isaac")
    s.runtime.arm.raw = lazy
    original = lazy.stream_profile
    with pytest.raises(RuntimeError, match="no motion"):
        with ordinary_call(s.runtime, "move_home", {}):
            assert not lazy.connected
            raise RuntimeError("no motion")
    assert lazy.stream_profile == original and constructions == []


def test_scope_rechecks_renderer_at_actual_curve_and_preserves_executor_kwargs(subject):
    from cascade.planning.backend_evidence import ordinary_call

    s = subject
    s.runtime.cfg = s.cfg
    s.runtime._kitchen_camera_renderer = "ovrtx"
    marker = object()
    context = {}
    with ordinary_call(s.runtime, "move_home", context):
        assert (
            s.runtime.arm.raw.stream_profile(
                s.profile, planned_state=s.state, approve=marker
            )
            is True
        )
        s.frames[1].capture["renderer"] = "isaac"
        with pytest.raises(RuntimeError, match="renderer"):
            s.runtime.arm.raw.stream_profile(
                s.profile, planned_state=s.state, approve=marker
            )
    assert len(s.calls) == 1 and s.calls[0][1]["approve"] is marker
    assert context["selected_backends"]["errors"]


def test_builder_rejects_bad_opt_in_before_any_runtime_construction(
    subject, monkeypatch, tmp_path
):
    from cascade.apps import demo

    monkeypatch.setenv("CASCADE_KITCHEN_CAMERA_RENDERER", "ovrtx")
    monkeypatch.setenv("CASCADE_REQUIRE_CUDA", "0")
    monkeypatch.setattr(
        demo,
        "_build_arm",
        lambda *_args: pytest.fail("arm constructed before validation"),
    )
    with pytest.raises(ValueError, match="strict CUDA"):
        demo.build_runtime(subject.cfg, tmp_path / "never-built", lazy_arm=True)
    assert not (tmp_path / "never-built").exists()


def test_constructor_cancellation_cannot_leave_an_unowned_stream_hook(
    subject, monkeypatch
):
    from cascade.planning import backend_evidence as backend

    s = subject
    s.runtime.cfg = s.cfg
    s.runtime._kitchen_camera_renderer = "ovrtx"
    original = s.runtime.arm.raw.stream_profile

    class Cancel(BaseException):
        pass

    actual = backend.BackendEvidence

    class Interrupted(actual):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            raise Cancel()

    monkeypatch.setattr(backend, "BackendEvidence", Interrupted)
    with pytest.raises(Cancel):
        with backend.ordinary_call(s.runtime, "move_home", {}):
            pytest.fail("cancelled constructor reached motion")
    assert s.runtime.arm.raw.stream_profile == original
    assert s.calls == []
