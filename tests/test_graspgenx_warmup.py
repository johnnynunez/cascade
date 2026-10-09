"""B15a / B36 signature 4: the GraspGen-X server warms its model before it binds.

Measured on the x86 rig (9 Oct 2026, RTX PRO 6000 Blackwell, torch 2.7.0+cu128):
a fresh server answered its first `infer_object` of a 5 cm cube in 15.53 s and
every later one in 0.09 s. CASCADE's client waits 8 s, so the first pick of a
session silently fell back to the analytic OBB planner -- on the bare Isaac scene
that planner closes across the cube's diagonal (B36). `scripts/graspgenx_server.py`
now runs one synthetic inference between model load and `serve_forever()`.

The upstream server and torch are stubbed: these tests pin the ORDER (warm-up
before the port opens), the opt-out, the health fields, and that the warm-up is
advisory -- a failed warm-up is logged and the server binds cold as before, and
a host without CUDA or without a loadable model refuses before any warm-up
(premises: those two pass on main by design).
"""
from __future__ import annotations

import importlib.util
import sys
import types

import numpy as np
import pytest

from conftest import REPO

SCRIPT = REPO / "scripts" / "graspgenx_server.py"
ARGS = ["--config", "/ckpt", "--assets-dir", "/assets", "--port", "46099"]


def _load():
    spec = importlib.util.spec_from_file_location("graspgenx_server_under_test", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def stubs(monkeypatch):
    """Fake torch + upstream server. `state` steers it: `cuda` (bool),
    `init_error` / `infer_error` (exception to raise), `reply` (infer reply),
    `probe_health` (ask health from inside serve_forever)."""
    calls: list[str] = []
    state: dict = {"cuda": True}

    class _Param:
        device = types.SimpleNamespace(type="cuda")

    class FakeServer:
        def __init__(self, **kwargs):
            calls.append("init")
            if state.get("init_error") is not None:
                raise state["init_error"]
            self.kwargs = kwargs
            self._shared_model = types.SimpleNamespace(parameters=lambda: iter([_Param()]))

        def _dispatch(self, request):
            calls.append(f"dispatch:{request.get('action')}")
            if request.get("action") == "health":
                return {"status": "ok"}
            if state.get("infer_error") is not None:
                raise state["infer_error"]
            if state.get("reply") is not None:
                return state["reply"]
            cloud = np.asarray(request["point_cloud"])
            assert cloud.ndim == 2 and cloud.shape[1] == 3 and cloud.dtype == np.float32
            return {"grasps": np.zeros((3, 4, 4)), "confidences": np.ones(3)}

        def serve_forever(self):
            calls.append("serve")
            if state.get("probe_health"):
                state["health"] = self._dispatch({"action": "health"})

    torch = types.ModuleType("torch")
    torch.cuda = types.SimpleNamespace(is_available=lambda: state["cuda"],
                                       get_device_name=lambda d: "fake GPU",
                                       get_device_capability=lambda d: (12, 0))
    torch.__version__ = "stub"
    torch.version = types.SimpleNamespace(cuda="stub")
    zmq_server = types.ModuleType("graspgenx.serving.zmq_server")
    zmq_server.GraspGenXZMQServer = FakeServer
    for name, mod in (("torch", torch), ("graspgenx", types.ModuleType("graspgenx")),
                      ("graspgenx.serving", types.ModuleType("graspgenx.serving")),
                      ("graspgenx.serving.zmq_server", zmq_server)):
        monkeypatch.setitem(sys.modules, name, mod)
    return calls, state


def _main(monkeypatch, *extra):
    """Run the script's `main()` the way the shell launcher does (argv)."""
    monkeypatch.setattr(sys, "argv", ["graspgenx_server.py", *ARGS, *extra])
    _load().main()


def test_the_server_infers_once_before_it_serves(stubs, monkeypatch, capsys):
    calls, _ = stubs
    _main(monkeypatch)
    assert calls == ["init", "dispatch:infer_object", "serve"], calls
    assert "warm-up inference" in capsys.readouterr().out


def test_no_warmup_keeps_the_old_start_up(stubs, monkeypatch, capsys):
    calls, state = stubs
    state["probe_health"] = True
    _main(monkeypatch, "--no-warmup")
    assert calls == ["init", "serve", "dispatch:health"], calls
    assert state["health"]["warmed_up"] is False and state["health"]["warmup_s"] is None
    assert "warm-up skipped" in capsys.readouterr().out


def test_health_reports_the_warmup(stubs, monkeypatch):
    _, state = stubs
    state["probe_health"] = True
    _main(monkeypatch)
    health = state["health"]
    assert health["learned"] is True
    assert health["warmed_up"] is True and health["warmup_s"] >= 0.0
    assert health["warmup_error"] is None


def test_a_warmup_without_grasps_is_logged_and_the_server_still_serves_cold(
        stubs, monkeypatch, capsys):
    """Fail open to the old start-up: the same reply would have reached the first
    client before this change; refusing to bind would turn it into a dead port."""
    calls, state = stubs
    state["reply"] = {"error": "CUDA error: no kernel image is available"}
    state["probe_health"] = True
    _main(monkeypatch)
    assert calls == ["init", "dispatch:infer_object", "serve", "dispatch:health"], calls
    err = capsys.readouterr().err
    assert "WARNING: warm-up inference failed" in err and "COLD" in err
    assert "no kernel image" in err
    health = state["health"]
    assert health["warmed_up"] is False and health["warmup_s"] is None
    assert "no kernel image" in health["warmup_error"]


def test_a_raising_warmup_is_logged_and_the_server_still_serves_cold(stubs, monkeypatch, capsys):
    """Upstream `_dispatch` raises (ValueError on a bad cloud, torch errors); the
    serve loop turns that into an error reply per request -- so must the warm-up."""
    calls, state = stubs
    state["infer_error"] = ValueError("point cloud has no points after outlier removal")
    state["probe_health"] = True
    _main(monkeypatch)
    assert calls[-2:] == ["serve", "dispatch:health"], calls
    err = capsys.readouterr().err
    assert "WARNING: warm-up inference failed (ValueError: point cloud has no points" in err
    assert state["health"]["warmed_up"] is False
    assert state["health"]["warmup_error"].startswith("ValueError: ")


def test_premise_no_cuda_refuses_before_any_warmup(stubs, monkeypatch):
    """Premise (passes on main): no GPU -> the old explicit refusal, no model
    load, no warm-up, never a hang. The CUDA-less hosts run the protocol stub."""
    calls, state = stubs
    state["cuda"] = False
    with pytest.raises(RuntimeError, match="requires CUDA"):
        _main(monkeypatch)
    assert calls == [], calls


def test_premise_a_model_that_fails_to_load_never_reaches_the_warmup(stubs, monkeypatch):
    """Premise (passes on main): missing checkpoints fail in the constructor,
    before the warm-up -- the warm-up's fail-open never hides them."""
    calls, state = stubs
    state["init_error"] = FileNotFoundError("release/gen/epoch_736.pth")
    with pytest.raises(FileNotFoundError, match="epoch_736"):
        _main(monkeypatch)
    assert calls == ["init"], calls


def test_the_warmup_request_is_a_resting_cube_with_the_rebot_sweep():
    mod = _load()
    req = mod.warmup_request()
    assert req["action"] == "infer_object" and req["num_grasps"] > 0
    cloud = req["point_cloud"]
    assert cloud.dtype == np.float32 and cloud.shape[1] == 3 and len(cloud) > 1000
    assert cloud[:, 2].min() > 0.0  # the bottom face is hidden by the table
    np.testing.assert_allclose(np.ptp(cloud, axis=0), [0.05, 0.05, 0.05], atol=2e-3)
    sweep = req["sweep_volume_params"]
    assert set(sweep) == {"extents_open", "offset_open", "extents_mid", "offset_mid",
                          "gripper_type", "fingertip_depth"}


def test_the_warmup_sweep_matches_what_the_client_sends_for_the_default_rig():
    """The warm-up volume is the one demo.yaml makes the client send (tip offset
    folded in), so the first real request reuses the warmed sampler."""
    from cascade.config import load_demo_config

    g = load_demo_config(camera="mock", arm="mock", llm="mock").grasp.graspgenx
    tip = float(g.get("tip_offset_m"))
    sweep = g.sweep.as_dict()
    for key in ("offset_open", "offset_mid"):
        sweep[key][2] += tip
    sweep["fingertip_depth"] += tip
    warm = _load().WARMUP_SWEEP
    assert set(warm) == set(sweep)
    for key, value in sweep.items():
        np.testing.assert_allclose(np.asarray(warm[key], float), np.asarray(value, float), atol=1e-12,
                                   err_msg=key)
