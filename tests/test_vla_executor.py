"""B49: an opt-in VLA executor behind `grasp_object` (ROADMAP mid term,
"VLA policy backend").

`grasp.executor: vla` serves `grasp_object(label)` -- and the composites that
grasp by label through it -- from a language-conditioned policy speaking the
openpi / LingBot-VLA-v2 websocket protocol (msgpack-numpy, `infer(obs) ->
action chunk`). The rules this file pins:

  - opt-in: `grasp.executor` defaults to `analytic` on every shipped profile
    and the analytic runtime exposes no VLA surface at all (golden);
  - the client needs the new `vla` extra; without it the route refuses
    explicitly, before any motion;
  - every chunk becomes joint targets that the SafetyHarness admits as a
    whole (a doomed chunk never starts) and then approves sample by sample
    through `SafeArm.move_joints` -- the harness stays the sole authority;
  - the stop latch and halts are honoured between chunks, a late chunk and an
    expired episode end the episode with no further motion;
  - the policy's own reply is never evidence: only the jaws and the
    unchanged three-state verifier (`effects.py`, kind `holding`) judge;
  - the capability matrix grows a `vla_policy` cell ONLY when the executor is
    attached, and withholds the label-grasp tools when no server answered.

Runtimes here dial only this item's port block (45900-45999); the protocol
stub binds an ephemeral loopback port like every other test server.
"""

from __future__ import annotations

import http.client
import sys
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest
from conftest import REPO, loopback_host, needs_pin

from cascade.config import load_demo_config

sys.path.insert(0, str(REPO / "scripts"))

#: inside this item's block, never bound by anything: a server that is down
_CLOSED_PORT = 45950
#: top-down tool orientation for the reBot mock (tool_axis_order down_open:
#: columns approach (0,0,-1), open (1,0,0), third (0,-1,0)) -- the orientation
#: the analytic OBB planner selects for the mock red cube
_TOPDOWN = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, -1.0], [-1.0, 0.0, 0.0]])


@pytest.fixture(autouse=True)
def _private_ports(monkeypatch):
    monkeypatch.setenv("CASCADE_GRASPGENX_PORT", "45901")
    monkeypatch.setenv("CASCADE_OCCUPANCY_PORT", "45902")
    monkeypatch.setenv("CASCADE_BRIDGE_PORT", "45903")
    monkeypatch.setenv("CASCADE_OCCUPANCY", "0")
    for var in ("CASCADE_GRASP_EXECUTOR", "CASCADE_VLA_PORT", "CASCADE_GRASP_BACKEND",
                "CASCADE_OBSERVED_FINGER_GATE"):
        monkeypatch.delenv(var, raising=False)


def _wire():
    """The `vla` extra. CI's test matrix installs it; the minimal install
    does not, and there these tests skip explicitly (the refusal itself is
    pinned without the extra by `test_missing_extra_is_an_explicit_refusal`)."""
    pytest.importorskip("msgpack", reason="the `vla` extra (msgpack) is not installed")
    pytest.importorskip("websockets.sync.client",
                        reason="the `vla` extra (websockets) is not installed")


# ── premises / goldens: true on main and after ─────────────────────────────


def _arm_profiles():
    from cascade.config import _load_profile_raw

    arms = REPO / "configs" / "arms"
    return [p.stem for p in sorted(arms.glob("*.yaml"))
            if _load_profile_raw("arms", p.stem, arms.parent).get("template") is not True]


def test_premise_every_shipped_profile_grasps_with_the_analytic_executor():
    """Golden: the VLA route is opt-in. No shipped profile selects it."""
    for name in _arm_profiles():
        grasp = load_demo_config(arm=name).grasp
        assert grasp.get("executor", "analytic") == "analytic", name
    assert load_demo_config().grasp.get("executor", "analytic") == "analytic"


def test_premise_grasp_object_is_judged_by_the_independent_holding_check():
    """The verdict on a grasp comes from `effects.py` (kind `holding`: the
    object's own pose must rise, the jaws are necessary, never sufficient).
    The VLA route stays inside `grasp_object`, so this verifier judges it
    unchanged -- the policy's reply is not an input to it."""
    from cascade.agent.effects import POSTCONDITIONS, REFUTED, PostconditionChecker

    assert POSTCONDITIONS["grasp_object"] == "holding"
    checker = PostconditionChecker(gripper_frac=lambda: 0.0)
    pc = checker.verify("grasp_object", {"label": "red cube"},
                        {"held": "red cube", "success": True, "done": True})
    assert pc.status == REFUTED and pc.channel == "gripper"


@needs_pin
def test_golden_analytic_runtime_exposes_no_vla_surface(tmp_path):
    """`grasp.executor: analytic` (the default) is the pipeline as it was:
    no executor object, no `grasp_executor` backend, no `vla_policy` cell,
    and grasp_object plans with the analytic planner."""
    from cascade.apps.capabilities import capability_matrix, format_matrix, withheld_tools
    from cascade.apps.demo import build_runtime, shutdown_runtime

    cfg = load_demo_config(camera="mock", arm="mock", llm="mock")
    cfg._data["grasp"]["backend"] = "obb"
    rt, arm = build_runtime(cfg, tmp_path / "run")
    try:
        assert getattr(rt, "vla_executor", None) is None
        assert set(rt.backends()) == {"grasp_planner", "occupancy", "occupancy_live"}
        matrix = capability_matrix(rt)
        assert "vla_policy" not in matrix
        assert "vla_policy" not in format_matrix(matrix)
        assert "grasp_object" not in withheld_tools(matrix)
        _wait_for(rt, "red cube")
        arm.object_stop_frac = 0.5
        with rt.watcher.paused():
            res = rt.execute("grasp_object", {"label": "red cube"})
        assert res["ok"] is True and res["held"] == "red cube", res
        assert "executor" not in res and "vla" not in res
        assert rt.grasp_planner_used == "obb"
    finally:
        shutdown_runtime(rt, arm)


# ── wire: codec, client, stub ──────────────────────────────────────────────


def test_codec_round_trips_the_openpi_msgpack_numpy_format():
    pytest.importorskip("msgpack", reason="the `vla` extra (msgpack) is not installed")
    import msgpack

    from cascade.grasping.vla_client import pack, unpack

    obs = {"observation/image": np.arange(12, dtype=np.uint8).reshape(2, 2, 3),
           "observation/state": np.array([0.1, -0.2, 1.0], dtype=np.float32),
           "prompt": "pick up the red cube", "step": np.int64(3),
           "nested": {"chunk": np.zeros((2, 7), dtype=np.float64)}}
    back = unpack(pack(obs))
    np.testing.assert_array_equal(back["observation/image"], obs["observation/image"])
    assert back["observation/image"].dtype == np.uint8
    assert back["observation/state"].dtype == np.float32
    assert back["prompt"] == "pick up the red cube"
    assert back["step"] == 3 and isinstance(back["step"], np.integer)
    assert back["nested"]["chunk"].shape == (2, 7)
    # a frame built exactly as openpi_client.msgpack_numpy.pack_array does
    raw = np.array([[1.5, 2.5]], dtype="<f4")
    frame = msgpack.packb({"actions": {b"__ndarray__": True, b"data": raw.tobytes(),
                                       b"dtype": raw.dtype.str, b"shape": raw.shape}})
    np.testing.assert_array_equal(unpack(frame)["actions"], raw)
    # object arrays would need pickle: refused, as openpi refuses them
    with pytest.raises(ValueError):
        pack({"bad": np.array([object()], dtype=object)})


def test_missing_extra_is_an_explicit_refusal(monkeypatch):
    """No silent fallback to the analytic pipeline: a `vla` route without its
    extra refuses, naming the extra to install."""
    from cascade.grasping import vla_client
    from cascade.types import SkillError

    for mod in ("websockets", "websockets.sync", "websockets.sync.client"):
        monkeypatch.setitem(sys.modules, mod, None)
    with pytest.raises(vla_client.VLAUnavailable) as err:
        vla_client.require_deps()
    assert issubclass(vla_client.VLAUnavailable, SkillError)
    assert "vla" in str(err.value) and "websockets" in str(err.value)
    assert "msgpack" not in str(err.value).split("(")[0]  # only what is missing is named
    status = vla_client.probe("127.0.0.1", _CLOSED_PORT, timeout_s=0.2)
    assert status["answered"] is False and "vla" in status["detail"]
    monkeypatch.setitem(sys.modules, "msgpack", None)
    with pytest.raises(vla_client.VLAUnavailable, match=r"needs websockets, msgpack \(install"):
        vla_client.require_deps()


def _stub(policy, metadata=None):
    from serve_vla_stub import StubPolicyServer

    return StubPolicyServer(policy, host="127.0.0.1", port=0,
                            metadata=metadata if metadata is not None else {"policy": "scripted-stub"}).start()


def test_stub_speaks_the_openpi_websocket_protocol():
    """Metadata frame on connect, msgpack obs -> msgpack actions with
    `server_timing`, a text frame for a server-side error, GET /healthz."""
    _wire()
    from serve_vla_stub import ScriptedPolicy

    from cascade.grasping.vla_client import PolicyClient, VLAServerError

    chunk = np.linspace(0.0, 1.0, 14, dtype=np.float32).reshape(2, 7)
    policy = ScriptedPolicy([chunk])
    server = _stub(policy, metadata={"policy": "scripted-stub", "action_horizon": 2})
    try:
        client = PolicyClient(loopback_host(), server.port, connect_timeout_s=2.0)
        meta = client.connect()
        assert meta == {"policy": "scripted-stub", "action_horizon": 2}
        reply = client.infer({"prompt": "pick up the red cube",
                              "observation/state": np.zeros(7, np.float32)}, timeout_s=2.0)
        np.testing.assert_array_equal(reply["actions"], chunk)
        assert "infer_ms" in reply["server_timing"]
        assert policy.requests[0]["prompt"] == "pick up the red cube"
        client.close()

        conn = http.client.HTTPConnection(loopback_host(), server.port, timeout=2.0)
        conn.request("GET", "/healthz")
        resp = conn.getresponse()
        assert resp.status == 200 and resp.read().strip() == b"OK"
        conn.close()

        def broken(obs):
            raise RuntimeError("policy exploded")

        policy.respond = broken
        client = PolicyClient(loopback_host(), server.port, connect_timeout_s=2.0)
        client.connect()
        with pytest.raises(VLAServerError, match="policy exploded"):
            client.infer({"prompt": "x"}, timeout_s=2.0)
        client.close()
    finally:
        server.stop()


def test_probe_reports_a_closed_port_as_not_answering():
    _wire()
    from cascade.grasping.vla_client import probe

    status = probe("127.0.0.1", _CLOSED_PORT, timeout_s=0.5)
    assert status["answered"] is False
    assert str(_CLOSED_PORT) in status["detail"]


class _RawServer:
    """A websocket server whose handler is the test's: protocol violations
    a real policy server could commit."""

    def __init__(self, handler):
        from websockets.sync.server import serve

        self._server = serve(handler, "127.0.0.1", 0, compression=None, max_size=None)
        self.port = int(self._server.socket.getsockname()[1])
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()

    def stop(self):
        self._server.shutdown()
        self._thread.join(timeout=5.0)


@pytest.mark.parametrize("violation, error, match", [
    ("silent", "VLAUnavailable", "not a policy server"),
    ("list_metadata", "VLAUnavailable", "non-dict metadata"),
    ("garbage_reply", "VLAServerError", "undecodable"),
    ("hang_up", "VLAServerError", "connection lost"),
    ("list_reply", "VLAServerError", "not a dict"),
])
def test_protocol_violations_are_named_failures(violation, error, match):
    _wire()
    from cascade.grasping import vla_client

    def handler(ws):
        if violation == "silent":
            time.sleep(1.0)
            return
        ws.send(vla_client.pack([1, 2]) if violation == "list_metadata" else vla_client.pack({}))
        try:
            ws.recv()
        except Exception:  # noqa: BLE001 -- the client may already be gone
            return
        if violation == "garbage_reply":
            ws.send(b"\xc1")  # a msgpack byte that is never valid
        elif violation == "list_reply":
            ws.send(vla_client.pack([0.0]))
        elif violation == "hang_up":
            ws.close()
        time.sleep(0.2)

    server = _RawServer(handler)
    client = vla_client.PolicyClient(loopback_host(), server.port, connect_timeout_s=0.3)
    try:
        with pytest.raises(getattr(vla_client, error), match=match):
            client.connect()
            client.infer({"prompt": "x"}, timeout_s=2.0)
    finally:
        client.close()
        server.stop()


def test_a_closed_session_cannot_infer():
    from cascade.grasping.vla_client import PolicyClient, VLAServerError

    with pytest.raises(VLAServerError, match="closed"):
        PolicyClient("127.0.0.1", _CLOSED_PORT).infer({}, timeout_s=0.1)


def test_a_late_reply_closes_the_session_and_is_never_read():
    """After a chunk deadline the session is closed: the late chunk cannot be
    read as the answer to a later request."""
    _wire()
    from cascade.grasping import vla_client

    def handler(ws):
        ws.send(vla_client.pack({}))
        try:
            ws.recv()
            time.sleep(0.4)
            ws.send(vla_client.pack({"actions": np.zeros((1, 7), np.float32)}))
            ws.recv()
        except Exception:  # noqa: BLE001 -- the client hung up, as it must
            return

    server = _RawServer(handler)
    client = vla_client.PolicyClient(loopback_host(), server.port, connect_timeout_s=1.0)
    try:
        client.connect()
        with pytest.raises(vla_client.VLATimeout):
            client.infer({"prompt": "x"}, timeout_s=0.1)
        time.sleep(0.5)  # the late chunk has been sent by now
        with pytest.raises(vla_client.VLAServerError, match="closed"):
            client.infer({"prompt": "x"}, timeout_s=1.0)
    finally:
        client.close()
        server.stop()


# ── configuration ──────────────────────────────────────────────────────────


def test_executor_config_is_validated_before_anything_runs():
    from cascade.grasping.vla_executor import VLAConfig, executor_name

    gcfg = load_demo_config().grasp
    assert executor_name(gcfg) == "analytic"
    assert executor_name({}) == "analytic"
    assert executor_name({"executor": "vla"}) == "vla"
    with pytest.raises(ValueError, match="grasp.executor"):
        executor_name({"executor": "policy"})
    vcfg = VLAConfig.from_cfg(gcfg.get("vla"))
    assert vcfg.action_space == "joint" and vcfg.gripper == "open_frac"
    assert vcfg.chunk_timeout_s > 0 and vcfg.episode_timeout_s > vcfg.chunk_timeout_s
    assert "{label}" in vcfg.prompt
    base = gcfg.get("vla").as_dict()
    for key, bad in (("action_space", "eef_delta"), ("gripper", "percent"),
                     ("chunk_timeout_s", 0), ("episode_timeout_s", -1.0),
                     ("max_chunks", 0), ("chunks_after_close", -1), ("port", 70000),
                     ("max_chunk_len", 0), ("waypoint_duration_s", float("nan"))):
        with pytest.raises(ValueError, match=key):
            VLAConfig.from_cfg({**base, key: bad})
    for over, match in (({"bogus": 1}, "unknown"), ({"prompt": "pick {obj}"}, "prompt"),
                        ({"reset": [1]}, "reset"), ({"extra_obs": "x"}, "extra_obs"),
                        ({"action_key": " "}, "action_key"), ({"image_size": 4}, "image_size"),
                        ({"gripper_column": -1}, "gripper_column"),
                        ({"max_chunks": 2.5}, "integer")):
        with pytest.raises(ValueError, match=match):
            VLAConfig.from_cfg({**base, **over})
    assert VLAConfig.from_cfg(None) == VLAConfig()
    assert VLAConfig.from_cfg(base).replace(port=45912).port == 45912
    with pytest.raises(ValueError, match="port"):
        VLAConfig.from_cfg(base).replace(port=0)


def test_route_preconditions_and_task_budget_are_explicit():
    from cascade.grasping.vla_executor import VLAExecutor, route_refusal
    from cascade.types import SkillError

    def rt(planner=None, occupancy=None):
        return SimpleNamespace(arm=SimpleNamespace(motion_planner=planner,
                                                   harness=SimpleNamespace(occupancy=occupancy)))

    assert route_refusal(rt()) is None
    assert route_refusal(rt(occupancy=SimpleNamespace(tracks_payload=False))) is None
    assert "payload" in route_refusal(rt(occupancy=SimpleNamespace(tracks_payload=True)))
    assert "motion planner" in route_refusal(rt(planner=object()))
    ex = VLAExecutor(_vcfg())
    assert ex._remaining(1e12, SimpleNamespace(_task_deadline=None)) > 0
    assert ex._remaining(1e12, SimpleNamespace(_task_deadline=time.monotonic() + 3.0)) <= 3.0
    with pytest.raises(SkillError, match="task budget"):
        ex._remaining(1e12, SimpleNamespace(_task_deadline=time.monotonic() - 1.0))
    with pytest.raises(SkillError, match="episode deadline"):
        ex._remaining(time.monotonic() - 1.0, SimpleNamespace())


def test_env_selects_the_executor_and_its_port_on_every_view(monkeypatch, tmp_path):
    monkeypatch.setenv("CASCADE_GRASP_EXECUTOR", "vla")
    monkeypatch.setenv("CASCADE_VLA_PORT", "45911")
    cfg = load_demo_config(arms=["so101_left", "so101_right"])
    for view in [cfg._data, *(a["resolved"] for a in cfg.arms)]:
        assert view["grasp"]["executor"] == "vla"
        assert view["grasp"]["vla"]["port"] == 45911
    # a config without a `grasp.vla` section does not grow one from the env
    import shutil

    import yaml

    shutil.copytree(REPO / "configs", tmp_path / "configs")
    demo = tmp_path / "configs" / "demo.yaml"
    data = yaml.safe_load(demo.read_text())
    del data["grasp"]["vla"]
    demo.write_text(yaml.safe_dump(data))
    bare = load_demo_config(config_dir=tmp_path / "configs")
    assert bare._data["grasp"]["executor"] == "vla" and "vla" not in bare._data["grasp"]
    monkeypatch.setenv("CASCADE_GRASP_EXECUTOR", "policy")
    with pytest.raises(ValueError, match="CASCADE_GRASP_EXECUTOR"):
        load_demo_config()
    monkeypatch.setenv("CASCADE_GRASP_EXECUTOR", "analytic")
    monkeypatch.setenv("CASCADE_VLA_PORT", "+80")
    with pytest.raises(ValueError, match="CASCADE_VLA_PORT"):
        load_demo_config()


# ── chunk -> joint targets ─────────────────────────────────────────────────


def _vcfg(**over):
    from cascade.grasping.vla_executor import VLAConfig

    return VLAConfig.from_cfg({**load_demo_config().grasp.get("vla").as_dict(), **over})


def test_chunk_rows_become_joint_targets_with_the_gripper_convention():
    from cascade.grasping.vla_executor import chunk_targets

    q = np.linspace(-0.3, 0.3, 6)
    chunk = np.vstack([np.r_[q, 1.0], np.r_[q + 0.01, 0.0], np.r_[q, 1.03], np.r_[q, -0.02]])
    out = chunk_targets({"actions": chunk}, _vcfg(), n_joints=6, kin=None, seed_q=q)
    assert len(out) == 4
    np.testing.assert_allclose(out[1][0], q + 0.01)
    assert [g for _, g in out] == [1.0, 0.0, 1.0, 0.0]  # small overshoot clipped
    # close_frac flips the column; an explicit column index is honoured
    flipped = chunk_targets({"actions": chunk[:2]}, _vcfg(gripper="close_frac"),
                            n_joints=6, kin=None, seed_q=q)
    assert [g for _, g in flipped] == [0.0, 1.0]
    wide = np.hstack([chunk[:2, :6], np.zeros((2, 1)), chunk[:2, 6:]])
    moved = chunk_targets({"act": wide}, _vcfg(gripper_column=7, action_key="act"),
                          n_joints=6, kin=None, seed_q=q)
    assert [g for _, g in moved] == [1.0, 0.0]


@pytest.mark.parametrize("reply, why", [
    ({"actions": np.full((2, 7), np.nan)}, "finite"),
    ({"actions": np.zeros(7)}, "2-D"),
    ({"actions": np.zeros((2, 5))}, "columns"),
    ({"actions": np.c_[np.zeros((2, 6)), [[0.5], [1.5]]]}, "gripper"),
    ({"actions": np.zeros((65, 7))}, "rows"),
    ({"actions": np.zeros((0, 7))}, "rows"),
    ({"action": np.zeros((2, 7))}, "actions"),
    ({"actions": None}, "actions"),
])
def test_malformed_chunks_are_refused(reply, why):
    from cascade.grasping.vla_executor import chunk_targets
    from cascade.types import SkillError

    with pytest.raises(SkillError, match=why):
        chunk_targets(reply, _vcfg(max_chunk_len=64), n_joints=6, kin=None, seed_q=np.zeros(6))


@needs_pin
def test_tcp_chunks_are_solved_by_ik_and_unreachable_rows_refused():
    from conftest import JOINT_SIGNS, URDF

    from cascade.control.kinematics import Kinematics
    from cascade.grasping.vla_executor import chunk_targets
    from cascade.types import SkillError

    kin = Kinematics(str(URDF), "gripper_end", joint_signs=JOINT_SIGNS)
    seed = np.asarray(load_demo_config(arm="mock").arm.home_q, float)
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = _TOPDOWN, [0.29, 0.0, 0.08]
    rpy = _rpy(_TOPDOWN)
    row = np.r_[T[:3, 3], rpy, 1.0]
    (q, grip), = chunk_targets({"actions": row[None]}, _vcfg(action_space="tcp"),
                               n_joints=6, kin=kin, seed_q=seed)
    np.testing.assert_allclose(kin.fk(q)[:3, 3], T[:3, 3], atol=2e-3)
    assert grip == 1.0
    far = np.r_[[3.0, 0.0, 0.08], rpy, 1.0]
    with pytest.raises(SkillError, match="IK"):
        chunk_targets({"actions": far[None]}, _vcfg(action_space="tcp"), n_joints=6, kin=kin, seed_q=seed)
    with pytest.raises(SkillError, match="kinematics"):
        chunk_targets({"actions": row[None]}, _vcfg(action_space="tcp"), n_joints=6, kin=None, seed_q=seed)
    # each row's IK is seeded from the previous row's solution (continuity)
    seeds = []

    def ik(T, s):
        seeds.append(np.asarray(s, float).copy())
        return SimpleNamespace(success=True, q=np.asarray(s, float) + 0.1)

    two = np.array([row, row])
    (q1, _), (q2, _) = chunk_targets({"actions": two}, _vcfg(action_space="tcp"), n_joints=6,
                                     kin=SimpleNamespace(ik=ik), seed_q=seed)
    np.testing.assert_allclose(seeds[0], seed)
    np.testing.assert_allclose(seeds[1], q1)
    np.testing.assert_allclose(q2, seed + 0.2)
    # an IK solution on another branch is not the pose sequence the policy described
    flipped = SimpleNamespace(ik=lambda T, s: SimpleNamespace(success=True, q=np.asarray(s) + 3.5))
    with pytest.raises(SkillError, match="branch"):
        chunk_targets({"actions": row[None]}, _vcfg(action_space="tcp"), n_joints=6, kin=flipped,
                      seed_q=seed)


def _rpy(R):
    """URDF rpy of a rotation (inverse of cascade.types.pose_to_transform)."""
    pitch = float(np.arcsin(-np.clip(R[2, 0], -1.0, 1.0)))
    roll = float(np.arctan2(R[2, 1], R[2, 2]))
    yaw = float(np.arctan2(R[1, 0], R[0, 0]))
    return np.array([roll, pitch, yaw])


# ── end to end on the mock stack ───────────────────────────────────────────


def _wait_for(rt, label):
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline and rt.beliefs.find(label) is None:
        time.sleep(0.05)
    assert rt.beliefs.find(label) is not None


def _topdown_q(rt, xyz, seed):
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = _TOPDOWN, xyz
    ik = rt.kin.ik(T, np.asarray(seed, float))
    assert ik.success, xyz
    return np.asarray(ik.q, float)


def _rows(q0, q1, n, grip):
    return [np.r_[q0 + (q1 - q0) * s, grip] for s in np.linspace(1.0 / n, 1.0, n)]


def _grasp_script(rt, *, close=True, lift=True):
    """What a policy fine-tuned on this arm would emit for the mock red cube:
    approach above it, descend, close, lift. Joint-space rows + open frac."""
    fix = rt.beliefs.find("red cube").position
    home = np.asarray(rt.cfg.arm.home_q, float)
    q_pre = _topdown_q(rt, [fix[0], fix[1], 0.081], home)
    q_grasp = _topdown_q(rt, [fix[0], fix[1], 0.041], q_pre)
    approach = np.array(_rows(home, q_pre, 5, 1.0))
    descend = np.array(_rows(q_pre, q_grasp, 4, 1.0)
                       + ([np.r_[q_grasp, 0.0]] * 2 if close else [np.r_[q_grasp, 1.0]] * 2))
    raise_ = np.array(_rows(q_grasp, q_pre, 4, 0.0 if close else 1.0))
    script = [approach, descend] + ([raise_] if lift else [])
    return script, SimpleNamespace(home=home, q_pre=q_pre, q_grasp=q_grasp,
                                   fix=np.asarray(fix, float))


@pytest.fixture
def vla_rig(tmp_path, monkeypatch):
    _wire()
    pytest.importorskip("pinocchio")
    from serve_vla_stub import ScriptedPolicy

    from cascade.apps.demo import build_runtime, shutdown_runtime

    policy = ScriptedPolicy([])
    server = _stub(policy)
    cfg = load_demo_config(camera="mock", arm="mock", llm="mock")
    cfg._data["grasp"]["backend"] = "obb"
    cfg._data["grasp"]["executor"] = "vla"
    cfg._data["grasp"]["vla"].update(host=loopback_host(), port=server.port,
                                     chunk_timeout_s=5.0, episode_timeout_s=60.0)
    rt, arm = build_runtime(cfg, tmp_path / "run")
    approved = []
    real_approve = rt.arm.harness.approve

    def spy(q_prev, q_next, dt, **kw):
        real_approve(q_prev, q_next, dt, **kw)
        approved.append(np.asarray(q_next, float).copy())

    monkeypatch.setattr(rt.arm.harness, "approve", spy)
    grips = []
    real_grip = arm.set_gripper

    def grip_spy(pos, effort=1.0):
        grips.append((round(float(pos), 6), round(float(effort), 6)))
        real_grip(pos, effort)

    monkeypatch.setattr(arm, "set_gripper", grip_spy)
    try:
        _wait_for(rt, "red cube")
        yield SimpleNamespace(rt=rt, arm=arm, policy=policy, server=server, approved=approved,
                              grips=grips)
    finally:
        rt.arm.harness.reset_estop()
        shutdown_runtime(rt, arm)
        server.stop()


def _grasp(rig, **args):
    with rig.rt.watcher.paused():
        return rig.rt.execute("grasp_object", {"label": "red cube", **args})


def _episode_start(rig):
    """Index of the first arm command after the policy's first chunk was
    requested: everything from here on is the VLA episode's motion."""
    return rig.policy.command_marks[0] if rig.policy.command_marks else len(rig.arm.commands)


def test_vla_grasp_holds_the_object_and_the_harness_approved_every_waypoint(vla_rig):
    rig = vla_rig
    script, poses = _grasp_script(rig.rt)
    rig.policy.chunks = script
    rig.policy.track(rig.arm)
    rig.arm.object_stop_frac = 0.5
    res = _grasp(rig)
    assert res["ok"] is True and res["held"] == "red cube", res
    assert res["executor"] == "vla" and res["grip_verified"] is True
    assert res["vla"]["chunks"] == 3 and res["vla"]["closed"] is True
    assert res["vla"]["stop"] == "closed" and len(res["vla"]["infer_ms"]) == 3
    assert rig.rt.held_object == "red cube"
    assert rig.rt._held_width_m == pytest.approx(0.5 * rig.rt._max_width)
    # aiming offset from the jaws' MEASURED pose at the close command
    tcp_close = rig.rt.kin.fk(poses.q_grasp)[:3, 3]
    np.testing.assert_allclose(rig.rt._held_offset + tcp_close, poses.fix, atol=5e-3)
    # the observation the policy saw: prompt with the label, RGB image, state
    first = rig.policy.requests[0]
    assert first["prompt"] == "pick up the red cube"
    import cv2

    expected = cv2.resize(np.ascontiguousarray(rig.rt.last_frame.rgb[:, :, ::-1]), (224, 224),
                          interpolation=cv2.INTER_AREA)
    assert first["observation/image"].dtype == np.uint8
    np.testing.assert_array_equal(first["observation/image"], expected)  # RGB, not the BGR frame
    np.testing.assert_allclose(first["observation/state"], np.r_[poses.home, 1.0], atol=1e-6)
    # jaws: opened before the episode, then ONE close at the material effort
    # (rigid 0.70) -- repeated rows of the same opening are not re-sent
    assert rig.grips == [(rig.rt._grip_open, 0.8), (rig.rt._grip_closed, 0.7)]
    # every streamed command of the episode went through harness.approve()
    episode = rig.arm.commands[_episode_start(rig):]
    assert len(episode) >= sum(len(c) for c in script)
    for cmd in episode:
        assert any(np.allclose(cmd, a) for a in rig.approved), cmd
    # the arm ended where the policy lifted it
    np.testing.assert_allclose(rig.arm.get_state().q, poses.q_pre, atol=1e-6)
    # the verdict is the independent verifier's, never the policy's
    assert res["postcondition"]["kind"] == "holding"


def test_harness_refuses_a_chunk_before_any_of_it_moves(vla_rig):
    rig = vla_rig
    script, poses = _grasp_script(rig.rt)
    fix = rig.rt.beliefs.find("red cube").position
    q_low = _topdown_q(rig.rt, [fix[0], fix[1], 0.005], poses.q_grasp)  # TCP below table clearance
    doomed = np.array(_rows(poses.q_pre, poses.q_grasp, 3, 1.0) + [np.r_[q_low, 1.0]])
    rig.policy.chunks = [script[0], doomed, script[2]]
    rig.policy.track(rig.arm)
    res = _grasp(rig)
    assert res["ok"] is False and "SafetyViolation" in res["error"], res
    assert "table clearance" in res["error"]
    # no command after chunk 1: the doomed chunk's safe prefix never streamed
    assert len(rig.arm.commands) == rig.policy.command_marks[1]
    assert rig.rt.held_object is None and getattr(rig.rt, "_held_provisional", None) is None


@pytest.mark.parametrize("stop", ["estop", "halt"])
def test_stop_latch_and_halt_are_honoured_between_chunks(vla_rig, stop):
    rig = vla_rig
    script, _ = _grasp_script(rig.rt)
    rig.policy.chunks = script
    rig.policy.track(rig.arm)
    harness = rig.rt.arm.harness

    vetted = []
    real_vet = harness.vet_pose

    def vet(q, *a, **k):
        vetted.append(np.asarray(q, float).copy())
        return real_vet(q, *a, **k)

    harness.vet_pose = vet
    at_stop = []

    def operator(obs, call):
        if call == 2:  # the stop arrives while the policy computes chunk 2
            at_stop.append(len(vetted))
            harness.estop("operator stop") if stop == "estop" else harness.halt("superseded")

    rig.policy.on_call = operator
    try:
        res = _grasp(rig)
    finally:
        del harness.vet_pose
    assert res["ok"] is False, res
    if stop == "estop":
        assert "e-stop latched: the VLA episode stopped between chunks" in res["error"], res
    else:
        assert "halt" in res["error"], res
    assert len(rig.arm.commands) == rig.policy.command_marks[1]
    assert len(rig.policy.requests) == 2
    # chunk 2 was computed across the stop: discarded unread, never admitted
    assert at_stop == [len(vetted)] and len(vetted) == len(rig.policy.chunks[0])


def test_a_halt_between_waypoints_stops_the_rest_of_the_chunk(vla_rig):
    """A halt during a chunk is not cleared by the next waypoint's motion:
    every waypoint and gripper command carries the skill's halt generation."""
    rig = vla_rig
    script, _ = _grasp_script(rig.rt)
    rig.policy.chunks = script
    ex, harness = rig.rt.vla_executor, rig.rt.arm.harness
    real = ex._remaining
    marks = []

    def remaining(deadline, runtime):
        marks.append(len(rig.arm.commands))
        if len(marks) == 4:  # 1 = before the request, 2..4 = before waypoints 1..3
            harness.halt("superseded mid-chunk")
        return real(deadline, runtime)

    ex._remaining = remaining
    res = _grasp(rig)
    assert res["ok"] is False and "halt" in res["error"], res
    assert len(rig.arm.commands) == marks[3]  # waypoint 3 never streamed
    assert len(rig.policy.requests) == 1


def test_a_waypoint_that_does_not_settle_ends_the_episode(vla_rig, monkeypatch):
    rig = vla_rig
    script, _ = _grasp_script(rig.rt)
    rig.policy.chunks = script
    real = rig.rt.arm.move_joints
    monkeypatch.setattr(rig.rt.arm, "move_joints",
                        lambda *a, **k: False if rig.policy.requests else real(*a, **k))
    res = _grasp(rig)
    assert res["ok"] is False and "did not settle at VLA chunk 1 waypoint 1" in res["error"], res
    assert len(rig.policy.requests) == 1
    assert [g for g in rig.grips if g[0] == rig.rt._grip_closed] == []


def test_a_late_chunk_aborts_the_episode_without_further_motion(vla_rig):
    rig = vla_rig
    script, _ = _grasp_script(rig.rt)
    rig.policy.chunks = script
    rig.policy.track(rig.arm)
    rig.rt.vla_executor.config = rig.rt.vla_executor.config.replace(chunk_timeout_s=0.2)
    rig.policy.delays = {2: 1.5}  # chunk 2 takes 7x the chunk deadline
    t0 = time.monotonic()
    res = _grasp(rig)
    assert res["ok"] is False and "chunk deadline" in res["error"], res
    assert time.monotonic() - t0 < 30.0
    assert len(rig.arm.commands) == rig.policy.command_marks[1]


def test_the_episode_deadline_cuts_a_chunk_short(vla_rig):
    """The executor's clock is read once to set the deadline, once before
    each request and once before each waypoint; it jumps past the 60 s
    budget at the THIRD waypoint of chunk 1: the arm stops at waypoint 2."""
    rig = vla_rig
    script, poses = _grasp_script(rig.rt)
    rig.policy.chunks = script
    rig.policy.track(rig.arm)
    reads = iter([0.0, 0.0, 0.0, 0.0] + [100.0] * 50)
    rig.rt.vla_executor._clock = lambda: next(reads)
    res = _grasp(rig)
    assert res["ok"] is False and "episode deadline" in res["error"], res
    assert len(rig.policy.requests) == 1
    np.testing.assert_allclose(rig.arm.get_state().q, script[0][1][:6], atol=1e-6)
    assert not np.allclose(rig.arm.get_state().q, script[0][-1][:6], atol=1e-3)


def test_the_episode_deadline_bounds_the_wait_for_a_chunk(vla_rig):
    rig = vla_rig
    script, _ = _grasp_script(rig.rt)
    rig.policy.chunks = script
    ex = rig.rt.vla_executor
    ex.config = ex.config.replace(episode_timeout_s=0.4, chunk_timeout_s=5.0)
    rig.policy.delays = {1: 2.0}  # the first chunk alone outlasts the episode
    rig.policy.track(rig.arm)
    t0 = time.monotonic()
    res = _grasp(rig)
    assert res["ok"] is False and "episode deadline" in res["error"], res
    assert "waiting for chunk 1" in res["error"]
    assert time.monotonic() - t0 < 5.0  # never the full chunk deadline
    assert len(rig.arm.commands) == rig.policy.command_marks[0]


def test_a_policy_that_reopens_the_jaws_holds_nothing(vla_rig):
    rig = vla_rig
    script, poses = _grasp_script(rig.rt)
    script[2] = np.array(_rows(poses.q_grasp, poses.q_pre, 4, 1.0))  # lifts with the jaws open
    rig.policy.chunks = script
    rig.arm.object_stop_frac = 0.5
    res = _grasp(rig)
    assert res["ok"] is False and "re-opened" in res["error"], res
    assert rig.rt.held_object is None and getattr(rig.rt, "_held_provisional", None) is None


def test_reset_message_and_static_observation_keys_reach_the_server(vla_rig):
    """LingBot needs `{reset: true, robo_name}` once per episode and a robot
    config name in every observation; both are configuration, not code."""
    rig = vla_rig
    script, _ = _grasp_script(rig.rt)
    rig.policy.chunks = script
    ex = rig.rt.vla_executor
    ex.config = ex.config.replace(reset={"reset": True, "robo_name": "rebot_b601"},
                                  extra_obs={"robo_name": "rebot_b601"}, image_size=None)
    rig.arm.object_stop_frac = 0.5
    res = _grasp(rig)
    assert res["ok"] is True, res
    assert [r["robo_name"] for r in rig.policy.resets] == ["rebot_b601"]
    assert all(r["robo_name"] == "rebot_b601" for r in rig.policy.requests)
    native = rig.rt.last_frame.rgb.shape
    assert rig.policy.requests[0]["observation/image"].shape == native  # image_size: null


def test_executor_built_on_first_use_and_status_follows_each_connect(vla_rig):
    from cascade.apps.capabilities import capability_matrix, withheld_tools

    rig = vla_rig
    rig.rt.vla_executor = None  # a runtime that skipped the startup probe
    script, _ = _grasp_script(rig.rt)
    rig.policy.chunks = script
    rig.arm.object_stop_frac = 0.5
    assert _grasp(rig)["ok"] is True
    assert rig.rt.vla_executor.status["answered"] is True
    rig.rt.execute("open_gripper", {})
    rig.rt.held_object = None
    rig.server.stop()  # the policy server goes away mid-session
    before = len(rig.arm.commands)
    res = _grasp(rig)
    assert res["ok"] is False and "no VLA policy server answered" in res["error"], res
    assert len(rig.arm.commands) == before
    matrix = capability_matrix(rig.rt)
    assert matrix["vla_policy"]["available"] is False
    assert "grasp_object" in withheld_tools(matrix)


def test_missing_jaw_feedback_ends_the_episode_before_a_chunk_is_requested(vla_rig, monkeypatch):
    rig = vla_rig
    script, _ = _grasp_script(rig.rt)
    rig.policy.chunks = script
    monkeypatch.setattr(rig.rt, "_gripper_width_frac", lambda: None)
    res = _grasp(rig)
    assert res["ok"] is False and "gripper feedback unavailable" in res["error"], res
    assert rig.policy.requests == []


def test_the_policy_self_report_is_never_evidence(vla_rig):
    """A reply that says success/done is ignored: the jaws closed on air, so
    the grasp fails, and the verifier refutes it. `done` on chunk 1 does not
    end the episode early either."""
    rig = vla_rig
    script, _ = _grasp_script(rig.rt)
    rig.policy.chunks = script
    rig.policy.extra = {"success": True, "done": True, "is_success": np.bool_(True),
                        "grasped": "red cube"}
    rig.policy.track(rig.arm)
    rig.arm.object_stop_frac = None  # nothing between the fingers
    res = _grasp(rig)
    assert res["ok"] is False and "air grasp" in res["error"], res
    assert len(rig.policy.requests) == 3
    assert rig.rt.held_object is None and getattr(rig.rt, "_held_provisional", None) is None
    assert res["postcondition"]["status"] == "refuted"
    assert "success" not in res and "grasped" not in res.get("vla", {})


def test_an_episode_that_never_closes_fails_without_inventing_a_hold(vla_rig):
    rig = vla_rig
    script, _ = _grasp_script(rig.rt, close=False)
    rig.policy.chunks = script
    rig.policy.track(rig.arm)
    rig.rt.vla_executor.config = rig.rt.vla_executor.config.replace(max_chunks=3)
    rig.arm.object_stop_frac = 0.5
    res = _grasp(rig)
    assert res["ok"] is False and "never closed" in res["error"], res
    assert "max_chunks, 3 chunks" in res["error"]
    assert rig.rt.held_object is None and getattr(rig.rt, "_held_provisional", None) is None


def test_an_abort_after_the_close_leaves_a_provisional_hold_to_reconcile(vla_rig):
    rig = vla_rig
    script, _ = _grasp_script(rig.rt)
    rig.policy.chunks = script
    rig.policy.track(rig.arm)
    rig.rt.vla_executor.config = rig.rt.vla_executor.config.replace(chunk_timeout_s=0.2)
    rig.policy.delays = {3: 1.5}  # the lift chunk is late; the jaws already closed
    rig.arm.object_stop_frac = 0.5
    res = _grasp(rig)
    assert res["ok"] is False and "chunk deadline" in res["error"], res
    assert rig.rt.held_object is None
    assert rig.rt._held_provisional[0] == "red cube"
    rig.rt._reconcile_held()  # jaws stalled on the cube: promoted, never dropped
    assert rig.rt.held_object == "red cube"


def test_a_down_server_refuses_without_motion_and_is_withheld(tmp_path):
    _wire()
    pytest.importorskip("pinocchio")
    from cascade.apps.capabilities import capability_matrix, format_matrix, withheld_tools
    from cascade.apps.demo import build_runtime, shutdown_runtime

    cfg = load_demo_config(camera="mock", arm="mock", llm="mock")
    cfg._data["grasp"]["backend"] = "obb"
    cfg._data["grasp"]["executor"] = "vla"
    cfg._data["grasp"]["vla"].update(host="127.0.0.1", port=_CLOSED_PORT, connect_timeout_s=0.3)
    rt, arm = build_runtime(cfg, tmp_path / "run")
    try:
        matrix = capability_matrix(rt)
        assert matrix["vla_policy"]["available"] is False
        assert str(_CLOSED_PORT) in matrix["vla_policy"]["why"]
        withheld = withheld_tools(matrix)
        assert {"grasp_object", "pick_and_place", "sort_by_color"} <= set(withheld)
        assert "vla_policy" in withheld["grasp_object"]
        assert "vla_policy=no" in format_matrix(matrix)
        assert "vla" in rt.backends()["grasp_executor"]
        _wait_for(rt, "red cube")
        before = len(arm.commands)
        with rt.watcher.paused():
            res = rt.execute("grasp_object", {"label": "red cube"})
        assert res["ok"] is False and "VLA" in res["error"], res
        assert len(arm.commands) == before  # refused before the arm moved
    finally:
        shutdown_runtime(rt, arm)


def test_build_runtime_probes_the_vla_server(vla_rig):
    from cascade.apps.capabilities import capability_matrix, withheld_tools

    rig = vla_rig
    matrix = capability_matrix(rig.rt)
    assert matrix["vla_policy"]["available"] is True
    assert "scripted-stub" in matrix["vla_policy"]["detail"]
    assert "grasp_object" not in withheld_tools(matrix)
    assert str(rig.server.port) in rig.rt.backends()["grasp_executor"]


def test_vla_route_refuses_rigs_whose_gates_it_does_not_implement(vla_rig, monkeypatch):
    """The observed-finger gate and a native motion planner own geometry
    contracts the VLA route does not implement: refused, never bypassed."""
    rig = vla_rig
    script, _ = _grasp_script(rig.rt)
    rig.policy.chunks = script
    before = len(rig.arm.commands)
    monkeypatch.setenv("CASCADE_OBSERVED_FINGER_GATE", "1")
    res = _grasp(rig)
    assert res["ok"] is False and "observed-finger" in res["error"], res
    monkeypatch.delenv("CASCADE_OBSERVED_FINGER_GATE")
    monkeypatch.setattr(rig.rt.arm, "motion_planner", object())
    res = _grasp(rig)
    assert res["ok"] is False and "motion planner" in res["error"], res
    assert len(rig.arm.commands) == before and rig.policy.requests == []


def test_pixel_addressed_grasps_stay_on_the_analytic_pipeline(vla_rig):
    rig = vla_rig
    rig.arm.object_stop_frac = 0.5
    frame, fix = rig.rt._localize("red cube")
    with rig.rt.watcher.paused():
        res = rig.rt.skill_grasp_object("red cube", _fix=fix, _frame=frame)
    assert res["held"] == "red cube" and "executor" not in res
    assert rig.policy.requests == []


# ── capability matrix (unit) ───────────────────────────────────────────────


def _bare(executor=None):
    return SimpleNamespace(backends=lambda: {}, arm_rig=None, memory=None,
                           **({} if executor is None else {"vla_executor": executor}))


def test_the_vla_cell_exists_only_with_an_attached_executor():
    from cascade.apps.capabilities import (CAP_VLA_POLICY, TOOL_REQUIREMENTS, capability_matrix,
                                           format_matrix, withheld_tools)

    for tool in ("grasp_object", "pick_and_place", "sort_by_color"):
        assert CAP_VLA_POLICY in TOOL_REQUIREMENTS[tool]
    plain = capability_matrix(_bare())
    base = withheld_tools(plain)  # what a bare runtime withholds anyway (memory, verifier, ...)
    assert CAP_VLA_POLICY not in plain and not {"grasp_object", "pick_and_place"} & set(base)
    assert CAP_VLA_POLICY not in capability_matrix(None)

    def ex(answered, detail="127.0.0.1:45950"):
        return SimpleNamespace(status={"answered": answered, "detail": detail},
                               config=SimpleNamespace(host="127.0.0.1", port=45950))

    unknown = capability_matrix(_bare(ex(None)))
    assert unknown[CAP_VLA_POLICY]["available"] is None and withheld_tools(unknown) == base
    up = capability_matrix(_bare(ex(True, "policy=pi05")))
    assert up[CAP_VLA_POLICY]["available"] is True and withheld_tools(up) == base
    down = capability_matrix(_bare(ex(False, "connection refused")))
    withheld = withheld_tools(down)
    assert set(withheld) - set(base) == {"grasp_object", "pick_and_place", "sort_by_color"}
    assert "connection refused" in withheld["grasp_object"]
    assert "analytic" in down[CAP_VLA_POLICY]["why"]  # names the refused fallback
    assert "vla_policy=no" in format_matrix(down) and "vla_policy" not in format_matrix(plain)


# ── packaging and docs ─────────────────────────────────────────────────────


def test_the_vla_extra_is_declared_and_ci_exercises_both_sides():
    import tomllib

    project = tomllib.loads((REPO / "pyproject.toml").read_text())["project"]
    extra = " ".join(project["optional-dependencies"]["vla"])
    assert "websockets" in extra and "msgpack" in extra
    assert not any("websockets" in d for d in project["dependencies"])
    ci = (REPO / ".github" / "workflows" / "ci.yml").read_text()
    tests_sync = next(line for line in ci.splitlines() if "uv sync" in line and "--extra grasping" in line)
    assert "--extra vla" in tests_sync
    minimal = ci.split("minimal-install:")[1]
    assert "websockets" in minimal.split("for mod in")[1].splitlines()[0]


def test_docs_record_the_landed_vla_executor_and_what_stays_open():
    roadmap = (REPO / "docs" / "ROADMAP.md").read_text()
    mid = roadmap.split("## Mid term")[1].split("## Long term")[0]
    assert "~~**VLA policy backend**" in mid and "landed 2026-10-09" in mid
    arch = (REPO / "docs" / "ARCHITECTURE.md").read_text()
    assert "grasp.executor: vla" in arch
    assert "We deliberately did **not** build a VLA-policy-in-the-loop executor" not in arch
    demo = (REPO / "configs" / "demo.yaml").read_text()
    assert "executor: analytic" in demo
    assert (REPO / "docs" / "VLA_EXECUTOR.md").exists()
