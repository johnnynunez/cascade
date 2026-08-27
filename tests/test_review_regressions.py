"""Regression tests for defects found in the adversarial review pass."""

import numpy as np
import pytest

from cascade.agent.llm import LLMResponse, MockLLM, ToolCall
from cascade.agent.orchestrator import AgentOrchestrator, _as_bool
from cascade.memory.beliefs import BeliefStore
from cascade.perception.detector import MockDetector
from cascade.perception.grounding import Extrinsics, localize_object
from cascade.perception.mock_camera import synthetic_tabletop
from cascade.safety.harness import SafetyHarness, SafetyLimits
from cascade.types import Detection, SafetyViolation


T_CAM2BASE = np.array(
    [
        [0.0, -1.0, 0.0, 0.28],
        [-1.0, 0.0, 0.0, 0.00],
        [0.0, 0.0, -1.0, 0.60],
        [0.0, 0.0, 0.0, 1.00],
    ]
)


def _mask(h, w, x0, y0, x1, y1):
    m = np.zeros((h, w), dtype=bool)
    m[y0:y1, x0:x1] = True
    return m


def test_spatial_hint_left_picks_left_object():
    """'left' must select the object with the LARGER base-frame y."""
    f = synthetic_tabletop()
    h, w = f.rgb.shape[:2]
    # Two cups: one left in the image, one right. With this camera transform
    # base y = -x_cam, so the image-left cup (smaller u) has larger base y.
    d_img_left = Detection("cup", 0.9, np.array([100, 200, 160, 260], np.float32),
                           mask=_mask(h, w, 100, 200, 160, 260))
    d_img_right = Detection("cup", 0.9, np.array([480, 200, 540, 260], np.float32),
                            mask=_mask(h, w, 480, 200, 540, 260))
    det = MockDetector(detections=[d_img_left, d_img_right])
    extr = Extrinsics(mode="eye_to_hand", T=T_CAM2BASE)

    fix_left = localize_object(f, "cup", det, extr, spatial_hint="left")
    fix_right = localize_object(f, "cup", det, extr, spatial_hint="right")
    assert fix_left.position[1] > fix_right.position[1]

    # near/far discriminate along base x = -y_cam, i.e. image rows.
    d_img_top = Detection("cup", 0.9, np.array([300, 60, 360, 120], np.float32),
                          mask=_mask(h, w, 300, 60, 360, 120))
    d_img_bottom = Detection("cup", 0.9, np.array([300, 360, 360, 420], np.float32),
                             mask=_mask(h, w, 300, 360, 360, 420))
    det2 = MockDetector(detections=[d_img_top, d_img_bottom])
    fix_near = localize_object(f, "cup", det2, extr, spatial_hint="near")
    fix_far = localize_object(f, "cup", det2, extr, spatial_hint="far")
    assert fix_near.position[0] < fix_far.position[0]


def test_hand_eye_npz_baseline_format(tmp_path):
    """The baseline saves mode as a 1-element string array; must load."""
    from cascade.config import Cfg

    npz = tmp_path / "hand_eye.npz"
    T = np.eye(4)
    T[:3, 3] = [0.01, 0.02, 0.03]
    np.savez(npz, T_result=T, mode=np.array(["eye_in_hand"]), n_samples=15)
    extr = Extrinsics.from_config(
        Cfg({"hand_eye_npz": str(npz)}), fk_tcp2base=lambda: np.eye(4)
    )
    assert extr.mode == "eye_in_hand"
    assert np.allclose(extr.cam_to_base(), T)


class FakeKin:
    joint_limits = (np.full(6, -3.0), np.full(6, 3.0))

    def fk(self, q):
        T = np.eye(4)
        T[:3, 3] = q[:3]
        return T

    def link_positions(self, q):
        return np.array([[0, 0, 0.2], [q[0], q[1], max(q[2], 0.1)]])


def _limits(**kw):
    base = dict(
        workspace_min=np.array([-1.0, -1.0, -0.5]),
        workspace_max=np.array([1.0, 1.0, 1.0]),
        table_z=0.0,
        table_clearance=0.02,
        max_joint_vel=10.0,
        watchdog_s=0.0,  # instantly stale
    )
    base.update(kw)
    return SafetyLimits(**base)


def test_watchdog_checked_at_motion_start_not_per_waypoint():
    h = SafetyHarness(_limits(), kinematics=FakeKin())
    # Stale perception: starting a motion must fail...
    with pytest.raises(SafetyViolation, match="watchdog"):
        h.begin_motion()
    # ...but once begun (fresh heartbeat), per-waypoint approve()s do not
    # re-check freshness even as time passes.
    h.heartbeat()
    h2 = SafetyHarness(_limits(watchdog_s=1e-3), kinematics=FakeKin())
    h2.heartbeat()
    h2.begin_motion()
    import time

    time.sleep(0.01)  # now stale relative to 1 ms watchdog
    q = np.array([0.3, 0.0, 0.2, 0, 0, 0])
    h2.approve(q, q, 0.02)  # must NOT raise mid-motion
    h2.end_motion()
    with pytest.raises(SafetyViolation, match="watchdog"):
        h2.approve(q, q, 0.02)  # outside a motion the check is live again


def test_escape_upward_from_below_clearance():
    h = SafetyHarness(_limits(watchdog_s=1e9), kinematics=FakeKin())
    below = np.array([0.3, 0.0, 0.005, 0, 0, 0])
    slightly_up = np.array([0.3, 0.0, 0.012, 0, 0, 0])
    deeper = np.array([0.3, 0.0, 0.001, 0, 0, 0])
    # No exemption: moving UP from below the floor is allowed...
    h.approve(below, slightly_up, dt=1e9)
    # ...but descending further is not.
    with pytest.raises(SafetyViolation, match="table"):
        h.approve(below, deeper, dt=1e9)


def test_belief_top_z_used_for_placement():
    store = BeliefStore()
    b = store.update("bowl", [0.3, 0.0, 0.04], conf=0.9, extent=np.array([0.2, 0.06, 0.05]),
                     top_z=0.08)
    assert b.top_z == 0.08
    # A later observation without top_z keeps the previous one.
    store.update("bowl", [0.3, 0.0, 0.04], conf=0.9)
    assert store.find("bowl").top_z == 0.08


def test_as_bool_string_false():
    assert _as_bool(True) is True
    assert _as_bool("true") is True
    assert _as_bool("false") is False
    assert _as_bool("no") is False
    assert _as_bool(0) is False


class _StubRuntime:
    def __init__(self):
        from cascade.memory import EpisodicMemory

        self.memory = EpisodicMemory()
        self.last_frame = None
        self.executed = []

        class _T:
            def finish(self, s):
                pass

        self.trace = _T()

    def execute(self, name, args):
        self.executed.append(name)
        if name == "task_done":
            return {"ok": True, "task_complete": True}
        return {"ok": True}

    def frame_jpeg(self):
        return None


def test_parallel_tool_calls_all_get_results():
    """Extra tool calls are answered (protocol) but not executed."""
    rt = _StubRuntime()
    llm = MockLLM(
        [
            LLMResponse(
                tool_calls=[
                    ToolCall("get_observation", {}, id="a"),
                    ToolCall("move_home", {}, id="b"),
                ]
            ),
            LLMResponse(tool_calls=[ToolCall("task_done", {"success": True, "summary": "x"}, id="c")]),
        ]
    )
    report = AgentOrchestrator(llm, rt, decompose=False, max_steps=5).run_task("t")
    assert report.success
    assert rt.executed == ["get_observation", "task_done"]  # move_home NOT executed
    # Second request must contain tool results for BOTH call ids a and b.
    msgs = llm.requests[1]["messages"]
    tool_ids = [m.get("tool_call_id") for m in msgs if m.get("role") == "tool"]
    assert "a" in tool_ids and "b" in tool_ids


def test_task_done_string_false_reports_failure():
    rt = _StubRuntime()
    llm = MockLLM(
        [LLMResponse(tool_calls=[ToolCall("task_done", {"success": "false", "summary": "s"})])]
    )
    report = AgentOrchestrator(llm, rt, decompose=False, max_steps=3).run_task("t")
    assert report.success is False


def test_image_pruning_keeps_only_latest():
    msgs = [
        {"role": "user", "content": "a", "images": [b"old"]},
        {"role": "user", "content": "b"},
        {"role": "user", "content": "c", "images": [b"new"]},
    ]
    pruned = AgentOrchestrator._prune_images(msgs)
    assert "images" not in pruned[0]
    assert pruned[2]["images"] == [b"new"]
    # Original list untouched.
    assert msgs[0]["images"] == [b"old"]
