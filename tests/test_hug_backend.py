"""HUG (Human Universal Grasping) backend: wire protocol, pinch mapping, ranking.

HUG (arXiv:2606.17054, github.com/KevinyWu/hug @ 8d1c52d, MIT) predicts a
RIGHT HUMAN HAND (MANO: 21 landmarks + wrist pose) for one RGB-D frame and a
query pixel on the object. It has no server, no score and no parallel-jaw
retargeting. CASCADE adds all three and labels them as its own:

  - `scripts/serve_hug.py` wraps HUG's documented inference path behind the
    same REQ/REP msgpack shape as GraspGen-X (batching N samples of one query
    is our extension);
  - `grasping/hug_backend.py` maps each hand to a parallel-jaw PINCH (thumb
    tip vs index tip -- an assumption, not a HUG output) and ranks the
    candidates by CASCADE geometry (HUG emits no confidence).

No weights, no GPU and no MANO model are needed here: these tests run the
client against `serve_hug.py --stub`, an analytic protocol double that says
so in its health reply, and the real engine's wrapper against a fake `hug`
package. They test that CASCADE speaks the protocol and converts frames
right, never grasp quality.
"""

from __future__ import annotations

import hashlib
import queue
import re
import socket
import subprocess
import sys
import threading
import time

import numpy as np
import pytest
from conftest import REPO

from cascade.config import Cfg
from cascade.types import Detection, Frame, ObjectFix


def _has_wire() -> bool:
    try:
        import msgpack  # noqa: F401
        import msgpack_numpy  # noqa: F401
        import zmq  # noqa: F401

        return True
    except ImportError:
        return False


needs_wire = pytest.mark.skipif(
    not _has_wire(), reason="needs the grasping extra: uv pip install -e '.[grasping]'")

SERVE = REPO / "scripts" / "serve_hug.py"


def _serve_module():
    sys.path.insert(0, str(REPO / "scripts"))
    try:
        import serve_hug
    finally:
        sys.path.remove(str(REPO / "scripts"))
    return serve_hug


def _free_port() -> int:
    """A port nobody listens on (bound, then released): a dead server."""
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


@pytest.fixture(autouse=True)
def _no_ambient_hug_endpoint(monkeypatch):
    """`CASCADE_HUG_PORT`/`HOST` override the configured endpoint (same
    contract as `CASCADE_GRASPGENX_PORT`). An operator shell exporting them
    for a live run must not re-point these tests at a real server."""
    monkeypatch.delenv("CASCADE_HUG_PORT", raising=False)
    monkeypatch.delenv("CASCADE_HUG_HOST", raising=False)


@pytest.fixture(scope="module")
def hug_stub():
    """`serve_hug.py --stub` on an EPHEMERAL port (it prints the endpoint)."""
    proc = subprocess.Popen(
        [sys.executable, str(SERVE), "--stub", "--port", "0"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    lines: queue.Queue = queue.Queue()
    threading.Thread(target=lambda: lines.put(proc.stdout.readline()), daemon=True).start()
    try:
        line = lines.get(timeout=20.0)
    except queue.Empty:
        proc.kill()
        pytest.fail("serve_hug.py --stub printed no endpoint within 20 s")
    match = re.search(r"tcp://127\.0\.0\.1:(\d+)", line or "")
    if match is None:
        proc.kill()
        err = proc.stderr.read()[:600] if proc.stderr else ""
        if not _has_wire():
            pytest.skip("grasping extra not installed")
        pytest.fail(f"no endpoint line from the stub: {line!r} {err}")
    yield int(match.group(1))
    proc.kill()
    proc.wait(timeout=5)


# ── a synthetic top-down RGB-D scene (the mock stack's camera convention) ─────

#: camera -> base for a camera looking straight down from 0.5 m (x_cam -> -y,
#: y_cam -> -x, z_cam -> -z), the same rotation as the mock camera profile
_R_DOWN = np.array([[0.0, -1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, -1.0]])


def _scene(centre_px=(160, 120), size=(320, 240), cube=0.04, cam_h=0.5):
    """Frame (BGR!), ObjectFix and T_base_cam of a red cube on a table.

    Only the top face is observed, as from a real top-down camera."""
    w, h = size
    K = np.array([[300.0, 0.0, w / 2.0], [0.0, 300.0, h / 2.0], [0.0, 0.0, 1.0]])
    T = np.eye(4)
    T[:3, :3] = _R_DOWN
    T[:3, 3] = [0.3, 0.0, cam_h]
    zt = cam_h - cube                       # top face depth
    u0, v0 = centre_px
    xc, yc = (u0 - K[0, 2]) * zt / K[0, 0], (v0 - K[1, 2]) * zt / K[1, 1]
    us, vs = np.meshgrid(np.arange(w), np.arange(h))
    X, Y = (us - K[0, 2]) * zt / K[0, 0], (vs - K[1, 2]) * zt / K[1, 1]
    on = (np.abs(X - xc) <= cube / 2) & (np.abs(Y - yc) <= cube / 2)
    depth = np.full((h, w), cam_h, np.float32)
    depth[on] = zt
    bgr = np.full((h, w, 3), 90, np.uint8)
    bgr[on] = (40, 40, 200)                 # red, in OpenCV's BGR order
    frame = Frame(rgb=bgr, depth_m=depth, K=K, depth_source="sensor")
    pts_cam = np.stack([X[on], Y[on], np.full(int(on.sum()), zt)], axis=1)
    pts = pts_cam @ T[:3, :3].T + T[:3, 3]
    ys, xs = np.nonzero(on)
    det = Detection(label="red cube", conf=0.9, mask=on,
                    bbox=np.array([xs.min(), ys.min(), xs.max(), ys.max()], np.float32))
    centre = pts.mean(axis=0) - np.array([0.0, 0.0, cube / 2])
    fix = ObjectFix(label="red cube", position=centre, points=pts, detection=det,
                    extent=np.array([cube, cube, cube]), axes=np.eye(3))
    return frame, fix, T


def _planner(port, **over):
    from cascade.grasping.hug_backend import HugPlanner

    hug = {"host": "127.0.0.1", "port": port, "timeout_ms": 8000, "num_samples": 8,
           "required": False}
    hug.update(over)
    return HugPlanner(Cfg({"hug": hug}))


def _hand(centre, close_axis, approach, aperture=0.04, palm_back=0.07):
    """21 MANO landmarks (HUG/manotorch order) of a pinch at `centre`:
    thumb tip (4) and index tip (8) straddle it along `close_axis`, the palm
    sits `palm_back` behind it against `approach`."""
    c = np.asarray(centre, float)
    x = np.asarray(close_axis, float) / np.linalg.norm(close_axis)
    a = np.asarray(approach, float) / np.linalg.norm(approach)
    y = np.cross(a, x)
    L = np.zeros((21, 3))
    L[0] = c - (palm_back + 0.03) * a                  # wrist
    thumb, index = c - aperture / 2 * x, c + aperture / 2 * x
    for i, f in zip((1, 2, 3), (0.25, 0.5, 0.75)):
        L[i] = L[0] + f * (thumb - L[0])
    L[4] = thumb
    for k, (mcp, off) in enumerate(((5, 0.03), (9, 0.01), (13, -0.01), (17, -0.03))):
        L[mcp] = c - palm_back * a + off * y
        tip = index if mcp == 5 else c + 0.01 * x - 0.02 * a + off * y
        for j in (1, 2, 3):
            L[mcp + j] = L[mcp] + j / 3 * (tip - L[mcp])
    return L


# ── pinch mapping: OUR assumption, pinned so it cannot drift silently ─────────


def test_landmark_indices_follow_hugs_manotorch_order():
    """HUG emits manotorch's SNAP order (manolayer.py@a2a70c5): 0 wrist,
    1-4 thumb (4 = tip, vertex 745), 5-8 index (8 = tip, vertex 317), then
    middle/ring/little; MCPs at 5/9/13/17. A shifted index silently pinches
    with the wrong finger."""
    from cascade.grasping import hug_backend as hb

    assert (hb.WRIST, hb.THUMB_TIP, hb.INDEX_TIP, hb.MIDDLE_TIP) == (0, 4, 8, 12)
    assert hb.PALM_LANDMARKS == (0, 5, 9, 13, 17)
    assert hb.N_LANDMARKS == 21


def test_pinch_centre_jaw_axis_and_aperture_come_from_thumb_and_index_tips():
    from cascade.grasping.hug_backend import pinch_from_landmarks

    L = _hand([0.3, 0.0, 0.03], [0, 1, 0], [0, 0, -1], aperture=0.05)
    p = pinch_from_landmarks(L)
    np.testing.assert_allclose(p["centre"], (L[4] + L[8]) / 2, atol=1e-12)
    np.testing.assert_allclose(abs(p["close_axis"] @ np.array([0, 1, 0])), 1.0, atol=1e-12)
    assert p["aperture"] == pytest.approx(0.05)


def test_pinch_approach_runs_from_the_palm_to_the_pinch_orthogonal_to_the_jaw():
    """The jaw comes in where the human palm came from (our retargeting
    assumption): palm centre -> pinch, minus its component along the jaw."""
    from cascade.grasping.hug_backend import pinch_from_landmarks

    tilted = np.array([0.4, 0.0, -1.0]) / np.linalg.norm([0.4, 0.0, -1.0])
    L = _hand([0.3, 0.0, 0.03], [0, 1, 0], tilted)
    p = pinch_from_landmarks(L)
    assert abs(p["approach"] @ p["close_axis"]) < 1e-9
    np.testing.assert_allclose(p["approach"], tilted, atol=1e-9)
    palm = L[[0, 5, 9, 13, 17]].mean(axis=0)
    assert (p["centre"] - palm) @ p["approach"] > 0


def test_vertical_pinch_mode_keeps_the_contacts_and_forces_a_top_down_approach():
    from cascade.grasping.hug_backend import pinch_from_landmarks

    tilted = np.array([0.4, 0.0, -1.0])
    jaw = np.array([0.0, 1.0, 0.3])
    L = _hand([0.3, 0.0, 0.03], jaw, tilted)
    p = pinch_from_landmarks(L, approach="vertical")
    np.testing.assert_allclose(p["approach"], [0, 0, -1], atol=1e-12)
    assert abs(p["close_axis"][2]) < 1e-12          # jaw in the horizontal plane
    np.testing.assert_allclose(p["centre"], (L[4] + L[8]) / 2, atol=1e-12)


@pytest.mark.parametrize("break_it", ["coincident_tips", "nan", "shape", "palm_on_jaw_axis"])
def test_degenerate_hands_are_refused_not_guessed(break_it):
    from cascade.grasping.hug_backend import pinch_from_landmarks

    L = _hand([0.3, 0.0, 0.03], [0, 1, 0], [0, 0, -1])
    if break_it == "coincident_tips":
        L[8] = L[4]
    elif break_it == "nan":
        L[8, 1] = np.nan
    elif break_it == "shape":
        L = L[:20]
    else:   # palm centre exactly on the jaw line: no approach direction left
        L[[0, 5, 9, 13, 17]] = (L[4] + L[8]) / 2 + 0.05 * (L[8] - L[4])
    with pytest.raises(ValueError):
        pinch_from_landmarks(L)


@pytest.mark.parametrize("order", ["down_open", "open_down", "third_open_down"])
def test_tool_rotation_follows_the_arms_axis_convention(order):
    """HUG grasps must use the same TCP convention as the OBB planner on the
    same arm; a column permutation rotates every grasp 90 degrees."""
    from cascade.grasping.hug_backend import tool_rotation
    from cascade.grasping.obb_grasp import _yaw_rotation

    yaw = 0.7
    open_axis = np.array([np.cos(yaw), np.sin(yaw), 0.0])
    R = tool_rotation(np.array([0.0, 0.0, -1.0]), open_axis, order)
    np.testing.assert_allclose(R, _yaw_rotation(yaw, axis_order=order), atol=1e-12)
    assert np.linalg.det(R) == pytest.approx(1.0)


# ── request encoding ───────────────────────────────────────────────────────────


def test_depth_goes_out_as_uint16_millimetres_with_invalid_as_zero():
    """HUG reads 16-bit millimetre depth and treats 0 (and >= 65535) as
    invalid; NaN/inf/negative metres must not wrap into plausible values."""
    from cascade.grasping.hug_backend import depth_to_mm

    d = np.array([[0.5506, 0.0, np.nan], [np.inf, -0.2, 70.0]], np.float32)
    mm = depth_to_mm(d)
    assert mm.dtype == np.uint16
    assert mm.tolist() == [[551, 0, 0], [0, 0, 0]]


def test_query_pixel_is_on_the_object_with_valid_depth():
    """A ring-shaped mask (a mug seen from above) has its centroid in the
    hole: the query must still land ON the object, where depth is valid."""
    from cascade.grasping.hug_backend import query_pixel

    h, w = 60, 80
    vv, uu = np.mgrid[0:h, 0:w]
    r = np.hypot(uu - 40, vv - 30)
    ring = (r >= 8) & (r <= 14)
    depth = np.where(ring, 0.45, 0.6).astype(np.float32)
    det = Detection(label="mug", conf=0.9, bbox=np.array([26, 16, 54, 44], np.float32), mask=ring)
    u, v = query_pixel(det, depth)
    assert ring[v, u] and depth[v, u] > 0
    # HUG samples training queries from a 3x3-ERODED mask: the query's whole
    # neighbourhood is on the object, not the ring's inner edge nearest the hole
    assert ring[v - 1:v + 2, u - 1:u + 2].all()
    depth[ring] = 0.0
    from cascade.grasping.hug_backend import HugError
    with pytest.raises(HugError, match="depth"):
        query_pixel(det, depth)


# ── client against the stub server ──────────────────────────────────────────


@needs_wire
def test_stub_health_is_labelled_as_a_stub(hug_stub):
    planner = _planner(hug_stub)
    status = planner.probe()
    assert status["stub"] is True and status.get("learned") is False
    assert "stub" in planner.describe()


@needs_wire
def test_required_hug_rejects_the_analytic_stub(hug_stub):
    from cascade.grasping.hug_backend import HugError

    with pytest.raises(HugError, match="stub rejected"):
        _planner(hug_stub, required=True).probe()


@needs_wire
def test_hug_pinches_land_on_the_object_in_the_base_frame(hug_stub):
    """End to end on the wire: camera-frame hands -> base-frame pinches. A
    wrong extrinsic or a dropped axis lands the jaws decimetres away."""
    frame, fix, T = _scene()
    planner = _planner(hug_stub)
    planner.probe()
    grasps = planner.plan(frame, fix, T_base_cam=T, max_width_m=0.09, width_pad_m=0.01)
    assert grasps
    top = fix.points[:, 2].max()
    for g in grasps:
        assert np.linalg.norm(g.position[:2] - fix.points[:, :2].mean(axis=0)) < 0.015
        assert top - 0.03 < g.position[2] < top + 1e-6       # between the jaws, not above
        assert g.approach[2] < -0.9                          # from above
        np.testing.assert_allclose(g.rotation[:, 0], g.approach, atol=1e-9)   # down_open
        assert 0.04 < g.width_m <= 0.04 * np.sqrt(2) + 0.01 + 1e-9           # object span + pad
    assert planner.last_counts["returned"] == 8


@needs_wire
def test_the_server_receives_rgb_and_millimetres_not_bgr_and_metres(hug_stub):
    """Frame.rgb is BGR (OpenCV). HUG's DINOv2 expects RGB: a red cube sent
    as BGR arrives blue and nothing fails loudly. The stub echoes what it
    received at the query pixel."""
    frame, fix, T = _scene()
    planner = _planner(hug_stub)
    planner.plan(frame, fix, T_base_cam=T)
    echo = planner.last_response["query"]
    assert echo["rgb"] == [200, 40, 40]
    assert echo["depth_mm"] == 460
    assert echo["crop"] == [40, 0, 240]


@needs_wire
def test_num_samples_is_one_batch_of_independent_hands(hug_stub):
    """Our batching extension: N samples of ONE query come back in one reply."""
    frame, fix, T = _scene()
    planner = _planner(hug_stub, num_samples=5)
    planner.plan(frame, fix, T_base_cam=T)
    assert planner.last_counts["returned"] == 5
    assert planner.last_response["landmarks_3d"].shape == (5, 21, 3)


@needs_wire
def test_an_object_outside_hugs_center_crop_is_refused_visibly(hug_stub):
    """HUG's documented preprocessing keeps only the centred square of the
    image. An object outside it is invisible to the model: refuse with the
    reason instead of conditioning on a clamped pixel."""
    from cascade.grasping.hug_backend import HugError

    frame, fix, T = _scene(centre_px=(20, 120))
    with pytest.raises(HugError, match="crop"):
        _planner(hug_stub).plan(frame, fix, T_base_cam=T)
    # ...and the explicit `query` crop (our extension) keeps it in view
    assert _planner(hug_stub, crop="query").plan(frame, fix, T_base_cam=T)


@needs_wire
def test_a_frame_whose_extrinsics_do_not_reproduce_the_fix_is_refused(hug_stub):
    """A secondary camera's frame with the primary camera's transform would
    put every pinch at the wrong place. The query pixel must back-project
    onto the localized object or nothing is planned."""
    from cascade.grasping.hug_backend import HugError

    frame, fix, T = _scene()
    wrong = T.copy()
    wrong[:3, 3] += [0.10, 0.0, 0.0]
    with pytest.raises(HugError, match="extrinsic"):
        _planner(hug_stub).plan(frame, fix, T_base_cam=wrong)


@needs_wire
def test_plane_depth_is_not_measured_geometry_for_hug(hug_stub):
    from cascade.grasping.hug_backend import HugError

    frame, fix, T = _scene()
    frame.depth_source = "plane"
    with pytest.raises(HugError, match="measured depth"):
        _planner(hug_stub).plan(frame, fix, T_base_cam=T)


@needs_wire
def test_a_dead_hug_server_raises_a_typed_error_promptly():
    from cascade.grasping.hug_backend import HugError

    frame, fix, T = _scene()
    planner = _planner(_free_port(), timeout_ms=500)
    t0 = time.monotonic()
    with pytest.raises(HugError, match="timed out"):
        planner.plan(frame, fix, T_base_cam=T)
    assert time.monotonic() - t0 < 10.0


@needs_wire
def test_bounded_requests_share_the_search_deadline_and_honour_cancellation(hug_stub):
    """Inside grasp_object's bounded search (same contract as GraspGen-X):
    the request never outlives the search deadline, the search's
    cancellation check interrupts it, and a cancelled REQ socket is
    replaced so the next request still answers."""
    from cascade.grasping.hug_backend import HugError

    frame, fix, T = _scene()
    planner = _planner(hug_stub)
    with pytest.raises(HugError, match="timed out"):
        planner.plan(frame, fix, T_base_cam=T, deadline=time.monotonic() - 1.0,
                     check=lambda: None)

    class Cancelled(Exception):
        pass

    calls = []

    def cancel_inside_the_request():
        calls.append(1)
        if len(calls) > 1:       # the plan's own first check passes
            raise Cancelled()

    with pytest.raises(Cancelled):
        planner.plan(frame, fix, T_base_cam=T, deadline=time.monotonic() + 10.0,
                     check=cancel_inside_the_request)
    grasps = planner.plan(frame, fix, T_base_cam=T, deadline=time.monotonic() + 10.0,
                          check=lambda: None)
    assert grasps and all(g.label == fix.label for g in grasps)
    planner.probe(deadline=time.monotonic() + 5.0, check=lambda: None)
    with pytest.raises(HugError, match="timed out"):
        planner.probe(deadline=time.monotonic() - 1.0, check=lambda: None)


# ── ranking: HUG has no score; the order is CASCADE geometry, then memory ────


def _respond(planner, hands_cam):
    """Make the client receive these camera-frame hands."""
    hands_cam = np.asarray(hands_cam, np.float32)
    planner._client.request = lambda payload, **kw: {
        "landmarks_3d": hands_cam, "T_camera_wrist": np.repeat(np.eye(4, dtype=np.float32)[None], len(hands_cam), 0),
        "query": {}, "stub": False}


def _to_cam(T, L_base):
    return (np.asarray(L_base) - T[:3, 3]) @ T[:3, :3]


@needs_wire
def test_candidates_are_ranked_by_cascade_geometry_and_impossible_pinches_dropped():
    """Five hands: centred, 2 cm beside the object, 6 cm beside it, above the
    object's top (closes on air) and from below the table. Only the first two
    survive, centred first, and the counts say why the others went."""
    frame, fix, T = _scene()
    top = fix.points[:, 2].max()
    c = np.r_[fix.points[:, :2].mean(axis=0), top - 0.015]
    down, jaw = np.array([0, 0, -1.0]), np.array([0, 1.0, 0])
    hands = [_hand(c + [0.02, 0, 0], jaw, down),        # beside, still over the cube edge
             _hand(c, jaw, down),                       # centred
             _hand(c + [0.06, 0, 0], jaw, down),        # 6 cm off the silhouette
             _hand(c + [0, 0, 0.035], jaw, down),       # above the top face: air
             _hand(c, jaw, [0, 0, 1.0])]                # approach from below
    planner = _planner(1)
    _respond(planner, [_to_cam(T, L) for L in hands])
    grasps = planner.plan(frame, fix, T_base_cam=T)
    assert len(grasps) == 2
    np.testing.assert_allclose(grasps[0].position, c, atol=1e-6)
    assert grasps[0].quality > grasps[1].quality
    assert planner.last_counts == {"returned": 5, "degenerate": 0, "approach": 1,
                                   "off_object": 1, "in_front_of_surface": 1, "kept": 2}


@pytest.mark.parametrize("bad", [
    {"landmarks_3d": np.zeros((2, 20, 3))},
    {"landmarks_3d": np.full((1, 21, 3), np.nan)},
    {"landmarks_3d": np.zeros((0, 21, 3))},
    {"error_free_but_missing": True},
    "not a mapping",
])
@needs_wire
def test_malformed_hug_batches_are_terminal(bad):
    from cascade.grasping.hug_backend import HugError

    frame, fix, T = _scene()
    planner = _planner(1)
    planner._client.request = lambda payload, **kw: bad
    with pytest.raises(HugError, match="malformed"):
        planner.plan(frame, fix, T_base_cam=T)


@needs_wire
def test_every_hand_rejected_is_an_error_with_the_counts():
    from cascade.grasping.hug_backend import HugError

    frame, fix, T = _scene()
    c = np.r_[fix.points[:, :2].mean(axis=0), fix.points[:, 2].max() - 0.01]
    planner = _planner(1)
    _respond(planner, [_to_cam(T, _hand(c, [0, 1, 0], [0, 0, 1.0]))])
    with pytest.raises(HugError, match="approach"):
        planner.plan(frame, fix, T_base_cam=T)


@needs_wire
def test_a_filtered_out_batch_inside_a_bounded_search_is_an_empty_batch():
    """HUG sampling is stochastic: inside the bounded search an all-rejected
    batch is GraspGen-X's `NoEligibleGrasps` (the search asks for a fresh
    batch), still a HugError everywhere else."""
    from cascade.grasping.graspgenx_backend import NoEligibleGrasps
    from cascade.grasping.hug_backend import HugError

    frame, fix, T = _scene()
    c = np.r_[fix.points[:, :2].mean(axis=0), fix.points[:, 2].max() - 0.01]
    planner = _planner(1)
    _respond(planner, [_to_cam(T, _hand(c, [0, 1, 0], [0, 0, 1.0]))])
    with pytest.raises(NoEligibleGrasps, match="approach") as info:
        planner.plan(frame, fix, T_base_cam=T, deadline=time.monotonic() + 5.0,
                     check=lambda: None)
    assert isinstance(info.value, HugError)
    with pytest.raises(HugError) as info:
        planner.plan(frame, fix, T_base_cam=T)
    assert not isinstance(info.value, NoEligibleGrasps)


# ── the serve script: preprocessing, device policy, licences, real wrapper ─────


def test_center_crop_matches_hugs_prepare_inputs():
    """`hug.prepare_inputs` (8d1c52d): square crop of the shorter side at
    ((w - s)//2, (h - s)//2), resize to 224, K shifted then scaled. The query
    pixel must be mapped with exactly that arithmetic."""
    serve = _serve_module()
    for h, w in ((480, 640), (720, 1280), (640, 480), (300, 300), (241, 640)):
        s = min(h, w)
        assert serve.crop_geometry(h, w) == ((w - s) // 2, (h - s) // 2, s)
    K = np.array([[600.0, 0, 330.0], [0, 610.0, 250.0], [0, 0, 1]])
    x_off, y_off, s = serve.crop_geometry(480, 640)
    K224 = serve.adjust_K(K, x_off, y_off, 224 / s)
    ref = K.copy()
    ref[0, 2] -= x_off
    ref[1, 2] -= y_off
    ref[:2, :] *= 224 / s
    np.testing.assert_allclose(K224, ref)
    u224, v224 = serve.query_to_224(400.0, 300.0, x_off, y_off, s)
    # same ray through K and through K_224
    np.testing.assert_allclose([(400 - K[0, 2]) / K[0, 0], (300 - K[1, 2]) / K[1, 1]],
                               [(u224 - K224[0, 2]) / K224[0, 0], (v224 - K224[1, 2]) / K224[1, 1]])
    with pytest.raises(serve.RequestError, match="crop"):
        serve.query_to_224(30.0, 300.0, x_off, y_off, s)


def test_query_crop_mode_shifts_the_square_to_contain_the_query():
    serve = _serve_module()
    x_off, y_off, s = serve.crop_geometry(480, 640, "query", (20.0, 200.0))
    assert (x_off, y_off, s) == (0, 0, 480)
    x_off, y_off, s = serve.crop_geometry(480, 640, "query", (630.0, 200.0))
    assert (x_off, y_off, s) == (160, 0, 480)
    with pytest.raises(serve.RequestError):
        serve.crop_geometry(480, 640, "bogus", (1.0, 1.0))


def test_serve_hug_refuses_silent_cpu_fallback():
    serve = _serve_module()
    with pytest.raises(RuntimeError, match="--device cpu"):
        serve.resolve_device("cuda", cuda_available=False)
    assert serve.resolve_device("cpu", cuda_available=False) == "cpu"
    assert serve.resolve_device("cuda:1", cuda_available=True) == "cuda:1"
    with pytest.raises(ValueError):
        serve.resolve_device("auto", cuda_available=True)   # auto would hide a fallback


def test_serve_hug_main_checks_cuda_before_touching_hug_or_weights(monkeypatch, capsys):
    """Without CUDA and without an explicit --device cpu the server must stop
    before importing HUG or reading any weights."""
    import types

    serve = _serve_module()
    fake_torch = types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: False))
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    built = []
    monkeypatch.setattr(serve, "HugEngine", lambda *a, **k: built.append(1))
    with pytest.raises(SystemExit) as exc:
        serve.main(["--checkpoint", "/nonexistent/hug_full.safetensors", "--port", "0"])
    assert exc.value.code != 0
    assert not built
    assert "--device cpu" in capsys.readouterr().err


def test_mano_is_required_from_the_operator_and_never_shipped(tmp_path):
    """MANO is licensed (registration, non-commercial) and NOT
    redistributable: the server names where to get it, and the repository
    never carries it."""
    serve = _serve_module()
    with pytest.raises(RuntimeError, match=re.escape(serve.MANO_URL)):
        serve.check_mano(tmp_path / "mano_models")
    (tmp_path / "mano_models" / "models").mkdir(parents=True)
    (tmp_path / "mano_models" / "models" / "MANO_RIGHT.pkl").write_bytes(b"operator-supplied")
    assert serve.check_mano(tmp_path / "mano_models").name == "MANO_RIGHT.pkl"
    tracked = subprocess.run(["git", "ls-files"], cwd=REPO, capture_output=True, text=True).stdout
    assert not re.search(r"(?i)(^|/)mano_(right|left)\.pkl$|(^|/)mano_models/", tracked, re.M)
    ignored = (REPO / ".gitignore").read_text()
    assert "/.hug-src/" in ignored and "/.hug/" in ignored


def test_checkpoint_digest_is_pinned_to_the_published_weights(tmp_path):
    serve = _serve_module()
    assert serve.PINNED_CHECKPOINT_SHA256 == (
        "515b5c3bc7987739aec019e754c15df5fbf3eff9daefb93924da098ae4bd1eae")
    ckpt = tmp_path / "hug_full.safetensors"
    ckpt.write_bytes(b"not the weights")
    with pytest.raises(RuntimeError, match="sha256"):
        serve.verify_checkpoint(ckpt, serve.PINNED_CHECKPOINT_SHA256)
    digest = hashlib.sha256(b"not the weights").hexdigest()
    assert serve.verify_checkpoint(ckpt, digest) == digest


_FAKE_HUG = {
    "hug/__init__.py": "",
    "hug/utils/__init__.py": "",
    "hug/utils/data_keys.py": (
        "from pathlib import Path\n"
        "PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent\n"
        "MANO_MODELS_FOLDER = PROJECT_ROOT / 'assets' / 'mano_models'\n"),
    "hug/prepare_inputs.py": (
        "import pickle\nimport numpy as np\nCALLS = []\n"
        "def prepare_pkl(rgb, depth, K, stem, out_dir, object_name=''):\n"
        "    CALLS.append((rgb.copy(), depth.copy(), K.copy(), stem))\n"
        "    h, w = rgb.shape[:2]; s = min(h, w); x, y = (w - s) // 2, (h - s) // 2\n"
        "    idx = (np.arange(224) * s / 224).astype(int)\n"
        "    K2 = K.astype(float).copy(); K2[0, 2] -= x; K2[1, 2] -= y; K2[:2, :] *= 224 / s\n"
        "    data = dict(rgb=rgb[y:y + s, x:x + s][idx][:, idx], depth=depth[y:y + s, x:x + s][idx][:, idx], K=K2)\n"
        "    (out_dir / f'{stem}.pkl').write_bytes(pickle.dumps(data))\n"),
    "hug/dataloader/__init__.py": "",
    "hug/dataloader/grasp_dataset.py": (
        "import pickle\nfrom pathlib import Path\nimport torch\n"
        "class GraspDataset:\n"
        "    def __init__(self, dataset_path, split='train', use_rgb=True, use_depth=True, **kw):\n"
        "        self.dataset_path = Path(dataset_path)\n"
        "    def get_inference_data(self, stem):\n"
        "        d = pickle.loads((self.dataset_path / f'{stem}.pkl').read_bytes())\n"
        "        return {'camera_K': d['K'], 'rgb_original': d['rgb'], 'depth_image': d['depth'],\n"
        "                'rgb': torch.zeros(3, 224, 224), 'pcl_xyz': torch.zeros(4096, 3),\n"
        "                'pcl_rgb': torch.zeros(4096, 3), 'width': 224, 'height': 224}\n"),
    "hug/utils/pcl_utils.py": (
        "import numpy as np\nimport torch\nCROPS = []\n"
        "def pixel_to_xyz(u, v, depth, K):\n"
        "    return np.array([(u - K[0, 2]) * depth / K[0, 0], (v - K[1, 2]) * depth / K[1, 1], depth], np.float32)\n"
        "def depth_to_pcl_tensors(depth_m, rgb, K, n_points=4096, max_depth=3.0, center=None, crop_radius=None):\n"
        "    CROPS.append((np.asarray(center).copy(), crop_radius))\n"
        "    return torch.zeros(n_points, 3), torch.zeros(n_points, 3)\n"),
    "hug/inference.py": (
        "from pathlib import Path\nimport numpy as np\nimport torch\nSAMPLES = []\n"
        "class _Model(torch.nn.Module):\n"
        "    use_rgb = True; use_depth = True; pcl_use_rgb = True; pcl_crop_radius = 0.3\n"
        "    def __init__(self):\n"
        "        super().__init__(); self.w = torch.nn.Parameter(torch.zeros(1))\n"
        "        self.register_buffer('fixed_betas', torch.zeros(1, 10))\n"
        "        self.mano = None; self.mesh_faces = np.zeros((1, 3), int)\n"
        "    def sample(self, point_uv, camera_K, steps=None, rgb=None, pcl_xyz=None, pcl_rgb=None):\n"
        "        SAMPLES.append(dict(point_uv=point_uv.clone(), camera_K=camera_K.clone(), steps=steps,\n"
        "                            rgb=tuple(rgb.shape), pcl_xyz=tuple(pcl_xyz.shape), pcl_rgb=tuple(pcl_rgb.shape)))\n"
        "        out = torch.zeros(point_uv.shape[0], 99); out[:, 2] = point_uv[:, 2]; return out\n"
        "def resolve_checkpoint_path(p):\n    return Path(p)\n"
        "def load_model(checkpoint_path, use_ema, device):\n    return _Model().to(device).eval()\n"),
    "hug/models/__init__.py": "",
    "hug/models/mano.py": (
        "import numpy as np\n"
        "def mano_params_to_grasp_dict(mano_params, betas, mano_model, camera_K, mesh_faces):\n"
        "    t = mano_params[:3].cpu().numpy()\n"
        "    T = np.eye(4, dtype=np.float32); T[:3, 3] = t\n"
        "    return {'landmarks_3d': np.zeros((21, 3), np.float32) + t, 'T_camera_wrist': T}\n"),
}


def test_the_real_engine_wraps_hugs_documented_inference_path(tmp_path, monkeypatch):
    """The real engine cannot run here (no weights, no MANO, no GPU), so it
    runs against a FAKE `hug` package with HUG's API surface. This pins the
    wrapper's plumbing: HUG's own preprocessing is called with the request's
    RGB/depth/K, the query depth is read from HUG's 224 px depth, the point
    cloud is cropped around the query at the model's radius, and N samples
    are ONE batched `model.sample` call."""
    torch = pytest.importorskip("torch")
    for rel, text in _FAKE_HUG.items():
        path = tmp_path / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    mano = tmp_path / "assets" / "mano_models" / "models"
    mano.mkdir(parents=True)
    (mano / "MANO_RIGHT.pkl").write_bytes(b"operator-supplied placeholder")
    ckpt = tmp_path / "hug_full.safetensors"
    ckpt.write_bytes(b"fake weights")
    monkeypatch.syspath_prepend(str(tmp_path))
    for name in [m for m in sys.modules if m == "hug" or m.startswith("hug.")]:
        monkeypatch.delitem(sys.modules, name)
    serve = _serve_module()
    engine = serve.HugEngine(ckpt, device="cpu", sampling_steps=1,
                             expected_sha256=hashlib.sha256(b"fake weights").hexdigest())
    try:
        _exercise_fake_engine(serve, engine, torch)
    finally:
        engine.close()
        for name in [m for m in sys.modules if m == "hug" or m.startswith("hug.")]:
            sys.modules.pop(name, None)


def _exercise_fake_engine(serve, engine, torch):
    health = serve.handle(engine, {"action": "health"})
    assert health["learned"] is True and health["stub"] is False and health["device"] == "cpu"
    frame, fix, T = _scene()
    rgb = np.ascontiguousarray(frame.rgb[..., ::-1])
    depth = np.round(frame.depth_m * 1000).astype(np.uint16)
    reply = serve.handle(engine, {"action": "infer", "rgb": rgb, "depth_mm": depth, "K": frame.K,
                                  "query_px": [160.0, 120.0], "num_samples": 6, "crop": "center"})
    assert "error" not in reply, reply
    assert reply["landmarks_3d"].shape == (6, 21, 3) and reply["T_camera_wrist"].shape == (6, 4, 4)
    import hug.inference
    import hug.prepare_inputs
    import hug.utils.pcl_utils
    (sent_rgb, sent_depth, sent_K, stem), = hug.prepare_inputs.CALLS
    assert np.array_equal(sent_rgb, rgb) and np.array_equal(sent_depth, depth)
    (call,) = hug.inference.SAMPLES
    assert call["point_uv"].shape == (6, 3) and call["camera_K"].shape == (6, 3, 3)
    assert call["rgb"] == (6, 3, 224, 224) and call["pcl_xyz"] == (6, 4096, 3)
    assert call["steps"] == 1
    assert torch.allclose(call["point_uv"][:, 2], torch.tensor(0.46))   # HUG's 224 px depth, metres
    (centre, radius), = hug.utils.pcl_utils.CROPS
    assert radius == 0.3 and centre[2] == pytest.approx(0.46)
    np.testing.assert_allclose(reply["landmarks_3d"][:, 0, 2], 0.46, atol=1e-6)
