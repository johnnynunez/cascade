"""HUG (Human Universal Grasping) grasp backend: a second learned backend
behind the same Grasp API as GraspGen-X.

HUG (Wu et al., arXiv:2606.17054; github.com/KevinyWu/hug @ 8d1c52d, MIT;
weights HF kevinywu/hug @ 1415c9e) predicts a RIGHT HUMAN HAND for one RGB-D
frame and a query pixel on the object: per sample a 99-D MANO vector that
`hug.models.mano.mano_params_to_grasp_dict` turns into 21 camera-frame
landmarks plus `T_camera_wrist`. The release has no server, no score and no
parallel-jaw retargeting. What CASCADE adds is labelled as CASCADE's:

  - `scripts/serve_hug.py` wraps HUG's documented inference path behind a
    REQ/REP msgpack protocol shaped like GraspGen-X's; N samples of one
    query per request is our extension (the paper names it, the app samples
    one per click). The heavy stack (Python 3.10, torch 2.9, DINOv2, MANO)
    lives in its own environment, like `.graspgenx`; this module needs only
    pyzmq + msgpack-numpy (the `grasping` extra).
  - `pinch_from_landmarks` maps a hand to a parallel-jaw PINCH. This is OUR
    assumption, not a HUG output: the jaws close along the thumb tip (4) ->
    index tip (8) line, the TCP is their midpoint, and the gripper arrives
    where the human palm came from (palm centre -> pinch, orthogonalised
    against the jaw axis). `approach: vertical` keeps only the contacts and
    forces a top-down approach.
  - HUG emits NO confidence. `Grasp.quality` of a HUG candidate is CASCADE's
    geometric score: how far the pinch centre sits, across the approach,
    from the observed object (`lateral`) and from its middle (`centre`).
    Learned ranking then comes only from the same `GraspOutcomeMemory`
    re-rank every backend gets, and the selector (width, IK, harness
    pre-vet) and the SafetyHarness stay the authority: a HUG candidate is a
    proposal, nothing more.

Frames: HUG works in the OpenCV camera frame of the image it is given (the
224 px crop/resize changes K, not the 3-D frame), so landmarks map to the
robot base with the same `T_base_cam` that lifted the object's mask points.
"""

from __future__ import annotations

import math
import time

import numpy as np

from ..types import Grasp
from . import evidence
from .graspgenx_backend import NoEligibleGrasps

#: manotorch's SNAP landmark order, which HUG emits (manolayer.py@a2a70c5):
#: 0 wrist, 1-4 thumb (4 = tip, MANO vertex 745), 5-8 index (8 = tip, vertex
#: 317), 9-12 middle, 13-16 ring, 17-20 little; MCP joints at 5/9/13/17.
WRIST, THUMB_TIP, INDEX_TIP, MIDDLE_TIP = 0, 4, 8, 12
PALM_LANDMARKS = (0, 5, 9, 13, 17)
N_LANDMARKS = 21
DEFAULT_PORT = 5558

#: CASCADE's geometric score (HUG has none): quality = 1 / (1 + lateral /
#: _SCORE_LATERAL_M + centre / _SCORE_CENTRE_M). A pinch over the object
#: beats one beside it; among those over it, central beats peripheral.
_SCORE_LATERAL_M = 0.01
_SCORE_CENTRE_M = 0.05

#: depth sources that are measurements; plane-cast depth puts every pixel on
#: the table and would hand HUG fabricated geometry
_MEASURED_DEPTH = ("sensor", "mono")


class HugError(RuntimeError):
    pass


class NoAdmissiblePinch(HugError, NoEligibleGrasps):
    """A valid HUG batch whose every hand the unchanged filters removed.
    Inside the runtime's bounded search this is GraspGen-X's
    `NoEligibleGrasps` -- an empty batch, so the search draws fresh hands
    (HUG sampling is stochastic); everywhere else it is a HugError."""


def _unit(v, what: str) -> np.ndarray:
    v = np.asarray(v, dtype=float).reshape(3)
    n = float(np.linalg.norm(v))
    if not math.isfinite(n) or n < 1e-6:
        raise ValueError(f"degenerate {what}")
    return v / n


def pinch_from_landmarks(landmarks, *, opposition: str = "index",
                         approach: str = "hand") -> dict:
    """21 hand landmarks (any frame; `vertical` needs the BASE frame) ->
    {centre, close_axis, approach, aperture} of a parallel-jaw pinch.

    OUR retargeting assumption, not HUG's: thumb tip vs index tip (or the
    mean of the index and middle tips, `opposition="index_middle"`); the
    approach runs from the palm centre (wrist + four MCPs) to the pinch with
    its jaw-axis component removed. Raises ValueError on a degenerate hand
    instead of guessing a direction."""
    L = np.asarray(landmarks, dtype=float)
    if L.shape != (N_LANDMARKS, 3) or not np.all(np.isfinite(L)):
        raise ValueError(f"expected {N_LANDMARKS} finite 3-D landmarks, got {L.shape}")
    thumb = L[THUMB_TIP]
    if opposition == "index":
        finger = L[INDEX_TIP]
    elif opposition == "index_middle":
        finger = (L[INDEX_TIP] + L[MIDDLE_TIP]) / 2.0
    else:
        raise ValueError(f"unknown pinch opposition {opposition!r} (index | index_middle)")
    aperture = float(np.linalg.norm(finger - thumb))
    centre = (thumb + finger) / 2.0
    close = _unit(finger - thumb, "pinch: thumb and finger tips coincide")
    if approach == "vertical":
        down = np.array([0.0, 0.0, -1.0])
        close = _unit(close - (close @ down) * down, "pinch: jaw axis is vertical")
        a = down
    elif approach == "hand":
        palm = L[list(PALM_LANDMARKS)].mean(axis=0)
        reach = centre - palm
        a = _unit(reach - (reach @ close) * close, "pinch: palm lies on the jaw axis")
    else:
        raise ValueError(f"unknown pinch approach {approach!r} (hand | vertical)")
    return {"centre": centre, "close_axis": close, "approach": a, "aperture": aperture}


def tool_rotation(approach, open_axis, axis_order: str = "down_open") -> np.ndarray:
    """TCP rotation from an approach and a jaw-opening axis in the arm's own
    `tool_axis_order` -- the same three conventions as
    `obb_grasp._yaw_rotation`, for any approach (not only straight down)."""
    a = _unit(approach, "approach")
    x = _unit(np.asarray(open_axis, float) - (np.asarray(open_axis, float) @ a) * a,
              "jaw axis along the approach")
    if axis_order == "down_open":
        return np.column_stack([a, x, np.cross(a, x)])
    if axis_order == "open_down":
        return np.column_stack([x, np.cross(a, x), a])
    if axis_order == "third_open_down":
        return np.column_stack([np.cross(x, a), x, a])
    raise ValueError(f"unknown tool_axis_order {axis_order!r} "
                     "(down_open | open_down | third_open_down)")


def depth_to_mm(depth_m) -> np.ndarray:
    """Metric float depth -> HUG's uint16 millimetres, 0 = invalid.

    HUG zeroes >= 65535; NaN/inf/non-positive metres must become 0, never a
    wrapped integer that reads as plausible depth."""
    d = np.asarray(depth_m, dtype=np.float64)
    with np.errstate(invalid="ignore", over="ignore"):
        mm = np.rint(d * 1000.0)
    ok = np.isfinite(mm) & (mm > 0) & (mm < 65535)
    return np.where(ok, mm, 0).astype(np.uint16)


def _erode(mask: np.ndarray) -> np.ndarray:
    """3x3 binary erosion (HUG samples its training query points from a
    3x3-eroded mask with valid depth)."""
    m = np.asarray(mask, dtype=bool)
    out = m.copy()
    out[1:, :] &= m[:-1, :]
    out[:-1, :] &= m[1:, :]
    out[:, 1:] &= m[:, :-1]
    out[:, :-1] &= m[:, 1:]
    out[1:, 1:] &= m[:-1, :-1]
    out[:-1, :-1] &= m[1:, 1:]
    out[1:, :-1] &= m[:-1, 1:]
    out[:-1, 1:] &= m[1:, :-1]
    out[[0, -1], :] = False
    out[:, [0, -1]] = False
    return out


def query_pixel(detection, depth_m) -> tuple[int, int]:
    """(u, v) on the object with valid depth: the 3x3-eroded mask pixel
    nearest the mask centroid (a mug's centroid is in its hole; the query
    must still be ON the mug). Falls back to the uneroded mask, and to the
    bbox when the detector gave no mask."""
    depth = np.asarray(depth_m, dtype=float)
    h, w = depth.shape[:2]
    mask = getattr(detection, "mask", None)
    if mask is None:
        x0, y0, x1, y1 = (int(round(float(c))) for c in np.asarray(detection.bbox).reshape(4))
        mask = np.zeros((h, w), dtype=bool)
        mask[max(y0, 0):min(y1 + 1, h), max(x0, 0):min(x1 + 1, w)] = True
    mask = np.asarray(mask, dtype=bool)
    if mask.shape != (h, w):
        raise HugError(f"detection mask {mask.shape} does not match the depth image {(h, w)}")
    valid = mask & np.isfinite(depth) & (depth > 0)
    pool = _erode(valid)
    if not pool.any():
        pool = valid
    if not pool.any():
        raise HugError("no pixel of the detection has valid depth; HUG needs a query "
                       "pixel on the object")
    vs, us = np.nonzero(pool)
    my, mx = np.nonzero(mask)
    cu, cv = float(mx.mean()), float(my.mean())
    i = int(np.argmin((us - cu) ** 2 + (vs - cv) ** 2))
    return int(us[i]), int(vs[i])


class HugClient:
    """Minimal REQ/REP msgpack wire client (mirrors `GraspGenXClient`)."""

    def __init__(self, host: str = "127.0.0.1", port: int = DEFAULT_PORT,
                 timeout_ms: int = 15000):
        try:
            import msgpack  # noqa: F401
            import msgpack_numpy
            import zmq  # noqa: F401
        except ImportError as e:
            raise HugError(f"hug backend needs pyzmq+msgpack-numpy in this venv ({e})") from e
        msgpack_numpy.patch()
        self._host, self._port, self._timeout = host, int(port), int(timeout_ms)
        self._sock = None

    def _connect(self):
        import zmq

        sock = zmq.Context.instance().socket(zmq.REQ)
        sock.setsockopt(zmq.RCVTIMEO, self._timeout)
        sock.setsockopt(zmq.SNDTIMEO, self._timeout)
        sock.setsockopt(zmq.LINGER, 0)
        sock.connect(f"tcp://{self._host}:{self._port}")
        self._sock = sock

    def request(self, payload: dict, timeout_ms: int | None = None, *,
                deadline: float | None = None, check=None) -> dict:
        import msgpack
        import zmq

        if deadline is not None:
            return self._request_bounded(payload, timeout_ms, deadline, check)
        if self._sock is None:
            self._connect()
        sock = self._sock
        if timeout_ms is not None:
            sock.setsockopt(zmq.RCVTIMEO, int(timeout_ms))
            sock.setsockopt(zmq.SNDTIMEO, int(timeout_ms))
        try:
            sock.send(msgpack.packb(payload, use_bin_type=True))
            raw = sock.recv()
        except zmq.error.Again as e:
            self.close()  # a REQ socket is wedged after a timeout
            raise HugError(
                f"hug server at {self._host}:{self._port} timed out "
                f"({timeout_ms if timeout_ms is not None else self._timeout} ms); "
                "is scripts/serve_hug.py running?") from e
        finally:
            if timeout_ms is not None and self._sock is not None:
                self._sock.setsockopt(zmq.RCVTIMEO, self._timeout)
                self._sock.setsockopt(zmq.SNDTIMEO, self._timeout)
        resp = msgpack.unpackb(raw, raw=False)
        if isinstance(resp, dict) and "error" in resp:
            raise HugError(f"hug server error: {resp['error']}")
        return resp

    def _request_bounded(self, payload, timeout_ms, deadline, check):
        """Send/receive share the search deadline and poll its cancellation
        check (GraspGen-X's bounded request); a cancelled or late REQ socket
        is closed, never reused for the next batch."""
        import msgpack
        import zmq

        if (not isinstance(deadline, (int, float)) or isinstance(deadline, bool)
                or not math.isfinite(deadline)):
            raise HugError("invalid grasp planning deadline")
        end = min(deadline, time.monotonic()
                  + (self._timeout if timeout_ms is None else timeout_ms) / 1000.0)

        def remaining_ms():
            if check is not None:
                check()  # pure cancellation/deadline callback, never a robot RPC
            left = end - time.monotonic()
            if left <= 0:
                raise HugError("hug planning request timed out")
            return max(1, min(50, math.ceil(left * 1000)))

        try:
            remaining_ms()
            encoded = msgpack.packb(payload, use_bin_type=True)
            remaining_ms()
            if self._sock is None:
                self._connect()
            sock = self._sock
            raw = None
            for event, operation in (
                    (zmq.POLLOUT, lambda: sock.send(encoded, flags=zmq.NOBLOCK)),
                    (zmq.POLLIN, lambda: sock.recv(flags=zmq.NOBLOCK))):
                while True:
                    ready = sock.poll(remaining_ms(), event)
                    remaining_ms()
                    if not ready:
                        continue
                    try:
                        raw = operation()
                        remaining_ms()
                        break
                    except zmq.error.Again:
                        continue
            response = msgpack.unpackb(raw, raw=False)
            remaining_ms()
            if isinstance(response, dict) and "error" in response:
                raise HugError(f"hug server error: {response['error']}")
            return response
        except BaseException:
            self.close()
            raise

    def probe(self, timeout_ms: int = 300, *, deadline=None, check=None) -> dict:
        kwargs = {} if deadline is None else {"deadline": deadline, "check": check}
        resp = self.request({"action": "health"}, timeout_ms=timeout_ms, **kwargs)
        if not (isinstance(resp, dict) and (resp.get("status") == "ok" or resp.get("ok") is True)):
            raise HugError(f"hug health returned {resp!r}")
        return resp

    def close(self):
        if self._sock is not None:
            self._sock.close()
            self._sock = None


class HugPlanner:
    """Frame + fix -> base-frame parallel-jaw pinches from HUG hands."""

    def __init__(self, cfg):
        from ..config import HUG_HOST_ENV, HUG_PORT_ENV, sidecar_endpoint

        h = cfg.get("hug", None)
        get = (lambda k, d: h.get(k, d)) if h is not None else (lambda k, d: d)
        # load_demo_config already applied CASCADE_HUG_HOST/_PORT to the section;
        # they fill only what a hand-made section lacks (applied once).
        host, port = sidecar_endpoint(h, port_env=HUG_PORT_ENV, host_env=HUG_HOST_ENV,
                                      default_port=DEFAULT_PORT)
        self._client = HugClient(host=host, port=port, timeout_ms=int(get("timeout_ms", 15000)))
        self.required = bool(get("required", True))
        self.num_samples = int(get("num_samples", 32))
        if not 1 <= self.num_samples <= 256:
            raise ValueError("hug.num_samples must be in 1..256")
        self.topk = int(get("topk", 32))
        self.approach_z_max = float(get("approach_z_max", 0.2))
        self.max_lateral_offset_m = float(get("max_lateral_offset_m", 0.03))
        self.surface_tolerance_m = float(get("surface_tolerance_m", 0.005))
        self.max_query_offset_m = float(get("max_query_offset_m", 0.02))
        self.opposition = str(get("pinch_opposition", "index"))
        self.approach_mode = str(get("pinch_approach", "hand"))
        self.crop = str(get("crop", "center"))
        if self.crop not in {"center", "query"}:
            raise ValueError(f"unknown hug.crop {self.crop!r} (center | query)")
        self.probe_timeout_ms = int(get("probe_timeout_ms", 300))
        #: what the startup probe saw, or None
        self.status: dict | None = None
        self.last_latency_s: float | None = None
        self.last_counts: dict[str, int] = {}
        self.last_response: dict | None = None

    def probe(self, timeout_ms: int | None = None, *, deadline=None, check=None) -> dict:
        kwargs = {} if deadline is None else {"deadline": deadline, "check": check}
        self.status = self._client.probe(
            timeout_ms=self.probe_timeout_ms if timeout_ms is None else timeout_ms, **kwargs)
        if self.required and self.status.get("stub"):
            self.status = None
            raise HugError("real HUG required; analytic protocol stub rejected")
        return self.status

    def describe(self) -> str:
        if self.status is None:
            return "hug (unprobed)"
        if self.status.get("stub"):
            return "hug-stub (analytic protocol double)"
        return f"hug (learned human hand -> cascade pinch, {self.status.get('device', '?')})"

    @staticmethod
    def _landmarks(response) -> np.ndarray:
        """Malformed batches are terminal: never pinch from a partial hand."""
        try:
            if not isinstance(response, dict):
                raise ValueError("expected a mapping")
            L = np.asarray(response["landmarks_3d"])
            if (L.ndim != 3 or L.shape[1:] != (N_LANDMARKS, 3) or L.shape[0] < 1
                    or L.dtype.kind not in "fiu" or not np.all(np.isfinite(L))):
                raise ValueError(f"expected finite numeric (N>=1, {N_LANDMARKS}, 3) "
                                 f"landmarks, got {L.shape}")
            if "T_camera_wrist" in response:
                T = np.asarray(response["T_camera_wrist"])
                if T.shape != (L.shape[0], 4, 4) or not np.all(np.isfinite(T)):
                    raise ValueError("T_camera_wrist does not match the landmarks")
            return L.astype(float)
        except (KeyError, ValueError, TypeError) as exc:
            raise HugError(f"malformed HUG batch: {exc}") from exc

    def plan(self, frame, fix, *, T_base_cam, max_width_m: float = 0.09,
             axis_order: str = "down_open", width_pad_m: float = 0.015,
             deadline: float | None = None, check=None) -> list[Grasp]:
        """RGB-D frame + the object's base-frame fix -> ranked pinch Grasps.

        `deadline`/`check` come from the runtime's bounded search: the request
        shares that deadline, and an all-filtered batch is then an empty
        batch (`NoAdmissiblePinch`) rather than a terminal error."""
        if check is not None:
            check()
        if frame is None or getattr(frame, "depth_m", None) is None:
            raise HugError("HUG needs the RGB-D frame the object was localized in")
        if str(getattr(frame, "depth_source", "none")) not in _MEASURED_DEPTH:
            raise HugError(f"HUG needs measured depth (sensor/mono), got "
                           f"{getattr(frame, 'depth_source', None)!r}")
        T = np.asarray(T_base_cam, dtype=float).reshape(4, 4)
        source_points = fix.points
        pts = np.asarray(source_points.detach().cpu().numpy() if hasattr(source_points, "detach")
                         else source_points, dtype=float).reshape(-1, 3)
        if pts.shape[0] < 10:
            raise HugError(f"only {pts.shape[0]} object points (<10)")
        depth = np.asarray(frame.depth_m, dtype=np.float32)
        K = np.asarray(frame.K, dtype=np.float64).reshape(3, 3)
        u, v = query_pixel(fix.detection, depth)
        # Fail closed if this frame's transform does not reproduce the fix:
        # the query pixel was taken inside the detection mask, so lifting it
        # with the right T lands on one of the object's own points. A
        # secondary camera's frame with the primary's extrinsic would not.
        d = float(depth[v, u])
        q_cam = np.array([(u - K[0, 2]) * d / K[0, 0], (v - K[1, 2]) * d / K[1, 1], d])
        q_base = T[:3, :3] @ q_cam + T[:3, 3]
        miss = float(np.min(np.linalg.norm(pts - q_base, axis=1)))
        if miss > self.max_query_offset_m:
            raise HugError(
                f"query pixel ({u}, {v}) back-projects {miss:.3f} m from the localized "
                "object: this frame's camera extrinsic does not match the fix")
        payload = {
            "action": "infer",
            # Frame.rgb is BGR (OpenCV); HUG and its DINOv2 expect RGB.
            "rgb": np.ascontiguousarray(np.asarray(frame.rgb, dtype=np.uint8)[..., ::-1]),
            "depth_mm": depth_to_mm(depth),
            "K": K,
            "query_px": [float(u), float(v)],
            "num_samples": self.num_samples,
            "crop": self.crop,
        }
        evidence.event("hug_request", query_px=[u, v], query_base_m=q_base,
                       num_samples=self.num_samples, crop=self.crop, status=self.status,
                       opposition=self.opposition, approach_mode=self.approach_mode)
        t0 = time.monotonic()
        resp = self._client.request(
            payload, **({} if deadline is None else {"deadline": deadline, "check": check}))
        self.last_latency_s = round(time.monotonic() - t0, 3)
        self.last_response = resp
        if check is not None:
            check()
        L_cam = self._landmarks(resp)
        evidence.array("hug_landmarks_cam_m", L_cam)
        L_base = L_cam @ T[:3, :3].T + T[:3, 3]

        counts = {"returned": len(L_base), "degenerate": 0, "approach": 0,
                  "off_object": 0, "in_front_of_surface": 0, "kept": 0}
        centroid = pts.mean(axis=0)
        grasps: list[Grasp] = []
        for hand in L_base:
            if check is not None:
                check()
            try:
                pinch = pinch_from_landmarks(hand, opposition=self.opposition,
                                             approach=self.approach_mode)
            except ValueError:
                counts["degenerate"] += 1
                continue
            a, x, c = pinch["approach"], pinch["close_axis"], pinch["centre"]
            # Tabletop sanity (same rule as GraspGen-X): no approach from below.
            if a[2] > self.approach_z_max:
                counts["approach"] += 1
                continue
            rel = pts - c
            across = rel - np.outer(rel @ a, a)
            lateral = float(np.min(np.linalg.norm(across, axis=1)))
            if lateral > self.max_lateral_offset_m:
                counts["off_object"] += 1        # the jaws would close beside it
                continue
            if float(c @ a) < float(np.min(pts @ a)) - self.surface_tolerance_m:
                counts["in_front_of_surface"] += 1   # closes on air before the object
                continue
            off_centre = centroid - c
            centre_err = float(np.linalg.norm(off_centre - (off_centre @ a) * a))
            span = (pts - c) @ x
            width = float(np.quantile(span, 0.95) - np.quantile(span, 0.05)) + float(width_pad_m)
            quality = 1.0 / (1.0 + lateral / _SCORE_LATERAL_M + centre_err / _SCORE_CENTRE_M)
            grasps.append(Grasp(position=c, rotation=tool_rotation(a, x, axis_order),
                                width_m=width, approach=a, quality=quality, label=fix.label))
        grasps.sort(key=lambda g: -g.quality)
        grasps = grasps[:self.topk]
        counts["kept"] = len(grasps)
        self.last_counts = counts
        evidence.event("hug_filtered_counts", **counts, approach_z_max=self.approach_z_max,
                       max_lateral_offset_m=self.max_lateral_offset_m,
                       latency_s=self.last_latency_s, stub=bool(resp.get("stub")),
                       ranking="cascade geometric score; HUG emits no confidence")
        if not grasps:
            error = HugError if deadline is None else NoAdmissiblePinch
            raise error(f"no HUG hand maps to an admissible pinch: {counts}")
        if check is not None:
            check()
        return grasps
