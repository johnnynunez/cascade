"""The cursor: a queryable point probe over the camera image.

Anthropic's *Claude Plays Robotics* (Jul 2026) ran an ablation over visual aids
for a Panda arm and found something that cuts against the obvious intuition:

* depth maps overlaid on the frame -- **roughly neutral**
* labeled segmentation overlays  -- **roughly neutral**
* a **cursor tool** ("a small red X on the gripper cam that the model can move
  and query for the object and distance at that point") -- **large uplift for
  every model tested**; on a 10-task manipulation subset their strongest model
  went from **6% to 32%** success.

Their reading: "models mainly need better orientation, not a different view of
the scene". The static overlays carry the right information but the signal is
*too diffuse* to act on. The cursor works because it answers a specific
question with a specific number, on demand.

That distinction is the whole design of this module. A probe is not a picture:

    probe_point(u, v)  ->  depth at that pixel, the 3D point in the robot's
                           base frame, which tracked object is there, whether
                           the arm can actually reach it, and how far it is
                           from the current gripper position.

Everything returned is a scalar the agent can compare, not a texture it has to
interpret. The inverse direction (`locate_pixel`) closes the loop: given an
object the agent knows about, where is it *in the image it is looking at*.

Note on this repo's existing surfaces: the dashboard's depth view and the VIA
annotated view are **human** diagnostics -- they are rendered for a person
watching a browser, not fed into the model's context. Anthropic's neutral
result is about overlays *given to the model*, so it does not condemn them;
but it does say plainly that adding more such overlays to the agent's context
is not where the win is. The win is here.

IMPORTANT SEMANTICS -- a probe returns the VISIBLE SURFACE, not the centroid.
Deprojecting a pixel gives the first surface the ray hits, so probing the
middle of a cube returns a point on its TOP FACE, offset toward the camera by
parallax. Measured on the live rig against Isaac physics truth: probing a
cube whose centre is at z=0.040 m returns z=0.062 m (+22 mm -- the top face of
a ~4.5 cm cube) and ~+24 mm in x (the ray meets the top face nearer the
camera). Round-trip 3D -> pixel -> 3D closes to **1.6 mm**, so the geometry is
sound; the offset is physics, not error.

Practical consequence: do NOT feed a probe straight into a grasp centre. Use
it for *relative* judgements -- is this reachable, which object is here, how
far is the gripper -- and let the grasp planner keep using segmented point
clouds for the object frame. `probe_top_z` is reported separately for the case
where the top face is exactly what you want (grasp depth).
"""

from __future__ import annotations

import numpy as np

#: Radius (px) of the median window used to read depth, so one dead pixel in
#: a sensor depth map does not produce a wild 3D point.
PATCH_R = 3

#: A probe within this distance (m) of a tracked object's centre is reported
#: as being "on" that object.
HIT_RADIUS_M = 0.06


def _median_depth(depth: np.ndarray, u: int, v: int, r: int = PATCH_R) -> float | None:
    """Median of the valid depth samples in a small window around (u, v).

    Single-pixel reads are the classic way to get a 3D point that is metres
    off: sensor depth maps have holes and flying pixels, especially on object
    silhouettes (documented on the L515 in perception/grounding.py).
    """
    h, w = depth.shape[:2]
    u0, u1 = max(0, u - r), min(w, u + r + 1)
    v0, v1 = max(0, v - r), min(h, v + r + 1)
    patch = depth[v0:v1, u0:u1]
    valid = patch[np.isfinite(patch) & (patch > 0)]
    if valid.size == 0:
        return None
    return float(np.median(valid))


def deproject(u: float, v: float, z: float, K: np.ndarray) -> np.ndarray:
    """Pixel + depth -> 3D point in the camera frame."""
    K = np.asarray(K, dtype=float)
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    return np.array([(u - cx) * z / fx, (v - cy) * z / fy, z], dtype=float)


class PointProbe:
    """Queryable cursor over a camera frame.

    ``runtime`` is a ``SkillRuntime``; everything else (intrinsics, extrinsics,
    beliefs, workspace limits, TCP) is read from it at call time so the probe
    never holds stale calibration.
    """

    def __init__(self, runtime):
        self.rt = runtime

    # ── geometry ─────────────────────────────────────────────────────────

    def _frame_and_extrinsics(self, camera: str | None):
        rig = getattr(self.rt, "rig", None)
        if camera and rig is not None:
            stream = rig.get(camera)          # KeyError -> caller reports it
            frame = stream.latest()
        else:
            frame = self.rt.last_frame or self.rt.observe()
        if frame is None:
            raise ValueError("no frame available yet")
        T = frame.T_base_cam
        if T is None:
            T = self.rt.extrinsics.cam_to_base()
        return frame, np.asarray(T, dtype=float)

    def _reach(self) -> tuple[dict, tuple[float, float]]:
        cfg = self.rt.cfg
        ws_raw = cfg.safety.get("workspace", {}) or {}
        ws = dict(getattr(ws_raw, "_data", ws_raw))
        g = cfg.grasp if hasattr(cfg, "grasp") else {}
        band = (float(g.get("reach_x_min", 0.155)), float(g.get("reach_x_max", 0.185)))
        return ws, band

    # ── the cursor ───────────────────────────────────────────────────────

    def probe(self, u, v, camera: str | None = None, normalized: bool = False) -> dict:
        """What is at this pixel? Returns metric facts, not an image.

        ``normalized=True`` accepts u, v in 0..1, which is what a model that
        reasoned over a resized frame will naturally produce.
        """
        frame, T = self._frame_and_extrinsics(camera)
        h, w = frame.rgb.shape[:2]
        u, v = float(u), float(v)
        # A model that reasoned over a resized frame naturally emits 0..1.
        # Auto-detect that rather than silently probing the top-left corner:
        # on a real frame, pixel (0.4, 0.7) is never a meaningful target.
        if normalized or (0.0 <= u <= 1.0 and 0.0 <= v <= 1.0 and max(w, h) > 4):
            u, v = u * (w - 1), v * (h - 1)
        ui, vi = int(round(u)), int(round(v))
        out: dict = {
            "pixel": [ui, vi],
            "camera": camera or getattr(self.rt.camera, "name", "primary"),
            "image_size": [w, h],
        }
        if not (0 <= ui < w and 0 <= vi < h):
            out["ok"] = False
            out["error"] = f"pixel ({ui}, {vi}) is outside the {w}x{h} image"
            return out

        if not frame.has_depth or frame.depth_m is None:
            out["ok"] = False
            out["error"] = "this camera has no depth; cannot lift a pixel to 3D"
            out["depth_source"] = frame.depth_source
            return out

        z = _median_depth(np.asarray(frame.depth_m, dtype=np.float32), ui, vi)
        out["depth_source"] = frame.depth_source
        if z is None:
            out["ok"] = False
            out["error"] = (
                f"no valid depth within {PATCH_R} px of ({ui}, {vi}) -- "
                "a depth hole; probe a nearby pixel on the object body"
            )
            return out

        p_cam = deproject(ui, vi, z, frame.K)
        p_base = (T @ np.append(p_cam, 1.0))[:3]
        out["ok"] = True
        out["distance_m"] = round(float(z), 4)
        out["position"] = [round(float(x), 4) for x in p_base]
        # Say this out loud in the payload: a ray hits the FIRST surface, so
        # probing the middle of a cube returns its top face (verified: +22 mm
        # in z, +24 mm in x toward the camera, on a 4.5 cm cube). An agent
        # that feeds this straight into a grasp centre will grasp high.
        out["measures"] = "visible surface at this pixel, not the object centre"

        # which tracked object is here (the "query for the object" half)
        hit, hit_d = None, None
        try:
            for b in self.rt.beliefs.all():
                d = float(np.linalg.norm(np.asarray(b.position, dtype=float)[:3] - p_base))
                if hit_d is None or d < hit_d:
                    hit, hit_d = b, d
        except Exception:
            pass
        if hit is not None and hit_d is not None and hit_d <= HIT_RADIUS_M:
            out["object"] = {
                "label": str(getattr(hit, "label", "?")),
                "offset_m": round(hit_d, 4),
                "center": [round(float(x), 4) for x in np.asarray(hit.position)[:3]],
            }
        else:
            out["object"] = None
            if hit is not None and hit_d is not None:
                out["nearest_object"] = {
                    "label": str(getattr(hit, "label", "?")),
                    "offset_m": round(hit_d, 4),
                }

        # can the arm actually act here? (the "orientation" the ablation says
        # models are missing -- a number, not a shaded region)
        ws, band = self._reach()
        try:
            lo, hi = ws["min"], ws["max"]
            inside = all(float(lo[i]) <= p_base[i] <= float(hi[i]) for i in range(3))
        except (KeyError, TypeError, IndexError):
            inside = None
        out["reachable"] = {
            "in_workspace": inside,
            "in_topdown_ik_band": bool(band[0] <= p_base[0] <= band[1]),
            "topdown_ik_band_x": [round(band[0], 3), round(band[1], 3)],
        }
        if inside is False:
            out["reachable"]["note"] = "outside the safety workspace AABB: motion will be refused"
        elif not out["reachable"]["in_topdown_ik_band"]:
            out["reachable"]["note"] = (
                f"outside the strict top-down IK band (x {band[0]:.3f}-{band[1]:.3f} m); "
                "a top-down grasp here will likely fail IK -- push the object closer first"
            )

        # how far is the gripper from this point right now
        try:
            tcp = np.asarray(self.rt._tcp(), dtype=float)[:3]
            delta = p_base - tcp
            out["from_gripper"] = {
                "distance_m": round(float(np.linalg.norm(delta)), 4),
                "delta_xyz_m": [round(float(x), 4) for x in delta],
            }
        except Exception:
            pass
        return out

    # ── the inverse: where is this object in the image? ───────────────────

    def locate_pixel(self, label: str, camera: str | None = None) -> dict:
        """Project a tracked object into image coordinates.

        Lets the agent move the cursor to something it knows by name, and
        reason about the scene in the same pixel space it is looking at.
        """
        frame, T = self._frame_and_extrinsics(camera)
        h, w = frame.rgb.shape[:2]
        want = set(str(label).lower().split())
        best, score = None, 0
        try:
            for b in self.rt.beliefs.all():
                have = set(str(getattr(b, "label", "")).lower().split())
                s = len(want & have)
                if s > score:
                    best, score = b, s
        except Exception:
            pass
        if best is None:
            return {"ok": False, "error": f"{label!r} is not in the world model",
                    "hint": "call get_observation or localize_object first"}

        p_base = np.asarray(best.position, dtype=float)[:3]
        p_cam = (np.linalg.inv(T) @ np.append(p_base, 1.0))[:3]
        if p_cam[2] <= 1e-6:
            return {"ok": False, "error": f"{best.label!r} is behind the camera"}
        K = np.asarray(frame.K, dtype=float)
        uv = K @ (p_cam / p_cam[2])
        u, v = int(round(float(uv[0]))), int(round(float(uv[1])))
        return {
            "ok": True,
            "label": str(best.label),
            "pixel": [u, v],
            "image_size": [w, h],
            "normalized": [round(u / max(w - 1, 1), 4), round(v / max(h - 1, 1), 4)],
            "position": [round(float(x), 4) for x in p_base],
            "distance_m": round(float(p_cam[2]), 4),
            "in_view": bool(0 <= u < w and 0 <= v < h),
        }


def draw_cursor(img: np.ndarray, u: int, v: int, text: str = "") -> np.ndarray:
    """Draw the red X the ablation used, for the human-facing views.

    The mark is for the person watching the dashboard; the *agent* consumes
    the numbers from ``PointProbe.probe``. Keeping those two channels separate
    is the point -- an overlay is neutral, a queryable number is not.
    """
    import cv2

    out = img if img.flags.writeable else img.copy()
    cv2.drawMarker(out, (int(u), int(v)), (0, 0, 255), cv2.MARKER_TILTED_CROSS, 22, 2)
    cv2.circle(out, (int(u), int(v)), 14, (0, 0, 255), 1, cv2.LINE_AA)
    if text:
        cv2.putText(out, text, (int(u) + 18, int(v) - 8), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(out, text, (int(u) + 18, int(v) - 8), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (60, 60, 255), 1, cv2.LINE_AA)
    return out
