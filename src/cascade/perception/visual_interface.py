"""VIA-style annotated visual interface: make the frame *readable* by an agent.

VIA (arXiv:2607.11119) recasts robot control as an agentic task: an
off-the-shelf frontier model drives a manipulator through a browser-based 3D
interface by taking screenshots, issuing intuitive commands, observing the
outcome and adjusting -- no robot-specific fine-tuning, no privileged state.
It reaches 96.7% on three LIBERO-Goal tasks with Claude Code / Codex, and its
central claim is that *the interface is the missing piece*: frontier agents
already possess the skills, given something they can actually read.

The lesson transfers without adopting their browser stack.  cascade hands
the VLM a raw camera JPEG and expects metric reasoning from it -- but a model
cannot say "grasp 3 cm to the left" reliably from bare pixels; it has no
scale, no origin, no way to name what it sees unambiguously.

This module renders the annotated view VIA relies on:

* **numbered marks** on every tracked object (set-of-mark prompting), so the
  agent refers to "object 2" instead of hoping its label string matches;
* a projected **base-frame grid** with metre ticks, so "further right" has a
  number attached;
* an optional **configured grasp band** drawn as a display hint. No default
  band is assumed; this overlay does not run inverse kinematics;
* the **gripper's current TCP** and the pending target, so the agent can
  judge its own motion visually rather than trusting a number it cannot check.

Everything is drawn from data the runtime already owns (beliefs, extrinsics,
kinematics), so this costs one image encode and no extra perception.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import cv2
import numpy as np

#: BGR palette for marks (distinct under JPEG, readable on a booth screen)
_COLORS = [
    (60, 220, 60), (60, 160, 255), (255, 120, 60), (200, 80, 255),
    (60, 240, 240), (255, 220, 60), (140, 140, 255), (80, 200, 160),
]
_WHITE = (255, 255, 255)
_BLACK = (0, 0, 0)


@dataclass
class Mark:
    """One numbered object in the annotated view."""

    index: int
    label: str
    uv: tuple[int, int]
    position: list[float]
    reachable: bool | None = None
    confirmed: bool = True
    extra: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "index": self.index,
            "label": self.label,
            "position": [round(float(v), 3) for v in self.position],
            "pixel": [int(self.uv[0]), int(self.uv[1])],
            "reachable": self.reachable,
            "confirmed": self.confirmed,
            **self.extra,
        }


def project(points_base: np.ndarray, T_base_cam: np.ndarray, K: np.ndarray) -> np.ndarray:
    """Base-frame points -> pixel coordinates.  Returns (N, 2) float array.

    Points behind the camera come back as NaN so callers can drop them.
    """
    pts = np.atleast_2d(np.asarray(points_base, dtype=float))
    T_cam_base = np.linalg.inv(np.asarray(T_base_cam, dtype=float))
    homo = np.hstack([pts, np.ones((pts.shape[0], 1))])
    cam = (T_cam_base @ homo.T).T[:, :3]
    z = cam[:, 2:3]
    with np.errstate(divide="ignore", invalid="ignore"):
        uv = (np.asarray(K, dtype=float) @ (cam / z).T).T[:, :2]
    uv[np.repeat(z <= 1e-6, 2, axis=1)] = np.nan
    return uv


def _put_label(img, text, org, color, scale=0.5, thick=1):
    """Text with a dark outline so it survives any background."""
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, _BLACK, thick + 2, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


class VisualInterface:
    """Renders the agent-facing annotated view of a frame.

    ``workspace`` is the safety AABB dict ({min: [...], max: [...]}) and
    ``reach_x`` is an optional configured (x_min, x_max) display band.
    Neither one establishes that a grasp or an IK solution exists.
    """

    def __init__(
        self,
        extrinsics=None,
        workspace: dict | None = None,
        reach_x: tuple[float, float] | None = None,
        table_z: float = 0.0,
        grid_step_m: float = 0.05,
        min_observations: int = 2,
        visible_horizon_s: float = 1.5,
    ):
        self.extrinsics = extrinsics
        self.workspace = workspace or {}
        self.reach_x = _valid_band(reach_x)
        self.table_z = float(table_z)
        self.grid_step = float(grid_step_m)
        #: A belief re-observed fewer than this many times AND not currently
        #: visible is one sighting the world never confirmed again -- drawn
        #: as "unconfirmed" instead of a full numbered target (see
        #: _draw_marks). ROADMAP: "annotated_view surfaced a stale 4th
        #: 'cube' mark" -- a single misdetection the belief store correctly
        #: never forgets (object permanence is deliberate), but the
        #: annotated view was handing it to the agent with the same
        #: confidence as a repeatedly-confirmed object.
        self.min_observations = int(min_observations)
        self.visible_horizon_s = float(visible_horizon_s)

    # ── the annotated frame ──────────────────────────────────────────────

    def render(
        self,
        frame,
        beliefs: list | None = None,
        tcp: np.ndarray | None = None,
        target: np.ndarray | None = None,
        grid: bool = True,
        envelope: bool = True,
        now: float | None = None,
    ) -> tuple[np.ndarray, list[Mark]]:
        """-> (annotated BGR image, marks).  Never mutates ``frame``."""
        img = np.ascontiguousarray(frame.rgb.copy())
        K = getattr(frame, "K", None)
        T = getattr(frame, "T_base_cam", None)
        if T is None and self.extrinsics is not None:
            T = self.extrinsics.cam_to_base()
        if K is None or T is None:
            return img, []

        if grid:
            self._draw_grid(img, T, K)
        if envelope:
            self._draw_envelope(img, T, K)

        marks = self._draw_marks(img, beliefs or [], T, K, now)

        if tcp is not None:
            self._draw_tcp(img, np.asarray(tcp, float), T, K)
        if target is not None:
            uv = project(np.asarray(target, float)[None, :3], T, K)[0]
            if np.isfinite(uv).all():
                p = (int(uv[0]), int(uv[1]))
                cv2.drawMarker(img, p, (0, 0, 255), cv2.MARKER_TILTED_CROSS, 18, 2)
                _put_label(img, "target", (p[0] + 10, p[1] - 8), (0, 0, 255))

        self._draw_legend(img, marks)
        return img, marks

    # ── layers ───────────────────────────────────────────────────────────

    def _draw_grid(self, img, T, K) -> None:
        ws = self.workspace
        try:
            x0, y0 = float(ws["min"][0]), float(ws["min"][1])
            x1, y1 = float(ws["max"][0]), float(ws["max"][1])
        except (KeyError, IndexError, TypeError, ValueError):
            return
        h, w = img.shape[:2]
        step = self.grid_step
        overlay = img.copy()

        def seg(a, b, color, thick=1):
            uv = project(np.array([a, b], dtype=float), T, K)
            if not np.isfinite(uv).all():
                return
            p0, p1 = (int(uv[0][0]), int(uv[0][1])), (int(uv[1][0]), int(uv[1][1]))
            if max(abs(p0[0]), abs(p1[0])) > 4 * w or max(abs(p0[1]), abs(p1[1])) > 4 * h:
                return
            cv2.line(overlay, p0, p1, color, thick, cv2.LINE_AA)

        n_x = int(round((x1 - x0) / step))
        n_y = int(round((y1 - y0) / step))
        for i in range(n_x + 1):
            x = x0 + i * step
            seg([x, y0, self.table_z], [x, y1, self.table_z], (90, 90, 90))
        for j in range(n_y + 1):
            y = y0 + j * step
            seg([x0, y, self.table_z], [x1, y, self.table_z], (90, 90, 90))
        cv2.addWeighted(overlay, 0.35, img, 0.65, 0, img)

        # metre ticks along the near edge, so distances are readable
        for i in range(n_x + 1):
            x = x0 + i * step
            uv = project(np.array([[x, y1, self.table_z]]), T, K)[0]
            if np.isfinite(uv).all():
                _put_label(img, f"{x:.2f}", (int(uv[0]) - 12, int(uv[1]) + 14), (170, 170, 170), 0.36)

    def _draw_envelope(self, img, T, K) -> None:
        """An explicitly configured display hint; no IK claim."""
        if self.reach_x is None:
            return
        ws = self.workspace
        try:
            y0, y1 = float(ws["min"][1]), float(ws["max"][1])
        except (KeyError, IndexError, TypeError, ValueError):
            return
        xa, xb = self.reach_x
        corners = np.array(
            [[xa, y0, self.table_z], [xb, y0, self.table_z],
             [xb, y1, self.table_z], [xa, y1, self.table_z]], dtype=float
        )
        uv = project(corners, T, K)
        if not np.isfinite(uv).all():
            return
        poly = uv.astype(np.int32).reshape(-1, 1, 2)
        overlay = img.copy()
        cv2.fillPoly(overlay, [poly], (0, 140, 0))
        cv2.addWeighted(overlay, 0.18, img, 0.82, 0, img)
        cv2.polylines(img, [poly], True, (0, 200, 0), 2, cv2.LINE_AA)
        anchor = uv[np.argmin(uv[:, 1])]
        _put_label(img, "configured grasp band", (int(anchor[0]), int(anchor[1]) - 8), (0, 220, 0), 0.45)

    def _draw_marks(self, img, beliefs, T, K, now: float | None = None) -> list[Mark]:
        now = time.monotonic() if now is None else now
        marks: list[Mark] = []
        for i, b in enumerate(beliefs, start=1):
            try:
                pos = np.asarray(b.position, dtype=float)[:3]
            except Exception:
                continue
            uv = project(pos[None, :], T, K)[0]
            if not np.isfinite(uv).all():
                continue
            u, v = int(uv[0]), int(uv[1])
            h, w = img.shape[:2]
            if not (-w < u < 2 * w and -h < v < 2 * h):
                continue
            color = _COLORS[(i - 1) % len(_COLORS)]
            in_workspace = self._in_workspace(pos)
            label = str(getattr(b, "label", "?"))
            last_seen_t = getattr(b, "last_seen_t", now)
            age_s = max(0.0, float(now - last_seen_t))
            observations = int(getattr(b, "observations", 1))
            # Currently visible, or re-observed at least once since first
            # spotted: a real object. A single stale sighting is drawn but
            # NOT handed to the agent as an equal-confidence numbered target.
            confirmed = age_s <= self.visible_horizon_s or observations >= self.min_observations

            in_image = bool(0 <= uv[0] < w and 0 <= uv[1] < h)
            marks.append(
                Mark(index=i, label=label, uv=(u, v), position=pos.tolist(),
                     confirmed=confirmed,
                     extra={"age_s": round(age_s, 1), "in_workspace": in_workspace,
                            "ik_checked": False, "projected_in_image": in_image})
            )
            if not in_image:
                continue

            if confirmed:
                cv2.circle(img, (u, v), 13, color, 2, cv2.LINE_AA)
                cv2.circle(img, (u, v), 3, color, -1, cv2.LINE_AA)
            else:
                cv2.circle(img, (u, v), 13, color, 1, cv2.LINE_AA)
            # the index badge is what the agent refers to
            cv2.rectangle(img, (u - 24, v - 26), (u - 6, v - 8), color, -1)
            _put_label(img, str(i), (u - 21, v - 12), _BLACK, 0.5, 2)
            tag = f"{label}  ({pos[0]:.2f}, {pos[1]:.2f})"
            if in_workspace is False:
                tag += " OUTSIDE WORKSPACE"
            if not confirmed:
                tag += f"  UNCONFIRMED {age_s:.0f}s"
            _put_label(img, tag, (u + 18, v + 4), color, 0.44)

        return marks

    def _draw_tcp(self, img, tcp, T, K) -> None:
        uv = project(tcp[None, :3], T, K)[0]
        if not np.isfinite(uv).all():
            return
        p = (int(uv[0]), int(uv[1]))
        cv2.drawMarker(img, p, (0, 255, 255), cv2.MARKER_CROSS, 22, 2)
        _put_label(img, f"TCP z={tcp[2]:.3f}", (p[0] + 12, p[1] + 16), (0, 255, 255), 0.44)

    def _draw_legend(self, img, marks: list[Mark]) -> None:
        h = img.shape[0]
        n_badges = sum(mark.extra["projected_in_image"] for mark in marks)
        _put_label(
            img, f"{n_badges} badges | {len(marks)} tracked objects | grid = 5 cm",
            (10, h - 12), _WHITE, 0.42,
        )

    # ── helpers ──────────────────────────────────────────────────────────

    def _in_workspace(self, pos) -> bool | None:
        ws = self.workspace
        try:
            lo, hi = ws["min"], ws["max"]
            return all(float(lo[i]) <= float(pos[i]) <= float(hi[i]) for i in range(3))
        except (KeyError, TypeError, ValueError, IndexError):
            return None

    # ── the text half of the interface ───────────────────────────────────

    @staticmethod
    def describe(marks: list[Mark]) -> str:
        """Text key for the annotated image (VIA pairs both).

        The image alone is ambiguous under JPEG compression; the key makes
        every mark referable and gives the metric values exactly.
        """
        if not marks:
            return (
                "Annotated view: no tracked-object projections available for this frame. "
                "This does not establish that the scene is empty."
            )
        lines = [
            "Annotated view: badges mark tracked positions projected inside this camera image. "
            "Projection does not establish current visibility or exclude occlusion. "
            "Entries outside the image have no badge. Coordinates are base-frame metres (x forward, "
            "y left, z up). Workspace bounds and any configured grasp band "
            "do not verify inverse kinematics or grasp reachability."
        ]
        for m in marks:
            flag = "  [outside the workspace AABB]" if m.extra.get("in_workspace") is False else ""
            if m.extra.get("projected_in_image") is False:
                flag += "  [projects outside this camera image; no badge]"
            if not m.confirmed:
                flag += "  [UNCONFIRMED: one stale sighting, may not be real]"
            lines.append(
                f"  {m.index}. {m.label} at ({m.position[0]:.3f}, "
                f"{m.position[1]:.3f}, {m.position[2]:.3f}){flag}"
            )
        return "\n".join(lines)


def _valid_band(value) -> tuple[float, float] | None:
    try:
        if value is None or len(value) != 2:
            return None
        low, high = map(float, value)
        return (low, high) if np.isfinite([low, high]).all() and low < high else None
    except (TypeError, ValueError):
        return None


def configured_grasp_band(grasp_cfg) -> tuple[float, float] | None:
    """Do not substitute a different arm's display range for missing config."""
    return _valid_band((grasp_cfg.get("reach_x_min"), grasp_cfg.get("reach_x_max")))


def annotate_frame(runtime, **kwargs) -> tuple[np.ndarray | None, list[Mark]]:
    """Convenience wrapper: build the annotated view from a live SkillRuntime."""
    frame = runtime.last_frame
    if frame is None:
        return None, []
    cfg = runtime.cfg
    grasp_cfg = cfg.grasp if hasattr(cfg, "grasp") else {}
    # `Cfg` is an attr/dict hybrid but NOT iterable, so dict(...) on a nested
    # block raises "'Cfg' object is not iterable". `_data` is the sanctioned
    # accessor for the raw mapping (same idiom as apps/mcp_server.py).
    ws_raw = cfg.safety.get("workspace", {}) or {}
    workspace = dict(getattr(ws_raw, "_data", ws_raw))
    vi = VisualInterface(
        extrinsics=runtime.extrinsics,
        workspace=workspace,
        table_z=float(cfg.safety.get("table_z", 0.0)),
        reach_x=configured_grasp_band(grasp_cfg),
    )
    try:
        tcp = runtime._tcp()
    except Exception:
        tcp = None
    beliefs = []
    try:
        beliefs = list(runtime.beliefs.all())
    except Exception:
        pass
    return vi.render(frame, beliefs=beliefs, tcp=tcp, **kwargs)
