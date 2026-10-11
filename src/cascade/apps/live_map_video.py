"""Live map video: what nvblox sees, next to what the cameras see.

A demo/diagnostic recorder that sits beside a running cascade MCP server and
composes, frame by frame:

    +------------------------+------------------------+
    | fixed camera (annotated stream)                  | wrist camera ...
    +------------------------+------------------------+
    | nvblox occupied voxels, 3-D (slowly orbiting)    | top-down height map + objects
    +------------------------+------------------------+

It only WATCHES. The server owns the arm, the cameras and the occupancy
bridge; recording must never change any of them:

- The bridge is reached through `ReadOnlyBridge`, which forwards `probe` and
  `query` and raises on everything else (`clear`, `integrate_*`): a recorder
  that could clear the map could make the safety gate's next refresh see an
  empty world.
- Cameras come from the live-view server's MJPEG streams (`/stream/<cam>`),
  not from the devices (one process may own a RealSense).
- A camera whose stream stops is drawn as MISSING, never as its last good
  frame: a frozen picture next to a live map would misreport the scene.

The renderer is a small numpy z-buffered point splatter (no OpenGL on the
Spark's snap browser stack, no Open3D wheel for this Python): voxel centres
are projected through a virtual OpenCV pinhole camera and drawn far-to-near,
so nearer voxels hide farther ones.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

#: The only bridge actions a recorder may send. Everything else writes the map.
READ_ONLY_ACTIONS = frozenset({"probe", "query"})

#: Panel background (RGB), the deck's dark slate.
BACKGROUND = (17, 20, 24)

#: Height colour range (metres, base frame): table level to above the spools.
Z_RANGE = (-0.02, 0.35)


class ReadOnlyBridge:
    """Wrap an occupancy wire client so that it can only read the map."""

    def __init__(self, wire):
        self._wire = wire

    def request(self, packet: dict, **kw) -> dict:
        action = packet.get("action")
        if action not in READ_ONLY_ACTIONS:
            raise PermissionError(f"the live-map recorder is read-only; refused bridge action {action!r}")
        return self._wire.request(packet, **kw)

    def occupied_points(self, region_min, region_max) -> np.ndarray:
        reply = self.request({"action": "query",
                              "region_min": np.asarray(region_min, np.float32),
                              "region_max": np.asarray(region_max, np.float32)})
        pts = np.asarray(reply.get("points", np.zeros((0, 3))), np.float32)
        return pts.reshape(-1, 3)


@dataclass
class View:
    """A virtual pinhole camera (OpenCV axes: x right, y down, z forward).

    `fov_deg` is horizontal. World up is +z (the robot base frame), so seen
    from the front (+x looking back at the base) the robot's left (+y) is on
    screen right, as it is for a person facing the robot.
    """

    eye: np.ndarray
    target: np.ndarray
    size: tuple[int, int] = (640, 360)
    fov_deg: float = 60.0
    R: np.ndarray = field(init=False)
    f: float = field(init=False)

    def __post_init__(self):
        eye, target = np.asarray(self.eye, float), np.asarray(self.target, float)
        forward = target - eye
        forward /= np.linalg.norm(forward)
        right = np.cross(forward, [0.0, 0.0, 1.0])
        if np.linalg.norm(right) < 1e-9:  # looking straight down/up: pick +y as screen-up
            right = np.cross(forward, [0.0, 1.0, 0.0])
        right /= np.linalg.norm(right)
        down = np.cross(forward, right)
        self.eye = eye
        self.R = np.stack([right, down, forward])
        self.f = (self.size[0] / 2) / np.tan(np.radians(self.fov_deg) / 2)

    def project(self, points: np.ndarray):
        pc = (np.asarray(points, float) - self.eye) @ self.R.T
        z = pc[:, 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            u = self.f * pc[:, 0] / z + self.size[0] / 2
            v = self.f * pc[:, 1] / z + self.size[1] / 2
        return u, v, z


def orbit_view(t_s: float, *, target=(0.30, 0.0, 0.12), radius=1.15, height=0.62,
               deg_per_s=6.0, start_deg=200.0, size=(640, 360)) -> View:
    """A slow orbit around the workspace (a full turn per minute at 6 deg/s)."""
    a = np.radians(start_deg + deg_per_s * t_s)
    target = np.asarray(target, float)
    eye = target + np.array([radius * np.cos(a), radius * np.sin(a), height])
    return View(eye=eye, target=target, size=size, fov_deg=55.0)


def height_colors(z: np.ndarray) -> np.ndarray:
    """Viridis-like RGB colours by height (uint8, N x 3)."""
    t = np.clip((np.asarray(z, float) - Z_RANGE[0]) / (Z_RANGE[1] - Z_RANGE[0]), 0, 1)
    lut = cv2.applyColorMap(np.arange(256, dtype=np.uint8)[:, None], cv2.COLORMAP_VIRIDIS)[:, 0, ::-1]
    return lut[(t * 255).astype(int)]


def render_points(points: np.ndarray, view: View, colors: np.ndarray | None = None, *,
                  point_px: int = 3, background=BACKGROUND) -> np.ndarray:
    """Z-buffered square splats of `points` (RGB uint8, H x W x 3).

    Points are drawn far-to-near, so the nearest one at a pixel wins. Points
    at or behind the virtual camera are dropped.
    """
    w, h = view.size
    img = np.empty((h, w, 3), np.uint8)
    img[:] = background
    pts = np.asarray(points, float).reshape(-1, 3)
    if len(pts) == 0:
        return img
    if colors is None:
        colors = height_colors(pts[:, 2])
    u, v, z = view.project(pts)
    ok = (z > 0.02) & np.isfinite(u) & np.isfinite(v)
    half = point_px // 2
    ok &= (u >= -half) & (u < w + half) & (v >= -half) & (v < h + half)
    idx = np.nonzero(ok)[0]
    idx = idx[np.argsort(-z[idx])]  # far first
    ui, vi = np.round(u[idx]).astype(int), np.round(v[idx]).astype(int)
    for i, x, y in zip(idx, ui, vi):
        img[max(0, y - half):max(0, y + half + 1), max(0, x - half):max(0, x + half + 1)] = colors[i]
    return img


def draw_box(img: np.ndarray, view: View, lo, hi, color=(118, 185, 0), thickness=1) -> None:
    """Wireframe of an axis-aligned box (e.g. the safety workspace)."""
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    corners = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
    u, v, z = view.project(corners)
    for a in range(8):
        for b in range(a + 1, 8):
            if np.sum(corners[a] != corners[b]) == 1 and z[a] > 0.02 and z[b] > 0.02:
                cv2.line(img, (int(u[a]), int(v[a])), (int(u[b]), int(v[b])), color, thickness, cv2.LINE_AA)


def topdown(points: np.ndarray, lo, hi, objects=(), size=(640, 360), px_per_m=None) -> np.ndarray:
    """Top-down height map of the occupied voxels (max height per cell), the
    workspace rectangle and the agent's visible objects, robot base marked.
    Screen up = robot forward (+x), screen left = robot left (+y)."""
    w, h = size
    img = np.empty((h, w, 3), np.uint8)
    img[:] = BACKGROUND
    x0, x1, y0, y1 = -0.15, 0.75, -0.5, 0.5
    s = px_per_m or min(h / (x1 - x0), w / (y1 - y0))
    cx = w / 2

    def to_px(x, y):
        return int(cx - y * s), int(h - (x - x0) * s)

    pts = np.asarray(points, float).reshape(-1, 3)
    if len(pts):
        order = np.argsort(pts[:, 2])  # highest drawn last
        cols = height_colors(pts[order, 2])
        r = max(1, int(0.005 * s))
        for p, c in zip(pts[order], cols):
            px, py = to_px(p[0], p[1])
            img[max(0, py - r):py + r + 1, max(0, px - r):px + r + 1] = c
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    cv2.rectangle(img, to_px(hi[0], hi[1]), to_px(lo[0], lo[1]), (118, 185, 0), 1, cv2.LINE_AA)
    cv2.circle(img, to_px(0.0, 0.0), 6, (255, 255, 255), -1, cv2.LINE_AA)
    cv2.putText(img, "base", (to_px(0.0, 0.0)[0] + 8, to_px(0.0, 0.0)[1] + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                (230, 230, 230), 1, cv2.LINE_AA)
    for ob in objects:
        pos = ob.get("position") if isinstance(ob, dict) else None
        if pos is None or len(pos) < 2:
            continue
        px, py = to_px(float(pos[0]), float(pos[1]))
        cv2.circle(img, (px, py), 7, (255, 0, 200), 2, cv2.LINE_AA)
        cv2.putText(img, str(ob.get("label", "?"))[:18], (px + 9, py - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                    (255, 180, 240), 1, cv2.LINE_AA)
    return img


def camera_panel(img_rgb: np.ndarray | None, size, name: str, note: str = "no frame") -> np.ndarray:
    w, h = size
    if img_rgb is None:
        out = np.empty((h, w, 3), np.uint8)
        out[:] = (30, 30, 34)
        cv2.putText(out, f"{name}: {note}", (16, h // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (230, 120, 120), 1,
                    cv2.LINE_AA)
        return out
    return cv2.resize(img_rgb, (w, h), interpolation=cv2.INTER_AREA)


def compose(panels, titles, tile=(640, 360)) -> np.ndarray:
    """2 x 2 grid of panels (RGB), each resized to `tile` with its title."""
    w, h = tile
    out = np.zeros((2 * h, 2 * w, 3), np.uint8)
    for i, (p, title) in enumerate(zip(panels, titles)):
        p = cv2.resize(np.asarray(p, np.uint8), (w, h), interpolation=cv2.INTER_AREA)
        y, x = (i // 2) * h, (i % 2) * w
        out[y:y + h, x:x + w] = p
        if title:
            (tw, th), _ = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            cv2.rectangle(out, (x, y), (x + tw + 14, y + th + 12), (0, 0, 0), -1)
            cv2.putText(out, title, (x + 7, y + th + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (118, 185, 0), 1,
                        cv2.LINE_AA)
    return out


def iter_mjpeg(stream):
    """Yield JPEG payloads from a multipart/x-mixed-replace stream that
    carries Content-Length headers (cascade's live-view server does)."""
    length = None
    while True:
        line = stream.readline()
        if not line:
            return
        line = line.strip()
        if line.lower().startswith(b"content-length:"):
            length = int(line.split(b":", 1)[1])
        elif not line and length is not None:
            data = stream.read(length)
            if len(data) < length:
                return
            yield data
            length = None
