"""Synthetic camera for hand-eye calibration: no hardware, real pixels.

Used by ``--dry-run`` and the end-to-end tests. Instead of synthesising
(FK, PnP) pairs directly (WRC's dry-run), this RENDERS the ArUco marker into
an image from a known ground-truth geometry, so a dry run exercises the same
detector, PnP, solver, record and loader the rig uses -- the only fake parts
are the photons and the motors.
"""

from __future__ import annotations

import numpy as np

from .aruco import aruco_dictionary, marker_object_points
from .frames import se3_inv


def render_marker(K, image_size, T_marker2cam, *, size_m: float = 0.10,
                  marker_id: int = 0, dictionary: str = "4x4_50",
                  background: int = 255, px_per_cell: int = 40,
                  margin_px: int = 4) -> np.ndarray | None:
    """BGR image of the marker at ``T_marker2cam``, or None when not visible.

    Not visible = behind the camera, printed face turned away, or any corner
    outside the image (a partially visible marker is undetectable anyway).
    The marker's OWN black border is mapped onto the projected corners, so
    ``size_m`` is the black-square side exactly as the operator measures it.
    """
    import cv2

    w, h = int(image_size[0]), int(image_size[1])
    T = np.asarray(T_marker2cam, dtype=float)
    obj = marker_object_points(size_m)
    pts_cam = obj @ T[:3, :3].T + T[:3, 3]
    if np.any(pts_cam[:, 2] <= 1e-3):
        return None
    # Face visibility: the marker's +z must point back towards the camera.
    if float(np.dot(T[:3, 2], -T[:3, 3])) <= 0.0:
        return None
    K = np.asarray(K, dtype=float)
    uv = (pts_cam @ K.T)
    uv = uv[:, :2] / uv[:, 2:3]
    if (np.any(uv[:, 0] < margin_px) or np.any(uv[:, 0] > w - 1 - margin_px)
            or np.any(uv[:, 1] < margin_px) or np.any(uv[:, 1] > h - 1 - margin_px)):
        return None
    d = aruco_dictionary(dictionary)
    cells = int(d.markerSize) + 2
    side = cells * int(px_per_cell)
    bitmap = cv2.aruco.generateImageMarker(d, int(marker_id), side)
    # Pixel i covers [i-0.5, i+0.5]: the black square's outer edge is at -0.5.
    src = np.array([[-0.5, -0.5], [side - 0.5, -0.5],
                    [side - 0.5, side - 0.5], [-0.5, side - 0.5]], dtype=np.float32)
    H = cv2.getPerspectiveTransform(src, uv.astype(np.float32))
    warped = cv2.warpPerspective(bitmap, H, (w, h), flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_CONSTANT, borderValue=background)
    return cv2.cvtColor(warped, cv2.COLOR_GRAY2BGR)


class SyntheticMarkerCamera:
    """A camera that sees one marker, placed by FK and a known hand-eye.

    eye_to_hand: camera fixed at ``T_hand_eye`` (= T_cam2base), marker on
                 the gripper at ``T_marker`` (= T_marker2gripper).
    eye_in_hand: camera on the gripper at ``T_hand_eye`` (= T_cam2gripper),
                 marker on the table at ``T_marker`` (= T_marker2base).

    ``tcp_pose()`` returns the CURRENT T_gripper2base (FK of the arm the
    session drives), so the image follows the arm like a real one would.
    ``noise_px`` adds Gaussian pixel noise; ``corrupt`` is a set of grab
    indices whose marker pose is displaced by ``corrupt_offset_m`` -- an
    injected outlier (a bumped mount, a misread), not noise.
    """

    has_depth = False

    def __init__(self, mode, T_hand_eye, T_marker, tcp_pose, *, K=None,
                 image_size=(1280, 720), size_m=0.10, marker_id=0,
                 dictionary="4x4_50", noise_px=0.0, seed=0,
                 corrupt=(), corrupt_offset_m=(0.0, 0.04, 0.0), serial="SYNTHETIC"):
        from .handeye import EYE_TO_HAND, MODES

        if mode not in MODES:
            raise ValueError(f"bad mode {mode!r}")
        self.mode = mode
        self._eth = mode == EYE_TO_HAND
        self.T_hand_eye = np.asarray(T_hand_eye, dtype=float)
        self.T_marker = np.asarray(T_marker, dtype=float)
        self._tcp = tcp_pose
        w, h = int(image_size[0]), int(image_size[1])
        self.image_size = (w, h)
        self.K = (np.array([[0.9 * w, 0.0, w / 2.0], [0.0, 0.9 * w, h / 2.0], [0, 0, 1.0]])
                  if K is None else np.asarray(K, dtype=float))
        self.size_m, self.marker_id, self.dictionary = size_m, marker_id, dictionary
        self.noise_px = float(noise_px)
        self._rng = np.random.default_rng(seed)
        self.corrupt = set(corrupt)
        self.corrupt_offset_m = np.asarray(corrupt_offset_m, dtype=float)
        self.serial = serial
        self.dist_coeffs = np.zeros(5)
        self.grabs = 0
        self.opened = False

    def marker_in_camera(self) -> np.ndarray:
        G = np.asarray(self._tcp(), dtype=float)
        if self._eth:  # G Y = X M  ->  M = X^-1 G Y
            return se3_inv(self.T_hand_eye) @ G @ self.T_marker
        return se3_inv(self.T_hand_eye) @ se3_inv(G) @ self.T_marker  # G X M = Z

    def open(self):
        self.opened = True

    def close(self):
        self.opened = False

    def get_frame(self):
        from ..types import Frame

        M = self.marker_in_camera()
        if self.grabs in self.corrupt:
            M = M.copy()
            M[:3, 3] += self.corrupt_offset_m
        self.grabs += 1
        img = render_marker(self.K, self.image_size, M, size_m=self.size_m,
                            marker_id=self.marker_id, dictionary=self.dictionary)
        w, h = self.image_size
        if img is None:
            img = np.full((h, w, 3), 255, np.uint8)
        if self.noise_px > 0:
            noisy = img.astype(np.float32) + self._rng.normal(0.0, self.noise_px * 4.0, img.shape)
            img = np.clip(noisy, 0, 255).astype(np.uint8)
        return Frame(rgb=img, depth_m=None, K=self.K.copy())
