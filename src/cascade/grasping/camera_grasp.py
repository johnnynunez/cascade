"""Camera-frame grasp planning from a segmentation mask (rebot_grasp style).

Ported from Seeed's WRC fork (`src/wrc_demo/grasping/camera_grasp.py`), which
mirrors `rebot_grasp/utils/ordinary_grasp.py::estimate_grasp` and
`rebot_grasp/utils/transforms.py::transform_grasp_pose_to_base` -- the
pipeline Seeed runs on its B601 rig (Orbbec Gemini 2 overhead camera, ~30 deg
tilt):

  mask --minAreaRect--> rect corners + short-edge direction (pixels)
    -> depth quantile inside the mask (median) --back-project--> p_cam
    -> approach_cam = -normalize(p_cam)        (along the camera's line of sight)
    -> open_cam = short edge lifted to 3D, made perpendicular to the approach
    -> reBot TCP axes [tool-forward, open, third] (`_grasp_axes_to_rebot_...`)
    -> T_grasp_base = T_cam2base @ T_grasp_cam, canonicalised (smaller roll;
       a near-vertical dive is snapped to exactly vertical)
    -> TCP pushed `insertion_depth_m` along the approach INTO the object
    -> laid out in the arm's `tool_axis_order`

Why a camera-frame plan at all (WRC's argument): with a tilted camera the
gripper dives along what the camera actually sees instead of a hard-coded
base-frame vector, and a 2D rectangle on the mask plus ONE depth statistic is
steadier than a 3D OBB fitted to thousands of noisy depth points.

In cascade this is an EXPLICITLY selected backend (`grasp.backend:
camera_frame`, `SkillRuntime._camera_frame_candidates`); it never replaces
GraspGen-X. Its candidates are ordinary `Grasp`s and go through the same
memory re-ranking, jaw-width check, IK and harness vetting as every other
backend's.

Cascade changes against WRC (tests/test_camera_grasp.py):
  * `axis_order` (the arm's `tool_axis_order`) -- WRC hard-coded the reBot
    columns, which rotates every grasp 90 deg on an `open_down` arm;
  * `finger_drop_m` is an argument (config `grasp.camera_frame.finger_drop_m`)
    instead of the `WRCSIM_FINGER_DROP_M` environment variable;
  * `camera_frame_grasps()` adds the runtime-facing checks: the mask must
    belong to the frame's depth image, and the planned surface point must
    land on the localized object (`max_fix_offset_m`) -- a mismatched frame or
    transform is refused rather than aimed at.
"""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from ..types import Grasp
from .obb_grasp import tool_rotation

#: TCP pushed this far INTO the object along the approach (rebot_grasp
#: `grasp_pipeline.grasp.insertion_depth_m`): the reBot RS fingers extend
#: ~1.5 cm past the TCP, so a TCP on the surface pinches the rim. NOTE: WRC's
#: live demo.yaml ran 0.0 because its hand-eye JSON had a 2 cm camera-frame
#: compensation baked in -- tune this together with the extrinsics.
DEFAULT_INSERTION_DEPTH_M = 0.015
#: depth statistic inside the mask (median; rebot_grasp default.yaml)
DEFAULT_DEPTH_QUANTILE = 0.50
#: jaw clearance added to the measured short edge (rebot_grasp selector)
DEFAULT_WIDTH_PAD_M = 0.015
#: mask pixels this much NEARER the camera than the mask median are dropped
#: (WRC: the 30 mm longer fingers land inside the YOLO mask and drag the
#: rectangle's centre toward the gripper)
DEFAULT_FINGER_DROP_M = 0.030


# ── small math helpers (rebot_grasp, minimal adaptation) ────────────────────


def _normalize(v: np.ndarray) -> Optional[np.ndarray]:
    n = float(np.linalg.norm(v))
    if n < 1e-8:
        return None
    return (np.asarray(v, dtype=np.float64) / n).astype(np.float64)


def _nearest_rotation_matrix(R: np.ndarray) -> np.ndarray:
    R = np.asarray(R, dtype=np.float64)
    if not np.all(np.isfinite(R)):
        raise ValueError("rotation matrix has non-finite entries")
    U, _, Vt = np.linalg.svd(R)
    R_ortho = U @ Vt
    if np.linalg.det(R_ortho) < 0.0:
        U[:, -1] *= -1.0
        R_ortho = U @ Vt
    return R_ortho


def _backproject(u: float, v: float, z_m: float, K: np.ndarray) -> np.ndarray:
    """Pixel (u, v) at depth z -> camera frame (OpenCV: +x right, +y down, +z fwd)."""
    return np.array([(u - K[0, 2]) * z_m / K[0, 0], (v - K[1, 2]) * z_m / K[1, 1], z_m],
                    dtype=np.float64)


def _pixel_vec_to_3d(vec_uv: np.ndarray, z_m: float, K: np.ndarray) -> np.ndarray:
    """A pixel displacement -> metric camera-frame displacement at depth z."""
    fx = max(float(K[0, 0]), 1e-6)
    fy = max(float(K[1, 1]), 1e-6)
    return np.array([float(vec_uv[0]) * z_m / fx, float(vec_uv[1]) * z_m / fy, 0.0])


_ROT_X_PI = np.diag([1.0, -1.0, -1.0])


def _rotation_matrix_to_euler_zyx(R: np.ndarray) -> np.ndarray:
    R = _nearest_rotation_matrix(R)
    sy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    if sy > 1e-6:
        return np.array([np.arctan2(R[2, 1], R[2, 2]), np.arctan2(-R[2, 0], sy),
                         np.arctan2(R[1, 0], R[0, 0])])
    return np.array([np.arctan2(-R[1, 2], R[1, 1]), np.arctan2(-R[2, 0], sy), 0.0])


def _canonicalize_parallel_gripper_tcp_rotation(R: np.ndarray) -> np.ndarray:
    """Of `R` and `R @ Rx(pi)` (grasp-equivalent for a symmetric parallel
    jaw) keep the smaller |roll|. Within 0.10 rad of a vertical dive the ZYX
    roll is numerically arbitrary (gimbal lock), so the tool-x is snapped to
    exactly vertical and the opening axis to its horizontal projection
    (WRC 2026-08-14). Input/output: reBot layout, column 0 = tool-forward.

    cascade's selector also tries each candidate's 180-degree twin
    (`selector._flip_twin`), so this choice affects the starting branch only.
    """
    R = _nearest_rotation_matrix(R)
    pitch = abs(float(_rotation_matrix_to_euler_zyx(R)[1]))
    if abs(pitch - np.pi / 2) < 0.10:
        R = R.copy()
        R[:, 0] = [0.0, 0.0, -1.0] if R[2, 0] < 0 else [0.0, 0.0, 1.0]
        h = R[:, 1].copy()
        h[2] = 0.0
        n = float(np.linalg.norm(h))
        R[:, 1] = h / n if n > 1e-9 else [1.0, 0.0, 0.0]
        R[:, 2] = np.cross(R[:, 0], R[:, 1])
        return R
    alt = R @ _ROT_X_PI
    roll = abs(float(_rotation_matrix_to_euler_zyx(R)[0]))
    alt_roll = abs(float(_rotation_matrix_to_euler_zyx(alt)[0]))
    return alt if alt_roll < roll else R


def _grasp_axes_to_rebot_tcp_rotation(grip_axis, open_axis, approach_axis) -> np.ndarray:
    """Vision axes [grip, open, approach(plane normal toward the camera)] ->
    reBot TCP [tool-forward = -approach, open, right-handed third], with the
    third axis kept on the grip axis's side."""
    g, o, a = _normalize(grip_axis), _normalize(open_axis), _normalize(approach_axis)
    if g is None or o is None or a is None:
        raise ValueError("grasp_axes_to_rebot_tcp_rotation: degenerate input")
    tcp_x = -a
    tcp_y = _normalize(o - float(np.dot(o, tcp_x)) * tcp_x)
    if tcp_y is None:
        tcp_y = np.array([0.0, 1.0, 0.0])
    tcp_z = _normalize(np.cross(tcp_x, tcp_y))
    if tcp_z is None:
        tcp_z = np.array([0.0, 0.0, 1.0])
    if float(np.dot(tcp_z, g)) < 0.0:
        tcp_y, tcp_z = -tcp_y, -tcp_z
    R = np.column_stack([tcp_x, tcp_y, tcp_z])
    if np.linalg.det(R) < 0.0:
        R[:, 2] *= -1.0
    return R


def _rect_from_mask(mask: np.ndarray) -> Optional[np.ndarray]:
    """`cv2.minAreaRect` corners (4, 2) of the mask's largest contour."""
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    if len(contour) < 3:
        return None
    return cv2.boxPoints(cv2.minAreaRect(contour.astype(np.float32))).astype(np.float32)


def _short_edge(rect: np.ndarray) -> tuple[np.ndarray, float]:
    best_vec = rect[1] - rect[0]
    best_len = float(np.linalg.norm(best_vec))
    for i in range(4):
        vec = rect[(i + 1) % 4] - rect[i]
        length = float(np.linalg.norm(vec))
        if length < best_len:
            best_vec, best_len = vec, length
    return best_vec.astype(np.float32), best_len


def _drop_finger_pixels(mask: np.ndarray, depth_m: np.ndarray, finger_drop_m: float):
    """Remove mask pixels more than `finger_drop_m` NEARER than the median."""
    if finger_drop_m <= 0:
        return mask
    ys, xs = np.nonzero(mask)
    if len(ys) <= 5:
        return mask
    ds = depth_m[ys, xs]
    valid = ds > 0
    if int(valid.sum()) <= 5:
        return mask
    keep = valid & (ds >= float(np.median(ds[valid])) - float(finger_drop_m))
    if int(keep.sum()) < 5:
        return mask
    clean = np.zeros_like(mask)
    clean[ys[keep], xs[keep]] = True
    return clean


# ── public API ──────────────────────────────────────────────────────────────


def plan_grasp_from_mask(
    mask: np.ndarray,
    depth_m: np.ndarray,
    K: np.ndarray,
    T_cam2base: np.ndarray,
    *,
    insertion_depth_m: float = DEFAULT_INSERTION_DEPTH_M,
    max_width_m: float = 0.09,
    width_pad_m: float = DEFAULT_WIDTH_PAD_M,
    depth_quantile: float = DEFAULT_DEPTH_QUANTILE,
    finger_drop_m: float = DEFAULT_FINGER_DROP_M,
    axis_order: str = "down_open",
    label: str = "",
    confidence: float = 1.0,
) -> list[Grasp]:
    """One base-frame parallel-jaw grasp from a mask, or [] if any input is
    missing/degenerate (no mask, no valid depth in it, no contour, no K, no
    transform). `Grasp.approach` is the tool's travel direction in the base
    frame, so `pregrasp_position(offset)` retreats along the line of sight.
    A grasp wider than `max_width_m` is returned with quality 0.2 x conf
    (annotated, not dropped -- the selector's width check refuses it).
    """
    if mask is None or depth_m is None or K is None or T_cam2base is None:
        return []
    mask = np.asarray(mask, dtype=bool)
    depth_m = np.asarray(depth_m)
    K = np.asarray(K, dtype=np.float64)
    T_cam2base = np.asarray(T_cam2base, dtype=np.float64)
    if mask.size == 0 or int(mask.sum()) < 5:
        return []
    if mask.shape != depth_m.shape:
        mask = cv2.resize(mask.astype(np.uint8), (depth_m.shape[1], depth_m.shape[0]),
                          interpolation=cv2.INTER_NEAREST).astype(bool)
    mask = _drop_finger_pixels(mask, depth_m, float(finger_drop_m))

    box = _rect_from_mask(mask)
    if box is None:
        return []
    short_vec_px, short_len_px = _short_edge(box)
    short_dir_px = _normalize(short_vec_px)
    if short_dir_px is None:
        return []
    center_px = box.mean(axis=0)

    zs = depth_m[mask]
    zs = zs[zs > 0]
    if len(zs) == 0:
        return []
    z_m = float(np.quantile(zs, depth_quantile))
    position_cam = _backproject(float(center_px[0]), float(center_px[1]), z_m, K)

    approach_cam = _normalize(-position_cam)
    if approach_cam is None:
        approach_cam = np.array([0.0, 0.0, -1.0])
    open_cam = _pixel_vec_to_3d(short_dir_px, z_m, K)
    open_cam = _normalize(open_cam - float(np.dot(open_cam, approach_cam)) * approach_cam)
    if open_cam is None:
        return []
    if open_cam[0] < 0:          # rebot_grasp canonicalises open_axis[0] > 0
        open_cam = -open_cam
    grip_cam = _normalize(np.cross(open_cam, approach_cam))
    if grip_cam is None:
        return []
    open_cam = _normalize(np.cross(approach_cam, grip_cam))
    if open_cam is None:
        return []
    try:
        tcp_R_cam = _grasp_axes_to_rebot_tcp_rotation(grip_cam, open_cam, approach_cam)
    except ValueError:
        return []

    T_cam = np.eye(4)
    T_cam[:3, :3] = tcp_R_cam
    T_cam[:3, 3] = position_cam
    T_base = T_cam2base @ T_cam
    R = _canonicalize_parallel_gripper_tcp_rotation(T_base[:3, :3])
    approach = R[:, 0].copy()
    position = T_base[:3, 3] + approach * float(insertion_depth_m)

    jaw_m = float(np.linalg.norm(_pixel_vec_to_3d(short_dir_px * short_len_px, z_m, K)))
    required = jaw_m + float(width_pad_m)
    feasible = required <= float(max_width_m)
    return [Grasp(
        position=position,
        rotation=tool_rotation(approach, R[:, 1], axis_order),
        width_m=float(required),
        approach=approach,
        quality=(1.0 if feasible else 0.2) * float(confidence),
        label=str(label),
    )]
