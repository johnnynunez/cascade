"""ArUco marker detection + square-marker PnP for hand-eye calibration.

Ported from WRC ``calibration/aruco_session.py``, camera-agnostic: the
intrinsics come from the camera's own ``Frame.K`` (every cascade backend
fills it), distortion from the backend's optional ``dist_coeffs`` attribute
(none of the shipped backends expose one yet, so zeros -- see
docs/HANDEYE_CALIBRATION.md). WRC read ``K``/``D`` off its Orbbec class.

Changes from WRC, each because the original returned a wrong pose:

* Corner model. WRC declared the marker frame "x right, y DOWN, z toward
  the camera", which is not right-handed: its z actually pointed INTO the
  board. Here the OpenCV ArUco convention is used verbatim (centre origin,
  x right, y up, z out of the printed face), so ``SOLVEPNP_IPPE_SQUARE``
  can be used as designed (WRC abandoned it because, with its corner order,
  it returned tvec ~ 0).
* Planar ambiguity. A square marker has two PnP solutions; IPPE returns
  both and the better-reprojecting one is kept, with the ratio recorded
  (``ambiguity``: near 1 means the reading could have flipped -- the joint
  solver's outlier rejection is the backstop).

``cv2`` (opencv-python-headless) is a base dependency; it is still imported
lazily so importing ``cascade.calibration`` stays cheap.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

#: name -> cv2.aruco constant name. Keys are what profiles/CLIs spell.
ARUCO_DICTS = {
    "4x4_50": "DICT_4X4_50", "4x4_100": "DICT_4X4_100", "4x4_250": "DICT_4X4_250",
    "4x4_1000": "DICT_4X4_1000", "5x5_50": "DICT_5X5_50", "5x5_100": "DICT_5X5_100",
    "6x6_50": "DICT_6X6_50", "6x6_250": "DICT_6X6_250",
    "apriltag_16h5": "DICT_APRILTAG_16h5", "apriltag_36h11": "DICT_APRILTAG_36h11",
}


def aruco_dictionary(name: str):
    import cv2

    key = str(name).lower()
    if key not in ARUCO_DICTS:
        raise ValueError(f"unknown ArUco dictionary {name!r}; valid: {sorted(ARUCO_DICTS)}")
    return cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, ARUCO_DICTS[key]))


def marker_object_points(size_m: float) -> np.ndarray:
    """Marker corners in the marker frame, in cv2.aruco's TL, TR, BR, BL order."""
    h = float(size_m) / 2.0
    return np.array([[-h, h, 0.0], [h, h, 0.0], [h, -h, 0.0], [-h, -h, 0.0]])


def camera_dist_coeffs(camera) -> np.ndarray | None:
    """Distortion coefficients a camera backend exposes, else None (= zeros)."""
    d = getattr(camera, "dist_coeffs", None)
    return None if d is None else np.asarray(d, dtype=float).ravel()


@dataclass(frozen=True)
class MarkerDetection:
    marker_id: int
    corners_px: np.ndarray       # (4, 2) TL, TR, BR, BL
    T_marker2cam: np.ndarray     # (4, 4)
    reprojection_px: float       # RMS corner reprojection of the kept solution
    ambiguity: float = 0.0       # best/second reprojection error (1.0 = ambiguous)


class ArucoSession:
    """cv2 (4.7+/5.x ``ArucoDetector`` API) wrapper returning marker poses."""

    def __init__(self, dictionary: str = "4x4_50") -> None:
        import cv2

        self.dictionary = str(dictionary).lower()
        self._dict = aruco_dictionary(self.dictionary)
        params = cv2.aruco.DetectorParameters()
        # Sub-pixel corners: the pose is only as good as the corners.
        params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        self._detector = cv2.aruco.ArucoDetector(self._dict, params)

    def detect(self, bgr, K, D, size_m: float, target_id: int | None = None
               ) -> MarkerDetection | None:
        """First (or ``target_id``) marker's pose in the camera, or None."""
        import cv2

        corners, ids, _ = self._detector.detectMarkers(bgr)
        if ids is None or len(ids) == 0:
            return None
        ids = [int(i) for i in np.asarray(ids).ravel()]
        if target_id is None:
            pick = 0
        elif int(target_id) in ids:
            pick = ids.index(int(target_id))
        else:
            return None
        img_pts = np.asarray(corners[pick], dtype=np.float64).reshape(4, 2)
        obj = marker_object_points(size_m)
        K = np.asarray(K, dtype=np.float64).reshape(3, 3)
        D = (np.zeros(5) if D is None or np.asarray(D).size == 0
             else np.asarray(D, dtype=np.float64).ravel())
        n, rvecs, tvecs, errs = cv2.solvePnPGeneric(
            obj, img_pts, K, D, flags=cv2.SOLVEPNP_IPPE_SQUARE)
        if not n:
            return None
        errs = np.asarray(errs, dtype=float).ravel()
        best = int(np.argmin(errs))
        R, _ = cv2.Rodrigues(np.asarray(rvecs[best], dtype=np.float64))
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = np.asarray(tvecs[best], dtype=float).ravel()
        if T[2, 3] <= 0 or not np.all(np.isfinite(T)):
            return None
        ambiguity = 0.0
        if len(errs) > 1:
            other = float(np.min(np.delete(errs, best)))
            ambiguity = float(errs[best] / other) if other > 0 else 1.0
        return MarkerDetection(marker_id=ids[pick], corners_px=img_pts, T_marker2cam=T,
                               reprojection_px=float(errs[best]), ambiguity=ambiguity)

    def detect_frame(self, frame, size_m: float, *, D=None, target_id: int | None = None
                     ) -> MarkerDetection | None:
        """Same, with the intrinsics taken from the ``Frame`` (BGR + K)."""
        return self.detect(frame.rgb, frame.K, D, size_m, target_id=target_id)

    def draw(self, bgr, det: MarkerDetection, K=None, size_m: float | None = None) -> np.ndarray:
        """Operator overlay: marker outline, id, reprojection (a copy)."""
        import cv2

        out = np.array(bgr, copy=True)
        pts = np.round(det.corners_px).astype(int)
        cv2.polylines(out, [pts.reshape(-1, 1, 2)], True, (0, 255, 0), 2)
        cv2.putText(out, f"id={det.marker_id} {det.reprojection_px:.2f}px "
                         f"z={det.T_marker2cam[2, 3]:.3f}m",
                    (int(pts[0][0]), max(12, int(pts[0][1]) - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
        if K is not None and size_m is not None:
            rvec, _ = cv2.Rodrigues(det.T_marker2cam[:3, :3])
            cv2.drawFrameAxes(out, np.asarray(K, dtype=np.float64), np.zeros(5), rvec,
                              det.T_marker2cam[:3, 3], float(size_m) / 2)
        return out
