"""Live perception viewer: what the demo's camera stack actually sees.

    cascade-view --camera l515                            # RGB + depth colormap
    cascade-view --camera l515 --detect                   # + the demo detector's overlays
    cascade-view --camera l515 --detect --classes "red cup,bowl"   # find these only

The detector is the one the demo config names for that camera
(`apps.demo._make_detector`: the brain detector from configs/demo.yaml, the
mock detector on the mock camera).

Left pane: RGB with detection boxes/labels (and center-pixel depth readout).
Right pane: depth colormap (JET, near=red) with invalid pixels black.
Keys: q/ESC quit, s save a PNG snapshot next to the cwd, d toggle detection.
"""

from __future__ import annotations

import argparse
import sys
import time

import cv2
import numpy as np

from ..config import load_profile
from ..perception.camera_base import make_camera
from .live_view import depth_colormap, draw_detections, draw_hud


def make_viewer_detector(camera: str):
    """The detector the demo config builds for `camera` (camera-pinned
    detectors win, as in the demo). Named objects are passed per call as
    `detect(frame, classes)`: refer queries for the brain detector."""
    from ..config import load_demo_config
    from .demo import _make_detector

    return _make_detector(load_demo_config(camera=camera))


def viewer_detections(detector, frame, classes):
    """One display frame's detections. A detector whose CLASS defines
    `detect_nowait` (the brain detector) answers from its cache and asks the
    brain in the background, so a multi-second brain call never freezes the
    window; others run `detect` inline."""
    nowait = getattr(type(detector), "detect_nowait", None)
    if nowait is not None:
        return nowait(detector, frame, classes)
    return detector.detect(frame, classes=classes)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="live camera/perception viewer")
    p.add_argument("--camera", default="l515")
    p.add_argument("--detect", action="store_true",
                   help="overlay the detections of the demo config's detector for this camera")
    p.add_argument("--classes", default=None,
                   help="comma-separated object names to look for (default: everything on the table)")
    p.add_argument("--scale", type=float, default=0.75, help="display scale")
    args = p.parse_args(argv)

    classes = [c.strip() for c in args.classes.split(",") if c.strip()] if args.classes else None

    cam = make_camera(load_profile("cameras", args.camera))
    cam.open()
    cam.warm_up(5)
    # After the camera streams, as in the demo: loading CUDA (the SAM warm-up)
    # before the depth cameras stream reset their USB link on the WRC rig.
    try:
        detector = make_viewer_detector(args.camera) if args.detect else None
    except BaseException:
        cam.close()
        raise
    win = f"cascade :: {args.camera}"

    detect_on = detector is not None
    fps, t_prev = 0.0, time.monotonic()
    try:
        # Inside the try: a headless OpenCV build raises here, and the camera
        # and the detector must still be released.
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        print(f"[cascade-view] streaming {args.camera}; q/ESC to quit", file=sys.stderr)
        while True:
            frame = cam.get_frame()
            now = time.monotonic()
            fps = 0.9 * fps + 0.1 * (1.0 / max(now - t_prev, 1e-6))
            t_prev = now

            rgb = frame.rgb.copy()
            dets = []
            if detect_on and detector is not None:
                try:
                    dets = viewer_detections(detector, frame, classes)
                except Exception as e:
                    detect_on = False
                    print(f"[cascade-view] detection disabled: {e}", file=sys.stderr)
            draw_detections(rgb, dets)

            h, w = rgb.shape[:2]
            cx, cy = w // 2, h // 2
            center_txt = "no depth"
            if frame.has_depth:
                d = float(frame.depth_m[cy, cx])
                center_txt = f"center {d:.2f} m" if d > 0 else "center invalid"
            cv2.drawMarker(rgb, (cx, cy), (0, 255, 0), cv2.MARKER_CROSS, 18, 2)
            draw_hud(rgb, [
                f"{args.camera}  {w}x{h}  {fps:4.1f} fps",
                f"depth: {frame.depth_source}  |  {center_txt}",
                f"objects: {len(dets)}" + ("" if detect_on else "  (detection off)"),
            ])

            if frame.has_depth:
                panel = np.hstack([rgb, depth_colormap(frame.depth_m)])
            else:
                panel = rgb
            if args.scale != 1.0:
                panel = cv2.resize(panel, None, fx=args.scale, fy=args.scale)
            cv2.imshow(win, panel)

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
            if key == ord("s"):
                name = time.strftime("wrc_view_%H%M%S.png")
                cv2.imwrite(name, panel)
                print(f"[cascade-view] saved {name}", file=sys.stderr)
            if key == ord("d"):
                detect_on = not detect_on and detector is not None
    finally:
        cam.close()
        close = getattr(type(detector), "close", None)
        if close is not None:
            close(detector)  # the brain detector joins its refreshes and SAM warm-up
        try:
            cv2.destroyAllWindows()
        except cv2.error:
            pass  # headless build: no window exists, and namedWindow's error is the one to show
    return 0


if __name__ == "__main__":
    sys.exit(main())
