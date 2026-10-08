"""Hand-eye calibration: camera <-> robot extrinsics from an ArUco marker.

Ported from Seeed's WRC fork (``wrc_demo.calibration``) into cascade's
architecture. See docs/HANDEYE_CALIBRATION.md for the operator procedure and
what was (and was not) carried over.

Layout:

* ``frames``  -- SE(3) helpers (pure numpy).
* ``handeye`` -- joint SE(3) solver: closed-form seed, Levenberg-Marquardt
  with Huber weights, 3*MAD outlier rejection, restarts; one solver for both
  eye-to-hand (camera fixed, marker on the gripper) and eye-in-hand (camera
  on the wrist, marker fixed in the world).
* ``dataset`` -- schema-v1 JSON record + the acceptance gate the loader
  enforces (``load_hand_eye`` refuses a rejected record, like
  ``perception.calibration.load_extrinsic`` does for the Kabsch fit).
* ``aruco``   -- cv2 (5.x) ArUco detection + square-marker PnP; intrinsics
  come from the camera's own ``Frame.K``.
* ``session`` -- the collection loop: every preset pose vetted by the arm's
  SafetyHarness and executed through ``SafeArm`` (never a raw backend).

Nothing in this package imports a camera SDK, an arm SDK or pinocchio at
module import time; heavy imports stay inside the functions that need them.
"""
