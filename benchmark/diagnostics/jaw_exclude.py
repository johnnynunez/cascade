"""Does excluding the finger-finger contact pair fix the 2.35 mm?

The premise I was carrying -- "CoACD decomposition inflates the hulls" -- does
not survive checking. CoACD appears in exactly ONE line of one evidence README
and nowhere in the pipeline. What the asset actually has:

    gripper_left_convex.stl    216 triangles, 110 unique vertices
    gripper_right_convex.stl   same

ONE convex mesh per finger, not a decomposition. And the MJCF has no
<contact><exclude> section at all, so the two fingers are free to collide with
each other.

The hull is 2.91x the visual volume because a C-shaped finger's convex hull
fills the C. MuJoCo convexifies mesh geoms anyway, so a mesh collider can never
represent that concavity -- decomposing it would be one answer, but for a
PARALLEL JAW there is a simpler and more honest one: the two fingers are
mechanically constrained by the mechanism and their joint limits. They should
never be tested against each other at all.

That is what <contact><exclude> is for: one line, no new geometry, no
re-authored meshes.

Test it head to head in plain MuJoCo (fast, deterministic, no Isaac):
command a series of openings on the stock model and on a copy with the pair
excluded, and compare achieved joint positions against the commanded targets.

Set REBOT_MJCF_ROOT to the reBot-Isaacsim mjcf/rebot_devarm directory.
"""
import os
import sys
from pathlib import Path

import numpy as np

try:
    import mujoco
except ImportError:
    print("mujoco not available in this interpreter")
    sys.exit(1)

REBOT_ROOT = Path(os.environ.get(
    "REBOT_MJCF_ROOT",
    Path(__file__).resolve().parents[3] / "reBot-Isaacsim/mjcf/rebot_devarm",
)).expanduser()
SRC = REBOT_ROOT / "rebot_devarm.xml"
if not SRC.is_file():
    raise SystemExit("Set REBOT_MJCF_ROOT to the directory containing rebot_devarm.xml.")
xml = SRC.read_text()

# build the variant with the finger pair excluded
if "<contact>" in xml:
    print("model already has a <contact> section; inspect before patching")
    sys.exit(1)

EXCLUDE = """  <contact>
    <exclude body1="gripper_left" body2="gripper_right"/>
  </contact>
"""
patched = xml.replace("</mujoco>", EXCLUDE + "</mujoco>")

work = Path("/tmp/jaw_test")
work.mkdir(exist_ok=True)
# mujoco resolves meshdir relative to the model file, so keep it in place
stock_p = SRC.parent / "_stock_tmp.xml"
excl_p = SRC.parent / "_excluded_tmp.xml"
stock_p.write_text(xml)
excl_p.write_text(patched)


def run(path, label):
    m = mujoco.MjModel.from_xml_path(str(path))
    d = mujoco.MjData(m)

    li = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "joint_left")
    ri = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "joint_right")
    la = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, "joint_left")
    ra = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, "joint_right")
    lq, rq = m.jnt_qposadr[li], m.jnt_qposadr[ri]
    lhi = m.jnt_range[li][1]
    rhi = m.jnt_range[ri][1]

    print(f"\n=== {label} ===")
    print(f"{'frac':>5} {'cmd L':>8} {'cmd R':>8} {'got L':>8} {'got R':>8} "
          f"{'err L':>8} {'err R':>8} {'ncon':>5}")
    worst = 0.0
    for frac in (0.0, 0.2, 0.4, 0.6, 0.8, 1.0):
        mujoco.mj_resetData(m, d)
        tl, tr = frac * lhi, frac * rhi
        d.ctrl[la], d.ctrl[ra] = tl, tr
        for _ in range(3000):                 # settle
            mujoco.mj_step(m, d)
        gl, gr = d.qpos[lq], d.qpos[rq]
        el, er = (gl - tl) * 1000, (gr - tr) * 1000
        worst = max(worst, abs(el), abs(er))
        print(f"{frac:>5.1f} {tl*1000:>8.2f} {tr*1000:>8.2f} "
              f"{gl*1000:>8.2f} {gr*1000:>8.2f} {el:>+8.3f} {er:>+8.3f} "
              f"{d.ncon:>5d}")
    print(f"worst |error|: {worst:.3f} mm")
    return worst


try:
    w_stock = run(stock_p, "STOCK (fingers collide with each other)")
    w_excl = run(excl_p, "EXCLUDED (finger pair not tested)")

    print("\n" + "=" * 60)
    print(f"stock worst error    : {w_stock:.3f} mm")
    print(f"excluded worst error : {w_excl:.3f} mm")
    if w_excl < 0.5 and w_stock > 1.0:
        print("-> the finger-finger contact IS the cause, and excluding the")
        print("   pair fixes it without touching any mesh")
    elif w_excl < w_stock * 0.5:
        print("-> large improvement, but not fully clean; inspect further")
    else:
        print("-> excluding the pair does NOT fix it; the cause is elsewhere")
finally:
    stock_p.unlink(missing_ok=True)
    excl_p.unlink(missing_ok=True)
