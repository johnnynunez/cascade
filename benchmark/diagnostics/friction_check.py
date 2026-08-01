"""The 3.5 mm residual does not decay. Is it friction, and does it accumulate?

The speed sweep produced a result that rules out the "fast motion shakes the
fingers" story:

    speed    peak |dq2|    finger dev        residual after 2 s
    slow      0.593 r/s    -3.49/+3.50 mm    -3.61/+3.61
    medium    1.085 r/s    +2.77/-2.75 mm    +3.30/-3.30
    fast      0.883 r/s    -1.29/+1.31 mm    -3.43/+3.47
    snap      0.919 r/s    +3.73/-3.70 mm    +3.63/-3.59

Peak deviation does NOT scale with joint speed (the FAST case is the smallest),
and every run settles to the same ~3.5 mm offset that never decays. A dynamic
overshoot would decay once the motion stops. A constant standing offset that
survives 2 s of stillness is a STATIC term -- the MJCF joint default carries
frictionloss=0.2, which is exactly a force the PD must overcome before the
joint will move at all.

Two properties distinguish friction from a broken drive, and both are testable:

  1. DIRECTION. Friction opposes the last motion, so the sign of the residual
     should flip when the arm sweeps the other way. A miscalibrated drive or a
     wrong target would keep the SAME sign regardless of direction.
  2. NO ACCUMULATION. Repeated sweeps should keep landing near the same
     magnitude rather than drifting further out each cycle.

Also worth knowing for the demo: does commanding the gripper again clear it?
If a fresh grip command pulls the fingers back onto target, the residual is
cosmetic for grasping -- the jaws re-close correctly before each pick.
"""
import sys
import time

sys.path.insert(0, "/home/johnny/Projects/demo/wrc_demo/src")

import numpy as np

from wrc_demo.sim.bridge_client import BridgeClient

GRIP_FRAC = 0.6
c = BridgeClient(port=8611)
c.connect()

c.request({"op": "gripper", "pos": GRIP_FRAC})
time.sleep(2.5)

PRE = f"""
import numpy as np
art = globals()["art"]
_n = list(art.dof_names)
LI, RI = _n.index("joint_left"), _n.index("joint_right")
_lo, _hi = [x.numpy()[0] for x in art.get_dof_limits()]
TL = _lo[LI] + {GRIP_FRAC} * (_hi[LI] - _lo[LI])
TR = _lo[RI] + {GRIP_FRAC} * (_hi[RI] - _lo[RI])
def q():
    return art.get_dof_positions().numpy()[0]
globals().update(dict(LI=LI, RI=RI, TL=TL, TR=TR, q=q))
print("ready")
"""
if "ready" not in c.request({"op": "exec", "code": PRE}).get("stdout", ""):
    print("setup failed")
    sys.exit(1)

base = np.asarray(c.request({"op": "state"})["q"], dtype=float)


def err():
    o = c.request({"op": "exec", "code":
                   "c=q()\nprint(f'{(c[LI]-TL)*1000:+.3f} {(c[RI]-TR)*1000:+.3f}')\n"})
    return [float(x) for x in o.get("stdout", "0 0").split()]


def sweep_to(a2, steps=6, dwell=0.10):
    start = float(np.asarray(c.request({"op": "state"})["q"], dtype=float)[1])
    for a in np.linspace(start, a2, steps):
        t = base.copy()
        t[1] = float(a)
        c.request({"op": "set_joints", "q": t.tolist()})
        time.sleep(dwell)
    time.sleep(2.0)


print("\n--- 1. does the residual FLIP SIGN with sweep direction? ---")
print("(friction opposes the last motion; a bad drive would not care)")
sweep_to(-1.2)
a = err()
print(f"  after sweeping toward -1.2 : {a[0]:+7.3f} / {a[1]:+7.3f} mm")
sweep_to(-0.4)
b = err()
print(f"  after sweeping toward -0.4 : {b[0]:+7.3f} / {b[1]:+7.3f} mm")
flipped = (a[0] * b[0]) < 0
print(f"  sign flipped: {flipped}   -> {'FRICTION (direction-dependent)' if flipped else 'NOT friction'}")

print("\n--- 2. does it ACCUMULATE over repeated cycles? ---")
mags = []
for i in range(3):
    sweep_to(-1.2)
    sweep_to(-0.4)
    e = err()
    mags.append(abs(e[0]))
    print(f"  cycle {i+1}: {e[0]:+7.3f} / {e[1]:+7.3f} mm   (|left| = {abs(e[0]):.3f})")
drift = max(mags) - min(mags)
print(f"  spread across cycles: {drift:.3f} mm -> "
      f"{'BOUNDED, no drift' if drift < 1.0 else 'ACCUMULATING'}")

print("\n--- 3. does re-commanding the gripper CLEAR it? ---")
print("(this is what decides whether it matters for grasping at all)")
before = err()
c.request({"op": "gripper", "pos": 0.2})
time.sleep(1.5)
c.request({"op": "gripper", "pos": GRIP_FRAC})
time.sleep(2.0)
after = err()
print(f"  before re-grip: {before[0]:+7.3f} / {before[1]:+7.3f} mm")
print(f"  after  re-grip: {after[0]:+7.3f} / {after[1]:+7.3f} mm")
if abs(after[0]) < abs(before[0]) * 0.5:
    print("  -> CLEARED. The jaws re-seat on every grip command, so the")
    print("     residual never reaches a real pick.")
else:
    print("  -> persists through a re-grip; it would affect real picks.")
