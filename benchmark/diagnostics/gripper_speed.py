"""Does the gripper still deviate when the arm moves FAST?

The earlier measurement (gripper_hold2.py) swept joint2 in 12 waypoints with
250 ms between them -- a slow, deliberate motion. It found ~3.6 mm peak
deviation on Newton and concluded "finite-stiffness PD response under inertial
load", which is correct for THAT speed but says nothing about fast motion.

Two things changed since then and neither has been re-measured against this
question:
  * num_substeps 1 -> 4 (the anti-tunnelling fix). Smaller effective timestep
    means the PD loop is integrated more finely, which should REDUCE overshoot.
  * the perception fix, which is irrelevant here but changed what the demo does.

So: sweep the same joint2 arc at several speeds and report peak finger
deviation for each. Speed is set by how many waypoints the arc is split into
and how long we dwell -- fewer waypoints + shorter dwell = faster motion.

The physically meaningful axis is joint velocity, so report the measured
max |dq/dt| alongside the deviation rather than trusting the commanded rate.

Everything is driven through the bridge's own `gripper` and `set_joints` ops,
because the bridge REWRITES gripper targets from grip_frac every tick -- a
direct set_dof_position_targets write is erased within one step.

Run against ONE engine; the caller restarts the bridge to switch.
"""
import sys
import time

sys.path.insert(0, "/home/johnny/Projects/demo/cascade/src")

import numpy as np

from cascade.sim.bridge_client import BridgeClient

GRIP_FRAC = 0.6
ARC = (-1.2, -0.4)          # joint2 arc, radians (same as the slow test)

c = BridgeClient(port=8611)
c.connect()

engine = "?"
try:
    engine = c.request({"op": "state"}).get("engine", "?")
except Exception:
    pass

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

def dq():
    return art.get_dof_velocities().numpy()[0]

globals().update(dict(LI=LI, RI=RI, TL=TL, TR=TR, q=q, dq=dq))
print("ready")
"""
r = c.request({"op": "exec", "code": PRE})
if "ready" not in r.get("stdout", ""):
    print("setup failed:", r)
    sys.exit(1)

base = np.asarray(c.request({"op": "state"})["q"], dtype=float)


def rest_error():
    out = c.request({"op": "exec", "code":
                     "c=q()\nprint(f'{(c[LI]-TL)*1000:+.4f} {(c[RI]-TR)*1000:+.4f}')\n"})
    return [float(x) for x in out.get("stdout", "0 0").split()]


def run_speed(n_waypoints, dwell_s):
    """Sweep the arc, tracking peak finger error AND peak joint2 speed."""
    # park at the start of the arc and let it settle
    tgt = base.copy()
    tgt[1] = ARC[0]
    c.request({"op": "set_joints", "q": tgt.tolist()})
    time.sleep(1.5)

    c.request({"op": "exec", "code": "_w=[0.0,0.0]\n_v=0.0\nprint('ok')\n"})

    t0 = time.monotonic()
    for a2 in np.linspace(ARC[0], ARC[1], n_waypoints):
        tgt = base.copy()
        tgt[1] = float(a2)
        c.request({"op": "set_joints", "q": tgt.tolist()})
        if dwell_s:
            time.sleep(dwell_s)
        # sample peak error and peak joint speed
        c.request({"op": "exec", "code":
                   "c=q(); v=dq()\n"
                   "if abs(c[LI]-TL)>abs(_w[0]): _w[0]=c[LI]-TL\n"
                   "if abs(c[RI]-TR)>abs(_w[1]): _w[1]=c[RI]-TR\n"
                   "if abs(v[1])>_v: _v=abs(v[1])\nprint('ok')\n"})
    elapsed = time.monotonic() - t0

    out = c.request({"op": "exec", "code":
                     "print(f'{_w[0]*1000:+.3f} {_w[1]*1000:+.3f} {_v:.4f}')\n"})
    peak_l, peak_r, peak_v = [float(x) for x in out.get("stdout", "0 0 0").split()]

    time.sleep(2.0)
    res = rest_error()
    return peak_l, peak_r, peak_v, elapsed, res


rl, rr = rest_error()
print(f"\nengine: {engine}")
print(f"at rest (no motion): {rl:+.4f} / {rr:+.4f} mm\n")

print(f"{'speed':<10} {'waypoints':>9} {'dwell':>7} {'peak |dq2|':>11} "
      f"{'finger dev':>18} {'residual':>16}")
print("-" * 78)

for name, nwp, dwell in [
    ("slow", 12, 0.25),      # the original measurement
    ("medium", 6, 0.10),
    ("fast", 3, 0.02),
    ("snap", 2, 0.0),        # single jump across the whole arc
]:
    pl, pr, pv, el, res = run_speed(nwp, dwell)
    print(f"{name:<10} {nwp:>9} {dwell:>7.2f} {pv:>9.3f} r/s "
          f"{pl:>+8.2f} /{pr:>+7.2f} mm {res[0]:>+7.3f} /{res[1]:>+6.3f}")

print("\nfinger dev = peak deviation from the commanded opening during motion")
print("residual   = deviation 2 s AFTER the motion stopped")
print("cube half-width is 25 mm; a finger off by >5 mm risks catching an edge")
