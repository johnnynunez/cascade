"""Are the finger MESHES touching, and is that what pins the 3.606 mm?

Ruled out so far, each by measurement:
  * asymmetric travel is intentional (manufacturer URDF) and the asset's own
    evidence validates both fingers to ~1e-08 m against those limits
  * the bridge scales each finger to its own range (isaac_bridge.py:1407)
  * no equality / tendon / mimic / coupling anywhere in the MJCF (0 occurrences)
  * the offset is not half the travel mismatch (3.606 vs 10.75 mm)

What remains: the two finger BODIES collide with each other. The repo's own
evidence already records gripper_left <-> gripper_right producing 5166
self-contacts at the home pose -- 75 % of every contact in the scene, a
consequence of the CoACD convex decomposition inflating the collision hulls
beyond the visual meshes.

If the collision hulls are fatter than the visual geometry, the fingers stop
short of their commanded positions the moment the hulls touch, and they stop at
a FIXED separation -- which is exactly an offset that does not vary with the
commanded opening.

The distinguishing measurement: the WORLD-SPACE distance between the two finger
bodies at each commanded opening. If a constraint/contact pins them, that
distance has a floor it never goes below. Compare the commanded separation
against the achieved one.
"""
import sys
import time

sys.path.insert(0, "/home/johnny/Projects/demo/wrc_demo/src")

import numpy as np

from wrc_demo.sim.bridge_client import BridgeClient

c = BridgeClient(port=8611)
c.connect()

SETUP = """
import numpy as np
art = globals()["art"]
_n = list(art.dof_names)
LI, RI = _n.index("joint_left"), _n.index("joint_right")
_lo, _hi = [x.numpy()[0] for x in art.get_dof_limits()]

# body-level poses: find the two finger links
_bn = None
for attr in ("body_names", "link_names"):
    v = getattr(art, attr, None)
    if v:
        _bn = list(v)
        break

def q():
    return art.get_dof_positions().numpy()[0]

globals().update(dict(LI=LI, RI=RI, LO=_lo, HI=_hi, q=q, BN=_bn))
print("BODIES", [b for b in (_bn or []) if "grip" in b.lower()][:4])
"""
r = c.request({"op": "exec", "code": SETUP})
print(r.get("stdout", "").strip())

print(f"\n{'frac':>5} {'cmd L':>8} {'cmd R':>8} {'real L':>8} {'real R':>8} "
      f"{'cmd sep':>9} {'real sep':>9}")
print("-" * 62)

rows = []
for gf in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
    c.request({"op": "gripper", "pos": 1.0})
    time.sleep(1.2)
    c.request({"op": "gripper", "pos": gf})
    time.sleep(2.5)

    out = c.request({"op": "exec", "code":
                     f"TL = LO[LI] + {gf}*(HI[LI]-LO[LI])\n"
                     f"TR = LO[RI] + {gf}*(HI[RI]-LO[RI])\n"
                     "cc = q()\n"
                     "print(f'{TL*1000:.3f} {TR*1000:.3f} "
                     "{cc[LI]*1000:.3f} {cc[RI]*1000:.3f}')\n"})
    v = out.get("stdout", "").split()
    if len(v) != 4:
        continue
    tl, tr, rl, rr = [float(x) for x in v]
    # both sliders move along the same axis; separation is their sum for a
    # facing pair, difference for a same-direction pair -- report both
    rows.append((gf, tl, tr, rl, rr))
    print(f"{gf:>5.1f} {tl:>8.3f} {tr:>8.3f} {rl:>8.3f} {rr:>8.3f} "
          f"{tl+tr:>9.3f} {rl+rr:>9.3f}")

if rows:
    print("\nerror per finger (real - commanded), mm:")
    for gf, tl, tr, rl, rr in rows:
        print(f"  frac {gf:.1f}:  L {rl-tl:+8.3f}   R {rr-tr:+8.3f}   "
              f"sum {(rl+rr)-(tl+tr):+8.3f}")
    sums = [(rl + rr) for _, _, _, rl, rr in rows]
    print(f"\nachieved sum ranges {min(sums):.2f} .. {max(sums):.2f} mm")
    print("if the SUM tracks the command while each finger is offset, the")
    print("pair is being held at a fixed relative position by a constraint")
