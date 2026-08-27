"""Is the 2.35 mm just an unsettled gripper?

MuJoCo, driving the SAME model, reports 0.002 mm error and ncon=0 -- the
fingers never touch and both reach their limits, including the right one at
71.5 mm. Isaac/Newton on the same asset reports 2.35 mm per finger and the
right finger stalling at 54 mm.

Same model, opposite results, so the asset is exonerated and the fault is in
the Isaac/Newton path. The USD carries the same limits (0..0.05 / 0..0.0715),
so it is not a limits mismatch either.

That leaves execution. In MuJoCo I stepped 3000 times (~10 s of sim) before
reading. Against the bridge I slept 2.5 s of WALL time -- and the bridge runs
at whatever real-time factor Isaac manages, which with 4 substeps and RTX
rendering can be far below 1.0. A PD drive that has not finished converging
looks exactly like a constant offset.

Test it directly: command one opening and sample the error as a function of
settle time, out to 30 s. If it decays, it was never a constraint at all.
"""
import sys
import time

sys.path.insert(0, "/home/johnny/Projects/demo/cascade/src")

from cascade.sim.bridge_client import BridgeClient

GF = 0.6
c = BridgeClient(port=8611)
c.connect()

SETUP = f"""
import numpy as np
art = globals()["art"]
_n = list(art.dof_names)
LI, RI = _n.index("joint_left"), _n.index("joint_right")
_lo, _hi = [x.numpy()[0] for x in art.get_dof_limits()]
TL = _lo[LI] + {GF}*(_hi[LI]-_lo[LI])
TR = _lo[RI] + {GF}*(_hi[RI]-_lo[RI])
def q():
    return art.get_dof_positions().numpy()[0]
globals().update(dict(LI=LI, RI=RI, TL=TL, TR=TR, q=q))
print("ready")
"""
if "ready" not in c.request({"op": "exec", "code": SETUP}).get("stdout", ""):
    print("setup failed")
    sys.exit(1)

# open wide, then command the test opening
c.request({"op": "gripper", "pos": 1.0})
time.sleep(3.0)
t0 = time.monotonic()
c.request({"op": "gripper", "pos": GF})

print(f"commanded grip_frac {GF}; sampling the error as it settles\n")
print(f"{'wall s':>8} {'err L mm':>10} {'err R mm':>10}")
print("-" * 30)

READ = ("c=q()\n"
        "print(f'{(c[LI]-TL)*1000:+.4f} {(c[RI]-TR)*1000:+.4f}')\n")

last = None
for target in (0.5, 1, 2, 3, 5, 8, 12, 18, 25, 32):
    while time.monotonic() - t0 < target:
        time.sleep(0.05)
    out = c.request({"op": "exec", "code": READ}).get("stdout", "").split()
    if len(out) != 2:
        continue
    el, er = float(out[0]), float(out[1])
    last = (el, er)
    print(f"{time.monotonic()-t0:>8.1f} {el:>+10.4f} {er:>+10.4f}")

print()
if last and max(abs(last[0]), abs(last[1])) < 0.5:
    print("-> the error DECAYS: it was settle time, not a constraint.")
    print("   The bridge simply read the joints before the PD converged.")
else:
    print("-> the error persists after 30 s of wall time: not settling.")
    print("   Something is genuinely holding the fingers off target.")
