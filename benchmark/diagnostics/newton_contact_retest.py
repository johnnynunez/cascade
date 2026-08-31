"""Re-test the 2026-07-19 Newton manipulation-contact bug (docs/ROADMAP.md).

    pip install "newton[examples]"      # package is `newton`, NOT newton-physics
    python benchmark/diagnostics/newton_contact_retest.py

Runs on CPU: Newton's SolverMuJoCo does NOT require CUDA, so this is testable
on a laptop (verified macOS arm64, Warp 1.17 with CUDA disabled).

ROADMAP recorded, on the Isaac 6.0 develop build:
  "constant +3.7 cm float even box-vs-box, fingers pass through objects,
   boot NaN, Triangle pair buffer overflowed"

Result on Newton 1.5.1: does not reproduce (-0.03 mm float; fingers stop
0.3-0.8 mm into the box at realistic drive force). See the ROADMAP entry.

TWO API TRAPS that produce FALSE PASSES here -- both hit while writing this:
 1. `add_body()` already creates a FREE joint. Adding a prismatic joint on top
    makes a parallel LOOP joint which MuJoCo silently drops ("no supported
    equality constraint mapping"); the actuator then does nothing, the fingers
    never move, and "they stopped on the box" is vacuously true. Articulated
    links need `add_link()` + `add_articulation(joints)`.
 2. A joint with no `parent_xform` is anchored at the WORLD ORIGIN, so both
    fingers spawn inside the box.
Hence the explicit "did it actually move / did it reach the full command"
guards below: a contact test that cannot fail proves nothing.
"""

import numpy as np
import warp as wp
import newton

wp.init()
DT = 1.0 / 60.0


def make_pipeline(model):
    pipe = newton.CollisionPipeline(model)
    # contacts buffer: API varies by version, so find the maker rather than
    # guessing (guessing is what produced two wrong calls already).
    for maker in ("contacts", "make_contacts", "allocate_contacts"):
        fn = getattr(pipe, maker, None) or getattr(model, maker, None)
        if callable(fn):
            try:
                return pipe, fn()
            except TypeError:
                continue
        if fn is not None and not callable(fn):
            return pipe, fn
    raise RuntimeError("cannot allocate a Contacts buffer")


def run(model, steps, control=None, gravity_off=False):
    if gravity_off:
        # model.gravity is a warp ARRAY (the solver calls .numpy() on it), so
        # assigning a bare wp.vec3 breaks _convert_to_mjc. Write through it.
        model.gravity.assign(np.zeros((1, 3), dtype=np.float32))
    solver = newton.solvers.SolverMuJoCo(model)
    s0, s1 = model.state(), model.state()
    control = control if control is not None else model.control()
    pipe, contacts = make_pipeline(model)
    for _ in range(steps):
        s0.clear_forces()
        pipe.collide(s0, contacts)
        solver.step(s0, s1, control, contacts, DT)
        s0, s1 = s1, s0
    return s0


print("=" * 64)
print("TEST 1: box-vs-box resting float (the '+3.7 cm' symptom)")
print("=" * 64)
b = newton.ModelBuilder()
plat = b.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 0.1), wp.quat_identity()),
                  is_kinematic=True)          # static, no parallel joint
b.add_shape_box(body=plat, hx=0.3, hy=0.3, hz=0.1)
dyn = b.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 0.30), wp.quat_identity()))
b.add_shape_box(body=dyn, hx=0.05, hy=0.05, hz=0.05)
b.add_ground_plane()
model = b.finalize()

s = run(model, 400)
z = float(s.body_q.numpy()[dyn][2])
expected = 0.1 + 0.1 + 0.05      # platform centre + half height + box half
print(f"  dynamic box z = {z:.4f}   expected ~{expected:.4f}")
print(f"  float above contact = {(z - expected) * 100:+.2f} cm   (bug was +3.7 cm)")
print(f"  finite: {bool(np.isfinite(z))}")
v1 = bool(np.isfinite(z)) and abs(z - expected) < 0.01
print(f"  -> {'PASS: rests exactly at contact' if v1 else 'FAIL'}")

print()
print("=" * 64)
print("TEST 2: two fingers squeezing a box (the 'pass through' symptom)")
print("=" * 64)
b2 = newton.ModelBuilder()
# The box is a free body: add_body() gives it a FREE joint, which is right.
box = b2.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 0.5), wp.quat_identity()),
                  mass=0.05)
b2.add_shape_box(body=box, hx=0.03, hy=0.03, hz=0.03)
# The fingers are ARTICULATED: use add_link() + an explicit joint. add_body()
# would give each one a FREE joint AND the prismatic, i.e. a parallel loop
# joint that MuJoCo drops with "no supported equality constraint mapping" --
# the actuator then does nothing and the fingers never move (observed).
fingers, fjoints = [], []
HOME = 0.10          # finger home |x|
# Contact geometry: box half 0.03 + finger half 0.01 -> touch at |x| = 0.04,
# i.e. after 0.06 m of inward travel from home. Command 0.09 m so that WITHOUT
# working contact each finger ends 0.01 m INSIDE the box. That difference is
# what makes this test discriminating; commanding less than 0.06 would let the
# fingers stop on their own target and look like a pass (this happened).
COMMAND = 0.09
CONTACT_TRAVEL = HOME - (0.03 + 0.01)      # 0.06
for sign in (-1.0, 1.0):
    f = b2.add_link(mass=0.2)
    b2.add_shape_box(body=f, hx=0.01, hy=0.04, hz=0.04)
    j = b2.add_joint_prismatic(
        parent=-1, child=f,
        # Anchor each finger's joint at its OWN home pose. Without a
        # parent_xform the joint origin is the world origin, q becomes absolute
        # x, and both fingers start at x=0 -- already inside the box.
        parent_xform=wp.transform(wp.vec3(sign * HOME, 0.0, 0.5), wp.quat_identity()),
        axis=wp.vec3(-sign, 0.0, 0.0),     # +q always travels INWARD
        actuator_mode=newton.JointTargetMode.POSITION,
        target_pos=COMMAND,
        target_ke=5000.0, target_kd=100.0,
        effort_limit=500.0,
        limit_lower=-0.02, limit_upper=0.12,
    )
    fingers.append(f)
    fjoints.append(j)
b2.add_articulation(fjoints)   # takes the joints it groups
model2 = b2.finalize()

control2 = model2.control()
for attr in ("joint_target_pos", "joint_target_q"):
    arr = getattr(control2, attr, None)
    if arr is not None:
        buf = arr.numpy()
        buf[-2:] = COMMAND
        arr.assign(buf)

s2 = run(model2, 800, control=control2, gravity_off=True)
bq = s2.body_q.numpy()
jq = s2.joint_q.numpy()[-2:]
sep = abs(float(bq[fingers[0]][0]) - float(bq[fingers[1]][0]))
min_gap = 2 * (0.03 + 0.01)      # 0.08: fingers resting on the box
moved = float(np.abs(jq).max())
print(f"  finger travel q = {np.round(jq, 4)} m   (commanded {COMMAND})")
print(f"  contact should stop them at q = {CONTACT_TRAVEL:.3f}")
print(f"  finger separation = {sep:.4f} m   (touching = {min_gap:.2f}, "
      f"ignoring contact = {2 * (HOME - COMMAND):.2f})")
print(f"  box centre x = {float(bq[box][0]):+.5f}   finite: {bool(np.isfinite(bq).all())}")
if moved < 0.005:
    v2 = False
    print("  -> INCONCLUSIVE: the fingers never moved, so this proves nothing")
elif moved >= COMMAND - 0.005:
    v2 = False
    print(f"  -> FAIL: reached the full commanded {COMMAND} m, i.e. straight "
          f"through the box")
else:
    v2 = bool(np.isfinite(bq).all()) and abs(sep - min_gap) < 0.01
    print(f"  -> {'PASS: stopped ON the box surface' if v2 else 'FAIL'}")

print()
print("=" * 64)
print(f"NEWTON {newton.__version__} / mujoco-warp on "
      f"{'CUDA' if wp.is_cuda_available() else 'CPU (no CUDA in this warp build)'}")
print(f"  box-vs-box float : {'OK' if v1 else 'BROKEN'}")
print(f"  finger squeeze   : {'OK' if v2 else 'BROKEN'}")
print("=" * 64)
