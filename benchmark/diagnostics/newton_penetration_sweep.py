"""Is the finger penetration NO contact, or SOFT contact?

If there is no contact at all, travel is the full commanded 0.09 regardless of
drive force. If contact exists but is soft, penetration scales with force.
That is the distinction between "Newton's manipulation contacts are broken"
(the 2026-07-19 ROADMAP claim) and "this probe over-drives a compliant solver".
"""
import numpy as np, warp as wp, newton
wp.init()
DT = 1 / 60

HOME, COMMAND = 0.10, 0.09
CONTACT_AT = HOME - 0.04     # 0.06 m of travel


def build(effort, ke):
    b = newton.ModelBuilder()
    box = b.add_body(xform=wp.transform(wp.vec3(0., 0., .5), wp.quat_identity()), mass=0.05)
    b.add_shape_box(body=box, hx=.03, hy=.03, hz=.03)
    js = []
    for sign in (-1., 1.):
        f = b.add_link(mass=0.2)
        b.add_shape_box(body=f, hx=.01, hy=.04, hz=.04)
        js.append(b.add_joint_prismatic(
            parent=-1, child=f,
            parent_xform=wp.transform(wp.vec3(sign * HOME, 0., .5), wp.quat_identity()),
            axis=wp.vec3(-sign, 0., 0.),
            actuator_mode=newton.JointTargetMode.POSITION,
            target_pos=COMMAND, target_ke=ke, target_kd=ke / 50,
            effort_limit=effort, limit_lower=-.02, limit_upper=.12))
    b.add_articulation(js)
    m = b.finalize()
    m.gravity.assign(np.zeros((1, 3), dtype=np.float32))
    return m, box


print(f"contact should stop the fingers at q = {CONTACT_AT:.3f} m "
      f"(commanded {COMMAND})")
print(f"{'effort N':>9} {'ke':>7} {'travel q':>9} {'penetration':>12} {'contacts':>9}")
for effort, ke in [(2.0, 200.0), (10.0, 500.0), (50.0, 2000.0), (500.0, 5000.0)]:
    m, box = build(effort, ke)
    solver = newton.solvers.SolverMuJoCo(m)
    s0, s1 = m.state(), m.state()
    ctrl = m.control()
    for attr in ("joint_target_pos", "joint_target_q"):
        a = getattr(ctrl, attr, None)
        if a is not None:
            v = a.numpy(); v[-2:] = COMMAND; a.assign(v)
    pipe = newton.CollisionPipeline(m)
    contacts = pipe.contacts()
    ncon = 0
    for _ in range(800):
        s0.clear_forces()
        pipe.collide(s0, contacts)
        n = getattr(contacts, "rigid_contact_count", None)
        if n is not None:
            ncon = max(ncon, int(n.numpy()[0]))
        solver.step(s0, s1, ctrl, contacts, DT)
        s0, s1 = s1, s0
    q = float(s0.joint_q.numpy()[-2:].max())
    print(f"{effort:9.1f} {ke:7.0f} {q:9.4f} {(q - CONTACT_AT)*1000:9.1f} mm {ncon:9d}")

print()
print("full 0.0900 at every force  -> no contact at all (the ROADMAP bug)")
print("travel shrinking with force -> contact works, solver is compliant")
