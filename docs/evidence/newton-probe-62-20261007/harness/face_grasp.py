"""B16 attribution 2: corner contact (radial jaw yaw) vs face-aligned jaw yaw at the far spot.

The probe aligns the jaw with the radial direction atan2(y, x). Axis-aligned cubes
are then gripped on their corners (28 deg off at the far spot, 41 deg at the near
spot). Re-run the probe's own grasp recipe at both spots with (a) the radial yaw
and (b) a jaw yaw aligned with the cube's measured faces, N trials each.
usage: face_grasp.py <port> <engine> <out.json> [trials]
"""
import importlib.util
import json
import math
import sys
import time

import numpy as np

port, engine, out = int(sys.argv[1]), sys.argv[2], sys.argv[3]
trials = int(sys.argv[4]) if len(sys.argv) > 4 else 2
sys.argv = ["physics_probe"]
spec = importlib.util.spec_from_file_location("pp", "scripts/physics_probe.py")
pp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pp)

p = pp.Probe(port, engine)
p.ensure_playing()
p.discover_objects()
near, far = pp.OBJECTS["pink_cube"][0], pp.OBJECTS["green_cube"][0]
park = (0.20, -0.20, near[2])


def cube_yaw(name):
    out = p.ex(
        "from isaacsim.core.experimental.prims import RigidPrim\n"
        f"rp = RigidPrim('/World_Props/{name}')\n"
        "_, q = rp.get_world_poses()\n"
        "q = q.numpy() if hasattr(q, 'numpy') else q\n"
        "print([float(x) for x in q[0]], flush=True)\n")
    w, x, y, z = json.loads(out.strip().splitlines()[-1])   # Isaac order: w, x, y, z
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def grasp(name, yaw):
    """The probe's t_grasp with an explicit jaw yaw; also returns the finger closure."""
    pos = p.poses([name])[name]
    tcp_z = pp.OBJECTS[name][1]
    R = pp._yaw_rotation(yaw)
    p.arm.set_gripper(p.grip_open, effort=1.0)
    if not p.goto(R, (pos[0], pos[1], tcp_z + 0.10), 2.0) or not p.goto(R, (pos[0], pos[1], tcp_z), 1.5):
        return {"skipped": "IK unreachable"}
    p.arm.set_gripper(p.grip_closed, effort=1.0)
    time.sleep(2.0)
    fingers = p.cli.request({"op": "state"})["gripper_joints"]["position_m"]
    z0 = p.poses([name])[name][2]
    yaw_in_grip = cube_yaw(name)
    p.goto(R, (pos[0], pos[1], tcp_z + 0.12), 1.5)
    time.sleep(0.5)
    dz = p.poses([name])[name][2] - z0
    p.arm.set_gripper(p.grip_open, effort=0.6)
    time.sleep(1.0)
    p.home()
    return {"finger_m": [round(f, 4) for f in fingers], "cube_yaw_in_grip_deg": round(math.degrees(yaw_in_grip), 1),
            "dz_lift_m": round(dz, 3), "lifted": dz > 0.05}


rows = []
for spot_name, spot in (("near r=0.227", near), ("far r=0.340", far)):
    for mode in ("radial", "face"):
        for t in range(trials):
            p.home()
            p.teleport("pink_cube", park, settle_s=1.0)
            p.teleport("green_cube", spot, settle_s=1.5)
            radial = math.atan2(spot[1], spot[0])
            cy = cube_yaw("green_cube")
            if mode == "radial":
                yaw = radial
            else:   # jaw yaw on the cube face closest to the radial yaw (keeps IK close to the probe's)
                k = round((radial - cy) / (math.pi / 2))
                yaw = cy + k * math.pi / 2
            r = grasp("green_cube", yaw)
            r.update({"spot": spot_name, "mode": mode, "trial": t, "jaw_yaw_deg": round(math.degrees(yaw), 1),
                      "cube_yaw_deg": round(math.degrees(cy), 1),
                      "jaw_minus_cube_deg": round(math.degrees(yaw - cy), 1)})
            rows.append(r)
            print(json.dumps(r), flush=True)
p.restore()
json.dump({"engine": engine, "rows": rows}, open(out, "w"), indent=1)
print("report:", out)
