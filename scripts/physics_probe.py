"""Systematic per-object physics validation against a live isaac_bridge.

For EVERY prop it measures, with ground-truth poses from the physics view
(no perception in the loop):

  settle  - object at rest: jitter/drift/penetration over 4 s
  drop    - +4 cm drop: settle time, bounce, tunneling, final pose
  grasp   - the arm closes its fingers on the object, holds, lifts:
            the Newton-NaN trigger, and the realism acid test under PhysX
  push    - TCP sweeps through the object: displacement sane, no explosion

Run against any bridge (engine chosen at bridge launch):
  .demo/bin/python scripts/physics_probe.py --port 8611 --engine physx
  .demo/bin/python scripts/physics_probe.py --port 8612 --engine newton \
      --solver-json '{"cone": "pyramidal", "impratio": 10.0}' --objects banana

Newton solver variants mutate the LIVE NewtonConfig.solver_cfg between a
timeline stop/play (the solver is rebuilt on play), so one boot can sweep
several solver configs. Report: table on stdout + JSON next to --report.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from wrc_demo.config import load_demo_config  # noqa: E402
from wrc_demo.control.arm_base import make_arm  # noqa: E402
from wrc_demo.control.kinematics import Kinematics  # noqa: E402
from wrc_demo.grasping.obb_grasp import _yaw_rotation  # noqa: E402
from wrc_demo.sim.bridge_client import BridgeClient  # noqa: E402
from wrc_demo.types import make_transform  # noqa: E402

HOME_Q = np.array([0.0, 1.2, 1.2, 0.0, 0.75, 0.0])

# name -> (spawn, grasp_tcp_z, graspable, pushable)
OBJECTS = {
    "banana": ((0.24, 0.14, 0.018), 0.025, True, False),
    "soup_can": ((0.20, -0.12, 0.052), 0.060, True, True),
    "pink_cube": ((0.28, 0.08, 0.026), 0.030, True, False),
    "cracker_box": ((0.36, -0.02, 0.107), 0.180, False, True),
}


class Probe:
    def __init__(self, port: int, engine: str):
        self.engine = engine
        self.cli = BridgeClient(port=port, timeout_s=60.0)
        self.cli.connect()
        cfg = load_demo_config(arm="isaac")
        cfg.arm._data["bridge_port"] = port
        self.kin = Kinematics(
            model_path=cfg.arm.model,
            ee_frame=cfg.arm.get("ee_frame", "gripper_end"),
            n_controlled=int(cfg.arm.get("n_joints", 6)),
            joint_signs=cfg.arm.get("joint_signs"),
        )
        self.arm = make_arm(cfg.arm, kinematics=self.kin)
        self.arm.connect()
        self.grip_open = float(cfg.arm.gripper.get("open", 1.0))
        self.grip_closed = float(cfg.arm.gripper.get("closed", 0.0))
        # The -plus asset hoists the whole booth ~3 m up (gravity-tune
        # heritage); arm IK is base-frame, but prop poses/teleports are
        # WORLD-frame. exec shares the bridge module globals.
        self.base_z = float(self.ex("print(float(BASE_Z), flush=True)").strip())

    # ── sim-side helpers (exec op runs on the bridge main thread) ───────
    def ex(self, code: str) -> str:
        r = self.cli.request({"op": "exec", "code": code})
        if not r.get("ok"):
            raise RuntimeError(f"exec failed: {r.get('error', '?')[:200]}")
        return r.get("stdout", "")

    def poses(self, names: list[str]) -> dict[str, list[float]]:
        pat = "|".join(names)
        out = self.ex(
            "from isaacsim.core.experimental.prims import RigidPrim\n"
            f"rp = RigidPrim('/World_Props/({pat})')\n"
            "pos, _ = rp.get_world_poses()\n"
            "pos = pos.numpy() if hasattr(pos, 'numpy') else pos\n"
            "import json\n"
            "print(json.dumps({p.rsplit('/',1)[-1]: [float(x) for x in q]\n"
            "                  for p, q in zip(rp.paths, pos)}), flush=True)\n"
        )
        raw = json.loads(out.strip().splitlines()[-1])
        return {k: [v[0], v[1], v[2] - self.base_z] for k, v in raw.items()}

    def teleport(self, name: str, pos, settle_s: float = 0.0) -> None:
        """PhysX: physics-view teleport. Newton: author while STOPPED (a
        live teleport leaves latent NaN in this build), timeline-aware
        bridge re-homes the arm on re-play."""
        x, y, z = float(pos[0]), float(pos[1]), float(pos[2]) + self.base_z
        if self.engine == "physx":
            self.ex(
                "import numpy as np\n"
                "from isaacsim.core.experimental.prims import RigidPrim\n"
                f"rp = RigidPrim('/World_Props/{name}')\n"
                f"rp.set_world_poses(np.array([[{x},{y},{z}]]))\n"
                "try:\n"
                "    rp.set_velocities(np.zeros((1,3)), np.zeros((1,3)))\n"
                "except Exception:\n"
                "    pass\n"
            )
        else:
            self.ex(
                "import omni.timeline, omni.usd\n"
                "from pxr import Gf, UsdGeom\n"
                "tl = omni.timeline.get_timeline_interface()\n"
                "tl.stop()\n"
            )
            time.sleep(1.0)
            self.ex(
                "import omni.usd\n"
                "from pxr import Gf, UsdGeom\n"
                "st = omni.usd.get_context().get_stage()\n"
                f"xf = UsdGeom.Xformable(st.GetPrimAtPath('/World_Props/{name}'))\n"
                "ops = xf.GetOrderedXformOps()\n"
                "m = ops[-1].Get() if ops else Gf.Matrix4d(1.0)\n"
                "m = Gf.Matrix4d(m)\n"
                f"m.SetTranslateOnly(Gf.Vec3d({x}, {y}, {z}))\n"
                "(ops[-1] if ops else xf.AddTransformOp()).Set(m)\n"
                "import omni.timeline\n"
                "omni.timeline.get_timeline_interface().play()\n"
            )
            time.sleep(3.0)  # bridge _resume_scene re-homes the arm
        if settle_s:
            time.sleep(settle_s)

    def sample(self, names: list[str], seconds: float, hz: float = 10.0):
        t0, rows = time.monotonic(), []
        while time.monotonic() - t0 < seconds:
            rows.append((time.monotonic() - t0, self.poses(names)))
            time.sleep(1.0 / hz)
        return rows

    def arm_nan(self) -> bool:
        q = np.asarray(self.arm.get_state().q, dtype=float)
        return bool(np.any(~np.isfinite(q)))

    # ── motions ──────────────────────────────────────────────────────────
    def goto(self, R, pos, duration_s: float = 2.0, q0=None) -> bool:
        q_now = np.asarray(self.arm.get_state().q, dtype=float)
        ik = self.kin.ik(make_transform(R, np.asarray(pos, dtype=float)),
                         q0 if q0 is not None else q_now)
        if not ik.success:
            return False
        self.arm.stream_to(ik.q, duration_s=duration_s)
        self.arm.wait_settled(ik.q, tol=0.06, timeout_s=4.0)
        return True

    def home(self):
        self.arm.stream_to(HOME_Q, duration_s=2.0)
        self.arm.wait_settled(HOME_Q, tol=0.08, timeout_s=4.0)

    # ── tests ────────────────────────────────────────────────────────────
    def t_settle(self, names: list[str]) -> dict:
        rows = self.sample(names, 4.0)
        rep = {}
        for n in names:
            zs = np.array([r[1][n] for r in rows])
            nan = bool(np.any(~np.isfinite(zs)))
            jit = float(np.std(zs[len(zs) // 2:, 2])) if not nan else float("nan")
            drift = float(np.linalg.norm(zs[-1, :2] - zs[0, :2])) if not nan else float("nan")
            sunk = (not nan) and bool(zs[-1, 2] < -0.005)
            rep[n] = {
                "nan": nan, "jitter_std_m": round(jit, 5),
                "drift_xy_m": round(drift, 4), "sunk": sunk,
                "pass": (not nan) and (not sunk) and jit < 0.002 and drift < 0.005,
            }
        return rep

    def t_drop(self, name: str) -> dict:
        spawn = OBJECTS[name][0]
        self.teleport(name, (spawn[0], spawn[1], spawn[2] + 0.04))
        rows = self.sample([name], 3.0)
        zs = np.array([r[1][name] for r in rows])
        nan = bool(np.any(~np.isfinite(zs)))
        if nan:
            return {"nan": True, "pass": False}
        settle_t = None
        for i in range(1, len(rows)):
            dt = rows[i][0] - rows[i - 1][0]
            v = np.linalg.norm(zs[i] - zs[i - 1]) / max(dt, 1e-3)
            if v < 0.01 and rows[i][0] > 0.3:
                settle_t = rows[i][0]
                break
        final_z = float(zs[-1, 2])
        return {
            "nan": False,
            "settle_s": round(settle_t, 2) if settle_t else None,
            "final_z": round(final_z, 4),
            "tunneled": final_z < -0.01,
            "flew": bool(np.max(zs[:, 2]) > spawn[2] + 0.09),
            "drift_xy_m": round(float(np.linalg.norm(zs[-1, :2] - np.array(spawn[:2]))), 4),
            "pass": (settle_t is not None and settle_t < 2.5
                     and -0.01 <= final_z and np.max(zs[:, 2]) <= spawn[2] + 0.09),
        }

    def t_grasp(self, name: str) -> dict:
        pos = self.poses([name])[name]
        if any(not np.isfinite(v) for v in pos):
            return {"skipped": "object already NaN", "pass": False}
        tcp_z = OBJECTS[name][1]
        yaw = float(np.arctan2(pos[1], pos[0]))
        R = _yaw_rotation(yaw)
        self.arm.set_gripper(self.grip_open, effort=1.0)
        if not self.goto(R, (pos[0], pos[1], tcp_z + 0.10), 2.0):
            return {"skipped": "hover IK unreachable", "pass": None}
        if not self.goto(R, (pos[0], pos[1], tcp_z), 1.5):
            return {"skipped": "grasp IK unreachable", "pass": None}
        self.arm.set_gripper(self.grip_closed, effort=1.0)
        rows = self.sample([name], 2.0)  # hold under full contact: NaN window
        zs = np.array([r[1][name] for r in rows])
        nan_hold = bool(np.any(~np.isfinite(zs))) or self.arm_nan()
        vmax = 0.0
        if not nan_hold:
            for i in range(1, len(rows)):
                dt = max(rows[i][0] - rows[i - 1][0], 1e-3)
                vmax = max(vmax, float(np.linalg.norm(zs[i] - zs[i - 1]) / dt))
        z_before = float(zs[-1, 2]) if not nan_hold else float("nan")
        self.goto(R, (pos[0], pos[1], tcp_z + 0.12), 1.5)
        time.sleep(0.5)
        after = self.poses([name])[name]
        nan_lift = any(not np.isfinite(v) for v in after) or self.arm_nan()
        lifted = (not nan_lift) and (after[2] - z_before > 0.05)
        self.arm.set_gripper(self.grip_open, effort=0.6)
        time.sleep(1.0)
        self.home()
        return {
            "nan": nan_hold or nan_lift,
            "hold_vmax_m_s": round(vmax, 3),
            "lifted": bool(lifted),
            "dz_lift_m": round(float(after[2] - z_before), 3) if not (nan_hold or nan_lift) else None,
            "pass": (not (nan_hold or nan_lift)) and vmax < 1.5 and lifted,
        }

    def t_push(self, name: str) -> dict:
        pos = self.poses([name])[name]
        if any(not np.isfinite(v) for v in pos):
            return {"skipped": "object already NaN", "pass": False}
        yaw = float(np.arctan2(pos[1], pos[0]))
        u = np.array([np.cos(yaw), np.sin(yaw)])  # push radially outward
        z = 0.03
        R = _yaw_rotation(yaw)
        start = np.array([pos[0], pos[1], z]) - np.array([*(u * 0.09), 0.0])
        end = np.array([pos[0], pos[1], z]) + np.array([*(u * 0.03), 0.0])
        self.arm.set_gripper(self.grip_closed, effort=0.8)
        if not self.goto(R, start + [0, 0, 0.08], 2.0) or not self.goto(R, start, 1.5):
            self.home()
            return {"skipped": "push IK unreachable", "pass": None}
        q_now = np.asarray(self.arm.get_state().q, dtype=float)
        ik = self.kin.ik(make_transform(R, end), q_now)
        if not ik.success:
            self.home()
            return {"skipped": "push end IK unreachable", "pass": None}
        self.arm.stream_to(ik.q, duration_s=2.0)
        rows = self.sample([name], 2.5)
        zs = np.array([r[1][name] for r in rows])
        nan = bool(np.any(~np.isfinite(zs))) or self.arm_nan()
        self.home()
        if nan:
            return {"nan": True, "pass": False}
        vmax = max(
            float(np.linalg.norm(zs[i] - zs[i - 1]) / max(rows[i][0] - rows[i - 1][0], 1e-3))
            for i in range(1, len(rows))
        )
        moved = float(np.linalg.norm(zs[-1, :2] - np.array(pos[:2])))
        return {
            "nan": False, "moved_m": round(moved, 4), "vmax_m_s": round(vmax, 3),
            "on_table": bool(-0.01 < zs[-1, 2] < 0.30),
            "pass": moved > 0.005 and vmax < 2.0 and -0.01 < zs[-1, 2] < 0.30,
        }

    def restore(self):
        self.home()
        for n, (spawn, *_rest) in OBJECTS.items():
            try:
                self.teleport(n, spawn)
            except Exception as e:
                print(f"[probe] restore {n}: {e}", file=sys.stderr)


def apply_solver_json(probe: Probe, solver_json: str):
    """Mutate the live NewtonConfig.solver_cfg between stop/play."""
    cfg = json.loads(solver_json)
    lines = [
        "import omni.timeline",
        "import isaacsim.physics.newton.impl.extension as _ne",
        "tl = omni.timeline.get_timeline_interface()",
        "tl.stop()",
    ]
    if cfg.pop("solver_type", None) == "xpbd":
        lines += [
            "from isaacsim.physics.newton.impl.solver_config import XPBDSolverConfig",
            "_ne._newton_stage.cfg.solver_cfg = XPBDSolverConfig()",
        ]
    for k, v in cfg.items():
        lines.append(f"setattr(_ne._newton_stage.cfg.solver_cfg, {k!r}, {v!r})")
    lines += ["print('solver:', _ne._newton_stage.cfg.solver_cfg, flush=True)"]
    probe.ex("\n".join(lines))
    time.sleep(1.0)
    probe.ex("import omni.timeline\nomni.timeline.get_timeline_interface().play()")
    # the bridge's timeline-aware resume recreates the Articulation on its
    # main loop; poll until the physics views are valid again
    for _ in range(30):
        time.sleep(2.0)
        try:
            probe.arm.get_state()
            probe.poses(["banana"])
            return
        except Exception:
            continue
    raise RuntimeError("bridge did not recover after solver swap")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8611)
    ap.add_argument("--engine", choices=["physx", "newton"], required=True)
    ap.add_argument("--objects", nargs="*", default=list(OBJECTS))
    ap.add_argument("--tests", nargs="*", default=["settle", "drop", "grasp", "push"])
    ap.add_argument("--solver-json", default=None,
                    help='e.g. \'{"cone": "pyramidal", "impratio": 10.0}\' (newton only)')
    ap.add_argument("--report", default=None)
    args = ap.parse_args()

    probe = Probe(args.port, args.engine)
    if args.solver_json:
        apply_solver_json(probe, args.solver_json)
    report = {"engine": args.engine, "solver": args.solver_json, "results": {}}
    names = [n for n in args.objects if n in OBJECTS]

    probe.home()
    if "settle" in args.tests:
        print(f"[probe] settle scan: {names}", flush=True)
        report["results"]["settle"] = probe.t_settle(names)
    for n in names:
        r = report["results"].setdefault(n, {})
        if "drop" in args.tests:
            print(f"[probe] drop: {n}", flush=True)
            r["drop"] = probe.t_drop(n)
        if "grasp" in args.tests and OBJECTS[n][2]:
            print(f"[probe] grasp: {n}", flush=True)
            r["grasp"] = probe.t_grasp(n)
        if "push" in args.tests and OBJECTS[n][3]:
            print(f"[probe] push: {n}", flush=True)
            r["push"] = probe.t_push(n)
    probe.restore()

    out = args.report or f"/tmp/physics_probe_{args.engine}.json"
    np_safe = lambda o: o.item() if hasattr(o, "item") else str(o)  # noqa: E731
    Path(out).write_text(json.dumps(report, indent=2, default=np_safe))
    print(f"\n=== physics probe [{args.engine}"
          f"{' ' + args.solver_json if args.solver_json else ''}] ===")
    for sect, res in report["results"].items():
        print(f"{sect}:")
        for k, v in res.items():
            print(f"  {k}: {json.dumps(v, default=np_safe)}")
    print(f"report: {out}")


if __name__ == "__main__":
    main()
