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

from cascade.config import load_demo_config  # noqa: E402
from cascade.control.arm_base import make_arm  # noqa: E402
from cascade.control.kinematics import Kinematics  # noqa: E402
from cascade.grasping.obb_grasp import _yaw_rotation  # noqa: E402
from cascade.sim.bridge_client import BridgeClient  # noqa: E402
from cascade.types import make_transform  # noqa: E402

HOME_Q = np.array([0.0, 1.2, 1.2, 0.0, 0.75, 0.0])

# name -> (spawn, grasp_tcp_z, graspable, pushable)
#
# The YCB rows are the July 2026 dev scene; the bridge has authored only the
# two 0.05 x 0.05 x 0.08 m boxes since the YCB experiment was disabled
# (isaac_bridge.py `PROPS`; green_cube is placed far out as a perception
# target). `discover_objects()` keeps whichever rows the live bridge has
# and takes the spawn from the bridge's own `_PROP_SPAWNS`, so a probe run
# never teleports a prop to a stale coordinate.
OBJECTS = {
    "banana": ((0.24, 0.14, 0.018), 0.025, True, False),
    "soup_can": ((0.20, -0.12, 0.052), 0.060, True, True),
    "pink_cube": ((0.17, 0.15, 0.04), 0.045, True, True),
    "green_cube": ((0.30, 0.16, 0.04), 0.045, True, True),
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

    def discover_objects(self) -> list[str]:
        """The props this bridge authored, in OBJECTS order, with OBJECTS'
        spawns replaced by the bridge's own spawn table."""
        out = self.ex("import json\nprint(json.dumps({k: list(v) for k, v in _PROP_SPAWNS.items()}), flush=True)")
        spawns = json.loads(out.strip().splitlines()[-1])
        live = []
        for name, (spawn, tcp_z, graspable, pushable) in list(OBJECTS.items()):
            if name in spawns:
                OBJECTS[name] = (tuple(float(v) for v in spawns[name]), tcp_z, graspable, pushable)
                live.append(name)
        self.live = list(live)
        return live

    def ensure_playing(self) -> bool:
        """A bridge left stopped (an aborted earlier probe, an editor Stop)
        has no physics tensors; `state` would raise. Press Play and wait
        for the articulation to come back. Returns whether Play was needed."""
        out = self.ex(
            "import omni.timeline\n"
            "_tl = omni.timeline.get_timeline_interface()\n"
            "_was = _tl.is_playing()\n"
            "if not _was:\n"
            "    _tl.play()\n"
            "print('PLAYING' if _was else 'RESUMED', flush=True)\n"
        )
        resumed = "RESUMED" in out
        if resumed:
            for _ in range(30):
                try:
                    self.arm.get_state()
                    break
                except Exception:  # noqa: BLE001 - tensors not back yet
                    time.sleep(1.0)
            else:
                raise RuntimeError("bridge resumed Play but the articulation never came back")
        return resumed

    def build_info(self) -> dict:
        """Isaac build + engine identity, read from the running Kit."""
        out = self.ex(
            "import json, carb\n"
            "info = {}\n"
            "try:\n"
            "    import isaacsim.core.version as _v\n"
            "    info['isaac'] = str(_v.get_version())\n"
            "except Exception as e:\n"
            "    info['isaac'] = f'unknown ({e})'\n"
            "try:\n"
            "    import newton, warp\n"
            "    info['newton'] = newton.__version__\n"
            "    info['warp'] = warp.__version__\n"
            "except Exception as e:\n"
            "    info['newton'] = f'unavailable ({e})'\n"
            "print(json.dumps(info), flush=True)\n"
        )
        try:
            return json.loads(out.strip().splitlines()[-1])
        except Exception:  # noqa: BLE001
            return {"raw": out[-300:]}

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
        """Put a prop at `pos` (base frame) through the bridge's own
        `place_prop` op -- the maintained, engine-aware path (PhysX: physics
        view teleport; Newton: reduced-coordinate reset of the prop's free
        joint in both state buffers). The July stop/author/play workaround
        this replaced assumed a matrix xformOp the current bridge does not
        author and bypassed the Newton reset the kitchen relies on."""
        r = self.cli.request({"op": "place_prop", "name": name,
                              "pos": [float(pos[0]), float(pos[1]), float(pos[2])]})
        if not r.get("ok"):
            raise RuntimeError(f"place_prop {name}: {r.get('error', '?')[:200]}")
        if "place FAILED" in str(r.get("stdout", "")):
            raise RuntimeError(f"place_prop {name}: bridge reported place FAILED")
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
        q_at = np.asarray(self.arm.get_state().q, dtype=float)
        tcp_at = np.asarray(self.kin.fk(q_at))[:3, 3]
        tcp_err = float(np.linalg.norm(tcp_at - np.array([pos[0], pos[1], tcp_z])))
        at_grasp = self.poses([name])[name]
        shoved = float(np.linalg.norm(np.array(at_grasp[:2]) - np.array(pos[:2])))
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
        # Lift to the highest reachable height. At the far spot (r = 0.34 m) a
        # top-down TCP 0.12 m above the grasp height is outside the IK envelope
        # on this asset (measured 2026-10-07 on PhysX and Newton alike: +0.10
        # solves, +0.12 does not); `goto` then returns False WITHOUT moving and
        # the old probe scored the untouched cube as a failed lift. An
        # unreachable lift is a probe-geometry fact, not a physics failure.
        lift_dz = next((dz for dz in (0.12, 0.10, 0.08, 0.06)
                        if self.goto(R, (pos[0], pos[1], tcp_z + dz), 1.5)), None)
        if lift_dz is None:
            self.arm.set_gripper(self.grip_open, effort=0.6)
            time.sleep(1.0)
            self.home()
            return {"skipped": "lift IK unreachable", "pass": None,
                    "tcp_err_m": round(tcp_err, 4), "nan": nan_hold}
        time.sleep(0.5)
        after = self.poses([name])[name]
        nan_lift = any(not np.isfinite(v) for v in after) or self.arm_nan()
        lifted = (not nan_lift) and (after[2] - z_before > 0.05)
        self.arm.set_gripper(self.grip_open, effort=0.6)
        time.sleep(1.0)
        self.home()
        return {
            "nan": nan_hold or nan_lift,
            "tcp_err_m": round(tcp_err, 4),
            "shoved_on_descent_m": round(shoved, 4),
            "hold_vmax_m_s": round(vmax, 3),
            "lift_target_dz_m": lift_dz,
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
        for n in self.live:
            try:
                self.teleport(n, OBJECTS[n][0])
            except Exception as e:
                print(f"[probe] restore {n}: {e}", file=sys.stderr)


def apply_solver_json(probe: Probe, solver_json: str, witness: str | None = None):
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
            if witness:
                probe.poses([witness])
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
    if probe.ensure_playing():
        print("[probe] bridge was stopped; pressed Play", flush=True)
    live = probe.discover_objects()
    if args.solver_json:
        apply_solver_json(probe, args.solver_json, next(iter(live), None))
    report = {"engine": args.engine, "solver": args.solver_json, "build": probe.build_info(),
              "objects": {n: {"spawn": list(OBJECTS[n][0]), "grasp_tcp_z": OBJECTS[n][1]} for n in live},
              "results": {}}
    names = [n for n in args.objects if n in live]
    missing = [n for n in args.objects if n not in live]
    if missing:
        print(f"[probe] not in this bridge's scene, skipped: {missing}", flush=True)
    if not names:
        raise SystemExit(f"none of {args.objects} is a prop of this bridge (it has {sorted(live)})")

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
