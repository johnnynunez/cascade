# SPDX-License-Identifier: Apache-2.0
"""Agentic MicroDuck episode: Cascade agents (one per robot) drive learned policies in Isaac Lab + Newton.

Run inside the Isaac Lab venv (see docs/MICRODUCK_AGENTIC_DEMOS.md)::

    python -m cascade.apps.microduck_agentic --scenario configs/microduck_agentic/backstage_obstacles.yaml \
        --llm codex_astra --output runs/backstage_001

Outputs (``--output``, must not exist): ``recording.npz`` (+ ``.json``) with every 50 Hz physics sample (16 body poses
per robot, controller per robot, held ToF frame, held 48x36 RGB, lens pose), ``events.jsonl`` (agent decisions, LLM
latency, tool receipts on the simulation clock), one trace directory per robot, ``maps_<robot>.npz`` (each robot's own
map over time), ``episode.json`` (task reports + simulator verification).
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import threading
import time

import yaml



def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def load_scenario(path: Path) -> dict:
    path = Path(path)
    data = yaml.safe_load(path.read_text())
    if data.get("version") != 1:
        raise ValueError("scenario requires version: 1")
    for key in ("name", "robots", "bounds"):
        if key not in data:
            raise ValueError(f"scenario missing {key!r}")
    scene = data.get("scene")
    if scene:
        p = Path(scene)
        data["scene_path"] = str(p if p.is_absolute() else (_repo_root() / p).resolve())
    elif data.get("scene_env"):
        # Not vendored here: an explicitly named local file, accepted only with the pinned digest.
        import hashlib
        import os

        value = os.environ.get(data["scene_env"], "")
        if not value:
            raise ValueError(f"set {data['scene_env']} to the scene file (sha256 {data.get('scene_sha256')})")
        p = Path(value).expanduser().resolve(strict=True)
        digest = hashlib.sha256(p.read_bytes()).hexdigest()
        if digest != data.get("scene_sha256"):
            raise ValueError(f"{p} sha256 {digest} does not match the scenario pin {data.get('scene_sha256')}")
        data["scene_path"] = str(p)
    data["source"] = str(path.resolve())
    return data


class LoggedLLM:
    """Pass-through LLM client that records every request/response on the simulation clock."""

    def __init__(self, client, port, robot):
        self.client, self.port, self.robot = client, port, robot
        self.supports_vision = getattr(client, "supports_vision", False)
        self.calls = 0
        self.wall_s = 0.0

    def chat(self, system, messages, tools=None, max_tokens=1024):
        t0 = time.time()
        self.port.log_event(self.robot, "llm_request", "", n_messages=len(messages), with_tools=bool(tools))
        resp = self.client.chat(system=system, messages=messages, tools=tools, max_tokens=max_tokens)
        dt = time.time() - t0
        self.calls += 1
        self.wall_s += dt
        calls = [{"name": c.name, "args": c.arguments} for c in (resp.tool_calls or [])]
        kind = "llm_reply" if tools else "plan"
        self.port.log_event(self.robot, kind, (resp.text or "")[:1200], tool_calls=calls, latency_s=round(dt, 2))
        return resp

    def close(self):
        close = getattr(self.client, "close", None)
        if close:
            close()

    def __getattr__(self, name):
        return getattr(self.client, name)


def verify(scenario: dict, finals: dict) -> dict:
    """Independent end-of-episode checks from the completed physics state (not from the agents' claims)."""
    out = {}
    for name, spec in (scenario.get("verify") or {}).items():
        st = finals.get(name)
        if st is None:
            out[name] = {"status": "missing"}
            continue
        kind = spec["type"]
        res = {"type": kind, "final_pose": {k: round(st[k], 3) for k in ("x", "y", "z")}, "fallen": st["fallen"]}
        if kind == "reach":
            d = math.hypot(st["x"] - spec["x"], st["y"] - spec["y"])
            res.update(distance_m=round(d, 3), radius_m=spec["radius"],
                       passed=bool(d <= spec["radius"] and not st["fallen"]))
        elif kind == "stay_above":
            res.update(min_z=spec["z"], passed=bool(st["z"] >= spec["z"] and not st["fallen"]))
        elif kind == "reach_and_stay_above":
            d = math.hypot(st["x"] - spec["x"], st["y"] - spec["y"])
            res.update(distance_m=round(d, 3), radius_m=spec["radius"], min_z=spec["z"],
                       passed=bool(d <= spec["radius"] and st["z"] >= spec["z"] and not st["fallen"]))
        elif kind == "near_object":
            x0, y0, x1, y1 = (float(v) for v in spec["footprint"])
            dx = max(x0 - st["x"], 0.0, st["x"] - x1)
            dy = max(y0 - st["y"], 0.0, st["y"] - y1)
            dist = math.hypot(dx, dy)
            res.update(object=spec.get("object"), distance_to_surface_m=round(dist, 3), max_m=spec["max_m"],
                       passed=bool(dist <= spec["max_m"] and not st["fallen"]))
        elif kind == "region":
            inside = (spec["x_min"] <= st["x"] <= spec["x_max"] and spec["y_min"] <= st["y"] <= spec["y_max"])
            res.update(region={k: spec[k] for k in ("x_min", "x_max", "y_min", "y_max")}, min_z=spec.get("z"),
                       passed=bool(inside and st["z"] >= spec.get("z", -1e9) and not st["fallen"]))
        elif kind == "upright":
            res.update(passed=not st["fallen"])
        else:
            res.update(passed=None, note="unknown check")
        out[name] = res
    return out


def main(argv=None):
    from isaaclab.app import add_launcher_args, launch_simulation

    from ..agent.llm import MockLLM, make_llm
    from ..agent.orchestrator import AgentOrchestrator
    from ..config import Cfg, _load_yaml, load_profile
    from ..sim.microduck_lab_world import CONTROL_DT_S, PolicyFiles, RobotSpawn, build_env_cfg, create_world
    from ..sim.microduck_world_port import PortClosed, WorldPort
    from ..skills.microduck_agent_runtime import FleetBoard, MicroduckAgentRuntime

    home = Path.home()
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--scenario", type=Path, required=True)
    p.add_argument("--llm", default="codex_astra")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--gtceu_root", type=Path, default=home / "Projects/microducks/microduck_gtceu-final")
    p.add_argument("--handoff_root", type=Path, default=home / "Projects/microducks/deliveries/johnny_handoff")
    p.add_argument("--wall_deadline_s", type=float, default=1800.0)
    p.add_argument("--seed", type=int, default=7)
    add_launcher_args(p)
    p.set_defaults(visualizer=[])
    args = p.parse_args(argv)
    scenario = load_scenario(args.scenario)
    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    (out / "scenario.yaml").write_text(Path(scenario["source"]).read_text())
    (out / "scenario_resolved.json").write_text(json.dumps(scenario, indent=1, default=str) + "\n")
    H = args.handoff_root
    use_nav = bool(scenario.get("navigation_policy", True))
    policies = PolicyFiles(
        walker_onnx=H / "microduck_johnny_navigation_v1/policies/walking_velocity_flat_existing/velocity_flat_existing.onnx",
        hero_pt=H / "microduck_johnny_perception_v1/policies/onboard_hero/onboard_hero.pt",
        getup_onnx=H / "microduck_johnny_balance_falls_v1/policies/getup_kitfloor_ft_model_1100/getup_kitfloor_ft_model_1100.onnx",
        nav_onnx=(H / "microduck_johnny_navigation_v1/policies/depth_navigation_v4_iter300/depth_navigation_v4.onnx")
        if use_nav else None,
    )
    robots = scenario["robots"]
    spawns = [RobotSpawn(r["name"], float(r["spawn"]["x"]), float(r["spawn"]["y"]), float(r["spawn"].get("yaw_deg", 0)),
                         r["spawn"].get("fallen"), float(r["spawn"].get("z", 0.0))) for r in robots]
    device = getattr(args, "device", None) or "cuda:0"
    llm_path = Path(args.llm).expanduser()
    llm_cfg = Cfg(_load_yaml(llm_path)) if llm_path.is_file() else load_profile("llm", args.llm)
    cfg = build_env_cfg(gtceu_root=args.gtceu_root, course_usd=scenario.get("scene_path"), num_robots=len(spawns),
                        device=device, seed=args.seed, nav_camera=use_nav)
    floor_z = float(scenario.get("floor_z", 0.0))
    t_wall = time.time()
    with launch_simulation(cfg, args):
        world = create_world(cfg, gtceu_root=args.gtceu_root, spawns=spawns, policies=policies, device=device,
                             course_usd=scenario.get("scene_path"),
                             meta={"scenario": scenario["name"], "llm": llm_cfg.type,
                                   "llm_model": getattr(llm_cfg, "model", None) if hasattr(llm_cfg, "model") else None})
        world.step(int(scenario.get("settle_s", 1.0) / CONTROL_DT_S))  # stand/settle before the agents start
        port = WorldPort(world, out / "events.jsonl", floor_z=floor_z,
                         max_sim_s=world.time_s + float(scenario.get("max_sim_s", 180.0)), control_dt_s=CONTROL_DT_S)
        fleet = FleetBoard() if len(robots) > 1 else None
        detector = None
        if scenario.get("detectable"):
            from ..sim.microduck_detector_standin import ObjectDetectorStandIn, load_from_usd

            objs, occ = load_from_usd(Path(scenario["scene_path"]), scenario["detectable"],
                                      tuple(scenario.get("occluders", ("Wall", "Outer", "Inner"))))
            detector = ObjectDetectorStandIn(objs, occ, max_range_m=float(scenario.get("detector_range_m", 2.5)))
        runtimes, reports, threads, clients = {}, {}, [], {}
        for i, r in enumerate(robots):
            rt = MicroduckAgentRuntime(r["name"], i, port, out / f"trace_{r['name']}", bounds_xy=scenario["bounds"],
                                       floor_z=floor_z, fleet=fleet, has_nav=use_nav, has_getup=True,
                                       extra_prompt=scenario.get("context", ""), detector=detector)
            runtimes[r["name"]] = rt
            client = MockLLM() if llm_cfg.type == "mock" else make_llm(llm_cfg)
            llm = LoggedLLM(client, port, r["name"])
            clients[r["name"]] = llm
            orch = AgentOrchestrator(llm, rt, max_steps=int(scenario.get("max_steps", 24)), decompose=True,
                                     attach_images=bool(getattr(client, "supports_vision", False)),
                                     verify_milestones=False, memory_frames_k=2)

            def work(name=r["name"], orch=orch, task=r["task"], rt=rt):
                port.log_event(name, "task", task)
                try:
                    rt._sense()  # an honest initial observation before the first LLM turn (fills map + view)
                    rep = orch.run_task(task)
                    reports[name] = {"success_claimed": rep.success, "summary": rep.summary, "steps": rep.steps,
                                     "milestones": rep.milestones, "duration_s": rep.duration_s}
                except PortClosed as exc:
                    reports[name] = {"error": f"simulation stopped: {exc}"}
                except Exception as exc:  # noqa: BLE001
                    reports[name] = {"error": f"{type(exc).__name__}: {exc}"}
                port.log_event(name, "task_end", json.dumps(reports[name], default=str)[:1500])

            threads.append(threading.Thread(target=work, name=f"agent-{r['name']}", daemon=True))
        for t in threads:
            t.start()
        why = port.serve(lambda: any(t.is_alive() for t in threads) and time.time() - t_wall < args.wall_deadline_s)
        for t in threads:
            t.join(timeout=5)
        finals = {r["name"]: world.base_state(i) for i, r in enumerate(robots)}
        checks = verify(scenario, finals)
        positions = world.rec["body_pose"]
        min_pair = None
        if len(robots) > 1 and positions:
            import numpy as np

            root = np.asarray(positions)[:, :, 0, :2]
            d = [np.linalg.norm(root[:, a] - root[:, b], axis=-1).min() for a in range(len(robots))
                 for b in range(a + 1, len(robots))]
            min_pair = float(min(d))
        clearance = None
        if positions and scenario.get("scene_path"):
            import numpy as np

            from ..sim.microduck_clearance import apply_to_verification, clearance_report, obstacle_boxes

            clearance = clearance_report(np.asarray(positions)[::5, :, :, :3].astype(np.float64),
                                         [r["name"] for r in robots], obstacle_boxes(scenario["scene_path"]))
            checks = apply_to_verification(checks, scenario.get("verify", {}), clearance)
        episode = {"scenario": scenario["name"], "stop": why, "sim_time_s": world.time_s,
                   "wall_s": round(time.time() - t_wall, 1), "reports": reports, "verification": checks,
                   "llm": {n: {"calls": c.calls, "wall_s": round(c.wall_s, 1)} for n, c in clients.items()},
                   "llm_profile": args.llm, "min_pairwise_root_distance_m": min_pair, "clearance": clearance,
                   "maps": {n: rt.map.summary() for n, rt in runtimes.items()},
                   "honesty": ["robots are separate Newton worlds: no robot-robot contact, robots invisible to each "
                               "other's onboard sensors",
                               "map pose source: simulator base pose (stand-in for SLAM)",
                               "simulation time is paused while every agent is waiting for its LLM (lockstep)"]}
        (out / "episode.json").write_text(json.dumps(episode, indent=1, default=str) + "\n")
        world.save(out / "recording.npz", {"episode": episode})
        import numpy as np

        for n, rt in runtimes.items():
            snaps = rt.map_snapshots
            np.savez_compressed(out / f"maps_{n}.npz", t=np.asarray([s[0] for s in snaps]),
                                logodds=np.asarray([s[1]["logodds"] for s in snaps]),
                                drop=np.asarray([s[1]["drop"] for s in snaps]),
                                bounds=np.asarray(scenario["bounds"]), res=rt.map.res,
                                visited=np.asarray(json.dumps([s[1]["visited"] for s in snaps])),
                                landmarks=np.asarray(json.dumps([s[1]["landmarks"] for s in snaps])))
        for c in clients.values():
            try:
                c.close()
            except Exception:  # noqa: BLE001
                pass
        print("EPISODE DONE", json.dumps({"stop": why, "sim_time_s": round(world.time_s, 2),
                                          "verification": checks}, default=str), flush=True)
        world.env.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
