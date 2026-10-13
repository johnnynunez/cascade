# SPDX-License-Identifier: Apache-2.0
"""Agentic SKATING MicroDuck episode: Cascade agents (one AgentOrchestrator, memory and trace per robot) drive
unchanged learned skating policies on the skate-wheel MicroDuck in ONE shared Isaac Lab 3.0 + Newton world.

    python scripts/microduck_agentic/run_skate_episode.py --scenario configs/microduck_agentic/skate_plaza.yaml \\
        --llm codex_astra --output runs/skate1

Outputs: events.jsonl (every LLM request/reply, tool call/result on the simulation clock), recording.npz/.json (all
body poses, commands, exact actor inputs and actions per control step), episode.json (agent claims next to independent
checks computed from the recorded physics), trace_<robot>/, maps_<robot>.npz (landmark memory + trail).
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import threading
import time

from .microduck_agentic import LoggedLLM, _repo_root, load_scenario, localization_eval

POLICIES = {"natural_roller": "skating_natural/natural_roller.onnx", "oval_roller": "skating_race/oval_roller.onnx"}


def verify_skate(scenario: dict, rec: dict, names: list[str], trunk: int, runtimes: dict) -> dict:
    """Independent checks from the RECORDED trunk trajectory (not from the agents' claims)."""
    import numpy as np

    pose = np.asarray(rec["body_pose"])  # (T, R, B, 7)
    t = np.asarray(rec["time_s"])
    root = pose[:, :, trunk, :3]
    grav = np.asarray(rec["proj_grav"])
    tilt = np.degrees(np.arccos(np.clip(-grav[..., 2], -1, 1)))
    fallen_any = (tilt > 60) | (root[..., 2] < 0.06)
    out = {}
    for ri, name in enumerate(names):
        spec = (scenario.get("verify") or {}).get(name)
        if spec is None:
            continue
        xy = root[:, ri, :2]
        fell = bool(fallen_any[:, ri].any())
        speed_end = float(np.linalg.norm(xy[-1] - xy[-6]) / max(t[-1] - t[-6], 1e-6)) if len(t) > 6 else 0.0
        res = {"type": spec["type"], "final_pose": {"x": round(float(xy[-1, 0]), 3), "y": round(float(xy[-1, 1]), 3)},
               "fell": fell, "final_speed_m_s": round(speed_end, 3)}
        if fell:
            res["fell_at_sim_time_s"] = round(float(t[np.argmax(fallen_any[:, ri])]), 2)
        if spec["type"] == "slalom":
            passes, ok_order, last_t = [], True, -1.0
            for cone in spec["cones"]:
                d = np.hypot(xy[:, 0] - cone["x"], xy[:, 1] - cone["y"])
                k = int(np.argmin(d))
                ok = bool(d[k] <= spec["pass_radius_m"])
                if ok:
                    # first time it came within the pass radius (order is judged on first passes)
                    k = int(np.argmax(d <= spec["pass_radius_m"]))
                passes.append({"cone": cone["name"], "closest_m": round(float(d.min()), 3), "passed": ok,
                               "first_pass_sim_time_s": round(float(t[k]), 2) if ok else None})
                if ok:
                    ok_order &= float(t[k]) > last_t
                    last_t = float(t[k])
            stop = spec["stop"]
            d_stop = float(math.hypot(xy[-1, 0] - stop["x"], xy[-1, 1] - stop["y"]))
            stopped = speed_end < spec.get("max_final_speed_m_s", 0.05)
            res.update(cones=passes, in_order=bool(ok_order), stop_distance_m=round(d_stop, 3), stop_radius_m=stop["radius"],
                       stopped=bool(stopped),
                       passed=bool(all(p["passed"] for p in passes) and ok_order and d_stop <= stop["radius"] and stopped
                                   and not fell))
        elif spec["type"] == "race_lap":
            rt = runtimes[name]
            prog = rt.race["progress_units"] / 4.0
            fin = rt.race["finish_time_s"]
            t0 = rt.race["start_time_s"]
            res.update(laps_done=round(prog, 3), finished=fin is not None,
                       race_time_s=round(fin - t0, 2) if (fin is not None and t0 is not None) else None)
            if rt.oval is not None:
                h = rt.oval.half
                cx = np.clip(xy[:, 0], -h, h)
                rad = np.hypot(xy[:, 0] - cx, xy[:, 1])
                lo, hi = spec.get("track_r_min", 0.45), spec.get("track_r_max", 1.35)
                res["off_track_fraction"] = round(float(((rad < lo) | (rad > hi)).mean()), 3)
            res["passed"] = bool(fin is not None and not fell)
        elif spec["type"] == "upright":
            res["passed"] = not fell
        else:
            res["passed"] = None
        out[name] = res
    if len(names) > 1:
        dmin = {}
        for a in range(len(names)):
            for b in range(a + 1, len(names)):
                d = np.linalg.norm(root[:, a, :2] - root[:, b, :2], axis=-1)
                dmin[f"{names[a]}-{names[b]}"] = round(float(d.min()), 3)
        out["_skaters"] = {"min_trunk_distance_m": dmin, "possible_skater_contact": any(v < 0.22 for v in dmin.values()),
                           "note": "trunk-centre distance; below 0.22 m the skaters' bodies may touch (shared world)"}
    return out


def main(argv=None):
    from isaaclab.app import add_launcher_args, launch_simulation

    from ..agent.llm import MockLLM, make_llm
    from ..agent.orchestrator import AgentOrchestrator
    from ..config import Cfg, _load_yaml, load_profile
    from ..sim.microduck_clearance import apply_to_verification, clearance_report, obstacle_boxes
    from ..sim.microduck_skate_world import CONTROL_DT_S, SkatePolicy, SkateSpawn, build_skate_cfg, create_skate_world
    from ..sim.microduck_world_port import PortClosed, WorldPort
    from ..skills.microduck_agent_runtime import FleetBoard
    from ..skills.microduck_skate_runtime import Geometry, MicroduckSkateRuntime, Oval

    home = Path.home()
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--scenario", type=Path, required=True)
    p.add_argument("--llm", default="codex_astra")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--gtceu_root", type=Path, default=home / "Projects/microducks/train/cascade/gtceu_main",
                   help="read-only export of microduck_gtceu main (skating policies + task module)")
    p.add_argument("--roller_usd", type=Path,
                   default=home / "Projects/microducks/train/skillsft/assets/microduck_rollers.usd")
    p.add_argument("--wall_deadline_s", type=float, default=1800.0)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--honest", action="store_true", help="onboard-only robot/agent loop (see microduck_honest.py)")
    add_launcher_args(p)
    p.set_defaults(visualizer=[])
    args = p.parse_args(argv)
    scenario = load_scenario(args.scenario)
    honest = bool(args.honest or scenario.get("honest"))
    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    (out / "scenario.yaml").write_text(Path(scenario["source"]).read_text())
    (out / "scenario_resolved.json").write_text(json.dumps(scenario, indent=1, default=str) + "\n")
    pol_name = scenario.get("skate_policy", "natural_roller")
    policy = SkatePolicy(pol_name, args.gtceu_root / "policies/skills" / POLICIES[pol_name])
    robots = scenario["robots"]
    names = [r["name"] for r in robots]
    spawns = [SkateSpawn(r["name"], float(r["spawn"]["x"]), float(r["spawn"]["y"]), float(r["spawn"].get("yaw_deg", 0)))
              for r in robots]
    device = getattr(args, "device", None) or "cuda:0"
    llm_path = Path(args.llm).expanduser()
    llm_cfg = Cfg(_load_yaml(llm_path)) if llm_path.is_file() else load_profile("llm", args.llm)
    cfg = build_skate_cfg(gtceu_root=args.gtceu_root, roller_usd=args.roller_usd, course_usd=scenario.get("scene_path"),
                          spawns=spawns, device=device, seed=args.seed)
    boxes = obstacle_boxes(scenario["scene_path"]) if scenario.get("scene_path") else {}
    geometry = Geometry(boxes)
    oval = None
    if scenario.get("oval"):
        o = scenario["oval"]
        oval = Oval(o["half"], o["lanes"], o.get("finish_x", 0.0))
    t_wall = time.time()
    with launch_simulation(cfg, args):
        world = create_skate_world(cfg, gtceu_root=args.gtceu_root, spawns=spawns, policy=policy,
                                   roller_usd=args.roller_usd,
                                   meta={"scenario": scenario["name"], "llm": llm_cfg.type})
        if world.body_names[0] != "trunk_base":
            raise RuntimeError(f"body 0 is {world.body_names[0]}, the replay expects the trunk first")
        rig = None
        if honest:
            from ..sim.microduck_honest import HonestWorld, OnboardDetector, OnboardRig, SkateAdapter, load_yoloe

            rig = OnboardRig(SkateAdapter(world), names, course_usd=scenario.get("scene_path"), robot_usd=args.roller_usd,
                             dt=CONTROL_DT_S, log_path=out / "sensors.log", video_dir=out)
            for _ in range(int(scenario.get("settle_s", 1.0) / CONTROL_DT_S)):
                world.step(1)
                rig.imu_tick()
            rig.start(world.time_s, [0.0] * len(names))
            world.heading_source = lambda i: rig.onboard[i].yaw_f  # heading error from the robot's own gyro estimate
            truth_start = {n: world.base_state(i) for i, n in enumerate(names)}  # offline evaluation only
            agent_world = HonestWorld(world, rig)
            yolo = load_yoloe(_repo_root())
            oval = None  # no track map is handed to an honest robot
        else:
            world.step(int(scenario.get("settle_s", 1.0) / CONTROL_DT_S))  # coast in place before the agents start
            agent_world = world
        port = WorldPort(agent_world, out / "events.jsonl", max_sim_s=world.time_s + float(scenario.get("max_sim_s", 120.0)),
                         control_dt_s=CONTROL_DT_S)
        fleet = FleetBoard() if len(robots) > 1 else None
        detector = None
        if scenario.get("detectable"):
            from ..sim.microduck_detector_standin import ObjectDetectorStandIn, load_from_usd

            objs, occ = load_from_usd(Path(scenario["scene_path"]), scenario["detectable"], ())
            detector = ObjectDetectorStandIn(objs, occ, max_range_m=float(scenario.get("detector_range_m", 3.0)))
        runtimes, reports, threads, clients = {}, {}, [], {}
        running = set(range(len(robots)))
        lock = threading.Lock()
        for i, r in enumerate(robots):
            if honest:
                rt = MicroduckSkateRuntime(r["name"], i, port, out / f"trace_{r['name']}",
                                           bounds_xy=scenario.get("map_bounds_start_frame", [-1.5, -3.0, 9.0, 3.0]),
                                           geometry=Geometry({}), fleet=fleet,
                                           detector=OnboardDetector(agent_world, i, yolo), oval=None,
                                           extra_prompt=scenario.get("context_honest", ""), robot_names=names,
                                           honest=True)
                r = dict(r, task=r.get("task_honest", r["task"]))
            else:
                rt = MicroduckSkateRuntime(r["name"], i, port, out / f"trace_{r['name']}", bounds_xy=scenario["bounds"],
                                           geometry=geometry, fleet=fleet, detector=detector, oval=oval,
                                           extra_prompt=scenario.get("context", ""), robot_names=names)
            runtimes[r["name"]] = rt
            client = MockLLM() if llm_cfg.type == "mock" else make_llm(llm_cfg)
            llm = LoggedLLM(client, port, r["name"])
            clients[r["name"]] = llm
            orch = AgentOrchestrator(llm, rt, max_steps=int(scenario.get("max_steps", 20)), decompose=True,
                                     attach_images=bool(honest and getattr(client, "supports_vision", False)),
                                     verify_milestones=False, memory_frames_k=0)

            def work(name=r["name"], orch=orch, task=r["task"], rt=rt, idx=i):
                port.log_event(name, "task", task)
                try:
                    rt._sense()
                    rep = orch.run_task(task)
                    reports[name] = {"success_claimed": rep.success, "summary": rep.summary, "steps": rep.steps,
                                     "milestones": rep.milestones, "duration_s": rep.duration_s}
                except PortClosed as exc:
                    reports[name] = {"error": f"simulation stopped: {exc}"}
                except Exception as exc:  # noqa: BLE001
                    reports[name] = {"error": f"{type(exc).__name__}: {exc}"}
                with lock:
                    running.discard(idx)
                port.log_event(name, "task_end", json.dumps(reports[name], default=str)[:1500])

            threads.append(threading.Thread(target=work, name=f"agent-{r['name']}", daemon=True))
        if len(robots) > 1:
            port.sync_running = lambda: set(running)
        if oval is not None:
            def progress(w):
                for i, n in enumerate(names):
                    runtimes[n].track_update(w.base_state(i))
            port.on_step.append(progress)
            for i, n in enumerate(names):
                runtimes[n].track_update(world.base_state(i))
        for rt in runtimes.values():
            rt.race["start_time_s"] = world.time_s  # common race clock: agents start together
        for t in threads:
            t.start()
        why = port.serve(lambda: any(t.is_alive() for t in threads) and time.time() - t_wall < args.wall_deadline_s)
        for t in threads:
            t.join(timeout=5)
        import numpy as np

        checks = verify_skate(scenario, world.rec, names, world.trunk, runtimes)
        clearance = None
        if boxes and world.rec["body_pose"]:
            clearance = clearance_report(np.asarray(world.rec["body_pose"])[::5, :, :, :3].astype(np.float64), names, boxes)
            checks = apply_to_verification(checks, scenario.get("verify", {}), clearance)
        episode = {"scenario": scenario["name"], "stop": why, "sim_time_s": world.time_s,
                   "wall_s": round(time.time() - t_wall, 1), "reports": reports, "verification": checks,
                   "llm": {n: {"calls": c.calls, "wall_s": round(c.wall_s, 1)} for n, c in clients.items()},
                   "llm_profile": args.llm, "clearance": clearance, "skate_policy": world.meta["policy"],
                   "robot_usd": {"path": world.meta["robot_usd"], "sha256": world.meta["robot_usd_sha256"]},
                   "maps": {n: rt.map.summary() for n, rt in runtimes.items()},
                   "race": ({n: rt.race for n, rt in runtimes.items()} if oval is not None else None),
                   "honesty": (["ONBOARD-ONLY loop: the agent and every controller used the robot's own sensors and "
                                "its own pose estimate (gyro heading + cuVSLAM RGBD translation, start frame); the "
                                "policy's heading error is computed from that estimate; simulator truth only in the "
                                "offline checks",
                                "sensors: OVRTX head RGB 640x360 15 Hz, VL53L8CX ToF model on the rendered depth, "
                                "trunk IMU model; detector YOLOE prompt-free on the head image",
                                "skate-wheel MicroDuck variant (microduck_rollers.usd): the canonical robot plus passive "
                                "wheels under the feet",
                                "scripted (on the estimate): steering loop, brake-to-halt, ToF brake reflex",
                                "no get-up policy for the skate robot: a fall ends the run",
                                "simulation time is paused while any agent is waiting for its LLM"] if honest else
                               ["PRIVILEGED-STATE PROTOTYPE (sim pose, ground-truth detector stand-in, collider range)",
                                "ONE shared Newton world: skaters can collide with each other and with the course",
                               "skate-wheel MicroDuck variant (microduck_rollers.usd): the canonical robot plus passive "
                               "wheels under the feet",
                               "agent senses are stand-ins: simulator pose/speed, detector = simulator ground truth in "
                               "the head-camera field of view, range-ahead from scene geometry (no ToF model on the "
                               "skate robot); the skating actors are blind",
                               "scripted: steering loop / race pilot heading command, brake-to-halt, obstacle reflex",
                               "simulation time is paused while any agent is waiting for its LLM"])}
        if honest:
            episode["localization_eval"] = localization_eval(rig, world, truth_start, robots)
            episode["onboard_calibration"] = rig.calibration
            episode["camera_frames"] = rig.cam_frames
        (out / "episode.json").write_text(json.dumps(episode, indent=1, default=str) + "\n")
        world.save(out / "recording.npz", {"episode": episode})
        for n, rt in runtimes.items():
            snaps = rt.map_snapshots
            np.savez_compressed(out / f"maps_{n}.npz", t=np.asarray([s[0] for s in snaps]),
                                logodds=np.asarray([s[1]["logodds"] for s in snaps]),
                                drop=np.asarray([s[1]["drop"] for s in snaps]),
                                bounds=np.asarray(scenario["bounds"]), res=rt.map.res,
                                visited=np.asarray(json.dumps([s[1]["visited"][-400:] for s in snaps])),
                                landmarks=np.asarray(json.dumps([s[1]["landmarks"] for s in snaps])))
        if rig is not None:
            np.savez_compressed(out / "onboard.npz", **{f"est_{n}": np.asarray(v, np.float64)
                                                         for n, v in rig.est_log.items()},
                                **{f"tof_{n}": np.asarray(v, np.int32) for n, v in rig.tof_log.items()},
                                cam_times=np.asarray(rig.cam_times))
            rig.close()
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
