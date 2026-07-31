#!/usr/bin/env python3
"""Run LIBERO with wrc_demo's own runtime, skills, harness and verification.

This answers the question the earlier LIBERO scripts could not: how does the
system you built score on the benchmark the papers use? Not a primitive
written for the benchmark -- YOUR SkillRuntime, YOUR SafetyHarness, YOUR grasp
pipeline, YOUR postconditions, driving the Franka in LIBERO.

WHAT RUNS UNCHANGED
    SkillRuntime.execute + every skill_* method
    SafeArm + SafetyHarness per-waypoint approval
    GraspPlanner / grasp selection + IK vetting
    PostconditionChecker (agent/effects.py)
    BeliefStore, TraceLogger

WHAT IS ADAPTED (backend only, see wrc_libero/backend.py)
    ArmBase      -> LiberoArm         (7-DoF Franka, stepped sim)
    Kinematics   -> MujocoKinematics  (validated: FK 0.0 mm, IK 0.1 mm)
    CameraBase   -> LiberoCamera      (agentview RGB-D + real intrinsics)

CONDITIONS
    skill_only    wrc_demo skill, postconditions OFF
    verified      postconditions ON, refuted effect drives up to 3 attempts

Success is LIBERO's own `done` flag -- the benchmark's criterion, not ours.
"""

from __future__ import annotations


import sys as _sys
from pathlib import Path as _Path

_BENCH = _Path(__file__).resolve().parent.parent
if str(_BENCH) not in _sys.path:
    _sys.path.insert(0, str(_BENCH))
# the sibling backend/kinematics modules live next to this file
_HERE = _Path(__file__).resolve().parent
if str(_HERE) not in _sys.path:
    _sys.path.insert(0, str(_HERE))
import paths  # noqa: E402  (benchmark/paths.py)

paths.add_paths()

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np

SUITE_MAX_STEPS = {"libero_spatial": 220, "libero_object": 280,
                   "libero_goal": 300, "libero_10": 520}


def build_wrc_runtime(env, task_language: str):
    """Wire wrc_demo's REAL runtime onto a LIBERO env.

    Rather than re-implementing the composition root (apps/demo.py's
    build_runtime wires ~12 components in a specific order and owns several
    non-obvious details -- LazyArm, detector locking, watcher pausing), we
    register our backends in wrc_demo's own factories and let build_runtime
    do its job. That keeps this file honest: if build_runtime changes, this
    benchmark changes with it, instead of silently testing a stale copy.
    """
    import wrc_demo.control.arm_base as arm_base
    import wrc_demo.perception.camera_base as camera_base
    from wrc_demo.apps.demo import build_runtime
    from wrc_demo.config import load_demo_config
    from backend import LiberoArm, LiberoCamera
    from kinematics_mj import MujocoKinematics

    inner = env.env
    # `env.sim` is only populated after the first reset(). CameraStream opens
    # its camera on a worker thread just after build_runtime returns, so the
    # env MUST already be reset here or every grab fails with
    # "'NoneType' object has no attribute 'model'" -- which the stream logs as
    # "[stream:mock] grab failed", making it look like the wrong backend was
    # selected rather than the wrong ORDER.
    if getattr(inner, "sim", None) is None:
        env.reset()
    kin = MujocoKinematics(inner.sim.model._model, inner.sim.data._data)

    # -- inject the LIBERO backends into wrc_demo's factories -------------
    real_make_arm = arm_base.make_arm
    real_make_cam = camera_base.make_camera
    #: capture the instances so the runner can wire arm -> camera publishing
    _created: dict = {}

    def make_camera(cfg):
        if cfg.get("type") == "libero":
            cam = LiberoCamera(inner)
            _created.setdefault("cameras", []).append(cam)
            _created["camera"] = cam
            print(f"[bridge] LiberoCamera #{len(_created['cameras'])}", flush=True)
            return cam
        return real_make_cam(cfg)

    def make_arm(cfg, kinematics=None):
        if cfg.get("type") == "libero":
            a = LiberoArm(inner, kin)
            _created["arm"] = a
            return a
        return real_make_arm(cfg, kinematics)

    arm_base.make_arm = make_arm
    camera_base.make_camera = make_camera
    # build_runtime imported these by value at module load, so patching the
    # defining module is NOT enough -- the name in apps.demo must be replaced
    # too or the original factory runs and quietly returns a MockCamera.
    import wrc_demo.apps.demo as demo_mod
    demo_mod.make_arm = make_arm
    demo_mod.make_camera = make_camera
    assert demo_mod.make_camera is make_camera, "camera factory not patched"

    cfg = load_demo_config(cameras=["mock"], arm="mock", llm="mock")
    cfg._data.setdefault("stream", {})["mode"] = "off"
    cfg.arm._data["type"] = "libero"
    # The mock profile ships a 6-element home_q for the RS arm; the Panda has
    # 7 joints, and every array op downstream (move_joints, harness approval)
    # broadcasts against it. Use LIBERO's own reset pose.
    cfg.arm._data["home_q"] = [0.0, -0.177, -0.037, -2.457, 0.006, 2.235, 0.799]
    # `_camera_cfgs` prefers cfg.cameras over cfg.camera, so a `cameras` list
    # left pointing at the mock profile silently wins and every frame grab
    # fails with "[stream:mock] grab failed". Drop the list and pin the
    # single profile, including the nested detector block the mock pins.
    cfg._data.pop("cameras", None)
    cfg.camera._data["type"] = "libero"
    # Keep the MOCK detector: this experiment isolates orchestration, and the
    # LIBERO venv has no ultralytics (removing the profile's detector block
    # falls through to the global YOLOE and ImportErrors). Object poses come
    # from MuJoCo, exactly as the sim-truth channel works on the real rig.
    cfg.camera._data["detector"] = {"type": "mock"}
    cfg._data["detector"] = {"type": "mock"}

    # The Panda is 7-DoF in a different workspace than the RS arm: the safety
    # harness must be given the limits of the robot we are actually driving.
    lo, hi = kin.joint_limits
    cfg.safety._data["joint_limits"] = [[float(x), float(y)]
                                        for x, y in zip(lo, hi)]
    cfg.safety._data["table_z"] = 0.80          # LIBERO table top, world frame

    # Kinematics is constructed inside build_runtime from a URDF; swap in the
    # MuJoCo one (validated: FK 0.0 mm, IK 0.1 mm against this exact model).
    real_kin_cls = demo_mod.Kinematics
    demo_mod.Kinematics = lambda *a, **k: kin
    try:
        rt, arm = build_runtime(cfg, Path("/tmp/wrc_on_libero"),
                                view=False, serve=False)
    finally:
        demo_mod.Kinematics = real_kin_cls

    # Wire arm -> camera so every env.step publishes a frame from the thread
    # that owns MuJoCo's GL context, and seed one frame before the stream
    # thread's first grab (otherwise it raises "no frame published yet").
    lib_arm = _created.get("arm")
    lib_cam = _created.get("camera")
    if lib_arm is not None and lib_cam is not None:
        lib_arm.camera = lib_cam
        obs = inner._get_observations()
        lib_arm.last_obs = obs
        lib_cam.publish(obs)

    # Return the LiberoArm itself, not build_runtime's wrapper: the runner
    # needs `_step`/`last_obs`/`camera`, which live on the backend. `rt` still
    # holds the SafeArm, so every skill goes through the harness as usual.
    return rt, (lib_arm if lib_arm is not None else arm), kin


def resolve_task_objects(inner, language: str) -> tuple[str | None, str | None]:
    """Map a LIBERO instruction onto (object, destination) MuJoCo body names.

    LIBERO gives a sentence ("pick up the black bowl between the plate and the
    ramekin and place it on the plate"); wrc_demo's skills take an object
    LABEL. Passing the sentence through makes `localize` fail with "no
    detections", which measures the label mismatch, not the robot.

    Grounding against scene body names -- rather than writing a parser -- is
    deliberate: it keeps the experiment about ORCHESTRATION, which is the
    variable under test, and matches how Pigey and ASPIRE supply perception
    externally. A parser written here would become part of what is measured.
    """
    m = inner.sim.model
    bodies = []
    for i in range(m.nbody):
        n = m.body_id2name(i)
        if n and not n.startswith(("robot0", "gripper0", "world", "table")):
            bodies.append(n)

    text = language.lower()
    # split on the LAST placement preposition: everything after it is the
    # destination, everything before is the object
    m_split = re.split(r"\b(?:on top of|into|onto|on|in)\b", text)
    obj_text = m_split[0] if m_split else text
    dest_text = m_split[-1] if len(m_split) > 1 else ""

    def best(phrase: str, exclude: str = "") -> str | None:
        toks = {t for t in re.findall(r"[a-z]+", phrase)
                if t not in {"the", "a", "up", "pick", "place", "put", "and",
                             "it", "between", "that", "is", "to", "of"}}
        if not toks:
            return None
        scored = []
        for b in bodies:
            if b == exclude:
                continue
            btoks = set(re.findall(r"[a-z]+", b.lower()))
            score = len(toks & btoks)
            if score:
                scored.append((score, -len(btoks), b))
        if not scored:
            return None
        scored.sort(reverse=True)
        return scored[0][2]

    obj = best(obj_text)
    dest = best(dest_text, exclude=obj or "") if dest_text else None
    return obj, dest


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default="libero_spatial")
    ap.add_argument("--tasks", type=int, default=3)
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--conditions", default="skill_only,verified")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()

    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    bm = benchmark.get_benchmark_dict()[a.suite]()
    max_steps = SUITE_MAX_STEPS.get(a.suite, 300)
    conds = a.conditions.split(",")
    out = {c: {"succ": 0, "n": 0, "claimed": 0, "false": 0, "tasks": []}
           for c in conds}

    print(f"=== wrc_demo ON LIBERO | {a.suite} | {a.tasks} tasks x "
          f"{a.episodes} eps ===", flush=True)

    for tid in range(min(a.tasks, bm.n_tasks)):
        task = bm.get_task(tid)
        bddl = os.path.join(get_libero_path("bddl_files"),
                            task.problem_folder, task.bddl_file)
        inits = bm.get_task_init_states(tid)
        env = OffScreenRenderEnv(bddl_file_name=bddl, controller="JOINT_POSITION",
                                 camera_heights=256, camera_widths=256,
                                 camera_depths=True)
        env.seed(0)
        env.reset()
        rt, arm, kin = build_wrc_runtime(env, task.language)
        inner = env.env
        obj_body, dest_body = resolve_task_objects(inner, task.language)
        print(f"  [{tid}] {task.language}\n"
              f"        -> object={obj_body!r} destination={dest_body!r}",
              flush=True)
        if obj_body is None:
            print(f"  [{tid}] SKIPPED: no scene body matched the instruction",
                  flush=True)
            env.close()
            continue

        def object_pos(name_hint: str):
            """Ground-truth body position -- the independent channel."""
            m = inner.sim.model
            for i in range(m.nbody):
                n = m.body_id2name(i)
                if n and name_hint and name_hint.lower() in n.lower():
                    return np.array(inner.sim.data.body_xpos[i])
            return None

        def seed_beliefs():
            """Publish scene object poses into wrc_demo's belief store.

            LIBERO's objects are kitchen items a YOLOE prompt list does not
            cover, and the whole point of this experiment is to measure the
            ORCHESTRATION layer, not perception. Pigey and ASPIRE do the same
            thing -- they supply perception externally (Gemini Robotics ER)
            and study the loop around it. Feeding exact poses here keeps the
            comparison about the loop.

            This is a PERCEPTION substitute, not a verification shortcut: the
            postcondition channel reads physics again AFTER the motion, so a
            skill still cannot confirm itself.
            """
            m = inner.sim.model
            for name in (obj_body, dest_body):
                if not name:
                    continue
                p = object_pos(name)
                if p is None:
                    continue
                rt.beliefs.update(label=name, position=np.asarray(p, float),
                                  conf=0.99,
                                  extent=np.array([0.06, 0.06, 0.06]))
            del m

        for cond in conds:
            ok = claimed = false = 0
            for ep in range(a.episodes):
                env.reset()
                obs = env.set_init_state(inits[ep % len(inits)])
                if isinstance(obs, tuple):
                    obs = obs[0]
                arm.last_obs = obs
                arm._terminated = False      # fresh episode, clear the latch
                # Seed the camera from the ENV thread before any skill runs:
                # the stream's worker thread will otherwise grab before the
                # first env.step and raise "no frame published yet".
                if getattr(arm, "camera", None) is not None:
                    arm.camera.publish(obs)
                for _ in range(10):                # settle
                    obs, _ = arm._step(np.zeros(8))
                arm.last_obs = obs
                seed_beliefs()      # external perception, Pigey-style

                if cond == "verified":
                    rt.attach_verifier(object_pose=object_pos)
                else:
                    rt.effects = None

                t0 = time.monotonic()
                self_ok = False
                attempts = 3 if cond == "verified" else 1
                try:
                    for _try in range(attempts):
                        r = rt.execute("pick_and_place",
                                       {"object": obj_body,
                                        "destination": dest_body or "table"})
                        self_ok = bool(r.get("ok"))
                        pc = r.get("postcondition") or {}
                        if pc.get("status") != "refuted":
                            break
                except Exception as e:
                    r = {"error": f"{type(e).__name__}: {e}"}
                if not self_ok:
                    # Surface WHY the skill declined: a silent 0% is
                    # indistinguishable from a broken harness.
                    print(f"        skill said: {str(r)[:220]}", flush=True)

                # LIBERO's OWN success flag decides
                done = getattr(arm, "_last_term", False)
                if not done:
                    for _ in range(20):
                        obs, term = arm._step(np.zeros(8))
                        if term:
                            done = True
                            break
                ok += int(done)
                claimed += int(self_ok)
                false += int(self_ok and not done)
                print(f"    [{tid}] {cond:10} ep{ep} done={done} "
                      f"self={self_ok} {time.monotonic()-t0:5.1f}s", flush=True)

            out[cond]["succ"] += ok
            out[cond]["n"] += a.episodes
            out[cond]["claimed"] += claimed
            out[cond]["false"] += false
            out[cond]["tasks"].append(
                {"task_id": tid, "language": task.language, "successes": ok})
            print(f"  [{tid}] {cond:10} {ok}/{a.episodes}  "
                  f"(claimed {claimed}, false {false})", flush=True)
        env.close()

    print("\n" + "=" * 62)
    for c in conds:
        d = out[c]
        print(f"  {c:12} {d['succ']:3}/{d['n']:<3} = {d['succ']/max(1,d['n']):.1%}"
              f"   claimed {d['claimed']}   FALSE CLAIMS {d['false']}")
    if a.json:
        Path(a.json).parent.mkdir(parents=True, exist_ok=True)
        Path(a.json).write_text(json.dumps(out, indent=2))
        print(f"[+] wrote {a.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
