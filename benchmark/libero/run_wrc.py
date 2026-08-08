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
import threading
import sys
import time
from pathlib import Path

import numpy as np

SUITE_MAX_STEPS = {"libero_spatial": 220, "libero_object": 280,
                   "libero_goal": 300, "libero_10": 520}


def build_wrc_runtime(env, task_language: str, perception: str = "oracle"):
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
    # Detector choice follows the perception mode. Oracle mode keeps the mock
    # detector (poses arrive via seed_beliefs, and the LIBERO venv has no
    # ultralytics). Camera mode runs the real open-vocabulary detector on the
    # rendered frames, which is what makes a number comparable to systems that
    # perceive their own scene.
    det = {"type": "mock"} if perception == "oracle" else {
        "type": "yoloe",
        "model": str(_Path(__file__).resolve().parents[2] / "models"
                     / "yoloe-11s-seg.pt"),
        "device": "cuda:0",
        "conf": 0.25,
        "prompt_free": True,
    }
    cfg.camera._data["detector"] = det
    cfg._data["detector"] = det

    # The Panda is 7-DoF in a different workspace than the RS arm: the safety
    # harness must be given the limits of the robot we are actually driving.
    lo, hi = kin.joint_limits
    cfg.safety._data["joint_limits"] = [[float(x), float(y)]
                                        for x, y in zip(lo, hi)]
    cfg.safety._data["table_z"] = 0.80          # LIBERO table top, world frame

    # The workspace AABB must move with the table. The shipped one is the RS
    # arm's, verified against its URDF reachability probe: x 0.10..0.50,
    # y +-0.30, z -0.01..0.55 in ITS base frame. LIBERO's table top is at
    # z = 0.80 world, so every reachable pregrasp here sits ABOVE that ceiling
    # and the safety layer rejected it with
    #     "pregrasp unsafe: TCP [0.138, -0.085, 1.009] outside workspace"
    # -- a harness misconfiguration reported as a skill failure. Size the box
    # around the Panda's actual reach in the world frame it is driven in.
    cfg.safety._data["workspace"] = {
        "min": [-0.60, -0.60, 0.75],
        "max": [0.60, 0.60, 1.35],
    }

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


def _human_label(body_name: str) -> str:
    """MuJoCo body name -> the words a person would say.

    `akita_black_bowl_2_main` -> `black bowl`. Instance indices, the `_main`
    suffix and vendor prefixes are scene structure, not something a camera can
    recover, so they are stripped rather than handed to the detector.

    Kept deliberately dumb: it is a naming convention adapter, not a parser of
    the instruction. The instruction itself is what the agent reasons over.
    """
    if not body_name:
        return ""
    s = re.sub(r"_main$", "", body_name)
    s = re.sub(r"_\d+$", "", s)
    parts = [p for p in s.split("_") if p and not p.isdigit()]
    # Vendor/collection prefixes that never appear in speech. Only dropped when
    # something remains, so an unknown asset still yields a usable label.
    for junk in ("akita", "libero", "ycb", "obj", "target", "region"):
        if len(parts) > 1 and parts[0] == junk:
            parts = parts[1:]
    return " ".join(parts)


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


def _task_success(env) -> bool:
    """LIBERO's own success predicate, read from live object state.

    The harness used to decide success from `arm._last_term`, a cached copy of
    the last step's terminated flag. That is a latch, and latches go stale:
    it was never reset between episodes, so an episode whose skill failed fast
    (mock detector finds nothing, ~1.3 s, zero detections) inherited the
    previous episode's True and scored a success without the robot moving.
    Verified in isolation -- benchmark/diagnostics/last_term_leak.py.

    `_check_success()` asks the task about object state right now, so it cannot
    carry over. Unwrap the env layers to reach it; return False if a build does
    not expose it, in which case the step-return path still applies.
    """
    e = env
    for _ in range(6):
        fn = getattr(e, "_check_success", None)
        if callable(fn):
            try:
                return bool(fn())
            except Exception:
                return False
        nxt = getattr(e, "env", None)
        if nxt is None:
            return False
        e = nxt
    return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default="libero_spatial")
    ap.add_argument("--tasks", type=int, default=3)
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--conditions", default="skill_only,verified")
    ap.add_argument("--json", default=None)
    ap.add_argument(
        "--perception", default="oracle", choices=["oracle", "camera"],
        help=(
            "oracle: seed beliefs from sim.data.body_xpos (measures the "
            "orchestration loop given perfect perception). camera: run the "
            "real detector on the rendered frames. ASPIRE (arXiv:2607.00272) "
            "FORBIDS ground-truth object positions and says using them "
            "'invalidates benchmark results'; any number published against "
            "ASPIRE, Pigey or Harness-VLA must therefore use --perception "
            "camera, or state the oracle prominently."
        ),
    )
    a = ap.parse_args()

    if a.perception == "oracle":
        print(
            "\n*** ORACLE PERCEPTION ***\n"
            "Object poses come from sim.data.body_xpos, not from the camera.\n"
            "These numbers measure ORCHESTRATION GIVEN PERFECT PERCEPTION and\n"
            "are NOT comparable to ASPIRE / Pigey / Harness-VLA / VIA, which\n"
            "perceive their own scenes (ASPIRE explicitly forbids this API).\n"
            "Use --perception camera for a comparable number.\n",
            file=sys.stderr, flush=True,
        )

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
        rt, arm, kin = build_wrc_runtime(env, task.language, a.perception)
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

            ORACLE PERCEPTION. This reads `sim.data.body_xpos`, the simulator's
            own object positions, and hands them to the agent. It measures the
            orchestration loop with perception held perfect, which is a
            legitimate ablation but is NOT what the comparable systems do.

            ASPIRE (arXiv:2607.00272) forbids this API by name in its agent
            instructions: "[FORBIDDEN] sim.data.body_xpos - no ground-truth
            object positions ... Using them invalidates benchmark results, as
            they don't transfer to real robots." ASPIRE uses SAM3 on the RGB
            frame instead. VIA states it takes "no access to privileged state
            information". Pigey holds "robot, cameras, scenes ... fixed".

            An earlier version of this docstring claimed Pigey and ASPIRE
            supply perception externally via Gemini Robotics ER. That claim
            could not be substantiated from either source and is contradicted
            by ASPIRE's own text; it has been removed rather than left to
            justify the shortcut.

            Verification is still independent: the postcondition channel reads
            physics again AFTER the motion, so a skill cannot confirm itself.

            Must be re-published DURING the episode, not only at the start.
            `_localize` accepts a belief as a detector fallback only while it
            is younger than `perception_loop.belief_fallback_age_s` (3 s), and
            the mock detector never refreshes it. Seeding once meant grasp
            retries 1-3 saw a valid fix and retries 4-8 got "no detections"
            purely because the clock ran out -- the sweep was measuring belief
            expiry, not orchestration.
            """
            if a.perception != "oracle":
                return
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

        # Background re-publisher: keeps the external perception feed alive for
        # the duration of a skill, the way a real perception service would.
        # Without it the belief expires 3 s in and later grasp retries fail on
        # staleness rather than on anything the orchestration layer did.
        _pub_stop = threading.Event()

        def _publish_loop():
            while not _pub_stop.is_set():
                try:
                    seed_beliefs()
                except Exception:
                    pass          # a torn-down env at episode end is expected
                _pub_stop.wait(1.0)

        _pub_thread = threading.Thread(target=_publish_loop, daemon=True)
        _pub_thread.start()

        for cond in conds:
            ok = claimed = false = 0
            for ep in range(a.episodes):
                env.reset()
                obs = env.set_init_state(inits[ep % len(inits)])
                if isinstance(obs, tuple):
                    obs = obs[0]
                arm.last_obs = obs
                arm._terminated = False      # fresh episode, clear the latch
                # `_last_term` is what decides success below, and it is NOT
                # implied by `_terminated`. Leaving it set carries the PREVIOUS
                # episode's termination into this one: a skill that fails fast
                # (mock detector finds nothing, ~1.3 s) never clears it, and
                # the episode is scored a success before the robot moves.
                # Verified in isolation -- benchmark/diagnostics/last_term_leak.py.
                arm._last_term = False
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
                # In camera mode the agent must refer to objects the way a
                # person would, not by MuJoCo body name: 'akita_black_bowl_2
                # _main' is privileged scene structure and no detector will
                # ever emit it. ASPIRE forbids reading sim assets for exactly
                # this reason. Oracle mode keeps the body name because that is
                # the key the seeded belief was stored under.
                obj_arg = obj_body if a.perception == "oracle" else _human_label(obj_body)
                dest_arg = (dest_body or "table") if a.perception == "oracle" \
                    else _human_label(dest_body or "table")
                try:
                    for _try in range(attempts):
                        r = rt.execute("pick_and_place",
                                       {"object": obj_arg,
                                        "destination": dest_arg})
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

                # LIBERO's OWN success flag decides -- read ONLY from the task
                # predicate, which interrogates object state right now.
                #
                # Neither `_last_term` nor the `term` returned by `_step` is
                # trustworthy here. `_last_term` is a latch that was never
                # reset between episodes, and once `_terminated` is set
                # `_step` short-circuits and returns True WITHOUT stepping the
                # sim (backend.py:79-80), so both can report success for an
                # episode in which the robot never moved. Reproduced:
                # benchmark/diagnostics/last_term_leak.py.
                done = _task_success(env)
                if not done:
                    for _ in range(20):
                        arm._step(np.zeros(8))
                        if _task_success(env):
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
        # stop the perception feed before tearing the env down, so the thread
        # cannot touch a closed sim
        _pub_stop.set()
        _pub_thread.join(timeout=3.0)
        env.close()

    print("\n" + "=" * 62)
    for c in conds:
        d = out[c]
        print(f"  {c:12} {d['succ']:3}/{d['n']:<3} = {d['succ']/max(1,d['n']):.1%}"
              f"   claimed {d['claimed']}   FALSE CLAIMS {d['false']}")
    if a.perception == "oracle":
        print("  [!] ORACLE PERCEPTION: not comparable to camera-based systems")
    if a.json:
        Path(a.json).parent.mkdir(parents=True, exist_ok=True)
        # Provenance travels WITH the numbers. A results file that does not say
        # how it perceived can be quoted years later as if it were comparable.
        payload = {
            "suite": a.suite,
            "episodes_per_task": a.episodes,
            "max_steps": max_steps,
            "perception": a.perception,
            "perception_note": (
                "oracle: object poses read from sim.data.body_xpos; measures "
                "orchestration given perfect perception; NOT comparable to "
                "ASPIRE/Pigey/Harness-VLA/VIA, and ASPIRE forbids this API "
                "(arXiv:2607.00272)"
                if a.perception == "oracle"
                else "camera: detector run on rendered frames"
            ),
            "results": out,
        }
        Path(a.json).write_text(json.dumps(payload, indent=2))
        print(f"[+] wrote {a.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
