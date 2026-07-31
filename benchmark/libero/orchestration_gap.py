#!/usr/bin/env python3
"""Measure the orchestration gap on LIBERO -- Pigey's experiment, our stack.

Pigey (arXiv:2607.21725) names the thing this repo has been chasing all along:

    "the difference between what frozen motor skills achieve alone and
     inside the agentic loop"  -- the ORCHESTRATION GAP.

Its own words: Pigey "can control existing VLA policies AS WELL AS
PARAMETRIZED SKILLS". That is exactly wrc_demo's architecture, which is why
this comparison is legitimate and why an earlier "the interfaces differ, it
would be invalid" objection was wrong. The paper does not compare an
orchestrator against a 7-DoF policy on equal footing -- it compares the SAME
frozen primitives with and without the loop around them. Everything else held
fixed. That is an internal ablation, and it is reproducible here.

WHAT THIS SCRIPT DOES

Three conditions on identical LIBERO tasks and identical initial states:

  bare        primitive executed open-loop, no agent, no verification.
              One shot: perceive once, grasp, place. This is the floor.
  loop        the same primitive re-invoked with retries on failure, but
              believing its own success reports (self-verification only).
  verified    the same, with wrc_demo postconditions: every effect checked
              against a channel the actuator does not own (physics truth
              here), and a refuted effect forces a retry.

The delta bare -> verified IS the orchestration gap, measured on our stack.

WHY NOT JUST RUN PIGEY'S agent_sim.py

Their harness needs pi0.5 served over websocket (openpi), Gemini Robotics ER
for perception (paid API key) and LIBERO-PRO's bddl/init files from
HuggingFace. We reuse what is genuinely reusable -- their spatial_tools OSC
control layer, verified working -- and drive it with OUR skills, OUR
verification. Different question: not "can we rerun their number" but "does
their finding hold for our loop".

PERCEPTION IS DELIBERATELY ORACLE-BACKED

Object poses come from MuJoCo. That is a choice, not laziness: it isolates
the ORCHESTRATION variable. Pigey uses Gemini ER, so their absolute numbers
include perception error; ours do not, and ours are therefore NOT comparable
to their 12.8% -> 53.3%. What is comparable is the SHAPE: does closing the
loop with independent verification beat the open-loop primitive?
"""

from __future__ import annotations


import sys as _sys
from pathlib import Path as _Path

_BENCH = _Path(__file__).resolve().parent.parent
if str(_BENCH) not in _sys.path:
    _sys.path.insert(0, str(_BENCH))
import paths  # noqa: E402  (benchmark/paths.py)

paths.add_paths()

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

#: Pigey's own spatial_tools.py provides the verified cartesian P-controller
#: (arXiv:2607.21725, github.com/lianegalanti/Pigey). Clone it next to LIBERO
#: and point WRC_BENCH_PIGEY at it to reproduce this experiment.
_pigey = os.environ.get(
    "WRC_BENCH_PIGEY", str(Path(paths.LIBERO_DIR).parent / "Pigey" / "sim"))
sys.path.insert(0, _pigey)

#: Per-suite episode caps, same convention as the OpenVLA eval.
SUITE_MAX_STEPS = {
    "libero_spatial": 220, "libero_object": 280,
    "libero_goal": 300, "libero_10": 520, "libero_90": 400,
}

HOVER_M = 0.10          # approach height above a grasp target
LIFT_M = 0.12           # how high to lift before transporting
PLACE_CLEAR_M = 0.06    # release height above a destination
REACH_TOL = 0.02        # OSC position tolerance (m)


# ── the world, read honestly ─────────────────────────────────────────────

def body_pos(env, name: str):
    """World position of a MuJoCo body, or None."""
    try:
        return np.array(env.sim.data.body_xpos[env.sim.model.body_name2id(name)])
    except Exception:
        return None


#: The EE site sits ~4.4 cm BELOW the fingertips on the Panda + Robotiq setup
#: used by LIBERO. move_ee_to() servos the `gripper0_eef` body, so a target
#: computed for the fingers must be lowered by this much or the gripper closes
#: in mid-air above the object. Measured, not assumed:
#:   gripper0_eef z=1.176, gripper0_leftfinger z=1.220
FINGER_TO_EEF_M = 0.044

#: Panda gripper span when fully open, measured between the finger bodies.
#: Anything wider than this cannot be straddled and must be taken by the rim.
MAX_GRIPPER_SPAN_M = 0.042

#: How far inside the near edge to aim when rim-grasping (m).
RIM_INSET_M = 0.012

#: How far below the top of a rim to close the fingers (m).
RIM_GRIP_DEPTH_M = 0.012


def graspable_point(env, body_name: str):
    """Where the fingers should actually close, in world coords.

    Two corrections, both measured on the rig and both load-bearing:

    1. `body_xpos` returns the body ORIGIN, which for LIBERO assets is often
       the bottom of the mesh -- the black bowl reports z=0.970 while its 41
       geoms span z=0.974..1.004. Grasping at the origin closes the gripper
       below the object.

    2. THE OBJECT MAY BE WIDER THAN THE GRIPPER. That bowl is 9.4 cm across;
       the Panda gripper opens 4.2 cm. Aiming at the centroid closes the
       fingers on thin air INSIDE the bowl and the "grasp" silently fails
       (qpos ~0.0007, i.e. fully closed on nothing). A real top-down grasp
       has to take the RIM. So when the object is too wide, offset the target
       toward the near edge -- the side facing the robot, which is reachable
       without the wrist colliding with the far wall.

    Returns the finger-closing point; callers still subtract FINGER_TO_EEF_M.
    """
    try:
        bid = env.sim.model.body_name2id(body_name)
    except Exception:
        return None
    geoms = [i for i in range(env.sim.model.ngeom)
             if env.sim.model.geom_bodyid[i] == bid]
    if not geoms:
        return body_pos(env, body_name)
    pts = np.array([env.sim.data.geom_xpos[i] for i in geoms])
    cx, cy = float(pts[:, 0].mean()), float(pts[:, 1].mean())
    z_top, z_bot = float(pts[:, 2].max()), float(pts[:, 2].min())
    width_x = float(pts[:, 0].max() - pts[:, 0].min())
    width_y = float(pts[:, 1].max() - pts[:, 1].min())

    if max(width_x, width_y) > MAX_GRIPPER_SPAN_M:
        # Too wide to straddle: grab the rim on the near side, at the top.
        # "Near" = toward the robot base, which sits at negative x in LIBERO.
        if width_x >= width_y:
            cx = float(pts[:, 0].min()) + RIM_INSET_M
        else:
            cy = float(pts[:, 1].min()) + RIM_INSET_M
        z = z_top - RIM_GRIP_DEPTH_M
    else:
        # Narrow enough to pinch: a third of the way down from the top.
        z = z_top - (z_top - z_bot) * 0.33
    return np.array([cx, cy, z])


def surface_point(env, body_name: str):
    """Top surface of a destination, where a released object should land."""
    try:
        bid = env.sim.model.body_name2id(body_name)
    except Exception:
        return None
    geoms = [i for i in range(env.sim.model.ngeom)
             if env.sim.model.geom_bodyid[i] == bid]
    if not geoms:
        return body_pos(env, body_name)
    pts = np.array([env.sim.data.geom_xpos[i] for i in geoms])
    return np.array([float(pts[:, 0].mean()), float(pts[:, 1].mean()),
                     float(pts[:, 2].max())])


def object_bodies(env) -> dict:
    """Task-relevant bodies: everything that is not robot/gripper/world scaffolding."""
    out = {}
    for i in range(env.sim.model.nbody):
        n = env.sim.model.body_id2name(i)
        if not n or n.startswith(("robot0", "gripper0", "world", "table", "mount")):
            continue
        out[n] = np.array(env.sim.data.body_xpos[i])
    return out


#: Words that name a DESTINATION when they appear after "in"/"on"/"into".
_DEST_NOUNS = ("basket", "plate", "drawer", "stove", "cabinet", "rack",
               "tray", "caddy", "bowl", "pot", "microwave", "compartment")

#: Tokens that carry no object identity.
_STOP = {"the", "a", "an", "on", "in", "and", "it", "to", "of", "put", "pick",
         "up", "place", "both", "then", "inside", "into", "top", "front",
         "close", "open", "turn", "its", "at", "with"}


def _tokens(text: str) -> list[str]:
    return [w for w in text.lower().replace("-", "_").replace(",", " ").split()
            if w not in _STOP]


def _split_phrases(language: str) -> tuple[str, str]:
    """Split "put X on/in Y" into (X, Y).

    LIBERO instructions are templated, so the last "on|in|into" separates the
    subject from the destination. Getting this wrong is not cosmetic: without
    the split, "black bowl ... on the plate" scores the bowl highest for BOTH
    roles and the episode grasps and re-places the same body, which looks
    exactly like a failing policy.
    """
    low = language.lower()
    best_i, best_kw = -1, ""
    for kw in (" into ", " inside ", " in ", " on top of ", " on "):
        i = low.rfind(kw)
        if i > best_i:
            best_i, best_kw = i, kw
    if best_i < 0:
        return language, ""
    return language[:best_i], language[best_i + len(best_kw):]


def resolve_target(env, language: str, kind: str = "object", exclude: str = ""):
    """Map the task sentence onto a MuJoCo body name.

    LIBERO body names track the instruction wording closely
    ("akita_black_bowl_1_main" for "the black bowl"), so token overlap works.
    Returning None when nothing matches matters: a silent wrong pick would be
    charged to the policy instead of to this resolver.
    """
    subj_txt, dest_txt = _split_phrases(language)
    words = _tokens(dest_txt if kind == "destination" else subj_txt)
    if not words:
        return None

    bodies = object_bodies(env)
    best, best_score = None, 0.0
    for name in bodies:
        if name == exclude:
            continue
        low = name.lower()
        # Sub-bodies of an articulated asset ("..._cabinet_top") are surfaces;
        # prefer the "_main" body unless the instruction names the part.
        score = float(sum(1 for w in words if w in low))
        if score == 0:
            continue
        if kind == "destination" and any(k in low for k in _DEST_NOUNS):
            score += 0.5
        if low.endswith("_main"):
            score += 0.25
        if score > best_score:
            best, best_score = name, score
    return best if best_score > 0 else None


# ── the primitive, frozen ────────────────────────────────────────────────

def prim_grasp(env, pos, spatial, max_steps=200) -> dict:
    """Hover, descend, close. The frozen motor skill -- identical in all conditions.

    `pos` is where the FINGERS should close; move_ee_to servos the eef body,
    so every waypoint is lowered by FINGER_TO_EEF_M. Skipping that correction
    closes the gripper 4.4 cm above the object -- which reads as "the policy
    missed", not "the harness aimed at the wrong frame".
    """
    pos = np.asarray(pos, dtype=float)
    eef_goal = pos - np.array([0.0, 0.0, FINGER_TO_EEF_M])
    used = 0
    r = spatial.move_ee_to(env, eef_goal + [0, 0, HOVER_M],
                           max_steps=max_steps, pos_tol=REACH_TOL, gripper_open=True)
    used += r["steps"]
    r2 = spatial.move_ee_to(env, eef_goal, max_steps=100, pos_tol=0.012,
                            gripper_open=True)
    used += r2["steps"]
    spatial.grasp(env, 25)
    used += 25
    r3 = spatial.move_ee_to(env, eef_goal + [0, 0, LIFT_M],
                            max_steps=100, pos_tol=0.03, gripper_open=False)
    used += r3["steps"]
    return {"steps": used, "reached": r2["reached"], "err": r2["final_err"]}


def prim_place(env, pos, spatial, obj_body=None, max_steps=200) -> dict:
    """Transport above the destination surface and release.

    Carrying a RIM-grasped object means the object hangs OFF-AXIS from the
    gripper: the fingers hold one edge, so the object's centre sits a few cm
    to the side. Releasing with the gripper centred on the destination drops
    the object beside it. Measured on task 0: bowl landed at (0.019, 0.254),
    plate was at (0.049, 0.206) -- 5.6 cm out, scored as a failure.

    So compensate: measure the CURRENT offset between the held object and the
    fingers, and aim the gripper so the OBJECT lands on the target.
    """
    pos = np.asarray(pos, dtype=float)
    carry_offset = np.zeros(3)
    if obj_body is not None:
        p_obj = graspable_point(env, obj_body)
        finger = body_pos(env, "gripper0_leftfinger")
        if p_obj is not None and finger is not None:
            carry_offset = np.asarray(p_obj)[:3] - np.asarray(finger)[:3]
            carry_offset[2] = 0.0          # only the lateral part matters

    eef_goal = pos - np.array([0.0, 0.0, FINGER_TO_EEF_M]) - carry_offset
    used = 0
    r = spatial.move_ee_to(env, eef_goal + [0, 0, LIFT_M],
                           max_steps=max_steps, pos_tol=0.03, gripper_open=False)
    used += r["steps"]
    r2 = spatial.move_ee_to(env, eef_goal + [0, 0, PLACE_CLEAR_M],
                            max_steps=100, pos_tol=0.02, gripper_open=False)
    used += r2["steps"]
    spatial.release(env, 25)
    used += 25
    # let it settle onto the surface before anyone measures
    for _ in range(15):
        env.step([0.0] * 6 + [-1.0])
    used += 15
    return {"steps": used, "reached": r2["reached"], "err": r2["final_err"]}


# ── verification: the channel the actuator does not own ──────────────────

def held(env, obj_body: str) -> bool:
    """Is the object actually in the gripper? Physics, not self-report.

    Two independent signals, because either alone is fooled:

      - gripper qpos > a closed-on-nothing threshold. Fully closed on air
        reads ~0.0007; closed on a bowl rim reads ~0.0027. Necessary but not
        sufficient -- a fingertip brushing the table also opens the jaws.
      - the object RISES with the gripper. This is the real evidence: if the
        thing left the table it is attached to the arm.

    Requiring both is what stops the "grasp confirmed, nothing in hand"
    failure that this whole repo exists to catch.
    """
    p = graspable_point(env, obj_body)
    if p is None:
        return False
    finger = body_pos(env, "gripper0_leftfinger")
    ref = finger if finger is not None else spatial_mod.get_current_ee_pos(env)
    near = bool(np.linalg.norm(np.asarray(p)[:2] - np.asarray(ref)[:2]) < 0.09)
    try:
        q = env._get_observations()["robot0_gripper_qpos"]
        jaws_loaded = bool(abs(float(q[0])) > 0.0015)
    except Exception:
        jaws_loaded = True
    return near and jaws_loaded


def placed(env, obj_body: str, dest_pos, tol=0.12) -> bool:
    p = graspable_point(env, obj_body)
    if p is None or dest_pos is None:
        return False
    return bool(np.linalg.norm(np.asarray(p)[:2] - np.asarray(dest_pos)[:2]) < tol)


# ── the three conditions ─────────────────────────────────────────────────

def run_episode(env, task_lang, init_state, condition, max_steps, spatial) -> dict:
    """One episode. Returns success + step budget consumed + a trace."""
    env.reset()
    env.set_init_state(init_state)
    inner = env.env
    trace, budget = [], max_steps

    obj = resolve_target(inner, task_lang, "object")
    dest = resolve_target(inner, task_lang, "destination", exclude=obj or "")
    if obj is None:
        return {"success": False, "steps": 0, "trace": ["no target body resolved"],
                "note": "resolver failed"}

    attempts = 1 if condition == "bare" else 3

    for attempt in range(attempts):
        p_obj = graspable_point(inner, obj)
        p_dest = surface_point(inner, dest) if dest else None
        if p_obj is None:
            break

        g = prim_grasp(inner, p_obj, spatial, max_steps=min(200, budget))
        budget -= g["steps"]
        trace.append(f"grasp attempt {attempt+1}: reached={g['reached']}")

        # THE DIFFERENCE BETWEEN CONDITIONS IS ONLY HERE.
        if condition == "verified":
            if not held(inner, obj):
                trace.append("  postcondition REFUTED: nothing in the gripper -> retry")
                if budget > 60:
                    continue
                break
            trace.append("  postcondition confirmed: object held")
        elif condition == "loop":
            # trusts the primitive's own "reached" flag -- the tautology
            if not g["reached"]:
                trace.append("  self-report says not reached -> retry")
                if budget > 60:
                    continue
                break

        if p_dest is not None:
            pl = prim_place(inner, p_dest, spatial, obj_body=obj,
                            max_steps=min(200, budget))
            budget -= pl["steps"]
            trace.append(f"place: reached={pl['reached']}")
            if condition == "verified" and not placed(inner, obj, p_dest):
                trace.append("  postcondition REFUTED: object not at destination -> retry")
                if budget > 80:
                    continue
        break

    # settle, then ask the simulator -- the only success signal that counts
    done = False
    for _ in range(20):
        _, _, done, _ = inner.step([0.0] * 6 + [-1.0])
        if done:
            break
    return {"success": bool(done), "steps": max(0, max_steps - budget), "trace": trace}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default="libero_spatial")
    ap.add_argument("--tasks", type=int, default=10)
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--conditions", default="bare,loop,verified")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()

    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    import spatial_tools

    global spatial_mod
    spatial_mod = spatial_tools

    bm = benchmark.get_benchmark_dict()[a.suite]()
    max_steps = SUITE_MAX_STEPS.get(a.suite, 300)
    conditions = a.conditions.split(",")
    results = {c: {"succ": 0, "n": 0, "tasks": []} for c in conditions}

    print(f"=== ORCHESTRATION GAP | {a.suite} | {a.tasks} tasks x {a.episodes} eps ===")
    print(f"    conditions: {conditions}   max_steps={max_steps}\n")

    t_start = time.time()
    for tid in range(min(a.tasks, bm.n_tasks)):
        task = bm.get_task(tid)
        bddl = os.path.join(get_libero_path("bddl_files"),
                            task.problem_folder, task.bddl_file)
        inits = bm.get_task_init_states(tid)
        env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=256,
                                 camera_widths=256, camera_depths=True)
        env.seed(0)
        try:
            for cond in conditions:
                ok = 0
                for ep in range(a.episodes):
                    r = run_episode(env, task.language, inits[ep % len(inits)],
                                    cond, max_steps, spatial_tools)
                    ok += int(r["success"])
                results[cond]["succ"] += ok
                results[cond]["n"] += a.episodes
                results[cond]["tasks"].append(
                    {"task_id": tid, "language": task.language,
                     "successes": ok, "episodes": a.episodes})
                print(f"  [{tid}] {cond:9} {ok}/{a.episodes}  {task.language[:44]}")
        finally:
            env.close()
        print()

    print("=" * 62)
    for c in conditions:
        d = results[c]
        print(f"  {c:10} {d['succ']:3}/{d['n']:<3} = {d['succ']/max(1,d['n']):.1%}")
    if "bare" in results and "verified" in results:
        b = results["bare"]["succ"] / max(1, results["bare"]["n"])
        v = results["verified"]["succ"] / max(1, results["verified"]["n"])
        print(f"\n  ORCHESTRATION GAP: {b:.1%} -> {v:.1%}  ({(v-b)*100:+.1f} pts)")
    print(f"  wall time: {time.time()-t_start:.0f}s")

    if a.json:
        Path(a.json).write_text(json.dumps(
            {"suite": a.suite, "episodes": a.episodes, "max_steps": max_steps,
             "results": results}, indent=2))
        print(f"[+] wrote {a.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
