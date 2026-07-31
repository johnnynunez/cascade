#!/usr/bin/env python3
"""Run the LIBERO benchmark the papers report on.

Protocol matches the OpenVLA / LIBERO convention so numbers are comparable
with published results:

  - each of the 10 tasks in a suite is run from N fixed init states (the
    benchmark ships them; episode i uses init state i, so runs are
    reproducible and every policy sees the same starts);
  - 10 no-op steps after set_init_state let objects settle before the policy
    acts (skipping this is the classic way to get an inflated failure rate --
    the first frames show a scene still falling);
  - success is the environment's own `done`, not a heuristic;
  - max 600 steps (LIBERO's own cap for these suites).

Policies:
  noop      zero action every step. This is the FLOOR: any task it "solves"
            was solvable by doing nothing, so a real policy's score must be
            read against it. Reported explicitly rather than assumed to be 0.
  random    uniform random actions -- the second floor, and a check that the
            env can be perturbed at all.
  openvla   the finetuned checkpoint the papers evaluate.

Usage:
    MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0 \
    python run_libero.py --suite libero_spatial --policy noop --episodes 5
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

#: LIBERO is installed editable from a clone; make the runner work from any
#: cwd rather than only from inside that clone.
_LIBERO_ROOT = Path(os.environ.get("LIBERO_ROOT", str(paths.LIBERO_DIR)))
if _LIBERO_ROOT.is_dir() and str(_LIBERO_ROOT) not in sys.path:
    sys.path.insert(0, str(_LIBERO_ROOT))

MAX_STEPS = 600
SETTLE_STEPS = 10

#: Per-suite episode caps from the OFFICIAL OpenVLA eval
#: (experiments/robot/libero/run_libero_eval.py). These are not arbitrary:
#: each is "longest training demo + margin", so a generic 600 both wastes
#: hours on doomed episodes and makes the numbers incomparable with published
#: results -- a policy that would have been scored a failure at 220 gets 380
#: extra steps to stumble into success.
SUITE_MAX_STEPS = {
    "libero_spatial": 220,   # longest training demo: 193
    "libero_object": 280,    # 254
    "libero_goal": 300,      # 270
    "libero_10": 520,        # 505
    "libero_90": 400,        # 373
}

#: OpenVLA was fine-tuned WITH random-crop augmentation, so its eval asserts
#: center_crop=True. Skipping it is a silent accuracy loss, not a style choice.
CROP_SCALE = 0.9
VLA_INPUT_PX = 224


def postprocess_gripper(action: np.ndarray) -> np.ndarray:
    """Map OpenVLA's gripper convention onto the environment's.

    Two steps, both from the official eval, and BOTH are load-bearing:

    1. normalize: the RLDS dataloader leaves the gripper dim in [0, 1] while
       every other dim is [-1, +1], so rescale and binarise to +/-1.
    2. invert:    the dataloader aligns gripper actions as 0=close, 1=open,
       but LIBERO wants -1=open, +1=close.

    Skipping these does not fail loudly -- the arm reaches correctly and then
    opens when it should close. Measured here: task 0 scored 0% without this
    step, against ~84% published.
    """
    a = np.asarray(action, dtype=float).copy()
    a[-1] = np.sign(2.0 * a[-1] - 1.0)     # [0,1] -> [-1,+1], binarised
    a[-1] *= -1.0                          # 0=close,1=open -> -1=open,+1=close
    return a


def center_crop_resize(img: np.ndarray, crop_scale: float = CROP_SCALE,
                       out_px: int = VLA_INPUT_PX) -> np.ndarray:
    """Numpy port of the official TF `crop_and_resize` preprocessing.

    Takes a centred crop of area `crop_scale` x original, then resizes to
    224x224. Note the crop side is sqrt(crop_scale), not crop_scale -- the
    upstream comment flags this explicitly as an easy thing to get wrong.

    Ported rather than pulled in: the reference path drags TensorFlow in for
    one crop, and TF would fight this venv's numpy/CUDA pins.
    """
    import cv2

    h, w = img.shape[:2]
    side = float(np.clip(np.sqrt(crop_scale), 0, 1))
    ch, cw = int(round(h * side)), int(round(w * side))
    top, left = (h - ch) // 2, (w - cw) // 2
    crop = img[top:top + ch, left:left + cw]
    # bilinear matches tf.image.crop_and_resize's default
    return cv2.resize(crop, (out_px, out_px), interpolation=cv2.INTER_LINEAR)


def make_policy(name: str, model_path: str | None, device: str = "cuda:0"):
    """Return (fn(obs, task_language) -> 7-vector action, description)."""
    if name == "noop":
        return (lambda obs, lang: np.zeros(7)), "zero action (floor)"

    if name == "random":
        rng = np.random.default_rng(0)

        def _rand(obs, lang):
            a = rng.uniform(-1, 1, 7)
            a[-1] = rng.choice([-1.0, 1.0])
            return a

        return _rand, "uniform random (floor)"

    if name == "openvla":
        import torch
        from PIL import Image
        from transformers import AutoModelForVision2Seq, AutoProcessor

        path = model_path or "openvla/openvla-7b-finetuned-libero-spatial"
        proc = AutoProcessor.from_pretrained(path, trust_remote_code=True)
        model = AutoModelForVision2Seq.from_pretrained(
            path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        ).to(device)
        model.eval()

        def _vla(obs, lang, unnorm_key):
            # 180-degree rotation to match training preprocessing, then the
            # centre crop the checkpoint was fine-tuned with. Both come from
            # the official eval; dropping either quietly costs accuracy.
            img = obs["agentview_image"][::-1, ::-1]
            img = center_crop_resize(img)
            prompt = f"In: What action should the robot take to {lang.lower()}?\nOut:"
            inputs = proc(prompt, Image.fromarray(img).convert("RGB")).to(
                device, dtype=torch.bfloat16
            )
            with torch.no_grad():
                a = model.predict_action(
                    **inputs, unnorm_key=unnorm_key, do_sample=False
                )
            return postprocess_gripper(a)

        return _vla, f"OpenVLA ({path}, center_crop={CROP_SCALE})"

    raise SystemExit(f"unknown policy {name!r}")


def run_suite(suite: str, policy_name: str, episodes: int, model_path: str | None,
              res: int = 256, verbose: bool = True) -> dict:
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    policy, policy_desc = make_policy(policy_name, model_path)
    bm = benchmark.get_benchmark_dict()[suite]()
    bddl_root = get_libero_path("bddl_files")
    max_steps = SUITE_MAX_STEPS.get(suite, MAX_STEPS)

    # The checkpoint's action de-normalisation key. The official eval falls
    # back to "<suite>_no_noops" when present -- using the wrong key silently
    # scales every action wrong and looks like a bad policy.
    unnorm_key = suite
    if policy_name == "openvla":
        import json as _json
        stats_p = Path(model_path or "") / "dataset_statistics.json"
        if stats_p.is_file():
            keys = _json.loads(stats_p.read_text()).keys()
            if suite not in keys and f"{suite}_no_noops" in keys:
                unnorm_key = f"{suite}_no_noops"
        if verbose:
            print(f"  unnorm_key={unnorm_key}  max_steps={max_steps}")

    per_task, t_start = [], time.time()
    for tid in range(bm.n_tasks):
        task = bm.get_task(tid)
        env = OffScreenRenderEnv(
            bddl_file_name=os.path.join(bddl_root, task.problem_folder, task.bddl_file),
            camera_heights=res,
            camera_widths=res,
        )
        env.seed(0)
        inits = bm.get_task_init_states(tid)
        wins, lengths = 0, []

        for ep in range(min(episodes, len(inits))):
            env.reset()
            obs = env.set_init_state(inits[ep])
            # let the scene settle -- acting on a still-falling scene is the
            # classic source of a fake-looking failure rate
            for _ in range(SETTLE_STEPS):
                obs, _, _, _ = env.step(np.zeros(7))

            done, steps = False, 0
            while not done and steps < max_steps:
                if policy_name == "openvla":
                    action = policy(obs, task.language, unnorm_key)
                else:
                    action = policy(obs, task.language)
                obs, _, done, _ = env.step(np.asarray(action, dtype=float).reshape(7))
                steps += 1
            wins += int(done)
            lengths.append(steps)

        env.close()
        sr = wins / max(min(episodes, len(inits)), 1)
        per_task.append({
            "task_id": tid,
            "language": task.language,
            "episodes": min(episodes, len(inits)),
            "successes": wins,
            "success_rate": round(sr, 4),
            "mean_steps": round(float(np.mean(lengths)), 1),
        })
        if verbose:
            el = time.time() - t_start
            print(f"  [{tid}] {sr:5.0%}  {task.language[:56]}  ({el/60:.0f}m)",
                  flush=True)

    total_ep = sum(t["episodes"] for t in per_task)
    total_ok = sum(t["successes"] for t in per_task)
    return {
        "suite": suite,
        "policy": policy_name,
        "policy_desc": policy_desc,
        "episodes_per_task": episodes,
        "resolution": res,
        "max_steps": max_steps,
        "settle_steps": SETTLE_STEPS,
        "unnorm_key": unnorm_key if policy_name == "openvla" else None,
        "tasks": per_task,
        "episodes_total": total_ep,
        "successes_total": total_ok,
        "success_rate": round(total_ok / max(total_ep, 1), 4),
        "wall_time_s": round(time.time() - t_start, 1),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="LIBERO benchmark runner")
    ap.add_argument("--suite", default="libero_spatial",
                    choices=["libero_spatial", "libero_object", "libero_goal",
                             "libero_10", "libero_90"])
    ap.add_argument("--policy", default="noop", choices=["noop", "random", "openvla"])
    ap.add_argument("--episodes", type=int, default=5, help="init states per task")
    ap.add_argument("--model-path", default=None)
    ap.add_argument("--res", type=int, default=256)
    ap.add_argument("--json", type=Path)
    args = ap.parse_args()

    if os.environ.get("MUJOCO_GL") != "egl":
        print("[!] set MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0 for headless rendering")

    print(f"\n=== LIBERO {args.suite} | policy={args.policy} | "
          f"{args.episodes} episodes/task ===")
    r = run_suite(args.suite, args.policy, args.episodes, args.model_path, args.res)
    print(f"\nSUCCESS RATE: {r['success_rate']:.1%} "
          f"({r['successes_total']}/{r['episodes_total']})  "
          f"[{r['wall_time_s']:.0f}s]\n")
    if args.json:
        args.json.write_text(json.dumps(r, indent=2))
        print(f"[+] wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
