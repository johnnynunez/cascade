# B32a on Isaac: the render self-mask gate in semantic fusion (8 Oct 2026)

**Question.** The B31 live check found the bare scene's whole open-vocabulary phantom rate (17.9–19.6 %) in ONE
belief. The side camera sees the arm's own upper link, and 98–99 % of that detection's pixels are robot pixels in
the bridge's render self-mask (`docs/evidence/b31-isaac-same-colour-20261008/`, per-detection diagnosis with
crops). Does gating fusion on that mask remove the phantom without costing a real prop?

**Answer: yes.** With the gate the phantom rate is 0 % and precision is 100 % in 3/3 runs. Without it, 17.9 / 17.9
/ 19.6 % in 3/3. Recall is 100 % in both arms, and localisation error is no worse with the gate. Measured on the
x86 test rig (RTX PRO 6000 GPU 0); these are not Spark numbers.

## Setup

- **Source.** Commit `d918180` (`main` 83a5ddd + the gate), clean tree. The bridge code is identical to `main`.
- **Simulator.** Isaac Sim `6.2.0-alpha.19+develop.0.48b2d951` (internal build), **PhysX**, private bridge on port
  18521 with `CASCADE_ISAAC_PIXEL_MASK=1`.
- **Scene.** The bare reBot scene: `pink_cube` (0.17, 0.15), `green_cube` (0.30, 0.16), and the bin at
  (0.18, -0.17). The props are reset before each run.
- **Perception.** YOLOE prompt-free (the shipped `detector:` block), cameras `isaac` + `isaac_side`, and the real
  `build_runtime` → WorldWatcher + `get_observation` path. Nothing moves the arm.
- **Metric.** `scripts/eval_detector.evaluate` over 12 frames, the metric behind
  `benchmark/results/perception_open_vocab.json`, with PhysX truth.
- **Arms.** `workspace_filter.self_mask: true` (the new default) vs `false` (the baseline, i.e. `main`).
- **Runs.** 3 per arm, interleaved on, off, on, off, on, off. Each run is one fresh process with a fresh belief
  store.
- **Files.** The harness is [`b32_live_ab.py`](b32_live_ab.py) with [`run_ab.sh`](run_ab.sh); the compact rows are
  in [`ab_summary.json`](ab_summary.json). The raw per-run JSON and logs stay in `cascade-lab`
  ([`manifest.json`](manifest.json), sha256).
- **Port workaround.** `load_demo_config` on this revision does not apply `CASCADE_BRIDGE_PORT` /
  `CASCADE_OCCUPANCY_PORT` (backlog B34). The harness rewrites the resolved ports and refuses to run if a shared
  port remains.

## Results

| arm | precision | phantom rate | recall | beliefs for 3 props (last frame) | localisation mean / max |
| --- | --- | --- | --- | --- | --- |
| self-mask gate **on** | **1.0 / 1.0 / 1.0** | **0 / 0 / 0 %** | 1.0 / 1.0 / 1.0 | 4 / 4 / 4 | 1.96 / 1.91 / 1.71 cm, ≤ 4.1 cm |
| gate off (`main`) | 0.821 / 0.821 / 0.804 | 17.9 / 17.9 / 19.6 % | 1.0 / 1.0 / 1.0 | 5 / 5 / 5 | 1.97 / 2.05 / 2.06 cm, ≤ 5.6 cm |

- Every off run ends with the phantom: a gray "biplane" or "fighter jet" 0.408 m from the nearest prop, i.e. the
  arm's upper link 0.42 m above the table. No on run has a belief farther than `HIT_M` from a prop.
- The fourth belief in every run, in BOTH arms, is the bin duplicate. The side camera names the bin "yellow" and
  the top camera names it "orange", and the colour rule keeps the two names apart. The metric scores it as a hit
  (it lies within 6 cm of the bin), but it makes the count wrong. It is **B32b**, untouched here.
- The threshold `self_mask_max_frac: 0.5` sits in a wide gap. In the per-detection diagnosis (24 rounds × 2
  cameras), robot detections were 0.885–0.995 robot pixels and props at most 0.046. The highest prop value was
  the side camera's pink cube, whose mask bleeds onto the base plate behind it. Its robot pixels are now removed
  before back-projection instead of being lifted onto the plate.

## Not claimed

- **Hardware.** No physical-camera measurement. Frames without a render self-mask (the real rig today) are fused
  exactly as before; fusion still needs a link-geometry mask there.
- **Other platforms.** No Spark or Newton measurement (PhysX only); no kitchen-scene measurement.
- **Bin duplicate.** The gate does not fix it (B32b), so the count is 4 for 3 props and the documented
  `count_stable` / 0 % row still differs on this build.
