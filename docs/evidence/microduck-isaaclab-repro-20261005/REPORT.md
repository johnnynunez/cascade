> Evidence copy (CASCADE `docs/evidence/microduck-isaaclab-repro-20261005/`). Absolute paths below refer to the
> x86 workstation where the runs were made. The raw per-step logs (`results/*.json`, 45 MB) are not committed; they are
> identified by path, size and SHA-256 in `raw-logs-manifest.json`, and `trajectories.csv.gz` holds a compact per-step
> trajectory of every episode (run, kind, step, phase, command, root position, yaw, tilt, world/body velocity).
> Nothing in this directory is a physical-hardware result. The generalized driver was offered to the fork as
> https://github.com/AntoineRichard/IsaacLab/pull/21 (`scripts/tools/microduck_stop_response.py`, draft).

# MicroDuck ±30 mm walk → zero-twist test, reproduced DIRECTLY in Isaac Lab (Newton MJWarp + native BAM) — 2026-10-05

Status: **DONE — 36 walk-then-zero episodes + 9 standing episodes + 8 steady-gait braking episodes ran in Isaac Lab; no fall, no blocker.**
Directory: `/home/johnny/Projects/demo/cascade-lab/ISAACLAB_MICRODUCK_REPRO_20261005/`
(`REPORT.md` this file · `report.json` aggregates + per-episode table · `results/*.json` raw per-step logs · `analysis_table.md` · scripts `lab_microduck_walk_test.py`, `analyze.py`, `build_report_json.py`, `probe_frame.py` · `logs/`.)

## 1. Verdict (for Antoine)

**The overrun reproduces in Isaac Lab's own MicroDuck environment, so it is a property of the policies (hypothesis A), not of CASCADE's observation/actuator pipeline (hypothesis B).** With the exact protocol CASCADE uses (settle 1 s → hold (±0.3, 0, 0) until the root has moved 25 mm along its initial heading → command (0, 0, 0) for 3 s), both `velocity_flat.onnx` and `velocity_rough.onnx` keep moving in the commanded direction after the zero command: forward +4 … +31 mm extra (mean +20 mm flat / +15 mm rough), reverse −11 … −46 mm extra (mean −25 mm flat / −29 mm rough), plus a lateral swerve of 5–27 mm, so the final along-heading travel of a "30 mm" walk is 30–59 mm forward and 36–71 mm in reverse; CASCADE's −27.6 → −48.5 mm (flat, reverse) and −26.7 → −55.4 mm (rough, reverse) sit inside that envelope. The asymmetry (reverse overruns more than forward) is there too. It is the same with Lab's play-mode randomization, with all randomization off, and with CASCADE's nominal BAM deployment (7.4 V, sag 0.1, no delay, no current limit — which gives the smallest forward overrun, +4…+11 mm, closest to CASCADE's +1 mm). Mechanism visible in the logs: the zero command arrives while the body is still moving fast — forward at 0.14–0.26 m/s (v_b,x +0.11…+0.15), reverse at 0.27–0.34 m/s (v_b,x −0.20…−0.24), i.e. in reverse exactly at the first-stride velocity peak, because the policy first swerves ~20 mm sideways before accelerating backwards, so the 25 mm threshold is crossed later and faster — and the policy then needs 0.3–0.5 s (≈ one half gait cycle) to brake; braking distance from a steady 0.19–0.21 m/s gait is 11–21 mm. The robot never drifts indefinitely: in all 36 episodes it is standing still (|v| < 0.02 m/s) 0.44–0.74 s after the zero command (one case 1.34 s), moves < 0.4 mm in the last second, and in the 10 s pure zero-command standing test the drift is < 0.35 mm and < 0.25° yaw — so "standing on a zero command" is NOT out of distribution; what the policy does not do is stop within one control step of a zero command issued mid-stride. The 61-value observation layout of CASCADE matches Lab's term by term (verified numerically on the Lab run, Section 7); the only structural difference is that Lab feeds the joint velocity with a one-step (20 ms) delay, which is part of the training recipe and does not change the overrun.

## 2. What was installed / executed (exact steps)

1. Cloned the fork (nothing else on the machine was touched; the upstream checkout `/home/johnny/Projects/isaac/IsaacLab` and both Isaac Sim builds were only read):
   `GIT_LFS_SKIP_SMUDGE=1 git clone --branch antoiner/feat/microduck-rough-velocity --single-branch https://github.com/AntoineRichard/IsaacLab fork`
   → HEAD `eafc80dfac8c45dc4392e84432d837b47ab54cc4` (#8270 head; contains #8161/#8265/#8266/#8267 and upstream develop `4aa39c103` "[Workflow] Use Newton 1.6 release branch (#8114)"). The fork working tree stays clean (`git status` empty); all driver code lives outside it.
2. Installed the fork's own uv environment (its documented workflow is `uv run --extra rsl-rl isaaclab play ...`), i.e. **kit-less Isaac Lab on the Newton backend**, no Isaac Sim:
   `systemd-run --user --scope --collect --unit=lab-repro-uvsync -p MemoryMax=64G -- bash -c 'cd fork && UV_PYTHON_PREFERENCE=only-managed uv sync --extra rsl-rl --extra test'` (log `logs/uv_sync.log`, EXIT=0, 201 packages; full list `venv_freeze.txt`).
   Key resolved versions (from the fork's `uv.lock`): newton **1.6.1rc1** (`git+https://github.com/newton-physics/newton.git@dff296359317779bd08e3ba4bc23bc5d4e20bbcd`, release-1.6 branch — the same Newton version Isaac Sim 6.2 bundles), mujoco-warp 3.12.0, mujoco 3.12.0, warp-lang 1.17.0, torch 2.12.0+cu130, usd-exchange 3.0.0 (pxr), rsl-rl-lib 5.5.1, Python 3.12.13 (uv-managed).
   Added only `uv pip install --python fork/.venv/bin/python onnxruntime` (1.30.0) to run the ONNX policies.
3. Why not inside the Isaac Sim 6.2 build's python: its `kit/python` site-packages is empty (only pip) and is shared with CASCADE's running bridges; `isaaclab.sh -i` would have installed torch/newton/isaaclab into it. The 6.1 build's python already holds the UPSTREAM Isaac Lab develop checkout as an editable install (`isaaclab.__file__` → `/home/johnny/Projects/isaac/IsaacLab/...`), so installing the fork there would have collided with it. The `uv run isaaclab --isaacsim_source` route rebuilds Isaac Sim (forbidden). The kit-less uv route is also exactly how the PR author validates (`uv run python -m pytest ...`, `uv run isaaclab play ...`). Consequence: Kit/RTX was never started; physics is Newton/MJWarp exactly as in the Lab task cfgs.
4. Assets: the Lab cfg points at `{ISAACLAB_NUCLEUS_DIR}/Robots/PollenRobotics/MicroDuck/microduck_walk.usd`; the public S3 root returns 404 for it (not synced, as the PRs say) and Nucleus needs the VPN. The driver overrides `env_cfg.scene.robot.spawn.usd_path` to the read-only local mirror `/home/johnny/Projects/demo/cascade-lab/MICRODUCK/external/isaaclab-microduck-usd-folder/isaaclab-microduck-usd/microduck_walk.usd` (sha256 recorded per run in `results/*.json` → `cfg.usd_sha256`). The ground-plane USD (`Environments/Grid/default_ground_plane_checker_v1/default_ground_plane.usda`) was fetched from the public S3 asset root by Lab itself.
5. Smoke test with the fork's own entry point: `fork/.venv/bin/python scripts/environments/zero_agent.py --task IsaacContrib-Velocity-Flat-MicroDuck --num_envs 1 --max_steps 100 env.scene.robot.spawn.usd_path=<local usd>` → EXIT=0 (`logs/smoke2_zero_agent.log`; note: this Lab version has no `--headless` flag, headless is the default without `--visualizer`).
6. Every Lab process ran as `systemd-run --user --scope --collect --unit=lab-repro-<tag> -p MemoryMax=64G -- bash -c "cd fork && CUDA_VISIBLE_DEVICES=GPU-4c811d02-a79b-2837-425e-5f62ac3c578a timeout 1800 .venv/bin/python ../lab_microduck_walk_test.py --task {flat|rough} --policy {velocity_flat|velocity_rough} --seed N --variant {lab_play|norand|cascade_nominal} --out ../results/<tag>.json"` (batch scripts `run_matrix.sh`, `run_matrix_rough.sh`, `run_matrix_1s.sh`; GPU 0 untouched; ~90 s cold start per process incl. warp kernel compile, 11 s CUDA-graph capture, then ~1 s per episode).

## 3. Environment / task configuration actually used

Built like `isaaclab play` does: `resolve_task_config(task, "", play_mode=True)` (→ `MicroDuckVelocityFlatEnvCfg.play_mode()`: num_envs capped, observation corruption OFF, `push_robot` interval event removed) → my overrides → `validate(env_cfg)` → `launch_simulation(env_cfg, args)` → `gym.make(task, cfg=env_cfg)`. The ONNX policy is run with onnxruntime (CPU, 1 thread) on the `policy` observation group returned by the env; its action tensor (1×14) goes straight into `env.step`. Hashes checked against the `.meta.json`: velocity_flat `6c3d0266…c7779`, velocity_rough `efdc851c…cb9f` (iteration 49,999, normalization embedded, opset 18).

| item | value |
|---|---|
| tasks | `IsaacContrib-Velocity-Flat-MicroDuck` (flat task) and `IsaacContrib-Velocity-Rough-MicroDuck` (rough task) |
| terrain | collision **plane**, static/dynamic friction 1.0 (both). The rough task's generator was first set to 100 % `MeshPlaneTerrainCfg` tiles, but MJWarp's MuJoCo conversion rejects a zero-volume terrain mesh (`ValueError: Error: mesh volume is too small: /World/ground/terrain/mesh_0 . Try setting inertia to shell`, `logs/run_rough_*` first attempt), so the rough task runs on the plane with its foot ray casters removed (critic/reward foot-height terms fall back to flat-ground height; the actor is blind anyway). Its softer terrain contact (solref 0.04) therefore does not apply; everything else of the rough task is kept. |
| num_envs / rate | 1 env; physics dt 0.005 s, decimation 4 → policy 50 Hz; `use_newton_actuators=True`; `episode_length_s` raised to 60 so the 10 s standing test cannot time out (timeouts only trigger resets) |
| solver (flat task) | MJWarp, iterations 10, ls_iterations 50, njmax 96, nconmax 16, pyramidal cone, impratio 1.0, implicitfast, `use_mujoco_contacts=False`, 1 substep |
| solver (rough task) | same but iterations 100, njmax 1024, nconmax 200 (gives identical trajectories to the flat-task solver on the plane, to 0.1 mm — the solver converges within 10 iterations here) |
| robot | `MICRODUCK_CFG`: `microduck_walk.usd` (2 enabled sole colliders, self-collision on), init height 0.125 m, default joint pose = CASCADE `HOME_Q` exactly (0, −0.0873, −0.4579, −0.0049, 0.4530, 0.3491, 0.3491, 0, 0, 0, 0.0873, 0.4579, 0.0049, −0.4530), soft limit factor 0.9 |
| action | `BiasedJointPositionAction`, 14 joints in policy order (left leg 5, neck/head 4, right leg 5), scale 1.0, `use_default_offset=True` → target = default + action − encoder_bias |
| BAM actuator (asset cfg) | XL330 m6 fit, kp_fw 200, `stiff_frictionloss=True`, motor max_current 1.75 A, friction_base 0.00477, friction_viscous 0.00536 |
| BAM deployment, `lab_play` | vin sampled once from (6.5, 8.2) V → **7.178 V** (seed 0) / **8.013 V** (seed 1); sag gain from (0, 0.2) → **0.194** / **0.134** V/N·m; vin_min 6.0; command delay **3–6 physics steps** (15–30 ms, resampled every step); max_effort 1.0676 N·m (= 8.2·kt/R); friction_scale per reset 0.9–1.1 (1.043 / 0.989 at the logged reset) — all read back live from the drive (`bam_readback_after_first_reset`) |
| BAM deployment, `norand` | same drive parameters as `lab_play` seed 0 (7.178 V, sag 0.194, delay 3–6, current limit 1.75 A), friction_scale fixed 1.0 |
| BAM deployment, `cascade_nominal` | vin **7.4 V** fixed, sag gain **0.1** V/N·m, vin_min 6.0, delay **0**, `max_current=0` (disabled) → max_effort **0.9634** N·m (= 7.4·kt/R) — CASCADE's `official_infer_nominal_no_delay` profile |
| randomization, `lab_play` | ON (play_mode keeps it): startup encoder bias ±0.015 rad, IMU misalignment ≤ 6°, trunk mass ×0.95–1.05, foot friction 0.7–1.3; per reset: trunk/head CoM ±3 mm, BAM friction scale 0.9–1.1, armature ×0.9–1.1, root pose x,y ±0.5 m, yaw uniform(−π, π); observation noise OFF; pushes OFF; IMU observation lag 0–1 step (redrawn every 64 steps), joint-velocity observation lag 1 step |
| randomization, `norand` / `cascade_nominal` | all ranges zeroed (identity IMU, no bias, nominal mass/CoM/friction/armature, reset pose identity), IMU lag fixed 0, joint-velocity lag 1 step (task constant) |
| commands | written by the driver each step; the three command terms' resampling disabled (`resampling_time_range=(1e6,1e6)`, `rel_standing/forward/turn=0`), head-pose (4) and body-pose (6) commands held at 0 |
| seeds | `--seed 0/1` → `env_cfg.seed`, `env.reset(seed=…)`, torch/numpy seeds; each episode starts from a fresh `env.reset()` |

## 4. Protocol (as CASCADE's `walk_distance ±0.03 m`)

Per episode: `env.reset()` → **settle** 50 policy steps (1.0 s) at command (0,0,0) with the policy running → record root position `p0` and heading `h0` = root-link +x axis (yaw) → **command** (±0.3, 0, 0):
closed loop (`fwd_cl`/`rev_cl`): hold until `sign·(p−p0)·h0 ≥ 0.025 m` (30 mm − 5 mm tolerance), cap 3 s; open loop (`fwd_ol`/`rev_ol`): fixed 15 steps (0.3 s); steady gait (`fwd_1s`/`rev_1s`): fixed 50 steps (1.0 s) → **zero** (0,0,0) for 150 steps (3.0 s). `stand`: 500 steps (10 s) at (0,0,0) after the settle.
Logged every policy step (`results/*.json → episodes[].log[]`): step, phase, t, command, the 61-value policy observation actually fed to the ONNX, the 14 raw actions, root position (link and COM), quaternion (**xyzw**, Isaac Lab 3.0 convention), yaw, tilt, root linear/angular velocity (world and body), joint positions/velocities (policy order), terminated/truncated. Metrics in the tables are recomputed from these logs by `analyze.py` (heading frame = `h0`; "still" = |planar root velocity| < 0.02 m/s for 10 consecutive steps = 0.2 s; "fell" = `bad_orientation` termination, 70°). The policy observation consistency check (`obs_patch_max_abs_diff`, recomputing the observation after the command write vs. the vector fed to the policy) is 0.0 in every run.

## 5. Results

### 5.1 Walk-then-zero episodes (36 ±30 mm episodes + 8 steady-gait episodes; heading = root +x at the end of the settle)

| policy | env | variant | seed | case | cmd steps | reached | during along [mm] | during lat [mm] | during yaw [deg] | post-zero along [mm] | post-zero lat [mm] | post-zero path [mm] | post-zero yaw [deg] | still (0.2 s) at [s] | last-1 s disp [mm] | max tilt [deg] | fell |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| velocity_flat | flat-task(plane) | cascade_nominal | 0 | fwd_cl | 13 | yes | 25.8 | 16.8 | -0.0 | 10.4 | -12.7 | 28.7 | 10.9 | 0.60 | 0.35 | 2.7 | no |
| velocity_flat | flat-task(plane) | cascade_nominal | 0 | rev_cl | 14 | yes | -25.3 | -4.7 | 1.6 | -10.9 | -10.0 | 29.0 | -4.9 | 0.64 | 0.07 | 2.6 | no |
| velocity_flat | flat-task(plane) | cascade_nominal | 0 | fwd_ol | 15 | - | 31.9 | 11.5 | -0.1 | 10.7 | -10.7 | 29.3 | 10.2 | 0.58 | 0.24 | 2.8 | no |
| velocity_flat | flat-task(plane) | cascade_nominal | 0 | rev_ol | 15 | - | -29.0 | -9.0 | 1.9 | -34.8 | -2.8 | 58.9 | -16.5 | 1.34 | 0.11 | 5.7 | no |
| velocity_flat | flat-task(plane) | cascade_nominal | 0 | fwd_1s | 50 | - | 142.1 | 0.6 | 2.6 | 14.2 | 4.7 | 33.6 | -3.1 | 0.50 | 0.05 | 3.3 | no |
| velocity_flat | flat-task(plane) | cascade_nominal | 0 | rev_1s | 50 | - | -157.7 | -3.7 | 4.2 | -10.6 | -15.0 | 26.1 | -1.8 | 0.56 | 0.02 | 2.7 | no |
| velocity_flat | flat-task(plane) | lab_play | 0 | fwd_cl | 14 | yes | 26.1 | 26.7 | 1.0 | 27.9 | -5.0 | 44.9 | 0.7 | 0.60 | 0.13 | 2.4 | no |
| velocity_flat | flat-task(plane) | lab_play | 0 | rev_cl | 18 | yes | -27.9 | 5.5 | 1.0 | -18.2 | -12.7 | 42.6 | -0.5 | 0.68 | 0.03 | 2.8 | no |
| velocity_flat | flat-task(plane) | lab_play | 0 | fwd_ol | 15 | - | 28.7 | 21.8 | 1.2 | 24.4 | -6.5 | 43.3 | 1.6 | 0.56 | 0.15 | 3.0 | no |
| velocity_flat | flat-task(plane) | lab_play | 0 | rev_ol | 15 | - | -18.5 | 12.0 | 2.0 | -22.5 | -19.3 | 45.2 | -1.5 | 0.62 | 0.07 | 2.7 | no |
| velocity_flat | flat-task(plane) | lab_play | 0 | fwd_1s | 50 | - | 162.3 | -1.7 | -1.0 | 17.4 | 8.4 | 29.5 | -2.6 | 0.44 | 0.01 | 3.0 | no |
| velocity_flat | flat-task(plane) | lab_play | 0 | rev_1s | 50 | - | -149.3 | -20.1 | 4.2 | -18.1 | 14.9 | 29.9 | 8.2 | 0.44 | 0.02 | 2.5 | no |
| velocity_flat | flat-task(plane) | lab_play | 1 | fwd_cl | 14 | yes | 25.7 | 26.8 | 1.3 | 16.7 | -15.7 | 39.7 | -1.0 | 0.50 | 0.08 | 1.5 | no |
| velocity_flat | flat-task(plane) | lab_play | 1 | rev_cl | 16 | yes | -27.0 | 9.8 | 2.2 | -33.1 | -21.1 | 53.8 | -3.0 | 0.48 | 0.05 | 1.8 | no |
| velocity_flat | flat-task(plane) | lab_play | 1 | fwd_ol | 15 | - | 28.3 | 21.2 | 1.2 | 15.8 | -17.0 | 39.7 | -1.1 | 0.50 | 0.08 | 1.7 | no |
| velocity_flat | flat-task(plane) | lab_play | 1 | rev_ol | 15 | - | -24.5 | 9.4 | 2.3 | -34.3 | -21.4 | 51.3 | -1.9 | 0.54 | 0.08 | 1.4 | no |
| velocity_flat | flat-task(plane) | norand | 0 | fwd_cl | 14 | yes | 27.5 | 26.9 | 1.4 | 31.2 | -6.7 | 45.8 | 1.1 | 0.50 | 0.07 | 2.7 | no |
| velocity_flat | flat-task(plane) | norand | 0 | rev_cl | 17 | yes | -27.1 | -1.7 | 0.8 | -23.9 | -11.6 | 45.9 | 2.4 | 0.66 | 0.09 | 2.6 | no |
| velocity_flat | flat-task(plane) | norand | 0 | fwd_ol | 15 | - | 29.3 | 20.6 | 1.4 | 23.1 | -10.9 | 43.3 | 1.4 | 0.60 | 0.03 | 2.9 | no |
| velocity_flat | flat-task(plane) | norand | 0 | rev_ol | 15 | - | -20.5 | 2.1 | 0.9 | -21.5 | -20.3 | 43.1 | -0.4 | 0.50 | 0.15 | 2.3 | no |
| velocity_rough | flat-task(plane) | lab_play | 0 | fwd_cl | 15 | yes | 27.9 | 18.8 | 0.9 | 17.6 | -10.9 | 41.2 | 2.3 | 0.52 | 0.04 | 3.1 | no |
| velocity_rough | flat-task(plane) | lab_play | 0 | rev_cl | 17 | yes | -25.1 | 10.0 | -1.8 | -28.8 | -17.9 | 53.7 | -3.5 | 0.60 | 0.12 | 3.7 | no |
| velocity_rough | flat-task(plane) | lab_play | 0 | fwd_ol | 15 | - | 30.1 | 19.2 | 1.1 | 21.8 | -9.5 | 37.8 | 2.2 | 0.46 | 0.05 | 3.6 | no |
| velocity_rough | flat-task(plane) | lab_play | 0 | rev_ol | 15 | - | -18.8 | 14.3 | -0.9 | -27.8 | -22.3 | 56.7 | -7.0 | 0.60 | 0.12 | 3.6 | no |
| velocity_rough | rough-task(plane) | cascade_nominal | 0 | fwd_cl | 14 | yes | 25.4 | -0.6 | 0.0 | 5.0 | -7.7 | 24.1 | 2.9 | 0.54 | 0.13 | 2.5 | no |
| velocity_rough | rough-task(plane) | cascade_nominal | 0 | rev_cl | 14 | yes | -28.7 | -11.7 | 0.3 | -12.0 | -2.4 | 31.4 | -3.0 | 0.68 | 0.03 | 2.5 | no |
| velocity_rough | rough-task(plane) | cascade_nominal | 0 | fwd_ol | 15 | - | 27.5 | -4.5 | 0.6 | 4.1 | -3.5 | 20.4 | 1.5 | 0.52 | 0.14 | 2.5 | no |
| velocity_rough | rough-task(plane) | cascade_nominal | 0 | rev_ol | 15 | - | -32.9 | -15.0 | 0.3 | -22.9 | 6.7 | 40.1 | -3.2 | 0.68 | 0.03 | 2.5 | no |
| velocity_rough | rough-task(plane) | cascade_nominal | 0 | fwd_1s | 50 | - | 140.3 | 16.8 | 2.6 | 2.4 | -8.3 | 30.4 | 8.6 | 0.66 | 0.07 | 3.8 | no |
| velocity_rough | rough-task(plane) | cascade_nominal | 0 | rev_1s | 50 | - | -172.5 | 3.1 | -2.1 | -13.5 | -13.0 | 35.4 | -4.2 | 0.68 | 0.04 | 2.8 | no |
| velocity_rough | rough-task(plane) | lab_play | 0 | fwd_cl | 15 | yes | 27.9 | 18.8 | 0.9 | 17.6 | -10.9 | 41.2 | 2.3 | 0.52 | 0.04 | 3.1 | no |
| velocity_rough | rough-task(plane) | lab_play | 0 | rev_cl | 17 | yes | -25.1 | 10.0 | -1.8 | -28.8 | -17.9 | 53.6 | -3.5 | 0.60 | 0.13 | 3.7 | no |
| velocity_rough | rough-task(plane) | lab_play | 0 | fwd_ol | 15 | - | 30.1 | 19.2 | 1.1 | 21.8 | -9.5 | 37.9 | 2.2 | 0.46 | 0.05 | 3.6 | no |
| velocity_rough | rough-task(plane) | lab_play | 0 | rev_ol | 15 | - | -18.8 | 14.3 | -0.9 | -27.7 | -22.3 | 56.5 | -7.1 | 0.62 | 0.11 | 3.6 | no |
| velocity_rough | rough-task(plane) | lab_play | 0 | fwd_1s | 50 | - | 159.2 | 4.4 | -1.1 | 16.8 | 10.0 | 34.2 | -3.7 | 0.50 | 0.04 | 3.5 | no |
| velocity_rough | rough-task(plane) | lab_play | 0 | rev_1s | 50 | - | -165.7 | -10.1 | -1.4 | -21.1 | 20.1 | 34.6 | 4.0 | 0.42 | 0.06 | 3.6 | no |
| velocity_rough | rough-task(plane) | lab_play | 1 | fwd_cl | 15 | yes | 26.3 | 16.1 | 1.6 | 15.1 | -19.1 | 41.9 | -0.6 | 0.54 | 0.04 | 1.7 | no |
| velocity_rough | rough-task(plane) | lab_play | 1 | rev_cl | 16 | yes | -26.7 | 12.8 | -1.0 | -43.7 | -21.5 | 70.9 | -4.1 | 0.66 | 0.26 | 2.9 | no |
| velocity_rough | rough-task(plane) | lab_play | 1 | fwd_ol | 15 | - | 27.3 | 12.1 | 1.1 | 11.6 | -15.9 | 40.6 | -0.4 | 0.54 | 0.04 | 2.1 | no |
| velocity_rough | rough-task(plane) | lab_play | 1 | rev_ol | 15 | - | -24.7 | 10.7 | -0.1 | -45.8 | -21.7 | 73.3 | -3.8 | 0.74 | 0.23 | 2.6 | no |
| velocity_rough | rough-task(plane) | norand | 0 | fwd_cl | 14 | yes | 25.6 | 21.3 | 0.9 | 24.5 | -12.0 | 38.6 | 1.4 | 0.46 | 0.01 | 3.2 | no |
| velocity_rough | rough-task(plane) | norand | 0 | rev_cl | 16 | yes | -25.5 | 2.1 | -1.4 | -26.0 | -15.8 | 50.1 | -3.9 | 0.54 | 0.06 | 3.2 | no |
| velocity_rough | rough-task(plane) | norand | 0 | fwd_ol | 15 | - | 28.5 | 8.3 | 0.8 | 14.3 | -12.3 | 38.7 | 0.4 | 0.44 | 0.04 | 3.2 | no |
| velocity_rough | rough-task(plane) | norand | 0 | rev_ol | 15 | - | -23.1 | 2.5 | -1.6 | -26.4 | -17.2 | 52.7 | -3.7 | 0.50 | 0.05 | 2.9 | no |

Column meaning: "during" = from the end of the settle to the last commanded step; "post-zero" = from the last commanded step to the end of the 3 s zero window (along/lateral in the initial heading frame, path = integrated planar path); "still at" = first time after the zero command at which |v_planar| < 0.02 m/s has held for 0.2 s; "last-1 s disp" = displacement during the final second of the zero window. Positive lateral = robot's left (+y of the root link).

Aggregates over the 36 ±30 mm episodes (`report.json → summary`):

| policy / direction | n | during along [mm] mean (min…max) | post-zero along [mm] mean (min…max) | post-zero lateral [mm] mean | post-zero yaw [deg] mean (min…max) | still after [s] mean (min…max) | all still | any fell |
|---|---|---|---|---|---|---|---|---|
| velocity_flat fwd | 8 | +27.9 (25.7…31.9) | **+20.0 (10.4…31.2)** | −10.7 | +3.0 (−1.1…10.9) | 0.55 (0.50…0.60) | yes | no |
| velocity_flat rev | 8 | −25.0 (−29.0…−18.5) | **−24.9 (−34.8…−10.9)** | −14.9 | −3.3 (−16.5…2.4) | 0.68 (0.48…1.34) | yes | no |
| velocity_rough fwd | 10 | +27.7 (25.4…30.1) | **+15.3 (4.1…24.5)** | −11.1 | +1.4 (−0.6…2.9) | 0.50 (0.44…0.54) | yes | no |
| velocity_rough rev | 10 | −24.9 (−32.9…−18.8) | **−29.0 (−45.8…−12.0)** | −15.2 | −4.3 (−7.1…−3.0) | 0.62 (0.50…0.74) | yes | no |

Steady-gait braking (1.0 s of ±0.3 m/s, then zero): the policies track only ≈ 0.19–0.21 m/s of the 0.3 m/s command (140–172 mm in 1 s); after the zero command they stop in 0.42–0.68 s with a braking distance of **+2…+17 mm forward / −11…−21 mm reverse** along the heading. The ±30 mm protocol overruns more than this because its threshold is crossed at the first-stride velocity peak (see 5.3).

### 5.2 10 s pure zero-command standing (after the 1 s settle)

| policy | env | variant | seed | drift [mm] | along [mm] | lateral [mm] | yaw change [deg] | max planar speed [m/s] | still fraction | max tilt [deg] | fell |
|---|---|---|---|---|---|---|---|---|---|---|---|
| velocity_flat | flat-task(plane) | cascade_nominal | 0 | 0.13 | -0.11 | 0.08 | 0.05 | 0.002 | 1.000 | 1.2 | no |
| velocity_flat | flat-task(plane) | lab_play | 0 | 0.27 | -0.26 | 0.08 | -0.00 | 0.001 | 1.000 | 1.4 | no |
| velocity_flat | flat-task(plane) | lab_play | 1 | 0.33 | -0.30 | 0.13 | -0.05 | 0.003 | 1.000 | 0.7 | no |
| velocity_flat | flat-task(plane) | norand | 0 | 0.30 | -0.30 | 0.04 | 0.06 | 0.002 | 1.000 | 1.5 | no |
| velocity_rough | flat-task(plane) | lab_play | 0 | 0.22 | 0.12 | 0.18 | -0.15 | 0.000 | 1.000 | 1.6 | no |
| velocity_rough | rough-task(plane) | cascade_nominal | 0 | 0.16 | 0.14 | 0.08 | -0.08 | 0.001 | 1.000 | 1.7 | no |
| velocity_rough | rough-task(plane) | lab_play | 0 | 0.22 | 0.12 | 0.18 | -0.15 | 0.000 | 1.000 | 1.6 | no |
| velocity_rough | rough-task(plane) | lab_play | 1 | 0.19 | 0.16 | 0.10 | -0.23 | 0.000 | 1.000 | 1.1 | no |
| velocity_rough | rough-task(plane) | norand | 0 | 0.10 | 0.02 | 0.10 | -0.09 | 0.001 | 1.000 | 1.8 | no |

A zero command from rest is a perfectly stable stand in Lab: < 0.35 mm drift and < 0.25° yaw in 10 s, planar speed < 3 mm/s, tilt ≤ 1.8°, every step "still".

### 5.3 What the time series show (flat policy, `lab_play` seed 0, `results/flat_velocity_flat_lab_play_seed0.json`)

Reverse, closed loop (command −0.3 for 18 steps = 0.36 s): for the first 0.2 s the robot does not move backwards at all but swerves +21 mm to its left (body-frame velocity ≈ (0, +0.2) m/s); it then accelerates backwards and reaches the −25 mm threshold at t = 0.34 s **while at its peak speed (0.33–0.35 m/s, v_b,x = −0.21)**. After the zero command it decelerates over 0.3 s, reaching −51 mm along the heading at t = 0.64 s, rebounds to −46 mm by t = 0.9 s and stands (|v| < 2 mm/s). Net: −27.9 mm commanded → −45.8 mm final (CASCADE flat reverse: −27.6 → −48.5 mm).
Forward, closed loop (14 steps = 0.28 s): the first stride goes diagonally (+26 mm along, +27 mm left), the threshold is crossed at 0.14 m/s with the body still accelerating along the heading; it keeps going at 0.14–0.22 m/s for 0.3 s and stops at +54.5 mm along / +22 mm left at t ≈ 0.6 s (CASCADE flat forward: +25.6 → +25.8 mm but +0.13 rad yaw; Lab's forward yaw stays within ±2.4° in this seed and reaches +10.9° in the `cascade_nominal` seed).
In all 44 episodes the robot stands within 0.42–0.74 s after the zero command (one outlier 1.34 s: flat policy, `cascade_nominal`, `rev_ol`, which also had the largest yaw drift −16.5° and tilt 5.7°) and never falls (max tilt 5.7°, typically 1.5–3.7°).

## 6. Comparison with CASCADE's numbers

| case | CASCADE (Isaac Sim 6.2 + Newton 1.6.1rc1, CASCADE BAM, plane) | Isaac Lab (this report, plane) |
|---|---|---|
| velocity_flat forward: during → final along | +25.6 → +25.8 mm, yaw +0.131 rad after | +25.7…+31.9 → +36…+59 mm (post-zero +10…+31 mm), yaw −1…+11° |
| velocity_flat reverse: during → final along | −27.6 → −48.5 mm, yaw −0.228 rad after | −18.5…−29.0 → −36…−64 mm (post-zero −11…−35 mm), yaw −16.5…+2.4° |
| velocity_rough forward | +25.9 → +26.9 mm (yaw 0.006 rad) | +25.4…+30.1 → +30…+52 mm (post-zero +4…+25 mm; `cascade_nominal`: +4…+5 mm) |
| velocity_rough reverse | −26.7 → −55.4 mm | −18.8…−32.9 → −41…−70 mm (post-zero −12…−46 mm; `cascade_nominal`: −12/−23 mm) |
| eventually still? | yes (settle drift ~0.1 mm, subsequent stop max speed 1–2 mm/s) | yes, 0.42–0.74 s after the zero command (< 0.4 mm in the last second) |
| falls | 0 | 0 |

Reading: the reverse overrun magnitude (CASCADE −21 mm flat / −29 mm rough extra) is reproduced (Lab mean −25 mm / −29 mm). The forward overrun in Lab (mean +20 / +15 mm) is larger than CASCADE's (+0.2 / +1 mm), except in the `cascade_nominal` variant (+4…+11 mm), i.e. the deployment parameters (delay, sag, current limit, supply) modulate how much the gait overshoots forward, while the reverse overrun is robust to them. The lateral swerve of the first stride is to the robot's left in Lab (+12…+27 mm for forward commands); CASCADE reports −5.8 mm (forward) / −10.1 mm (reverse) lateral — smaller and to the right. Whether that is a dynamics difference, the leading-foot choice at the given initial state, or a y-axis sign convention in CASCADE's measurement frame could not be settled from the available CASCADE summary (no per-step CASCADE log was available here).

## 7. Observation layout: Isaac Lab policy group vs. CASCADE `microduck_policy.observation()`

Verified on the Lab logs (`results/flat_velocity_flat_norand_seed0.json`, fwd_cl): the 61-vector Lab fed to the ONNX equals, term by term, what CASCADE's `observation(q, dq, angular_velocity_body, gravity_body, previous_action, command)` builds from the same state, except for the joint-velocity delay:

| slice | Isaac Lab term (fork) | CASCADE (`src/cascade/control/microduck_policy.py` @ origin/main) | numerical check on the Lab run (`norand`) |
|---|---|---|---|
| 0:3 | `base_ang_vel`: root angular velocity in the root frame through the IMU-misalignment quaternion (identity when the event range is 0), lag 0–1 step (0 in `norand`) | `angular_velocity_body = Rᵀ ω_world` (trunk frame), no misalignment, no lag | max abs diff vs. logged `root_ang_vel_b` of the previous step: **0.0** (`lab_play`: up to 0.80 rad/s from the 6° misalignment + lag) |
| 3:6 | `projected_gravity` through the same quaternion/lag | `gravity_body = Rᵀ (0,0,−1)` | **4.7e-7** (`lab_play`: up to 0.026) |
| 6:20 | `joint_pos` − `default_joint_pos` (+ encoder bias, 0 in `norand`), policy joint order | `q − HOME_Q`, `POLICY_JOINTS` order — identical order and identical HOME values (checked: Lab default pose = HOME_Q) | **2.4e-8** (`lab_play`: up to 0.0144 rad = the sampled encoder bias) |
| 20:34 | `joint_vel` with `delay_min_lag = delay_max_lag = 1` → **the joint velocity of the previous policy step** | `dq` as supplied by the host (the policy module does not delay it) | obs equals the joint velocity logged TWO states back exactly (0.0) and differs from the latest by up to 2.9 rad/s → Lab's actor sees dq delayed by one step (20 ms); CASCADE must apply the same one-step delay to be bit-equivalent (check `microduck_sdk`/stepper on CASCADE's side) |
| 34:48 | `last_action` (raw previous policy output) | `previous_action` raw | **0.0** |
| 48:51 | `velocity_commands` (vx, vy, wz) body frame | `command[0:3]` | identical by construction |
| 51:55, 55:61 | `head_pose_commands` (4), `body_pose_commands` (6) — sampled uniformly in play mode unless held; held at 0 here | `command[3:13]` (zeros) | identical |
| action → target | `default + 1.0·action − encoder_bias` (bias 0 in `norand`/CASCADE) | `HOME_Q + 1.0·action` (`direct-v1`) | same |

Other differences that are NOT in the observation vector but in the pipeline: Lab's quaternions are stored **xyzw** (`root_quat_w` doc: "(x, y, z, w)"); CASCADE's are wxyz — each side is self-consistent (Lab's own `quat_apply` reproduces `projected_gravity_b` from the stored quaternion). Lab's tracking reward uses `root_link_lin_vel_b` (root link, not COM). The Lab actuator applies a 3–6 step command delay, battery sag and a 1.75 A current limit by default; CASCADE's nominal profile has none of these — and the `cascade_nominal` variant shows this changes the forward overrun (+4…+11 mm vs +15…+31 mm) but not the existence of the overrun nor the reverse magnitude.

## 8. Not done / caveats

- Not run inside Isaac Sim (Kit): the fork's documented workflow is kit-less and both Isaac Sim builds are shared with other work, so the physics here is Newton 1.6.1rc1/MJWarp 3.12 run by Isaac Lab without Kit. It is the same Newton version Isaac Sim 6.2 bundles, but not literally the 6.2 build's bundled wheel.
- The rough task could not be run on an all-flat MESH terrain (MJWarp conversion error above), so it ran on the collision plane without its softer terrain contact and without the foot ray casters; the rough policy was additionally run in the flat task (identical results on the plane).
- The policies were executed through onnxruntime (CPU) on the exported ONNX, not through `isaaclab play --rl_library rsl_rl` (which needs the `.pt` checkpoints that were not supplied).
- 1 environment per run, 2 seeds per policy for `lab_play`, 1 seed for `norand`/`cascade_nominal`; no video.
- First run `results/OLD_wrongquat_flat_velocity_flat_lab_play_seed0.json` logged the quaternion under the key `quat_wxyz` while it is xyzw; its in-run yaw/tilt summaries are wrong (kept for traceability, excluded from the tables; the run was repeated).

## 9. Files

- `REPORT.md` (this), `report.json` (metadata, variants, CASCADE reference numbers, aggregates, per-episode table, per-run cfg + BAM read-backs, list of raw logs), `analysis.json`, `analysis_table.md`
- `results/<task>_<policy>_<variant>_seed<N>[_1s].json` — raw per-step logs (`episodes[].log[]`, ~230–560 records per episode) + cfg summary + BAM read-back + policy hashes (13 files, 45 MB)
- `lab_microduck_walk_test.py` (driver), `analyze.py`, `build_report_json.py`, `probe_frame.py` (frame probe: root +x is the sagittal axis, +y is the robot's left; head above/slightly ahead of the trunk), `run_matrix*.sh`
- `logs/` (uv sync, smoke tests, every Lab process), `venv_freeze.txt`, `cascade_microduck_policy_origin_main.py` and `cascade_locomotion_summary_origin_main.json` (read-only copies from `git show origin/main:` of the CASCADE repo)
- `fork/` — the clone of AntoineRichard/IsaacLab @ eafc80d with its `.venv` (unmodified sources)
