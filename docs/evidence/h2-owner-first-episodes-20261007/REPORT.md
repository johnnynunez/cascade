# Unitree H2 — first CASCADE-owned episodes on PhysX (7 October 2026)

First CASCADE-owned Unitree H2 episodes: scripts/isaac_h2_bridge.py (PhysX, Isaac Sim 6.2 build, GPU 0 of the x86 rig) publishes on the MOBILE wire; CASCADE RobotRuntime -> SafeBase walk_velocity drives it under configs/bases/h2_velocity_candidate.yaml; the independent BasePostconditionChecker judges each command from the truth channel with the owner's solved support contacts.

**Not claimed:** No physical admission. Candidate verifier limits (not measured limits). Two of five commands REFUTED by the verifier (post-command settle: residual velocity/pose drift) and recorded as such; nothing was retried, relaxed or re-run to pass. x86 two-GPU rig numbers, not Spark numbers.

## Owner (scripts/isaac_h2_bridge.py)

- hello: kind `h2`, engine physx, device `cuda:0` (physics on `cuda:0`), policy `0a47182bbe20…`, asset root `040d2140bb0d…`, model identity `a7800af7fc57…`.
- Deployment chain: `isaacsim.robot.policy.examples.RobotPolicyRunner` with `ArticulationActuators` (training `DelayedDCMotor`: e.g. left knee stiffness 300.0, damping 8.0, effort 360.0 N·m, delay ≤ 4 steps) — equal to the pinned contract per joint; physics drives of the 14 policy joints at zero gain (effort mode).
- Known deviation: the 17 joints the policy never commands keep the asset's authored drives (waist_yaw stiffness 57296 vs training 300.0), as in NVIDIA's reference example.
- Self-collisions True, solver iterations [8, 4] (training export). Consumed stage layers hashed (8, root = pinned `H2.usda`).
- 5512 manual PhysX solves (`SimulationManager.step`), 1378 policy evaluations (every 4th), 552 overview frames; 0.336× real time with 20 fps rendering; ended by the harness SIGTERM (exit 143 by design), receipt complete.
- Pelvis height 0.909–1.021 m, max tilt 4.1°, 0 fallen rows of 1379.

## Episode (CASCADE RobotRuntime → SafeBase → owner; independent verifier)

| # | walk_velocity | wall s | verifier | net world displacement | reason |
| --- | --- | --- | --- | --- | --- |
| 0 | vx=+0.3 vy=+0.0 wz=+0.0 × 3 s | 11.35 | **confirmed** | 0.64 m | independent measured motion matches caller intent |
| 1 | vx=+0.0 vy=+0.0 wz=+0.5 × 2 s | 9.14 | **confirmed** | 0.15 m | independent measured motion matches caller intent |
| 2 | vx=+0.3 vy=+0.0 wz=+0.0 × 3 s | 11.44 | **refuted** | 0.67 m | did not settle: active controller, residual velocity or pose drift |
| 3 | vx=+0.0 vy=+0.2 wz=+0.0 × 2 s | 8.88 | **refuted** | 0.37 m | did not settle: active controller, residual velocity or pose drift |
| 4 | vx=-0.3 vy=+0.0 wz=+0.0 × 2 s | 9.35 | **confirmed** | 0.47 m | independent measured motion matches caller intent |

3 confirmed, 2 refuted of 5. Refutations are the post-command settle window of the candidate profile (stop speed 0.08 m/s, stop drift 0.03 m within 0.4 s, 4 s timeout): the policy keeps stepping/swaying briefly after the twist returns to zero. They are measurements, not failures to hide; the candidate limits are what the campaign will measure.

Video (overlay: sim time, step, admitted command, verifier verdict): `h2-owner/runs/ep3/h2-episode-ep3.mp4` (sha in manifest); contact sheet `contact-sheet.jpg`.

Raw rows (physics/policy/frames jsonl, receipts, verifier samples, owner log, video) stay in `cascade-lab/HERMES_AUDIT_20261007/h2-owner/runs/ep3/`; `manifest.json` pins their SHA-256.
