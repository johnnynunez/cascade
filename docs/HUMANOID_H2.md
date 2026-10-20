# Unitree H2: the whole-body humanoid vertical

**Next action:** run the CPU contract and turn-ramp tests (no Isaac; about 30 s):

```bash
uv run --extra dev pytest tests/test_h2_policy_contract.py tests/test_turn_goal_ramp.py tests/test_turn_goal_ramp_budget.py -q
```

**Status: candidate; gate 1 (binding) passed in simulation on 7 October 2026, no physical
admission** ([gates](#admission-gates-robot_modularitymd--admission-work-for-an-actual-humanoid)).
This page fixes the embodiment, policy, engine order and owner architecture of CASCADE's first
whole-body humanoid and records what has been shown. The `h1`/`h1_2`/`g1` profiles drive only the
ARM of a standing Unitree humanoid over ROS2/Arm-SDK and are unrelated.

## Decision (7 October 2026): H2, PhysX first

| item | choice | why |
| --- | --- | --- |
| embodiment | **Unitree H2** (31 revolute joints) | the user's target; public descriptions exist (`unitree_ros` `h2_description`, BSD-3) |
| sim asset | NVIDIA's public `Isaac/Robots/Unitree/H2/H2.usda` (URDF-USD-Converter output of `h2_description`; `Physics` variant with physx/physics/mujoco payloads, 31 `MjcActuator`) | the only H2 USD paired with a policy; the stage the policy was trained on |
| policy | NVIDIA **`Velocity-H2-History-v0`**: `Isaac/Samples/Policies/h2/{policy.pt, env.yaml, IO_descriptors.yaml}`, TorchScript MLP 255 → 14, 14 leg + waist-roll/pitch joints, history 5, 50 Hz control / 200 Hz physics, trained on **PhysX** in AGILE | the only public pretrained H2 walking policy; SHA-256 `0a47182b…` (the pin of Isaac Sim's `h2_standalone.py`) |
| engine | **PhysX first** in the internal Isaac Sim 6.2 build | the policy's training engine; there is no Newton-trained H2 policy yet |
| Newton route | `newton-physics/newton-assets` **PR #53** (open): closed-loop-linkage H2 USD from `H2_loop.xml` (`mujoco-usd-converter 0.5.0`, 44 bodies, six loop closures) | for an MJWarp-trained H2 policy, or this policy re-validated on Newton as an explicit experiment |
| not chosen | G1 29-DoF from newton-assets (`g1.usd` + `mjw_g1_29DOF.onnx`, MJWarp-trained) | Newton-first and complete, but not the H2; its 29-DoF + Dex3 MJCF drives the Newton [humanoid fastening](HUMANOID_FASTENING.md) task (pelvis fixed) |

Provenance, digests and licences: [`configs/h2/bundle.json`](../configs/h2/bundle.json) (the two
YAML files vendored under `assets/h2/bundle/`, Apache-2.0; `policy.pt` and the USD are fetched or
opened from the asset root and verified, never committed; `scripts/h2_assets.py --check` verifies
without downloading).

## What has been shown

1. **Reference smoke** ([evidence](evidence/h2-reference-smoke-20261007/manifest.json)): NVIDIA's
   `h2_standalone.py --headless --test` on the internal 6.2 build, PhysX: the pinned policy walks
   **0.63 m forward in 2 s**, exit 0. No CASCADE controller or verifier: not admission.
2. **Contract on CPU** (`src/cascade/control/h2_policy_contract.py`,
   `tests/test_h2_policy_contract.py`): 31-joint order (14 commanded, 17 held), six observation
   terms with history 5 (255 total; scales 0.2 base angular velocity, 0.05 joint velocity), action
   decode `offset + 0.5 × clip(raw)`, PD gains per actuator group (legs 200/300 N·m/rad, feet
   40/20, waist 300, arms, head), training command ranges (±0.5 m/s, ±1.0 rad/s) and fall criteria
   (torso tilt > 30°, pelvis below 0.615 m). Every drift from the pinned export is refused.
3. **First CASCADE-owned episodes** ([evidence](evidence/h2-owner-first-episodes-20261007/REPORT.md)):
   `scripts/isaac_h2_bridge.py` owns one H2 (manual `SimulationManager.step` at 200 Hz, NVIDIA's
   `RobotPolicyRunner` fed only the admitted twist, every state on the MOBILE wire with solved
   ground reactions). `RobotRuntime → SafeBase.walk_velocity` under `h2_velocity_candidate`: the
   independent verifier **confirmed 3 of 5** and refuted 2 on post-command settle; no fall, no
   fault. Known deviation (also in NVIDIA's example): the 17 held joints keep the asset's stiffer
   authored drives.
4. **Geometric skills and MCP** ([evidence](evidence/h2-geometric-candidate-20261007/REPORT.md)):
   `walk_distance` (0.3 m/s, up to 0.6 m inside the 3 s command budget) and `turn` (0.35 m
   translation veto). Settle refutations came from a 0.3–0.9 rad/s yaw oscillation after a
   mid-step goal cut lasting past the 4 s wall settle budget (≈1.9 s of sim at ≈0.47× real time).
   **Revision 2** raised `settle_timeout_s` 4 → 8 s wall (`max_wall_duration_s` 14 → 20,
   `max_samples` 801 → 1201), rest thresholds unchanged: 6 of 7 confirmed on both paths (MCP via
   `CASCADE_ROBOT=h2`, 13 tools), but `turn` 0.8 rad stayed refuted (yaw 0.69 → 0.32 rad/s over
   3.75 s sim). A walk issued after the previous runtime's close latched the stop was refused.
5. **Not shown:** any Newton run of the H2, anything on hardware, measured (not candidate) verifier limits,
   `walk_distance` beyond ~0.6 m (bounded by the 3 s command budget).

## Candidate revision 3: turn goal ramp (software only, 7 October 2026)

The turn decelerates before the goal instead of cutting 0.5 rad/s to zero mid-step; settle budget
and rest thresholds are unchanged (`SafeBase.turn`, opt-in per profile):

1. `turn_control.goal_ramp: {decel_rad_s2, min_rate_rad_s, rate_step_rad_s}` (exact keys; explicit
   `null` turns it off for an `extends:` child). On every fresh advancing state, with the same
   measured unwrapped yaw that decides goal arrival: `|wz| = max(min_rate, min(previous,
   sqrt(2·decel·(|remaining| − turn_tolerance))))`. The rate never rises, never drops below the
   floor before the unchanged zero-twist `stop(latch=False)`, and updates only by ≥ `rate_step` or
   at the floor. Stale or missing state still fails closed (latched stop).
2. **One admission, scaled inside its envelope**: `scale_velocity(scale, generation)` sets
   `0 < scale ≤ 1` of the ADMITTED twist for the same owner/command/epoch/generation; it never
   admits, extends, renews or replays motion and is refused when latched, faulted, stale, expired,
   heading-held or inactive. Opt-in on the bridge (`MobileBridgeController(velocity_scaling=True)`,
   advertised in `hello.capabilities` only then, so MicroDuck hellos are byte-identical) and on the
   H2 owner (`--velocity-scaling`); a ramped profile on a backend without it is refused before any
   read or command.
3. Unchanged: every veto (posture/support, 0.35 m translation, overshoot, 3 s command, 8 s wall),
   the verifier, the `turn(angle_rad)` tool and every other profile. A ramped result adds
   `turn_rate_updates` and `commanded_rate_at_stop_rad_s`.
4. Values (`h2_velocity_candidate`, derivation in the YAML): `decel_rad_s2: 0.4`,
   `min_rate_rad_s: 0.15`, `rate_step_rad_s: 0.025`; 1.0 rad needs ≈2.56 s of command at the
   measured 93 % tracking, inside the 3 s `max_duration_s`.
5. Cost: ≈0.4 s more command time than a constant-rate turn; a robot tracking much worse (75 % on
   1.0 rad) hits the 3 s deadline and fails closed (execution error, latched stop), never a larger
   budget.

### Live result (8 October 2026)

[Evidence](evidence/h2-turn-ramp-live-20261008/REPORT.md) (yaw profiles, harness, SHA-256 manifest):
owner with `--velocity-scaling` on GPU 0 (6.2 build, PhysX), ramp-on and `goal_ramp: null` arms on
the same owner differing only in `goal_ramp`, 39 turns, limits and 3 s command unchanged.

| angle | ramp on | ramp off |
| --- | --- | --- |
| ±0.6 rad | 3/4 (1 timeout) | 3/4 (1 refuted) |
| ±0.8 rad | **8/8** | 1/6 (5 refuted: settle heading drift 0.053–0.084 rad) |
| ±1.0 rad | 1/5 (3 timeouts, 1 refuted) | 5/6 |
| ±0.8 rad through MCP | 4/6 | — |

1. The ramp fixes 0.8 rad (cut at the 0.15 rad/s floor; settle drift 0.028–0.050 rad, one at
   0.0499: thin margin).
2. All four timeouts are ramp-on left turns: a policy yaw-rate dip below 0.2 rad/s around
   1.0–1.75 s (24 of 39 turns, both arms) plus the deceleration no longer fit in 3 s.
3. 1.0 rad turns sit at the translation limit in both arms (0.166–0.227 m vs `max_lateral_drift_m`
   0.20 m).
4. **Verdict: not admitted.** Nothing was retuned after the measurement.

## Candidate revision 4: time-budgeted goal ramp (software only, 8 October 2026)

The ramp is budgeted against the ADMITTED command time left, so it cannot decelerate a turn past
the 3 s command; when time is short it holds the rate and cuts higher.

1. `turn_control.goal_ramp.time_budget: {reserve_s, tracking}` (exact keys; `null` restores
   revision 3; `goal_ramp: null` still disables the ramp). Law: `|wz| = max(min_rate, min(previous,
   max(sqrt(2·decel·d), d / (tracking · (left − reserve)))))` with `d = |remaining| −
   turn_tolerance` and `left = ACK end_sim_time_s − sample sim time`; inside the last `reserve_s`
   the rate is only held. Same `rate_step` gate, `scale_velocity` and zero-twist stop.
2. It reads no yaw rate, never raises the rate, extends or renews the command or skips the stop;
   a dip longer than any budget still ends in the fail-closed timeout. Results add
   `turn_rate_budget: {held_samples, first_held}`, `command_time_left_s` and `budget_rate_rad_s`.
3. Values: `reserve_s: 0.3` (s of sim), `tracking: 0.93`. Replayed on the revision-3 A/B, the budget
   binds exactly on the three timed-out +1.0 rad turns, one late −1.0 rad turn and one update of a
   confirmed 0.8 rad turn.
4. CPU evidence (`tests/test_turn_goal_ramp_budget.py`, toy at 0.93 tracking with a ≤ 0.1 rad/s dip
   at 1.0–1.75 s): 0.6 and 0.8 rad identical to revision 3; ±1.0 rad 2.72 s (cut at 0.5) instead of a
   timeout. Golden digest over all nine shipped base profiles: 75/76 command sequences identical
   (only the candidate's 1.0 rad dip turn changes).
5. Cost: when the budget binds the turn is cut above the floor, up to revision 2's 0.5 rad/s (which
   refuted 5 of 6 ramp-off 0.8 rad turns on settle). **Live A/B pending**: same owner command line,
   harness importing this checkout (`CASCADE_H2_REPO=<checkout>`), the private base
   (`bases/h2_episode.yaml`, `extends: h2_velocity_candidate`) pinned from each owner's
   `kit/model-identity.json` (revision 4 is client-side only, so the owner identity does not move),
   arms R (as shipped), V (`turn_control: {goal_ramp: {time_budget: null}}`), N (`goal_ramp: null`)
   checked with `--dry-config`; record verdict, settle drift/ω, `turn_rate_budget`, cut rate and
   command time.

## Owner architecture (`scripts/isaac_h2_bridge.py`)

One process per world inside Isaac Sim, PhysX at 200 Hz, policy every fourth solve (`decimation:
4`), following the MicroDuck [shared-scene boundary](MICRODUCK.md#shared-scene-implementation-boundary):

1. **Open and bind** (gate 1): open `H2.usda`, record the composed stage identity and active
   colliders (the training `env.yaml` names `collision_profile=feet_only`, `hands=none` variants the
   public stub lacks), map the DOF order onto `H2PolicyContract.joint_names` (fail closed on a
   missing joint), apply the contract's PD gains, limits and armature, hold the 17 non-policy
   joints. Policy: `runs/.install-cache/h2/policy.pt` (`scripts/h2_assets.py`, SHA-256 verified).
2. **Step**: read root angular velocity, projected gravity and the 14 joint states; every fourth
   step build the sample (`H2PolicyContract.scaled_sample`, `TermHistory`), run the model,
   `decode_action`; non-finite observations or outputs abort the step with no actuation.
3. **Publish** the completed `BaseState` through `MobileBridgeController` (freshness, epochs,
   generation, fall flag from the contract's criteria) so `SafeBase` skills and the verifier work
   unchanged; admitted limits (`configs/bases/h2_velocity_physx.yaml`) stay inside the training
   command ranges.
4. **Verify**: a walk is `confirmed` only by the verifier's own state measurements under unchanged
   deadlines (state age 0.5 s, progress 0.4 s, RPC 0.5 s).
5. The 6.2 build's `isaacsim.robot.policy.examples` (`RobotPolicyRunner`, `PolicyArtifact`,
   `derive_binding`, `ObservationHistory`) may only be wrapped after reading its failure behaviour
   against CASCADE's fail-closed rules; it must never become a second command path.

## Admission gates (`ROBOT_MODULARITY.md` → *Admission work for an actual humanoid*)

| gate | status | next evidence |
| --- | --- | --- |
| 1 Binding (model, policy, endpoint, mappings, exclusive ownership, controller clock) | **passed in simulation (7 Oct 2026)** — the owner binds the pinned asset/policy, maps the 31 joints, verifies the deployed actuation against the contract, owns the only command path and the physics clock; receipt + `model-identity.json` | hardware binding (Unitree SDK2 `rt/lowcmd`) is a separate gate |
| 2 Dynamic transforms with epochs/freshness | **partially shown** — states carry epoch/generation/age; the verifier's freshness and settle windows ran on the truth channel | measured limits replacing the candidate profile |
| 3 Whole-body limits, balance, self/environment collision | open by construction (policy commands 14 of 31 joints; arms/head held) | fall detection through the training criteria; later a second owner for the arms |
| 4 Sensors | open | none added yet |
| 5 Rehearsal through MCP/controller/physics/verifier with bound video | **first pass through controller/physics/verifier with video** (3/5 confirmed; no MCP/LLM in the loop yet) | MCP-driven episodes, `walk_distance`, the settle behaviour after a command, measured verifier limits |

## Risks recorded

1. PhysX-trained policy vs CASCADE's Newton priority: a Newton run of this policy is an
   experiment, not a port.
2. Training used `DelayedDCMotor` (0–4 step delay, per-group saturation); Isaac Sim deploys PD
   drives with the same gains and limits. A deviation shows up as a fall under the verifier, never
   as a relaxed limit.
3. The AGILE H2 training code is not public: the policy is frozen; retraining needs
   `unitree_rl_mjlab` (29-DoF MJCF, no head) or a new Isaac Lab task.
4. `policy.pt` and the USD carry Isaac Sim asset terms (no licence file in the bundle); the YAML
   files are Apache-2.0; the Unitree description is BSD-3.
5. Turn ramps trade command time for a slower cut; the answer to a deadline hit is a re-derived
   ramp or a standing hand-over, never a larger `max_duration_s`.
