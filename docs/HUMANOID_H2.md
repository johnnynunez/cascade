# Unitree H2: the whole-body humanoid vertical (design, 7 October 2026)

**Status: candidate, no admission gate passed.** This page fixes the embodiment,
the policy, the engine order and the owner architecture for CASCADE's first
whole-body humanoid, and records what has actually been shown. The existing
`h1`/`h1_2`/`g1` profiles drive only the ARM of a standing Unitree humanoid
over ROS2/Arm-SDK and are unrelated to this vertical.

## Decision (user, 7 October 2026): H2, PhysX first

| item | choice | why |
| --- | --- | --- |
| embodiment | **Unitree H2** (31 revolute joints) | the user's target; public descriptions exist (`unitree_ros` `h2_description`, BSD-3) |
| sim asset | NVIDIA's public Isaac Sim asset `Isaac/Robots/Unitree/H2/H2.usda` (URDF-USD-Converter output of `h2_description`; `Physics` variant with physx/physics/mujoco payloads, 31 `MjcActuator`) | the only H2 USD paired with a policy; the same stage the policy was trained on |
| policy | NVIDIA **`Velocity-H2-History-v0`**: `Isaac/Samples/Policies/h2/{policy.pt, env.yaml, IO_descriptors.yaml}`, TorchScript MLP 255 → 14, 14 leg + waist-roll/pitch joints, history 5, 50 Hz control / 200 Hz physics, trained on **PhysX** in AGILE | the only public pretrained H2 walking policy; pinned by SHA-256 `0a47182b…` (the pin Isaac Sim's own `h2_standalone.py` carries) |
| engine | **PhysX first** in the internal Isaac Sim 6.2 build | the policy's training engine; Newton remains CASCADE's priority target but there is no Newton-trained H2 policy yet — the Newton route is tracked separately (below) |
| Newton route | `newton-physics/newton-assets` **PR #53** (open, reviewed by eric-heiden / andrewkaufman): closed-loop-linkage H2 structured USD converted from `H2_loop.xml` with `mujoco-usd-converter 0.5.0`, 44 bodies, six loop closures | the asset to use when an H2 policy trained on MJWarp exists or when this policy is re-validated on Newton as an explicit experiment |
| not chosen | G1 29-DoF from newton-assets (`g1.usd` + `mjw_g1_29DOF.onnx`, MJWarp-trained, on disk) | Newton-first and complete, but it is not the H2 |

Provenance, digests and licences: [`configs/h2/bundle.json`](../configs/h2/bundle.json)
(the two YAML files are vendored under `assets/h2/bundle/`, Apache-2.0; `policy.pt`
and the USD are fetched/opened from the asset root and verified, never
committed). Inventory with every source checked and every "not found":
`cascade-lab/HERMES_AUDIT_20261007/h2-inventory/REPORT.md`.

## What has been shown (and what has not)

- **Reference smoke on this build** ([evidence](evidence/h2-reference-smoke-20261007/manifest.json)):
  NVIDIA's `h2_standalone.py --headless --test` on the internal Isaac Sim 6.2
  build with the public 6.2 asset root pinned, PhysX, GPU 0 of the x86 rig:
  the policy loads from the asset root under its SHA-256 pin and the robot
  walks **0.63 m forward in 2 s of forward command**, exit 0. This proves the
  asset + policy pair runs on the build CASCADE targets. It involves no
  CASCADE controller, verifier or deadline, so it is not admission.
- **Contract parsed and tested on the CPU**
  (`src/cascade/control/h2_policy_contract.py`, `tests/test_h2_policy_contract.py`):
  joint order (31; 14 commanded, 17 held), observation layout (six terms,
  history 5, oldest → newest per term, scales 0.2 on base angular velocity and
  0.05 on joint velocity; 255 total), action decode (`offset + 0.5 × clip(raw)`,
  offsets = default standing pose), PD gains per actuator group (legs 200/300
  N·m/rad, feet 40/20, waist 300, arms, head), the training command ranges
  (±0.5 m/s, ±1.0 rad/s) and the training fall criteria (torso tilt > 30°,
  pelvis below 0.615 m). Every drift from the pinned export is refused.
- **Not shown:** anything under CASCADE's own owner, RPC controller, deadlines
  or verifier; any Newton run; anything on hardware.

## Owner architecture (to build; mirrors the MicroDuck shared owner)

The H2 owner is one process per world inside Isaac Sim, stepping PhysX at
200 Hz with the policy every fourth solve (the bundle's `decimation: 4`),
following the MicroDuck pattern (`docs/MICRODUCK.md` → shared-scene
implementation boundary) rather than a new control path:

1. **Open and bind** (gate 1): open `H2.usda` from the asset root, record the
   composed stage identity and the active collider set (the training `env.yaml`
   names variants `collision_profile=feet_only`, `hands=none` that the public
   stub does not define — the receipt must say which colliders are actually
   active), map the articulation DOF order onto `H2PolicyContract.joint_names`
   (fail closed on any missing/renamed joint), apply the contract's PD gains,
   effort/velocity limits and armature, and hold the 17 non-policy joints at
   their default positions. Policy = `runs/.install-cache/h2/policy.pt`
   (`scripts/h2_assets.py`, SHA-256 verified) loaded with TorchScript.
2. **Step**: per physics step read root angular velocity (body frame),
   projected gravity, the 14 joint positions/velocities; every fourth step
   build the sample (`H2PolicyContract.scaled_sample`), expand the history
   (`TermHistory`), run the model, `decode_action` → joint position targets;
   refuse non-finite observations or outputs (no actuation, step aborted).
3. **Publish**: the completed `BaseState` through the existing
   `MobileBridgeController` (freshness, epochs, generation, fall flag from the
   contract's tilt/height criteria, status) so `SafeBase` skills
   (`walk_velocity`, `turn`, `stop_navigation`, later `walk_distance`) and the
   independent verifier work unchanged. Admitted limits
   (`configs/bases/h2_velocity_physx.yaml`) stay inside the training command
   ranges.
4. **Verify**: the verifier reads states, not the policy; a walk is
   `confirmed` only by its own measurement under the unchanged deadlines
   (state age 0.5 s, progress 0.4 s, RPC 0.5 s), as for MicroDuck.

Reuse to consider explicitly before writing code: the 6.2 build's
`isaacsim.robot.policy.examples` (`RobotPolicyRunner`, `PolicyArtifact`,
`derive_binding`, `ObservationHistory`) implements steps 1–2 for the reference
example; CASCADE's contract module reproduces its layout so the owner can
either wrap it (one SDK dependency, its own validation) or drive the
articulation directly from the contract. Decide by reading its failure
behaviour (what it does on a non-finite output, on a missing joint) against
CASCADE's fail-closed rules; never let it become a second command path.

## Admission gates (`ROBOT_MODULARITY.md` → *Admission work for an actual humanoid*)

| gate | status | next evidence |
| --- | --- | --- |
| 1 Binding (model, policy, endpoint, mappings, exclusive ownership, controller clock) | **open; attemptable now** — contract parsed and pinned, reference loop runs | owner opens the stage, binds joints and gains from the contract, publishes states; receipt with stage identity, active colliders, policy digest |
| 2 Dynamic transforms with epochs/freshness | open | not before gate 1 |
| 3 Whole-body limits, balance, self/environment collision | open by construction (policy commands 14 of 31 joints; arms/head held) | fall detection through the training criteria; later a second owner for the arms |
| 4 Sensors | open | none added yet |
| 5 Rehearsal through MCP/controller/physics/verifier with bound video | open | the first `walk_velocity`/`walk_distance` episodes with the independent verifier and the route-style evidence pack |

## Risks recorded

- PhysX-trained policy vs CASCADE's Newton priority: the first episodes are
  PhysX; a Newton run of this policy is an experiment, not a port.
- Actuator model in training was `DelayedDCMotor` (0–4 step delay, saturation
  torque per group); Isaac Sim deploys it as PD drives with the same gains and
  limits (the reference example walks). Any deviation shows up as a fall under
  the verifier, never as a relaxed limit.
- Training code (AGILE H2 task) is not public: the policy is a frozen artefact;
  retraining needs `unitree_rl_mjlab` (29-DoF MJCF, no head) or a new Isaac Lab
  task.
- `policy.pt` and the USD carry Isaac Sim asset terms (no licence file in the
  bundle); the YAML files are Apache-2.0; the Unitree description is BSD-3.
