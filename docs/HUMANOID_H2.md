# Unitree H2: the whole-body humanoid vertical (design, 7 October 2026)

**Status: candidate; gate 1 (binding) passed in simulation on 7 October 2026,
no physical admission** ([gates](#admission-gates-robot_modularitymd--admission-work-for-an-actual-humanoid)).
This page fixes the embodiment,
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
- **First CASCADE-owned episodes on PhysX**
  ([evidence](evidence/h2-owner-first-episodes-20261007/REPORT.md), 7 October 2026):
  `scripts/isaac_h2_bridge.py` owns one H2 in the internal 6.2 build (manual
  `SimulationManager.step` at 200 Hz, NVIDIA's `RobotPolicyRunner` as the
  deployment chain fed only the admitted twist, every completed state published
  on the MOBILE wire with the solved ground reactions of every link); CASCADE's
  `RobotRuntime → SafeBase.walk_velocity` drove five commands under the candidate
  profile `h2_velocity_candidate` and the independent verifier **confirmed 3 of 5**
  (forward 0.3 m/s × 3 s → 0.64 m net, turn 0.5 rad/s × 2 s, backward 0.3 m/s × 2 s)
  and **refuted 2** (post-command settle: the policy keeps stepping briefly after the
  twist returns to zero). No fall, no fault. The two independent validations agree at
  open: the runner deploys the training's `DelayedDCMotor` groups as an actuator
  model whose per-joint parameters equal the pinned contract, drives of the 14
  policy joints at zero gain, self-collision and solver iterations as trained. Known
  deviation (also in NVIDIA's example): the 17 held joints keep the asset's authored
  drives, far stiffer than the training's gains.
- **Geometric skills, settle measurement and the chat-host path**
  ([evidence](evidence/h2-geometric-candidate-20261007/REPORT.md), 7 October 2026, same
  owner): `h2_velocity_candidate` now carries `walk_distance`/`turn`
  (`distance_control` 0.3 m/s up to 0.6 m inside the unchanged 3 s command budget;
  `turn_control` translation veto 0.35 m — candidates derived from the first episodes).
  Through `RobotRuntime → SafeBase`: `walk_distance` +0.5 → **+0.452 m**, `turn` 0.8 →
  **+0.771 rad** (pelvis path 0.185 m), `walk_distance` −0.5 → **−0.453 m**, all executed
  and all three **refuted by the verifier's settle check**; `walk_velocity` 0.3 × 3 s
  confirmed again (+0.643 m). Through **MCP** (`python -m cascade.apps.mcp_server`,
  `CASCADE_ROBOT=h2`, the chat-host path): 13 tools listed, `walk_velocity` +0.63 m
  **confirmed**, `turn` 0.6 → 0.574 rad **refuted** (settle ω 0.205 vs 0.20), `walk_velocity`
  back −0.41 m **confirmed**; a `walk_velocity` issued while the previous runtime's close had
  latched the stop was correctly refused (`backend stop is latched`). No fall in either run.
- **Measured cause of every settle refutation so far:** the policy cut to zero twist at
  goal arrival (mid-step) keeps a yaw oscillation of 0.3–0.9 rad/s for longer than the
  ≈1.9 s of sim that the 4 s **wall** `settle_timeout_s` covers at this owner's ≈0.47× real
  time, whereas a duration-ended `walk_velocity` is below 0.12 rad/s by then. Standing right
  after readiness decays 0.34 → 0.04 rad/s over 3.5 s sim, and the reader sampled ≥ 1.9 s
  sim after a stop sees ω ≤ 0.06 rad/s: the robot does come to rest; it needs ~2.5–3.5 s of
  sim to pass the candidate 0.20 rad/s. Nothing was relaxed to pass. **Open decision:** size
  the candidate settle budget from this measurement in sim seconds, or have the geometric
  skills hand over to an explicit standing phase before the verdict.
- **Candidate revision 2 (same day, run geo3-v2 in the same evidence):** decision taken —
  `settle_timeout_s` 4 → 8 s wall (≈ 3.8 s sim here), `max_wall_duration_s` 14 → 20,
  `max_samples` 801 → 1201; **rest thresholds unchanged**. Result on the same owner, both
  paths in one run: **6 of 7 commands confirmed** — `walk_distance` +0.5/−0.5 → +0.452/−0.455 m
  confirmed, `walk_velocity` ±0.3 confirmed on both paths, MCP `turn` 0.6 → 0.577 rad
  confirmed; `turn` 0.8 rad (+0.773 rad) still **refuted**, and that one is a measurement, not
  a budget artefact: its yaw oscillation decays 0.69 → 0.32 rad/s over 3.75 s sim (the 0.6 rad
  turn is at 0.04 by then). The budget is not inflated further; the next revision is control-side
  (ramp the turn rate down before the goal, or hand over to standing) and will be measured on
  its own. Until then turns ≥ 0.8 rad on the H2 come back `refuted` and the host treats them so.
- **Candidate revision 3 (same day; owner episodes 8 October — mixed, not admitted):** the turn
  now decelerates before the goal instead of cutting 0.5 rad/s to zero mid-step
  ([below](#candidate-revision-3-turn-goal-ramp-software-only-7-october-2026)). CPU tests pin
  the control law, the in-admission scaling primitive and every fail-closed path. On the owner
  (39 independent turns, ramp on vs `goal_ramp: null` on the same owner, limits unchanged) the
  0.8 rad turn now passes the unchanged settle check 8/8 (1/6 without the ramp), but 1.0 rad
  drops to 1/5 (5/6 without): the ramp's deceleration plus a mid-turn yaw-rate dip of the
  policy exhaust the unchanged 3 s command ([live result](#live-result-8-october-2026)).
- **Candidate revision 4 (8 October; software only, live A/B pending):** the ramp is budgeted
  against the ADMITTED command time left, so it can no longer decelerate a turn past the unchanged
  3 s command; when time is short it holds the rate and cuts higher instead
  ([below](#candidate-revision-4-time-budgeted-goal-ramp-software-only-8-october-2026)). On a CPU toy
  with the measured mid-turn dip, 1.0 rad goes from a timeout to 2.72 s while 0.6/0.8 rad keep
  revision 3's ramp unchanged. Not measured on the owner.
- **Not shown:** any Newton run; anything on hardware; measured (not candidate) verifier
  limits; `walk_distance` beyond ~0.6 m (bounded by the 3 s command budget, not by the robot).

## Candidate revision 3: turn goal ramp (software only, 7 October 2026)

**Why.** Revision 2 left one refutation: `turn` 0.8 rad, cut from the admitted 0.5 rad/s to
zero twist at goal arrival. In the geo3-v2 owner rows the one goal-stopped turn that settled
(MCP 0.6 rad) was cut at ≈0.26 rad/s (0.1 s-mean yaw rate over its last 0.3 s; that turn had
stalled on the way), the refuted 0.8 rad turn at ≈0.47 rad/s; after the cut the yaw oscillates
at ≈5 Hz (period 0.16–0.23 s) and the 0.8 rad turn's decays only 0.69 → 0.32 rad/s over 3.75 s
sim. Settle budget and rest thresholds stay unchanged; the change is on the control side.

**Mechanism** (`SafeBase.turn`, opt-in per profile):

- `turn_control.goal_ramp: {decel_rad_s2, min_rate_rad_s, rate_step_rad_s}` (exact keys;
  an explicit `null` turns it off for an `extends:` child). On every freshly validated,
  advancing state after the admitted baseline, with the same unwrapped **measured** yaw that
  decides goal arrival: `|wz| = max(min_rate, min(previous, sqrt(2·decel·(|remaining| −
  turn_tolerance))))`, sign of the admitted command. The deceleration aims at the tolerance
  boundary where the unchanged zero-twist `stop(latch=False)` fires; the rate never rises
  (a yaw-oscillation dip cannot re-accelerate the robot) and never drops below the floor
  before that stop. A new rate is sent only when it is `rate_step` lower or reaches the floor.
  Nothing is extrapolated between samples; a stale or missing state still fails closed
  (latched stop) before any rate is computed from it.
- **One admission, scaled inside its envelope**, not a sequence of commands: the bridge
  refuses a command while one is active, a stop zeroes the twist until the next admission
  (the very cut being removed), and the unchanged verifier binds a motion to one admission
  generation G and completion G+1 (`BasePostconditionChecker._admitted_states`), so a
  multi-admission turn would be `unverified` by construction. The new transport primitive
  `scale_velocity(scale, generation)` sets `0 < scale ≤ 1` of the ADMITTED twist for the same
  owner/command_id/epoch/generation; it never admits, extends, renews or replays motion, does
  not consume the generation, and is refused when latched, faulted, stale, expired,
  heading-held or inactive. It is opt-in on the bridge (`MobileBridgeController(
  velocity_scaling=True)`, advertised in `hello.capabilities` only then, so every MicroDuck
  hello is byte-identical) and on the H2 owner (`--velocity-scaling`). A ramped profile on a
  backend that does not advertise it is refused before any read or command.
- Unchanged: every veto (posture/support, 0.35 m translation path, overshoot, the 3 s command
  and 8 s wall deadlines), the verifier and all its limits, the `turn(angle_rad)` tool, and every
  other profile — on the toy fixture all 19 shipped-profile skill episodes
  (`turn`/`walk_velocity`/`walk_distance`) reproduce the pre-change tree exactly except the H2
  candidate's turn, which does too with `goal_ramp: null`. A ramped result additionally carries
  `turn_rate_updates` (step, sim time, measured remaining, rate) and
  `commanded_rate_at_stop_rad_s`.

**Values** (`h2_velocity_candidate`, derivation in the YAML comment): `decel_rad_s2: 0.4`,
`min_rate_rad_s: 0.15`, `rate_step_rad_s: 0.025`. The floor sits below both measured cut rates
(≈0.26 settled, ≈0.47 refuted); 0.5 → 0.15 rad/s takes 0.875 s, ≈4 oscillation periods and ≈8×
the policy's ≈0.11 s response (0.1 s-mean yaw rate at 0.4 rad/s 0.11 s after admission); the
longest admitted turn (1.0 rad) needs ≈2.38 s of command, ≈2.56 s at the measured 93 % tracking
(0.465 of 0.5 rad/s) — inside the unchanged 3 s `max_duration_s` (the CPU test drives it through
the real loop: 2.54 s); at most 14 updates per turn, one per ≈0.06 s sim.

**Cost.** The ramp spends ≈0.4 s more of the unchanged 3 s command than a constant-rate turn
(toy fixture, 0.8 rad: 1.96 s instead of 1.56 s; 1.0 rad at 93 % tracking: 2.54 s instead of
2.12 s). A robot tracking much worse than measured — 75 % for 1.0 rad, or the 53 % average of the
MCP 0.6 rad turn that stalled, for 0.8 rad — completes without the ramp (2.62 s / 2.94 s on the toy)
but hits the 3 s deadline with it: the existing fail-closed timeout (execution error, latched
stop), never a larger budget. Each update is one control-channel RPC to an owner running at
≈0.47× real time.

**Live recipe (run by the parent on GPU 0 on 8 October 2026; result below).**

1. Start the owner from this revision's checkout with the geo3-v2 command line plus
   `--velocity-scaling`; `BRIDGE_LISTENING.json` → `hello.capabilities` must list
   `velocity_scaling` (otherwise every ramped turn is refused before motion, by design).
2. Re-pin the private `bases/h2_episode.yaml` (it `extends: h2_velocity_candidate`, so it
   inherits `goal_ramp`) from the new `kit/model-identity.json`: the identity changes with this
   revision because `mobile_bridge.py`, `mobile_base.py` and `isaac_h2_bridge.py` are hashed into it.
3. Run `turn` 0.8 rad through `RobotRuntime → SafeBase` and through MCP (`CASCADE_ROBOT=h2`),
   plus −0.8 and 1.0 rad; per turn record the verifier verdict, settle ω and drift (limits
   unchanged: 0.20 rad/s, 0.05 rad), `turn_rate_updates`, `commanded_rate_at_stop_rad_s`, command
   time used of the 3 s, and the owner's policy rows (commanded wz 0.5 → 0.15 inside ONE generation).
4. A/B on the same owner: the 0.8 rad turn from an `extends:` child with `turn_control:
   {goal_ramp: null}` (revision 2 behaviour).

### Live result (8 October 2026)

[Evidence](evidence/h2-turn-ramp-live-20261008/REPORT.md): every turn, yaw-rate profiles, the
harness and a SHA-256 manifest of the raw rows. The owner from this revision ran with
`--velocity-scaling` on GPU 0 (the 6.2 build, PhysX). The ramp-on and `goal_ramp: null` arms
ran on the same owner, with resolved bases that differ only in `goal_ramp` (checked before any
motion). 39 independent turns; limits and the 3 s command unchanged.

| angle | ramp on | ramp off |
| --- | --- | --- |
| ±0.6 rad | 3/4 (1 timeout) | 3/4 (1 refuted) |
| ±0.8 rad | **8/8** | 1/6 (5 refuted: settle heading drift 0.053–0.084 rad) |
| ±1.0 rad | 1/5 (3 timeouts, 1 refuted) | 5/6 |
| ±0.8 rad through MCP | 4/6 | — |

- **The ramp fixes what it was built for.** The commanded rate is at the 0.15 rad/s floor at
  every ramped stop, and the 0.8 rad settle drifts are 0.028–0.050 rad. One is at 0.0499, so
  the margin is thin.
- **All four timeouts are ramp-on left turns.** The policy's yaw rate dips below 0.2 rad/s
  around 1.0–1.75 s in 24 of the 39 turns, in both arms. On a 1.0 rad turn, that dip plus the
  ramp's deceleration no longer fit in the 3 s command. Without the ramp, all six 1.0 rad turns
  came within tolerance at 2.21–2.43 s.
- **1.0 rad turns sit at the translation limit in both arms.** The verifier path is
  0.166–0.227 m against `max_lateral_drift_m` 0.20 m, and both −1.0 rad refutations are
  "unrequested translation during turn".
- **Verdict: not admitted.** Nothing was retuned after the measurement. The next control
  revision will be designed and measured on its own: either a ramp that budgets the command time
  left, or one that holds the rate through a dip. → [Revision 4](#candidate-revision-4-time-budgeted-goal-ramp-software-only-8-october-2026)
  takes the first option (software only, live A/B pending).

## Candidate revision 4: time-budgeted goal ramp (software only, 8 October 2026)

**Why.** In the revision-3 owner A/B the three +1.0 rad timeouts dipped at full command
(0.08–0.12 rad/s around 1.25–1.75 s), so the ramp, which starts on the measured remaining yaw
(≈0.31 rad), began only at 2.10–2.24 s and its ≈0.8 s deceleration ran past the unchanged 3 s
command. Without the ramp the same turns reached tolerance at 2.21–2.43 s.

**Mechanism** (`SafeBase.turn`, opt-in per profile, inside the same single admission):

- `turn_control.goal_ramp.time_budget: {reserve_s, tracking}` (exact keys; an explicit `null`
  restores revision 3 for an `extends:` child, `goal_ramp: null` still turns the whole ramp off).
- On every freshly validated sample the revision-3 law gets one more term:
  `|wz| = max(min_rate, min(previous, max(sqrt(2·decel·d), d / (tracking · (left − reserve)))))`
  with `d = |remaining| − turn_tolerance` (measured yaw) and `left = ACK end_sim_time_s − sample
  sim time`, i.e. the constant rate that still covers the remaining measured yaw in the ADMITTED
  command time left. Inside the last `reserve_s` the rate is only held. The same `rate_step` gate,
  the same `scale_velocity(scale, generation)` and the same zero-twist stop follow.
- It reads no yaw *rate*, so a transient dip moves it only by the yaw it actually cost (the
  "dip-aware" requirement holds by construction). It can only withhold a deceleration; it never
  raises the rate (`min(previous, …)` and the update gate), extends or renews the command, or skips
  the stop. A dip longer than any budget can absorb still ends in the unchanged fail-closed timeout.
- A budgeted result also carries `turn_rate_budget: {held_samples, first_held}` (how often the time
  left, not the distance left, set the rate) and `command_time_left_s` / `budget_rate_rad_s` per update.

**Values** (`h2_velocity_candidate`): `reserve_s: 0.3` (s of sim; ≈3 policy responses, ≈6 owner
samples), `tracking: 0.93` (the measured 0.465 of 0.5 rad/s that revision 3 also used). Replayed on
the logged revision-3 updates of the A/B (verifier clock), the budget binds on exactly the three
timed-out +1.0 rad turns, the −1.0 rad turn that came within tolerance only at 2.90 s, and the last
update of one confirmed 0.8 rad turn (budget 0.186 vs the law's 0.171 rad/s, so that update is
withheld); the 18 other ramped turns are unchanged.

**CPU evidence** (`tests/test_turn_goal_ramp_budget.py`, a kinematic toy at 0.93 tracking whose
measured yaw rate dips to ≤ 0.1 rad/s 1.0–1.75 s after admission, with and without the ramp):

| angle | no ramp | revision 3 | revision 4 |
| --- | --- | --- | --- |
| 0.6 rad | 1.86 s | 2.16 s, cut at 0.15 | identical to revision 3 |
| 0.8 rad | 2.28 s | 2.70 s, cut at 0.15 | identical to revision 3 |
| ±1.0 rad | 2.72 s | **timeout** (3 s command) | 2.72 s, cut at 0.5 (budget holds the admitted rate) |

With a milder dip (0.2 rad/s) revision 4 keeps part of the ramp (1.0 rad: 2.70 s, cut at 0.35);
without a dip it reproduces revision 3 decision for decision. Golden digest over all nine shipped
base profiles (76 toy episodes, both trees): 75/76 backend command sequences identical to origin/main
133876c (only the candidate's 1.0 rad dip turn changes); the candidate with `time_budget: null` or
`goal_ramp: null` is identical to main in full.

**Cost, stated.** When the budget binds, the turn is cut above the 0.15 rad/s floor, up to the
admitted 0.5 rad/s, i.e. revision 2's cut. For 1.0 rad that is the arm that confirmed 5/6 on the owner;
whether a late 0.8 rad turn cut that way still settles is exactly what the live A/B has to measure.
Dip-*hold* (holding the rate while the measured rate is low) was not built: in all three 1.0 rad
timeouts the dip came before the ramp started, so holding through it buys no time, and the one
0.6 rad timeout (stalled at 0.02–0.06 rad/s with 0.29 rad/s commanded) is out of reach of any
monotone ramp.

**Live recipe (for the parent, GPU 0).** Same owner command line as revision 3 (`--velocity-scaling`).
The change is client-side only (`SafeBase` + the candidate YAML; nothing in the owner's
`SOURCE_FILES`), so the owner identity does not move, but the harness must import CASCADE from this
revision (`CASCADE_H2_REPO=<this checkout>`) and pins the private base from each owner's
`kit/model-identity.json` as before.
Arms from one config dir, checked with `--dry-config` before any owner starts: R = as shipped
(revision 4), V = `turn_control: {goal_ramp: {time_budget: null}}` (revision 3), N = `goal_ramp: null`
(revision 2). Record per turn the verdict, the settle drift/ω, `turn_rate_budget`, the commanded
rate at stop and the command time used; the limits, the 3 s command and the 8 s wall settle budget
stay unchanged.

## Owner architecture (built 7 October 2026 as `scripts/isaac_h2_bridge.py`; mirrors the MicroDuck shared owner)

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
| 1 Binding (model, policy, endpoint, mappings, exclusive ownership, controller clock) | **passed in simulation (7 Oct 2026)** — the owner binds the pinned asset/policy, maps the 31 joints, verifies the deployed actuation against the contract, owns the only command path and the physics clock; receipt + `model-identity.json` | hardware binding (Unitree SDK2 `rt/lowcmd`) is a separate gate |
| 2 Dynamic transforms with epochs/freshness | **partially shown** — states carry epoch/generation/age; the verifier's freshness and settle windows ran on the truth channel | measured limits replacing the candidate profile |
| 3 Whole-body limits, balance, self/environment collision | open by construction (policy commands 14 of 31 joints; arms/head held) | fall detection through the training criteria; later a second owner for the arms |
| 4 Sensors | open | none added yet |
| 5 Rehearsal through MCP/controller/physics/verifier with bound video | **first pass through controller/physics/verifier with video** (3/5 confirmed; no MCP/LLM in the loop yet) | MCP-driven episodes, `walk_distance`, the settle behaviour after a command, measured verifier limits |

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
- The revision-3 turn ramp trades command time for a slower cut: a long turn on a robot that
  tracks much worse than the measured 93 % reaches the unchanged 3 s command deadline and fails
  closed. If the owner episode shows that, the answer is a re-derived ramp (or a deadline-aware
  floor) or a standing hand-over, not a larger `max_duration_s`. (The owner showed it on 8 October;
  revision 4 is that deadline-aware floor. Its own trade: a late turn is cut faster, up to revision
  2's 0.5 rad/s, which is what refuted 5 of 6 ramp-off 0.8 rad turns on settle.)
