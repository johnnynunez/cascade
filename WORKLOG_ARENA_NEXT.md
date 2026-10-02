# Arena native preflight checkpoint

Owned worktree: `ARENA_NEXT/cascade`, branch `feat/arena-native-preflight`,
base `f73686216deafbcb219be7cbc214d6dfbc0c0734`. Other simulator checkouts,
installed environments, learned stores and processes remain untouched.

## Contract and skill mapping

The requested result is a real, bounded CASCADE → Arena policy adapter →
native simulator → independent observation episode. A successful launch or
upstream task flag alone is insufficient. No task success, physical stop,
universal control or whole-body safety is inferred from this preflight.

Applied `isaac-sim-orchestrator` for the staged foundation/integration/receipt
workflow. Foundation capabilities are: exact Arena/Isaac Lab/Isaac Sim source
and dependency admission; one Franka articulation with upstream action units;
ordinary composed runtime ownership and stop; solved-state observation with
episode/step/time binding; and bounded offscreen capture. Native launch,
physics/control, rendering and final validation will use their relevant
specialist skills if that foundation is available.

## Read-only foundation findings

- Reviewed Arena pin: `c8d04e2199b86abbbb301bb22cec0effd83c4e63`.
- Its Isaac Lab submodule pin is
  `ae37b028ea415c91ea2bc32609efcd759ed2b974`.
- README advertises Isaac Sim 6.0 / Lab3; the committed `uv.lock` actually
  selects Isaac Sim6.1.0.0, Torch2.11.0+cu128, Newton1.5.2,
  MuJoCo3.11.0, Warp1.16.0 and NumPy2.3.1 on Python3.12/Linux x86_64.
  These exact inputs, rather than the README badge, determine a candidate.
- `cube_goal_pose` is a registered real Franka differential-IK environment,
  with a stand, table, cube and explicit initial joint pose. Its normal
  factory and evaluator are available; its task is not a preflight success
  criterion.
- The CASCADE adapter already supplies the actual `PolicyBase` subclass,
  single action-owner resource, generation checks and priority stop route.
  A final consumer gate is still needed between returned action and native
  `env.step`; no zero-action substitution is an established stop policy.
- GPU1 UUID `GPU-4c811d02-a79b-2837-425e-5f62ac3c578a` is the only authorized
  GPU. The initial observation showed45,923MiB occupied out of97,887MiB,
  including foreign processes that must remain untouched. Disk had2.6TiB free.

## Selected foundation variant

No additional explicit EULA receipt was available for a new Isaac binary
installation. The operator selected the already authorized existing SDK instead:
`MICRODUCK/isaac-newton160` (Isaac6.1rc26), kept separate from the task's private
Arena and Isaac Lab source checkouts. The task imports the exact Arena and Lab
pins above. Runtime differs from the Arena lock: Torch2.12.0+cu130,
Warp1.17.0, gymnasium1.3.0 and the SDK's own packages. PhysX is selected
explicitly. This is never described as locked-recipe parity.

The headless and physics-simulation skills guide the bounded native foundation.
The owned scope is limited to24GiB host memory, four CPU cores and240s with a
15s kill grace. Task caches and persistent paths are private; Python bytecode
writes are disabled. No foreign process or controller is used.

- Foundation01: AppLauncher started on the expected GPU UUID; no physics
  steps. The initial sparse checkout omitted the optional cuRobo registration
  package, which Arena imports unconditionally. Materialized that source only.
- Foundation02: same actual startup, zero physics steps; eager G1 registration
  then required Pinocchio. Native SDK `close()` exits the process, so the
  runner now saves its last receipt and passes its failure exit code before
  requesting close. Exit status alone never admits a run.
- Private dependency closure:18 packages (431MiB installed), exact versions
  and artifact hashes from Arena's lock, installed with `--no-deps` and
  `--require-hashes`. NumPy/Torch/SDK packages were not replaced. The private
  additions provide Pinocchio, Pink and ONNX Runtime required by eager imports;
  no G1 policy is executed.

Raw attempts and installation evidence are outside Git under `ARENA_NEXT/`.
Foundation04 passed the framework/API checks below. Native CASCADE integration remains a separate gate until its own receipt passes.


## Framework foundation and integration gates

Foundation03 built the real PhysX environment, then rejected an unavailable
`SimulationContext.current_time` property before rollout. The retained error
led to the public physical-step count multiplied by the effective timestep.
Foundation04 then completed20 native control steps /160 PhysX solves, with
finite articulation state and clean owned process exit. Exact held joint values
under upstream ZeroActionPolicy do not establish motion tracking. The stronger
integration witness adds actual cube pose/velocity, rather than treating a
counter or held joint position as a motion measurement.

The optional benchmark-local owner exposes only `run_policy_steps` (1–20,
single use). The real Arena adapter routes this tool through ordinary
RobotRuntime. Its final action-consumer fence checks stop generation and local
deadline under the same lock used for stop acknowledgement. A previously
admitted step may finish; no braking, sustained lease, hardware control, task
success or physical-rest claim is made. Failures and terminal/autoreset flags
never permit another step. The episode always remains task-unverified.

Owner/adapter software tests:48 passed, including returned-action cancellation,
expiry, invalid output, terminal, active-close and task-done refusal. An
independent read-only review found no blocker and retained the same stop scope.
The optional native runner uses the upstream CubeGoalPose factory, exact pinned
PolicyBase/ZeroActionPolicy and PhysX APIs. Capture adds a declared fixed RGB
observer camera; it does not enter policy inputs or overwrite physical state.
Source inventories bind CASCADE/Arena/Lab Python/config/lock files before/after;
they do not attest every SDK binary or remote texture dependency.

The Isaac validator's Level1 literal-import check rejects AppLauncher-based
scripts for not spelling `from isaacsim import SimulationApp`. This static
limitation is retained; a second application is not inserted to satisfy it.
The <=20-step scope is also shorter than its generic3s settling rubric and
makes no settled-contact claim. Native API evidence remains a separate check.


## Final native checkpoint

Native01 completed20/160 native steps with measured cube gravity response, but
its second reset failed because a PyTorch inference buffer was reset outside
inference mode. The renderer also exposed an incorrect observer quaternion
order: this pinned Lab API uses XYZW. The entire failed run, gray frames and
source/closure receipts were retained. Native02 fixes only those host/API
issues and adds blank-frame rejection.

Native02 passes both episodes:20 control/160 PhysX steps and8/64 before the
injected stop vetoes the ninth produced action. The source inventory of4295
files is unchanged. Cube height0.200000→0.0209997m and minimum vertical velocity
−1.30800m/s are read from native tensors; q stays constant. Both task_done calls
refuse success, and both repeat commands are denied. Runtime/environment close
completed, native exit0, owned scope inactive, all recorded process births gone.
An offline independent file audit rechecks hashes, source inventories, finite
state, epoch/time/step binding, solve deltas and physical gravity response.

Presentation video starts at frame1 because frame0 follows reset before the
first solve and shows a stale robot render pose; all raw frames are retained.
RGB is illustrative, not a verified pixel-to-pose measurement. The observer
uses an independent read path in the same simulator process. No task, tracking,
physical braking/rest, periodic lease, SDK parity or general arm admission is
claimed. Final focused owner/adapter/runtime suite:77 passed in1.63s.
