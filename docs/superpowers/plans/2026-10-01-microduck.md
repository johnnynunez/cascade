# MicroDuck Isaac Integration Implementation Plan

> **For agentic workers:** Execute task-by-task with test-driven development. The parent owns integration and live physics; scoped workers may implement independent CPU foundations. Never edit another worker's files. A fresh reviewer reviews the complete branch before publication.

**Goal:** Add an opt-in MicroDuck base-only CASCADE runtime with real Isaac Sim 6.1 PhysX and Newton locomotion, bounded tools, priority stop, independent evidence, and a verified PR.

**Architecture:** A mobile runtime shares the public CASCADE composition/MCP entry points, logging and memory but never creates a dummy arm. A private Isaac bridge owns the policy, actuator, simulation clock and autonomous watchdog. Mobility-only tool selection and separate motion/read/stop channels retain the kitchen's existing behavior.

**Tech Stack:** Python 3.12 test environment, existing CASCADE stdlib JSON-RPC MCP and NumPy, ONNX Runtime CPU policy inference, Isaac Sim 6.1 experimental prims, PhysX/Newton CUDA, pinned upstream MJCF/BAM/ONNX inputs.

**Spec:** `docs/superpowers/specs/2026-10-01-microduck-design.md` (approved in Telegram). Original outside checkout: `../DESIGN.md`.

## Global Constraints

- Worktree only: `/home/johnny/Projects/demo/cascade-lab/MICRODUCK/cascade`, branch `feat/microduck-isaac`, initial base `5359405a607a14e9d30182309c40d30c3b95f80b`.
- No other checkout, process, port, GPU experiment, model service or Spark may be changed. CPU work first; inspect live ownership before allocating a private Isaac run.
- No meshes/USD/ONNX or external source trees committed. Preserve code/model/weight licensing separately; model terms remain `Creative Commons BY-SA-NC` with unspecified version.
- No actor-driven teleport/support during scored motion, no fabricated observed pose, no automatic recovery/reset or checkpoint replacement after failure.
- Preserve kitchen defaults and proofs; never relax safety/acceptance to pass.
- Lazy optional imports; minimal installs must not acquire Isaac, ONNX, torch or Pinocchio solely to inspect/use a mobile mock.
- Physical reference: 0.005 s physics interval and 0.020 s actor interval; action ordering by name, no mouth/passive joints, integrated ONNX normalization, previous action raw.
- Reject unknown/recurrent policy layouts in the initial feed-forward profile; do not advertise unsupported skills.
- Report software tests, policy/actuator fidelity, physics per engine, agent integration and hardware as separate acceptance stages.
- User authorized implementation and opening a PR, not merging it. No additional administrative approval loop for reversible, in-scope work; material scope changes or outside-worktree effects still require review.

## Upstream context supplied during implementation

- User-supplied IsaacLab [#8161](https://github.com/isaac-sim/IsaacLab/pull/8161), pinned head `28aa1fca5843208ff9a67935695a4d5376e44d50`, is the primary Newton BAM integration reference. Prefer reuse over a duplicate production drive; task 1's numeric law remains an admission/reference check. Preserve upstream BSD-3-Clause notices for any Isaac Lab code reused.
- Before task 4's physical implementation, prove compatibility between its Newton 1.6 DriveBase/MJWarp bridge and the actual isolated Isaac Sim 6.1 runtime. Kitless upstream tests do not prove Kit compatibility. Keep PhysX validation independent; this PR explicitly rejects BAM on PhysX/OV-PhysX.
- Its linked MicroDuck branch `b74ae74c0927da538846076289031c0881431ff0` has assets/tasks but a different BAM configuration API. Do not mix them without a tested port or import the branch wholesale.
- Missing local USD/NPZ fixtures and author-reported benchmarks are not reproduced evidence. Obtain pinned authentic fixtures or generate them through a real declared reference; no synthetic stand-ins.
- That related asset branch mislabels the model license relative to its own upstream README. Keep the approved per-artifact license handling and no vendored models.
- Full review and downloaded source ledger are preserved outside the checkout at `../research/isaaclab-pr8161/CONTEXT_FOR_CASCADE.md`. Tasks 2/3/5/6 keep the same mobile interfaces and acceptance criteria.

## Review Focus

1. A stop racing startup/command delivery must invalidate the pending action, not merely the current tool; test generation/latch handling in tasks 2, 4 and 5.
2. A live socket serving stale/repeated simulator state must not authorize motion or confirm completion; test source/epoch/step and advance deadlines in tasks 2, 4 and 6.
3. Manipulation-only helpers silently creating a mock arm in base-only mode must be caught by explicit constructor traps; test tasks 3 and 5.
4. A policy that stands/falls/slides while commands report success must be refuted by physical velocity/pose windows; task 6 includes inert actor, wrong direction, fall and missing evidence.
5. Simulator pause/reset or transport loss must not revive the last walking command when resumed; tasks 2/4/5 test wall leases independently of simulated time.

## Task 1: Source admission and policy/actuator foundations

**Files:** create `src/cascade/control/microduck_policy.py`, `src/cascade/control/microduck_actuator.py`, `scripts/microduck_assets.py`, `assets/microduck/manifest.json`, `assets/microduck/NOTICE.md`, `tests/test_microduck_policy.py`, `tests/test_microduck_actuator.py`, `tests/test_microduck_assets.py`. Assets themselves remain outside tracked git paths.

**Interfaces:**
- Policy module exports `POLICY_JOINTS: tuple[str,...]`, `HOME_Q: np.ndarray`, `observation(q,dq,angular_velocity_body,gravity_body,previous_action,command)->np.ndarray` shaped `(1,61)`; `MicroduckPolicy(path, expected_sha256, action_scale=1.0)` with `infer(obs)->np.ndarray` raw shape `(14,)`, `targets(action)->np.ndarray`, `reset()`.
- Actuator module exports an explicitly named, configured BAM-compatible motor-law adapter with pure numeric input/output; never calls a solver or changes body poses. Its documented result distinguishes commanded effort and friction/armature required by the solver. Exact class signature is finalized by the foundation owner and recorded in COORDINATION before bridge integration.
- Asset CLI takes explicit destination, `--fetch` or `--check`, preserves immutable source/policy hashes and licensing; conversion is a separate explicit operation. No import-time network, arbitrary archive extraction or asset overwrite.

- [ ] Write failing tests for exact joint map/HOME/61D layout, reordered joints, raw previous action, one normalizer, dimensions/type/hash/NaN rejection, lazy ONNX import, malformed manifests and path traversal.
- [ ] Run `.venv/bin/python -m pytest tests/test_microduck_policy.py tests/test_microduck_actuator.py tests/test_microduck_assets.py -q`; record RED before implementation.
- [ ] Implement the pinned contract from the archived official sources, preserving Apache notices for copied/adapted code. Fetch complete source asset dependencies explicitly and verify hashes.
- [ ] Test motor law against the actual pinned BAM implementation across error, speed, load, voltage and current-limit cases; resolve the `None` versus `1.75 A` discrepancy by recording both effective reference configurations, not silently changing upstream.
- [ ] Run real ONNX CPU inference on named fixtures; tests must compare actual upstream/reference outputs, not made-up goldens.
- [ ] Commit only code/tests/manifests/notices after GREEN; report which actuator semantics remain unproven in an engine.

## Task 2: Mobile contracts, mock and fail-closed safety

**Files:** create `src/cascade/control/mobile_base.py`, `src/cascade/control/mobile_rig.py`, `src/cascade/control/mock_base.py`, `src/cascade/safety/base_harness.py`, `tests/test_mobile_base.py`, `tests/test_mobile_safety.py`.

**Interfaces:**
- `BaseState` includes robot/source/epoch identity, step, simulation time, client receipt time, producer-state age, world position, wxyz orientation, linear/angular velocities, joint names/q/dq, controller status, generation and contact/fall diagnostics. `as_dict()` and strict `from_dict()` preserve provenance and reject malformed dimensions/nonfinite data.
- `VelocityCommand(vx,vy,wz,duration_s)` represents body-frame SI units and finite duration.
- `MobileBase`: side-effect-free metadata/capabilities; `connect()`, `disconnect()`, `get_state()->BaseState`, `command_velocity(command:VelocityCommand, *, generation:int)->dict`, `stop(*, latch:bool=True)->dict`, `reset_stop()->dict`. The transport owns server generation readback; stop never clears a latch and reset never resumes a command.
- `MobileRig`: named bases, deterministic primary, unknown name rejected, fan-out stop/close continuing after one failure.
- `SafeBase(raw:MobileBase, limits:dict)`: `walk_velocity(vx,vy,wz,duration_s)->dict`, `turn(angle_rad)->dict`, `stop(latch=True)->dict`, `reset_stop()->dict`, `get_state()->BaseState`; priority stop must not wait for a motion/read lock. Public results include measured state/window and explicit outcome, never ACK=completed.

- [ ] Write RED tests for finite input/limits, exact identity/epoch, repeated state, deadman, stop-before-start, stop-during-command/read, reset with queued commands, unknown bases and concurrent command refusal.
- [ ] Implement mock as explicitly kinematic test fixture, never physical evidence. Its failure injection hooks remain in tests where practical.
- [ ] Implement generation/latch safety, finite simulation duration plus independent wall deadline and distinct stop channel; normal stop cancels travel but preserves a validated balancing controller.
- [ ] Test no command after cancel/epoch change; missing or stale feedback fails closed; bridge errors preserve uncertain delivery, do not retry motion.
- [ ] Run targeted tests GREEN and commit code/tests only. Keep exact interfaces in COORDINATION for consumers.

## Task 3: Base-only composition and runtime

**Files:** modify `src/cascade/config.py`, `src/cascade/apps/demo.py`; create `src/cascade/apps/mobile_runtime.py`, `src/cascade/skills/mobile_runtime.py`, `configs/bases/microduck_mock.yaml`, `configs/bases/microduck_isaac.yaml`, `tests/test_mobile_runtime.py`.

**Interfaces:**
- `load_demo_config(..., base: str|None=None, bases:list[str]|None=None)` opts into mobility explicitly; omitted base retains existing arm config. Base-only composition consumes task 2 and creates no arm/FK/grasp pipeline.
- `build_runtime(...)` retains its public API and chooses a mobile composition when configured. Mobile runtime provides the existing observation/memory/trace/tier surface and explicit `tool_specs`, `motion_skills`, stop/reset/shutdown methods. No manipulation methods are exposed.
- `MobileSkillRuntime.execute(name,args)->dict` is the mobile dispatch choke point; schemas include `base` for mobile motion, not `arm`. It uses `TraceLogger`, `EpisodicMemory` and independent verifier from task 6.

- [ ] RED: monkeypatch arm/kinematics constructors to raise and prove base-only never invokes them; default arm config unchanged; empty/unknown/duplicate bases fail explicitly.
- [ ] Implement base profiles/heredity, resolved safety per base and bounded runtime motion tools (`list_bases`, `get_base_state`, `walk_velocity`, `turn`, `stop_navigation`, observation/task_done where appropriate).
- [ ] Add frame/memory/tracing behavior without assuming a fixed manipulation frame or creating a fictitious TCP/gripper. Teardown closes every owned worker/socket.
- [ ] Test mock episode, error outcomes, source/epoch binding, separate run directories and unchanged existing arm episodes; run all runtime/arm-rig tests.
- [ ] Commit after GREEN.

## Task 4: Isaac backend, policy bridge and reproducible conversion

**Files:** create `src/cascade/control/isaac_base.py`, `src/cascade/sim/mobile_bridge.py`, `scripts/isaac_microduck_bridge.py`, `scripts/convert_microduck.py`, `tests/test_isaac_base.py`, `tests/test_mobile_bridge.py`, `tests/test_microduck_conversion.py`; extend only allowlisted environment forwarding in `scripts/isaac_launch.py` if needed.

**Interfaces:**
- `IsaacBase(profile)` implements task 2 over separate bounded command, stop and observation `BridgeClient` connections. No reBot fallback.
- JSON protocol v1: `hello`, `state`, `command_velocity`, `stop`, `reset_stop`, `frame`; any fixture/world reset is separate privileged entry point, not an agent motion tool. Handshake exposes robot type/id, source/epoch, engine/solver, actual device, asset/policy hashes, dt and capabilities. Each motion binds generation and epoch.
- Pure `MobileBridgeController` owns command admission/leases/fault state independently of Kit. Snapshot publication comes only from completed simulation steps; observations never step or alter physics.
- Isaac main thread owns all prim/tensor operations, inference and efforts. Policy at 50 Hz, actuator/physics at 200 Hz; rendering and RPC work cannot create fictitious physics ticks. Stop can invalidate targets from another thread; next legitimate physics tick consumes that invalidation.

- [ ] RED tests for identity/hash/protocol mismatch, no default 8611 dial, bounded RPC failure, stop priority, wall lease expiry during physics pause, stale generation after reset and rejected arbitrary `exec`/joint ops.
- [ ] Import official MJCF with free root, preserve contacts/model metadata, enumerate dependencies and verify kinematic/physical contract before rollout. No community USD treated as authoritative.
- [ ] Test actuator representability against both live engines before declaring a faithful profile. If solver-required static friction/load semantics are unavailable, report a gate failure; do not replace with disguised PD.
- [ ] Implement stepping, camera capture and read-only state publication using existing supported Isaac APIs. Reset only before/outside scored episodes; invalidate policy history and epoch.
- [ ] Run isolated live standing/commanded trials with source-bound receipts and exact process cleanup. Physical failures remain failures, not exception-suppressed READY.
- [ ] Commit tested code/config only, external assets excluded.

## Task 5: Native MCP/CLI capability routing and priority stop

**Files:** modify `src/cascade/apps/mcp_server.py`, `src/cascade/agent/orchestrator.py`, CLI parsing in `apps/demo.py`; create `tests/test_mobile_mcp.py`, `tests/test_mobile_cli.py`. Add optional dependency/entry point documentation only where required in `pyproject.toml` and lockfile.

**Interfaces:** `CASCADE_BASE` selects the base-only profile; no environment setting means existing behavior. MCP `list_tools` can inspect selected profiles without actuating. `emergency_stop`, cancellation and `stop_navigation` reach mobile stop without waiting for execution lock. EOF must invalidate mobile activity immediately, not queue behind walking. Existing kitchen behavior is preserved.

- [ ] RED subprocess MCP tests: no manipulation tools, call to hidden/incompatible tools rejected, initialized/listed without actuation, stop pending during build, stop/cancel while move blocks, EOF while moving, no command revival after reset.
- [ ] Add explicit runtime capability branching using task 3, never duck-typing a fake arm. Maintain locked stdout protocol and stderr logging.
- [ ] CLI/orchestrator use runtime-specific tools and prompts; reflex/habit cannot dispatch arm actions to the base.
- [ ] Exercise the actual stdio subprocess through initialize/tools/list/tools/call and a complete mock mobile episode; run existing MCP regression tests too.
- [ ] Commit after GREEN.

## Task 6: Independent outcomes and acceptance campaigns

**Files:** create `src/cascade/sim/base_truth.py`, `src/cascade/agent/base_effects.py`, `benchmark/diagnostics/microduck_acceptance.py`, `tests/test_mobile_effects.py`; integrate verifier into task 3.

**Interfaces:** `BaseTruthReader` is a separate read-only connection bound to exact robot/source/epoch. `BasePostconditionChecker` retains history/digest/contradictions in the shape MCP expects. It compares caller intent with advancing measured state, never with integrated targets. Campaign outputs per-case raw JSONL, compact receipts, source hashes and video references; any failed/unverified required case returns nonzero.

- [ ] RED for stationary actor during walk, wrong sign, fall, missing sensor, wrong source/epoch, duplicate steps, stale stream, verifier crash and ACK-only completion.
- [ ] Implement balance/walk/turn/stop windows; preserve original actuation errors even if final state is acceptable. Freeze numeric trial tolerances in campaign config before execution, based on admitted reference behavior; record them in every receipt.
- [ ] Run fresh-start/repeated reset campaigns separately for PhysX and Newton through actual MCP. Include forward/reverse, turns both ways, stop/cancel/disconnect, paused clock and deliberate failed actor controls. No successful subset can stand for the full matrix.
- [ ] Record actual view of each engine, inspect start/middle/end frames and send milestones. Record host identity; do not label x86 as Spark or sim as hardware.
- [ ] Run kitchen cases/resets on both engines if shared execution/stop/Isaac behavior changed; compare against source-bound baseline without trying to repair Codex's unrelated in-flight failures.

## Task 7: Review, documentation and PR

**Files:** create `docs/MICRODUCK.md`; update README, docs index, ARCHITECTURE and mobility design with implemented-vs-planned boundaries; add compact reproducible evidence under `benchmark/results/` without local paths/secrets/third-party assets.

- [ ] Document exact install/fetch/conversion/launch/MCP/test commands and physical limitations; show separately whether BAM, each engine, host integration and hardware passed.
- [ ] Run full locked-env suite and booth JS tests; execute minimal-install mobile mock and existing SO-101 mock in an isolated minimal env. Run `ruff check --select F,E9` on changed Python, `git diff --check`.
- [ ] Fresh independent whole-branch code/spec/security review, with special attention to the five Review Focus conditions. Fix material findings with RED→GREEN regressions; full suite after fixes.
- [ ] Refresh origin/main and reconcile new changes without touching Codex's checkout; rerun affected checks for the published tree.
- [ ] Commit/push only this feature branch, create PR to `johnnynunez/cascade` main, read it back and verify head SHA/base/body/files/checks. Do not merge.
- [ ] Report PR URL, actual test/engine outcomes and remaining limitations; no claim of completion while a required engine/case is unverified.
