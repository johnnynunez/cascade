# External evaluation adapters

**Next action:** import an Arena result or inspect one VAB trial on the CPU (seconds; no simulator):

```sh
uv run python benchmark/arena/import_results.py /path/to/arena_experiment_result.json
uv run python benchmark/vab/inspect_trials.py /path/to/VAB \
  tasks/libero_object_all_variance/pick_up_the_alphabet_soup_and_place_it_in_the_basket.yaml \
  --init-index 0
```

These optional adapters connect reviewed external formats to CASCADE's runtime and independent
verification boundary. CPU tests establish serialization, dispatch and provenance contracts. Native
preflights below exercised real simulators but **no benchmark task completed**; nothing here
establishes humanoid locomotion, contact performance or task success.

## Isaac Lab-Arena

Inspected source: [`c8d04e2199b86abbbb301bb22cec0effd83c4e63`](https://github.com/isaac-sim/IsaacLab-Arena/tree/c8d04e2199b86abbbb301bb22cec0effd83c4e63).

1. **Import:** `ArenaExperimentResult.to_dict()` is `runs -> rebuilds -> episodes` (environment
   and episode indices, seed, success (may be null), length, language, timestamp, variations,
   progress). The importer is stricter than upstream: malformed, failed, empty or duplicate
   campaigns reject. Arena's success lacks model identity, epoch and observation binding, so the
   CLI reports **unverified** even for `success: true`; `import_experiment(..., bindings=...,
   bound_artifact_sha256=...)` accepts an external observer's per-record sidecar only for the exact
   result file (record IDs `[run_name,rebuild_index,env_id,episode_in_env]`). `source_revision`
   labels the reviewed API; the JSON does not attest the producer's commit.
2. **Policy adapter:** `ArenaPolicyAdapter` delegates `get_action(env, observation)`,
   `reset(env_ids)` and `close()` to an explicit action controller; `execute_skill(name,args)` calls
   the composed CASCADE runtime and tool results never become action tensors.
   `make_policy_type(runtime_factory, controller_factory, command_resources=...,
   actuation_owner="arena_policy")` lazily returns a real `PolicyBase` subclass. The controller
   provides the embodiment's actuation and safety path; the adapter approves no raw tensors and
   supplies no whole-body controller.
3. **Stop:** the external owner implements `stop()` and `reset_stop()` (which must return literal
   `True`). Generation/latch checks surround action computation; cancellation raises and stops the
   owner and never invents a zero action for a biped; `reset(env_ids)` alone cannot clear a stop.
   The rollout must abort on an adapter exception, and the action consumer owns cancellation at the
   final `env.step` (a returned tensor cannot be revoked).
4. **Ownership:** register `ArenaControllerDomain(controller, command_resources=...)` in
   `RobotRuntime` first (`actuation_owner` is its domain ID) so direct MCP stop reaches the
   controller through the priority mailbox. `controller_factory(config)` builds the owner and
   `runtime_factory(config, owner)` must include it. Each command resource is an explicit
   `ResourceDescriptor` (`controller_id`, single `writer_id`); `ResourceCatalog` rejects another
   writer on that endpoint. The stop-only domain exposes no motion tools; its claims are not
   physical or safety admission.
5. **Native foundation:** Arena's committed lock at the inspected source pins an Isaac Sim older than
   6.2, so this preflight is not lock parity. `native_preflight.py --out <dir> ...` runs under an
   Isaac Sim 6.2.0 SDK's `python.sh` (hashed deps in `private_dependencies.txt`) and refuses any
   other SDK. The [earlier receipt](../../docs/evidence/arena-native/20261002.json) exercised real
   Arena/PhysX through the CASCADE runtime, preserved task-unverified results and tested
   cancellation at the final action consumer, on an SDK this preflight no longer accepts; it has not
   been re-run on 6.2. No task success, SafeArm tracking, Newton parity or physical braking. Code is Apache-2.0; simulator components, models and assets keep their
   licenses.

## VAB

Inspected source: [`edcc4bf005446839c0c6f43f8a3bf416702af030`](https://github.com/ehehee/Variational-Automation-Benchmark/tree/edcc4bf005446839c0c6f43f8a3bf416702af030)
(upstream calls `libero.vab.load_task(path)`, `task.make_env()`, `env.reset(init_index=i)`, the
four-value `env.step(action)`).

1. **Inspection** binds the Git revision, unchanged YAML bytes, initial-state index, control
   frequency, horizon and controller assumptions, and executes no upstream Python. Only single-Panda
   OSC_POSE containment trials are admitted; packing (teleports delivered objects), popcorn and crate
   (partial final predicates) reject. Even `contained_in` checks body-centre proximity, not contact,
   release or support.
2. **View:** `VabLiberoView` maps only public RGB/proprioception to the keys of
   `benchmark/libero/backend.py`'s `LiberoArm` and `LiberoCamera`, keeps absent depth absent,
   validates action dimensions/limits and enforces the horizon independently of upstream `done`.
3. **Trials:** `run_trial(trial, runtime_factory=..., skill_plan=..., output_path=...)` resets a
   fresh env, builds the supplied runtime (factory gets `(view, trial, trial_id)`, returns
   `(runtime, cleanup)`), calls `begin_task` and namespaced `execute`, and closes owned resources
   while retaining failures; it never substitutes a benchmark-specific grasp controller.
4. **A geometry-specific runtime factory is required.** The historical `build_wrc_runtime` assumes a
   table at z=0.80 m, 256 px cameras, a shared output path and global patches (VAB floor tasks: z≈0,
   128 px, no depth): never call it unchanged. A reviewed factory binds kinematics, camera
   calibration, scene/workspace geometry and perception mode and keeps SafeArm/SafetyHarness;
   enabling depth is a recorded variant; oracle object poses never enter policy observations.
5. **Native CPU preflight:** from the repo root, `<vab-env>/bin/python -m
   benchmark.vab.native_preflight <vab-checkout> <out> [--cancel-after-step 8]` with the pinned VAB
   checkout and `src/` on `PYTHONPATH`, where `<vab-env>` is a separate venv installed from
   `benchmark/vab/requirements-native-lock.txt` (robosuite 1.4.0, MuJoCo 2.3.7; original task
   bytes, RGB-only). It runs an explicit floor/Panda factory through real MuJoCo, CASCADE relative
   motion and an independent solved-state witness, including cancellation; not a grasp/placement
   result. It needs the full pinned VAB checkout and
   assets, an offscreen renderer, a private `LIBERO_CONFIG_PATH/config.yaml` and private CASCADE
   memory/output paths. The legacy local env (robosuite 1.4.1 / MuJoCo 3.2.3) was not changed or
   certified.

## Independent admission

`cascade.eval.trials.verify_episode` consumes a trusted independent verifier callback and explicit
required checks. Its receipt must match the exact trial, configuration, model identity, epoch,
physical step/time window, two observation digests, external file hash and record hash, and both
evidence files must remain present and unchanged; missing, mismatched or partial evidence stays
unverified and measured refutations stay refuted. The callback owns measurement semantics: choose
required checks (support/contact, retained placement, stop) before the run. CASCADE's physics
verifiers remain authoritative; upstream benchmark success never replaces them.
