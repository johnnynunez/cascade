# External evaluation adapters

These optional adapters connect reviewed external formats to CASCADE's runtime
and independent verification boundary. Their CPU tests establish serialization,
dispatch and provenance contracts. No native VAB or Arena episode was run for
this change; they do not establish humanoid locomotion or contact performance.

## Isaac Lab-Arena

Inspected source: [`c8d04e2199b86abbbb301bb22cec0effd83c4e63`](https://github.com/isaac-sim/IsaacLab-Arena/tree/c8d04e2199b86abbbb301bb22cec0effd83c4e63).
The real `ArenaExperimentResult.to_dict()` format is
`runs -> rebuilds -> episodes`. The episode recorder includes environment and
episode indices, seed, success (which may be null), episode length, language,
timestamp, variations and progress. The importer is deliberately stricter than
the upstream collector: malformed, failed, empty or duplicate campaigns reject.

```sh
python benchmark/arena/import_results.py /path/to/arena_experiment_result.json
```

Arena's success field lacks a physical model identity, epoch and observation
binding. This CLI therefore reports **unverified**, even for `success: true`.
`import_experiment(..., bindings=..., bound_artifact_sha256=...)` accepts an
external observer's per-record sidecar only for the exact result file. Record
IDs are JSON arrays `[run_name,rebuild_index,env_id,episode_in_env]`.
The selected `source_revision` labels the reviewed import API; the upstream
JSON does not itself attest the producer's source commit.

`ArenaPolicyAdapter` delegates `get_action(env, observation)`, `reset(env_ids)`
and `close()` to an explicitly supplied action controller. Its
`execute_skill(name,args)` calls the ordinary composed CASCADE runtime. Tool
results never become action tensors. `make_policy_type(runtime_factory,
controller_factory, command_resources=..., actuation_owner="arena_policy")` lazily returns an actual subclass of
Arena's `PolicyBase` for registration in an installed Arena application. The
controller must provide the embodiment's existing actuation and safety path;
this adapter does not approve raw tensors or supply a whole-body controller.
The external owner must additionally implement `stop()` and `reset_stop()`;
the latter must return literal `True` to acknowledge an explicit CASCADE stop
reset. Generation/latch checks surround action computation. Cancellation raises
and stops the external owner; it never invents a zero action for a biped.
`reset(env_ids)` alone cannot clear a stop. The rollout must abort on an adapter
exception, and its action consumer retains responsibility for cancellation at
the final `env.step` boundary: a previously returned tensor cannot be revoked.
`ArenaControllerDomain(controller, command_resources=...)` must be registered in `RobotRuntime` before
constructing the adapter; `actuation_owner` is that exact domain ID. Direct MCP
stop therefore reaches the controller through the existing priority mailbox even
while action computation blocks. `runtime_factory(config, owner)` must include
that supplied owner; `controller_factory(config)` constructs it first. Each
command resource is an explicit `ResourceDescriptor` with the reviewed command
endpoint (`controller_id`) and single writer (`writer_id`). No endpoint is inferred
from a live driver. ResourceCatalog rejects another writer on that endpoint,
including a separate arm using the same whole-body DDS command. Failed
construction releases acquired resources. The stop-only domain exposes no motion
tools and its configured resource claims are not physical or safety admission.

Arena's inspected dependency stack is Python 3.12/Linux x86_64, Isaac Lab 3.0,
Isaac Sim 6.0, PyTorch 2.11/CUDA12.8 and Newton1.5.2. It has not been installed
into CASCADE's environment. Source code is Apache-2.0; simulator components,
models and assets retain their respective licenses.

## VAB

Inspected source: [`edcc4bf005446839c0c6f43f8a3bf416702af030`](https://github.com/ehehee/Variational-Automation-Benchmark/tree/edcc4bf005446839c0c6f43f8a3bf416702af030).
The upstream calls are `libero.vab.load_task(path)`, `task.make_env()`,
`env.reset(init_index=i)` and the four-value `env.step(action)`.

```sh
python benchmark/vab/inspect_trials.py /path/to/VAB \
  tasks/libero_object_all_variance/pick_up_the_alphabet_soup_and_place_it_in_the_basket.yaml \
  --init-index 0
```

Inspection binds the Git revision, unchanged YAML bytes, initial-state index,
control frequency, horizon and controller assumptions. It executes no upstream
Python. This adapter admits single-Panda OSC_POSE containment trials only.
Packing teleports delivered objects upstream; popcorn and crate tasks have
partial final predicates. Those tasks reject, rather than inheriting an invalid
physical interpretation. Even `contained_in` checks body-center proximity,
not contact, release or retained support.

`VabLiberoView` translates only public RGB/proprioception into the flat keys
used by `benchmark/libero/backend.py`'s `LiberoArm` and `LiberoCamera`. It keeps
depth absent when absent, validates native action dimensions/limits and enforces
the episode horizon independently of upstream `done` semantics.

`run_trial(trial, runtime_factory=..., skill_plan=..., output_path=...)` resets a
fresh environment, builds the supplied CASCADE runtime, calls its `begin_task`
and namespaced `execute`, and closes owned resources while retaining failures.
It never substitutes a benchmark-specific grasp controller. The factory receives
`(view, trial, trial_id)` and returns `(runtime, cleanup)`.

**A geometry-specific runtime factory remains required.** The historical
`build_wrc_runtime` assumes a table at z=.80m, camera256px, a shared output path
and global factory patches; VAB floor tasks have z≈0 and128px cameras, and omit
depth by default. Do not call that builder unchanged. A reviewed factory must
bind the exact kinematics, camera calibration, scene/workspace geometry and
perception mode, and preserve SafeArm/SafetyHarness. Enabling depth changes the
original task configuration and must be recorded as a variant. Oracle object
poses must not enter benchmark policy observations.

Native execution is optional and was not exercised here. It requires the
complete pinned VAB checkout/assets, compatible robosuite/MuJoCo packages and
offscreen renderer, an explicit private `LIBERO_CONFIG_PATH/config.yaml`
pointing at those assets, private CASCADE memory/output paths, and the reviewed
runtime factory. The inspected VAB requirements pin robosuite1.4.0; the existing
local legacy environment has1.4.1/MuJoCo3.2.3 and was not changed or certified.

## Independent admission

`cascade.eval.trials.verify_episode` consumes a trusted independent verifier
callback and explicit required checks. Its receipt must match the exact trial,
configuration, model identity, epoch, physical step/time window, two observation
digests, external file hash and record hash. Both evidence files must remain
present and unchanged. Missing/mismatched/partial evidence stays unverified;
measured refutations remain refuted.

The callback owns measurement semantics. Contract validation does not establish
that a caller-authored receipt represents physical contact. Required checks
must be chosen before the run for the actual task, including support/contact,
retained placement or stop where applicable. CASCADE's existing physics
verifiers remain authoritative; upstream benchmark success never replaces them.
