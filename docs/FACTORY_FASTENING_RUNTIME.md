# Mounted Factory fastening runtime

**Next action:** run the thread-contact fixture (an SO-101 drives a hex socket in Newton; a few
minutes on a GPU host) in a standalone Newton environment, not cascade's `.venv`:

```bash
<newton-env>/bin/python scripts/validate_screw_contact.py \
  --robot-asset assets/mjcf/so101/so101.xml --output runs/thread-contact-01 --cache <dir>
```

`<newton-env>` is a separate venv built per [Newton's install guide](https://newton-physics.github.io/newton/1.6.0/guide/installation.html)
with the recorded run's packages ([environment](evidence/factory-thread-contact/environment.json):
newton 1.6.0, warp-lang 1.17.0, mujoco and mujoco-warp 3.12.0, scipy, trimesh). Newton and trimesh
are not cascade extras, so `uv run` cannot supply them; the script puts `src/` on `sys.path`
itself. `--output` is a new directory (an existing one is refused).

Status:

1. **Works (simulation):** one native mounted turn-and-rest episode through the ordinary
   configuration/MCP route passed the unchanged verifier (2026-10-04), and two more on 5 October
   with and without the startup-heap freeze.
2. **Not shown:** repeatability, pickup, initial engagement, seating, withdrawal, calibrated
   preload, hardware, scene video. The ordinary wrist routine stays
   [unverified](#ordinary-turn_screw-result).
3. Shipped profiles stay inert: `device: null` and `model_identity_sha256: null` until a private
   profile pins them.
4. **Humanoid variant (simulation):** a Unitree G1 turns the same nut with a hand-held driver by
   rolling its wrist, through the same tool and unchanged verifier
   ([HUMANOID_FASTENING.md](HUMANOID_FASTENING.md), recipe `g1_wrist_driver_m20_v1`).

## Native mounted turn and rest, 2026-10-04

Source `c7f6ec3b3495b961060033e919f148e743d6d3dd`, model
`fe61b52ae92f0080398a5a0f9b59e72f88ae3429b9468504f008648dc35acf8d`, task epoch
`cc282224897e40f48f13ec1399fd14a4`. Preparation loaded all 39 compiler variants and passed all 22
predicates with steps and clocks at zero; readiness supplied 0.500000012 simulated seconds of quiet
in 6.7573 wall seconds (10 s budget). One `reset_stop` and `fastening.turn_screw(turns=1.0,
direction="tighten")` then passed the existing verifier:

| Measurement | This episode |
| --- | --- |
| Rotation before stop | 0.950691 turns; existing one-turn tolerance is 0.05 |
| Final rotation / axial advance | 0.958650 turns / 2.398416 mm |
| Maximum pitch residual / radial offset | 0.028865 mm / 0.150427 mm |
| Contact-supported angular travel | 100% |
| Observed rest | 0.500000012 simulated seconds, exact zero spindle command/effort |
| Command admission to stop ACK | 39.848196 wall seconds, within 40 seconds |
| Complete retained native stream | 2,047 solves and 554,646 candidate/solved contact records |

The command margin was only **0.151804 s**; one shared-GPU episode, no repeatability. Max
controller-acceptance age 0.108428 s under the 0.2 s limit; three generation-2 GC pauses on the
calling thread; nothing relaxed. The private profile selected `sdk_recipe:
isaacsim_48b2d951_newton_1_6_1rc1` (Isaac Sim source `48b2d951919bbb7330de3c2aad54e15369fa0183`,
Newton 1.6.1rc1, MJWarp/MuJoCo 3.12, Warp 1.17; unknown selections and mixed source sets reject; the
pin JSON ships in the wheel) and `precompile: factory_nv19_contact_writer_compile_v3`. The MCP
handler ran in-process (no stdio or LLM); socket mounted and nut pre-engaged
([figure](evidence/factory-turn-rest-20261004/measurements.png),
[receipt](evidence/factory-turn-rest-20261004/receipt.json)).

## Thread contact fixture

The command is at the top: an SO-101 (MJCF, from `fetch_robot_assets.py so101`) drives a motorized
hex socket in Newton against the Factory nut and bolt (`--assets`, default `assets/factory/nut_bolt`
from `fetch_factory_assets.py`), and the nut moves only through solved thread/tool contacts;
`sim/threading_verification.py` judges the result. `--no-drive` commands zero socket torque,
`--misaligned` moves the socket off the bolt axis.

## Ordinary `turn_screw` result

`agent/effects.POSTCONDITIONS` registers `turn_screw` as `threaded`. No fastener
observer exists in the ordinary runtime, so a completed wrist routine stays
`unverified` (`verified: false`) and never trains a reflex.

## Configure and run the mounted domain

1. `configs/robots/factory_m20_mounted.yaml` declares `kind: fastening`, the fixed recipe and the
   official installer asset paths (`fetch_factory_assets.py` → `assets/factory/nut_bolt`,
   `fetch_robot_assets.py so101` → `assets/mjcf/so101/so101.xml`). `load_robot_config` /
   `describe_robot` expose `fastening.turn_screw` without opening a device; discovery is not
   permission to actuate.
2. A private profile must select an explicit `cuda:N` (`auto` and CPU are unsupported, no
   fallback). An owned process calls `prepare_factory_model(profile, cache_dir)` to construct and
   bind the model and get its digest without solves or uploads; a later fresh process must
   reproduce that digest from the pinned profile before the owner starts. Preparation never
   fetches assets or substitutes SDK implementations.
3. `build_robot_runtime` builds the domain via `apps/factory_runtime.py`; the existing MCP server
   selects it with `CASCADE_ROBOT=factory_m20_mounted` (no alternate server or fake ArmBase).
   Dynamic roots, internal multi-DoF joints, conflicting writers and unknown fields are refused;
   reused output epochs are refused.
4. The owner starts latched at generation zero. Readiness allows 10 host seconds for 0.5
   continuous physical seconds of measured rest with loaded thread/tool engagement and exact zero
   spindle effort (model construction is timed separately); the owner's lifetime is 120 host
   seconds from `owner.start()`. No automatic reset or motion during readiness.
5. Variants: `factory_m20_mounted_margin_v2` (`factory_m20_fixed_axis_margin_v2`) moves the whole
   static bolt, pre-engaged nut and seat ring 10 mm toward the base to `(0.23, 0)` m so the
   initial `wrist_flex` clears the unchanged 0.02 rad guard (IK with 0.025 rad clearance; CPU
   kinematics only, still needs native preparation);
   `configs/robots/factory_m20_shoulder_seating_heap_freeze.yaml` adds the
   [heap freeze](#opt-in-startup-heap-freeze-software-candidate-5-october-2026).

## Domain, owner and verifier contracts

1. **Domain** (`FasteningDomain`, `skills/fastening_runtime.py`): one command resource covering arm
   and spindle (a second writer is rejected); its only motion is `fastening.turn_screw(turns=1,
   direction="tighten")` with the socket already mounted and the nut engaged. `control/fastening.py`
   binds the recipe digest, exact body/tool/joint/collider identities, permitted contact pairs,
   timestep and limits digest for the fixed world +Z Factory M20 fixture with 2.5 mm pitch (not
   an arbitrary-thread adapter).
2. **Observations** (`FasteningSolve`, `sim/factory_observation.py`): every solve keeps capture
   time, epoch, generation, step, physical time, joint states and efforts, collision bounds, actual
   nut/fixture/tool poses, spindle speed/effort and all solved collider pairs. Counts at capacity,
   unknown buffers and positively loaded forbidden pairs reject; zero-force records cannot create
   engagement. Collision coverage checks raise (Newton's `verify_buffers=True` only warns); the
   hydroelastic route is unsupported. Solved forces describe the interval [previous step, completed
   step]; no instantaneous torque sensor is implied.
3. **Owner** (`sim/factory_model.py`, `sim/factory_owner.py`): one bounded admission, upload and
   solve per cycle, checking the native `solver._step` and exact float32 `d.time += dt` recurrence
   (CUDA graph replay unsupported). The guarded upload is the admission fence: a stop during
   planning prevents the old write; only a later zero-spindle upload produces the stop generation.
   Stopped, faulted and closing paths submit exact zero spindle effort and keep approved arm
   targets; observed effort must satisfy `abs(raw) <= max(cap, float32(cap))` (0.05 Nm spindle).
   Closure joins for at most 2 s; a blocked SDK call needs an external process watchdog.
4. **Verification** starts strictly after admission, uses a separate passive reader and
   `verify_threading` on actual fastener poses, then stops and observes the new stop generation
   through a full rest window. The threaded outcome must survive stopping: `pre_stop_threading`
   keeps the achievement, `final_threading` (used by `postcondition.threading`) covers the whole
   observed interval, so a passive unwind after the turn fails the tool. Persistence failure
   revokes authority and leaves results unverified; a failed closure is never success.
5. **Diagnostics:** `FasteningObservationAgeFault` keeps the rejection and adds a bounded
   `observation_age` payload (capture/check times, limit, stage, step, generation, epoch, hashes).
   Raw `backend.advance` values are retained as private pickle-protocol-5 records created only by
   the trusted in-process backend (20,000-record capacity; encode cost counts against the age
   limit; expansion failure latches stop). No file, transport or caller-supplied byte stream is
   ever decoded: adding an encoded-byte API or accepting untrusted objects with pickle reducers
   would break that premise (Python encapsulation is no in-process security boundary).

Write-guard limits (initial candidate contract; physical and host budgets are distinct and claim no
real-time operation):

| Check | Initial candidate contract |
| --- | --- |
| Joint speed / margin | 0.8 rad/s / 0.02 rad, from the SO-101 safety profile |
| Joint following error | 0.1 rad maximum; not a hardware calibration |
| Spindle effort / measured angular speed | 0.05 Nm / 10 rad/s |
| Geometry | Complete moving arm/tool bounds within declared workspace and above table; native target FK required |
| One-turn command budget | 4 physical seconds and 40 host seconds, both required |
| Maximum observation age | 0.2 host seconds |
| Observed rest | 0.5 continuous physical seconds, zero commanded/solved spindle effort, speeds ≤0.001 m/s and ≤0.02 rad/s |
| Rest budget | 3 physical seconds and 30 host seconds, beginning at the stop ACK |

`ThreadContract` is unchanged: 0.05-turn tolerance, 0.3 mm pitch residual, 1 mm radial / 0.05 rad
tilt gates, fixture-motion bounds and at least 80% contact-supported angular travel. Stop revokes
authority; its immediate ACK is a latch receipt, not proof of a zero upload or rest.

### Opt-in startup-heap freeze (software candidate, 5 October 2026)

`gc_policy: freeze-startup-heap` (`configs/robots/factory_m20_shoulder_seating_heap_freeze.yaml`,
the shoulder-seating profile plus that key) applies `src/cascade/sim/heap_freeze.py` after model
construction and the pin check, before the owner thread starts, so later full collections skip the
startup heap (earlier readiness failures had rejected 0.43–0.46 s observation ages while a 433–449
ms generation-2 collection ran in `observer.read`).

1. `gc-policy.json` binds the apply receipt (counters, durations, implementation digest) to the
   model pin; `readiness.json` carries the selection only when selected; `closure.json` records
   the release, which happens only after `owner_thread_closed` (an unconfirmed join leaves the heap
   frozen and closure not ok).
2. Collector thresholds and callbacks are unchanged; the physics `model_identity_sha256` is
   unchanged by design (a host-process setting bound by the profile marker and receipts).
3. Native pair on 5 October (same model pin `e03bf144…`, `turn-01` on, `turn-02` off): both passed
   the unchanged verifier (readiness 6.147 / 6.335 s of 10 s, stop ACK at 35.658 / 35.600 s of 40 s,
   axial advance 2.3992 / 2.3977 mm) ([records](evidence/factory-heap-freeze-20261005/)).
4. Neither episode had an automatic generation-2 collection in its control window, so this proves
   the apply/bind/release mechanism, not a removed pause. Remaining observation gaps above 0.2 s sit
   outside the command window (observation drain, `observation.asdict`) and stay an open item.
5. CPU mechanism: [heap-freeze receipt](../benchmark/results/heap_freeze_cpu_20261005.json); native
   interpreter numbers in [MicroDuck](MICRODUCK.md#shared-scene-implementation-boundary).

## CPU candidates (not selected by any native profile)

1. Process boundary: `sim/factory_process_wire.py` and `sim/factory_process_protocol.py` carry the
   private CPU adapter's transport; launchers keep their owner, journal and queue.
2. Precompilation: `configs/robots/factory_m20_precompile_writer_v3.yaml` is an opt-in
   compiler-preparation variant with `device` and `model_identity_sha256` left `null`; a rejection
   is saved as `precompile.json`.

## Receipts (2026-10-03 checkpoints)

| Checkpoint | Receipt |
| --- | --- |
| stage one, readback, owner, configuration | [stage1](../benchmark/results/factory_runtime_stage1_20261003.json), [observation](../benchmark/results/factory_observation_20261003.json), [owner](../benchmark/results/factory_owner_20261003.json), [config](../benchmark/results/factory_config_20261003.json), [device selection](../benchmark/results/factory_device_selection_20261003.json) |
| native preparation failures and fixes | [SDK interface](../benchmark/results/factory_sdk_interface_20261003.json), [optional array](../benchmark/results/factory_optional_array_20261003.json), [initial margin](../benchmark/results/factory_initial_margin_20261003.json), [binding scalars](../benchmark/results/factory_binding_scalars_20261003.json) |
| readiness failure and age diagnostics | [readiness v3 failure](../benchmark/results/factory_readiness_v3_failure_20261003.json), [observation age](../benchmark/results/factory_observation_age_20261003.json) |
| outcome and retention | [final outcome](../benchmark/results/factory_final_outcome_20261003.json), [raw archive](../benchmark/results/factory_raw_archive_20261003.json), [archive replay](../benchmark/results/factory_raw_archive_replay_20261003.json) |
