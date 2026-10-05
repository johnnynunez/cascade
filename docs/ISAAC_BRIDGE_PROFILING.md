# Isaac bridge Python profiling

Local [profiling attempt 08](LOCAL_RTX_VALIDATION.md#profiling-attempt-08-placement-timeout-and-real-trace)
captured real Kit CPU and Vulkan GPU zones but left most elapsed time in an
exploratory trace slice outside instrumented main-thread regions. It did not
establish which Python operation caused the delay. The bridge's optional
Python spans measure the existing operations needed to investigate that gap.

## Collector accounting and the opt-in startup-heap freeze (5 October 2026)

With `CASCADE_ISAAC_PYTHON_TIMINGS=1` the helper also registers one passive
`gc.callbacks` entry and adds a bounded `gc` block to every timing summary:
completed collections per generation with count, total and maximum interval,
whether the longest ran on the registering thread, unmatched or pending
boundaries, the frozen-object count and the collector settings observed at
registration and at the summary. Intervals include scheduling and other
callbacks and are not isolated CPU time; the helper never enables, disables,
retunes, forces or freezes the collector and removes only its own callback at
shutdown, after the final summary.

`scripts/isaac_bridge.py --gc-policy freeze-startup-heap` (or
`CASCADE_ISAAC_GC_POLICY`) applies `src/cascade/sim/heap_freeze.py` after every
startup allocation, once the TCP server already answers and before the first
main-loop step, so later automatic generation-2 collections traverse only
objects allocated after that point. The bridge exposes a compact apply record
in its `ping` identity (`gc_policy`) and logs `[bridge] gc policy applied`;
at shutdown it releases after the server has stopped and before Kit closes,
logging `[bridge] gc policy released` with the full collection measured while
still frozen. Both engines, PhysX and Newton, run through this bridge, so the
same flag covers both; it changes no physics setting, camera cadence, deadline
or motion behavior and is not acceptance of anything.

### Direct passive bridge probes on PhysX (5 October 2026)

Two direct launches of the kitchen bridge (`demo/scene/kitchen_config.json`,
headless, `CASCADE_REQUIRE_CUDA=1`, timing summaries on, GPU 1, systemd user
scope) each served a read-only client for 90 s: `state` at about 4 Hz and one
camera frame per second. [Compact evidence](evidence/heap-freeze-20261005/bridge-summary.json)
retains the plans, receipts and log hashes. Without the policy, the collector
ran ten generation-2 collections of up to 209.4 ms, 2.0 s in total, all on the
main thread and all during startup before the first ten-second summary; none
occurred during the 90 s observation window. With `--gc-policy
freeze-startup-heap` the explicit pre-freeze collection took 209.2 ms and
froze 582,533 objects in 0.021 ms, the window again contained no automatic
full collection, and at release a full collection over the 4,843 tracked
objects outside the frozen set took 1.99 ms against 204.4 ms once unfrozen;
4,285 frozen objects had been freed by reference counting meanwhile. Client
medians were 43.6 and 22.0 ms for `state` and about 70 ms for a frame in both
runs, with one `state` read above 3 s in each run that overlapped no
collection and whose timing those two runs did not record. So on PhysX the
policy bounds the cost of a future full collection without changing a passive
window in which none happens; it is not a kitchen-campaign or deadline result.

### Newton engine selection on the local 6.2.0 source build

Three direct Newton launches of the bridge with its default full Newton editor
experience, with and without the kitchen scene, never printed a bridge line:
startup stopped after the `isaacsim.ros2.bridge` extension at about 15.7 s and
stayed silent until the client deadline closed each scope. The PhysX path does
not load that experience, and the MicroDuck shared backend boots Newton through
its own minimal experience. `CASCADE_ISAAC_EXPERIENCE` therefore selects an
explicit `.kit` file for the bridge, validated for a supported package version
and recorded by the launcher that sets it; a minimal Newton experience modelled
on the MicroDuck one served the bridge with Newton on `cuda:0`. The stalled
default remains a retained failure of this build and environment, not a
diagnosed cause.

### Direct passive bridge probes on Newton (5 October 2026)

With that minimal experience and the bridge's default reBot scene, two direct
90 s launches on Newton completed with ordinary closure (same client, same
scope and GPU as the PhysX pair; the kitchen scene was not used on Newton).
Without the policy the collector ran two full collections of 360.7 and 412.9 ms
at 3.4 and 4.2 s into startup and none during the observation window; client
`state` reads had a 15.8 ms median and one 2.6 s read 5.3 s after serving,
which the bridge reported as a physics read that timed out waiting for the
main loop, with frames at 35 ms. With `--gc-policy freeze-startup-heap` the
explicit pre-freeze collection took 453.7 ms and froze 1,229,864 objects in
0.021 ms, the window again contained no automatic full collection, `state`
medians were 15.3 ms with the same single startup stall, and at release a full
collection over the 5,292 tracked objects outside the frozen set took 1.71 ms
against 461.6 ms once unfrozen; 4,258 frozen objects had been freed by
reference counting. On both engines, therefore, a passive window contains no
automatic generation-2 collection and the policy bounds the cost of one that
does occur from 204 to 462 ms down to about 2 ms; neither pair is a campaign,
deadline or physical result, and single runs establish no latency comparison.

## Activation and scope

`CASCADE_ISAAC_PYTHON_SPANS=1` enables the spans. They are disabled by default;
the helper uses only the standard library until explicitly enabled, and imports
`carb.profiler` only after `SimulationApp` exists. Configure a compatible Kit
profiler backend separately before application startup. This environment flag
adds bridge zones; it does not start a backend, connect a collector, or prove
that a capture contains them.

`CASCADE_ISAAC_PYTHON_TIMINGS=1` separately enables bounded summaries of
completed zones without requiring a native profiler. Every ten seconds and at
shutdown, `[bridge-python-spans]` records report counts, exceptions, and wall
and main-thread CPU totals/minima/maxima, bound to process and source identity.
These totals are inclusive: nested zones overlap and must not be summed as
independent costs. Thread CPU excludes other threads and GPU execution; use
the native capture below to investigate those. Diagnostic errors invalidate
the summaries without changing simulation, motion or control deadlines.

Use a fresh diagnostic checkout and bind its exact source, helper, SDK,
process ownership and capture settings. An instrumented checkout has a
different source identity from attempt 08. Keep camera resolution, cadence,
physics step, model inputs and control deadlines explicit in any comparison.
Collect actual GPU tracks as well as CPU zones, and distinguish the profiler's
overhead and existing shared-machine load from product performance.

| Zone | Existing work measured |
|---|---|
| `bridge.loop` | Main-loop body, including its nested zones |
| `bridge.targets.get` / `numpy` / `compose` / `set` | Commanded target retrieval, conversion, composition and write |
| `bridge.wrist` | Wrist camera transform update before rendering |
| `bridge.app_update` | One existing synchronous Kit update |
| `bridge.frame_state.q` / `contacts` | Existing measured state and enabled contact snapshots |
| `bridge.history` | Frame-history clock and record work, including state capture |
| `bridge.refresh` and its child zones | Existing render tokens, RGB, depth, masks and publication |
| `bridge.exec_jobs` and its child zones | Existing queued callable or code execution |

The spans add no tensor getters, CUDA synchronization, robot commands, reset,
retry or physics updates. They preserve measured feedback and camera capture
anchors. Timing zones are not safety or freshness evidence. A profiler error
invalidates the diagnostic recording, disables new zones and preserves the
original enclosed operation and its exceptions.

## Bind CPU event order to the task clock

Before the main loop, the enabled helper attempts one static
`bridge.clock_anchor` zone. A `[bridge-python-spans]` JSON log record contains
the monotonic nanosecond values immediately before and after that zone,
process and native thread IDs, and bridge/helper source hashes. Retain this
log with the raw trace. Inside `bridge.loop`, the helper also emits
`bridge.clock_sample.000001` through at most `.002048`: one sample on the first
loop, then at most one per second measured from the preceding sample's end.
There is no catch-up burst or additional sleep. Each immediate `clock_sample`
JSON record carries `seq`, its monotonic bracket, process/thread/source
identity, `diagnostic_valid` and `error_count`. The helper retains only its
counter and last sample. Attempting a 2049th sample invalidates further
diagnostics and emits `diagnostic_error`; other profiler failures do likewise.
A shutdown record reports errors and the last sample. Without that record,
final status is pending only while the log contains no explicit failed sample
or diagnostic error. A positive sticky error count establishes failure even
if the first error's own log write was lost. A closed earlier CPU window may
still qualify; later failure must remain visible in the final report.

Retain the initial anchor's startup preflight. For the captured interval,
require a contiguous, unique sequence of complete sample zones paired with
their log records, with identical process, native thread and source hashes.
Disclose any samples omitted before or after capture; reject interior gaps,
duplicates, regressions, invalid records and excessively wide brackets.
Select completed main-thread CPU zones
by their actual order in the trace tree between samples whose complete
monotonic brackets lie inside the independently recorded task interval.
Sorting zones by timestamps does not establish this order. Exclude buffered
startup and boundary-crossing zones. Require at least 100 completed updates
for a loaded comparison; insufficient samples or updates leave it unqualified.

These brackets establish CPU event ordering without assuming that Tracy's
clock has a constant offset or a known drift relative to `monotonic_ns`.
Report Tracy durations as relative trace times unless a separate calibration
establishes conversion and rate error bounds. A narrow startup bracket alone
does not bound drift across the task. Never infer an offset from connection
time, capture epoch or the last event. CPU ordering markers do not qualify a
GPU task window or establish CPU/GPU causality; global profile validation
remains incomplete without independent GPU clock evidence.
Disclose later diagnostic errors or cap exhaustion separately; a qualified
earlier closed CPU window does not make the whole recording valid.

Report inclusive and self elapsed times separately; do not add parents to
children or sum overlapping threads. Graphics GPU zones do not measure all
CUDA work or total device utilization.
Kit's bridge zones also do not measure the separate MCP process's detector,
watcher locks or camera decoding. Missing zones never mean zero work.

## Validation status

Local [attempt 09](LOCAL_RTX_VALIDATION.md#profiling-attempt-09-python-spans-and-placement-timeout)
ran source `3ccdc2e8` with these spans. Its technical capture succeeded for
120.460 seconds, while the physical task later timed out during placement.
The first offline matcher rejected the SDK's ` (Python)` suffix; that failure
is preserved. A separately tested adapter accepts that suffix only for known
bridge names on the exact source file and main-thread identity. Reusing the
decoded events, it qualified a closed **118.593-second CPU window with 968
complete updates**, bounded by samples 90 and 197.

`bridge.exec_job.code` accounts for 75.398 seconds inclusive and 73.802 seconds
self elapsed over 548 calls (maximum 0.536 seconds). RGB refresh self elapsed
is 8.449 seconds and mask refresh 3.327 seconds; target retrieval and NumPy
conversion total 0.774 seconds. The
[per-zone statistics](evidence/isaac-profile09/window-stats.json) retain counts,
inclusive/self totals, medians, nearest-rank p95, maxima and update gaps without
adding nested parents and children. These are Tracy-reported durations, not
pure Python execution time or CPU utilization. Execution-job subcomponents
and possible observation overhead require further measurement before a cause
or optimization can be claimed.

The [CPU-window record](evidence/isaac-profile09/cpu-window.json) is qualified;
global profile and GPU absolute alignment remain false. The final diagnostic
status is `pending` because no shutdown summary was observed, despite separate
successful administrative closure. No performance improvement or successful
physical task follows from these timings. Exact source, trace, decode, failed
analysis and closure records are
[retained separately](evidence/isaac-profile09/retained-inputs.json).

A later [passive observer diagnostic](ISAAC_FRAME_ENCODING.md) timed the
snapshot's existing sections without a native task. It identified expensive
image-hash paths and source inspection found unused depth encoding on RGB-only
reads. A separate CPU replay supports independent component encoding for that
case. Passive profile 11 on the new source passed the same 100-sample
measurement, with new-capture payload median reduced from 272.239 to 68.170 ms
across the two runs. This does not assign every active profile 09 job to the
observer or prove a live task speedup; physical acceptance remains pending.

Attempt 08 used the previous bridge without these Python spans. Its successful
capture, failed native task and incomplete clock mapping remain separate
[retained results](evidence/isaac-profile08/retained-inputs.json). This diagnostic
interface does not establish a speedup or successful manipulation; any later
run requires its own source-bound software checks and physical evidence.
The scene source inventory includes the bridge and its profiling helper;
changes require regenerating `demo/scene/own_assets.json` with the maintainer
`demo.scene_identity.source_manifest` operation before source admission.
