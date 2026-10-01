# Isaac bridge Python profiling

Local [profiling attempt 08](LOCAL_RTX_VALIDATION.md#profiling-attempt-08-placement-timeout-and-real-trace)
captured real Kit CPU and Vulkan GPU zones but left most elapsed time in an
exploratory trace slice outside instrumented main-thread regions. It did not
establish which Python operation caused the delay. The bridge's optional
Python spans measure the existing operations needed to investigate that gap.

## Activation and scope

`CASCADE_ISAAC_PYTHON_SPANS=1` enables the spans. They are disabled by default;
the helper uses only the standard library until explicitly enabled, and imports
`carb.profiler` only after `SimulationApp` exists. Configure a compatible Kit
profiler backend separately before application startup. This environment flag
adds bridge zones; it does not start a backend, connect a collector, or prove
that a capture contains them.

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
