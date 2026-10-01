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

## Bind the trace to the task clock

Before the main loop, the enabled helper attempts one static
`bridge.clock_anchor` zone. A `[bridge-python-spans]` JSON log record contains
the monotonic nanosecond values immediately before and after that zone,
process and native thread IDs, and bridge/helper source hashes. Retain this
log with the raw trace. A shutdown record reports profiler errors; absence of
a final record must be reported as incomplete diagnostic evidence.

For a usable mapping, require exactly one matching anchor zone in the trace,
the same process/thread/source, an error-free anchor record, and an acceptable
bracket width. Each anchor boundary must map inside that recorded monotonic
interval. Preserve the resulting clock-origin interval and its uncertainty;
do not invent an exact offset from the capture connection time or the trace's
wall-clock epoch. Missing, ambiguous or excessively wide anchors cannot
qualify a task window. Buffered startup must be excluded explicitly.

Only count completed update and GPU zones wholly inside the proven task
window, accounting for the mapping uncertainty. Use at least 100 completed
updates for a loaded timing comparison. Report inclusive and self elapsed
times separately; do not add parents to children or sum overlapping threads.
Graphics GPU zones do not measure all CUDA work or total device utilization.
Kit's bridge zones also do not measure the separate MCP process's detector,
watcher locks or camera decoding. Missing zones never mean zero work.

## Validation status

Attempt 08 used the previous bridge without these Python spans. Its successful
capture, failed native task and incomplete clock mapping remain separate
[retained results](evidence/isaac-profile08/retained-inputs.json). This diagnostic
interface does not establish a speedup or successful manipulation; any later
run requires its own source-bound software checks and physical evidence.
