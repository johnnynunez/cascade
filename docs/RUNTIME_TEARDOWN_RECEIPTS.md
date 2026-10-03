# Runtime teardown receipts

`shutdown_runtime(runtime, arm)` returns a structured software lifecycle
receipt. A failed park, owner close, camera close, disconnect, or live owned
worker makes `ok` and `complete` false. Teardown still visits the remaining
owners in the existing order: park, beliefs, watcher, stream server, viewer,
cameras, then arms. `ArmRig.disconnect()` and `CameraRig.close()` retain each
member's result rather than discarding exceptions after logging them.

The receipt contains `schema: 1`, `scope: software_teardown`, stage results,
pending thread descriptions, and `physical_rest_verified: false`. Existing
synchronous driver methods returning `None` mean their software close call
returned without an observed exception. A structured `ok: false`, incomplete
result, pending-owner list, invalid result, or raised exception is retained.
This does not independently measure torque, support, rest, object release, or
external simulator termination.

Parking follows the existing backend policy. An e-stopped or unmaterialized
arm remains untouched. Isaac's transport-only disconnect still skips parking
when retained load/contact/release state exists. Hardware keeps its existing
park-before-disconnect policy. No locomotion driver is treated as an arm and
no new torque-off operation is introduced. A false park completion is now
reported without removing the following existing teardown calls.

The legacy manipulation runtime retains its first receipt. Repeated calls do
not repeat parking or disconnects. Delegated mobile/composed runtimes may retry
their incomplete owner cleanup; already-complete domains are not closed again.
`complete` describes the current software cleanup, while `ok` remains false
after any failed attempt. An `attempts` history retains the earlier receipts,
including pending IO, exceptions and receipt-persistence failures. Thus a
later `complete: true` never repairs the episode's failed lifecycle verdict.
This distinction propagates through `RobotRuntime.close()`, the runtime wrapper
and MCP; a cached completed receipt retains its historical failure. Composed
domain close methods must return a dictionary, whereas existing synchronous
legacy driver methods retain their `None` convention.

Callers that historically ignored the return value remain valid Python
callers, but must inspect `ok` before asserting successful teardown. The
manipulation adapter in composed mode propagates the receipt directly.

MCP saves `teardown.json` atomically in its own run directory, with the actual
Python PID and whether a runtime was built. Saving failure itself makes the
receipt unsuccessful and is also reported to stderr. Normal EOF returns zero
only after a successful receipt. The existing handled SIGTERM exit143 is
preserved only if teardown succeeds; failed teardown returns exit1. A killed
process or missing receipt must never be promoted from process absence alone.
The MCP wire protocol and tool results are unchanged.

An explicit repeat of MCP shutdown may finish incomplete delegated cleanup.
It does not repeat the initial stop request. Both the first failure and the
latest completion remain in the persisted receipt; MCP still refuses a clean
exit after any failed attempt. No background retry, new deadline or automatic
movement is introduced.
`complete` aggregates all observed stages: an unresolved initial stop may keep
it false even after delegated cleanup finishes. Process absence does not repair
that missing stage evidence or cause another close of already-finished owners.

CPU validation on base `125e248e406181f66c1f871ece8de3d150374db5`:
eight causal controls fail on the old source because errors or receipts are
lost. The corrected selection passes 165 tests in 45.11 s, including existing
arm shutdown, retained-load/backend policy, owned-thread joins, composed/MCP
wiring and mobile signal-stop tests. Source and protected-store hashes are
unchanged across that run. This is software validation; no hardware, physics,
LLM-host campaign, or GPU run was performed for this change.

Independent review then reproduced a recovery defect in that first checkpoint:
after a real five-second composed drain timeout, the wrapper cache blocked a
later close even after the reader returned. Three causal controls failed on
`cfff2f2`, with the legacy no-repeat control passing. Two additional controls
on an exported copy of that source demonstrated lost nested completion and a
lost direct-owner failure. The follow-up passes 139 tests in 45.74 s, including
actual held-reader interleavings through the wrapper and MCP, direct and nested
receipts, persistence failures, existing composition, retained-load policy and
mobile signal-stop paths. All 568 inventoried source/test/config files and
protected stores remained unchanged during the final run. Ruff F/E9 and diff
checks pass. Evidence: `benchmark/results/runtime_teardown_retry_20261003.json`.
