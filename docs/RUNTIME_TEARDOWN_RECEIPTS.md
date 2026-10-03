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

The first runtime receipt is retained on that runtime object. Repeated
shutdown calls return a copy; they do not repeat a park or erase a prior
failure. Callers that historically ignored the return value remain valid
Python callers, but must inspect the receipt before asserting successful
teardown. The manipulation adapter in composed mode propagates it directly.

MCP saves `teardown.json` atomically in its own run directory, with the actual
Python PID and whether a runtime was built. Saving failure itself makes the
receipt unsuccessful and is also reported to stderr. Normal EOF returns zero
only after a successful receipt. The existing handled SIGTERM exit143 is
preserved only if teardown succeeds; failed teardown returns exit1. A killed
process or missing receipt must never be promoted from process absence alone.
The MCP wire protocol and tool results are unchanged.

CPU validation on base `125e248e406181f66c1f871ece8de3d150374db5`:
eight causal controls fail on the old source because errors or receipts are
lost. The corrected selection passes 165 tests in 45.11 s, including existing
arm shutdown, retained-load/backend policy, owned-thread joins, composed/MCP
wiring and mobile signal-stop tests. Source and protected-store hashes are
unchanged across that run. This is software validation; no hardware, physics,
LLM-host campaign, or GPU run was performed for this change.
