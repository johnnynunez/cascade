# MicroDuck stop verification and native finalization

This is a safety/correctness follow-up to the candidate mobile implementation.
It does not admit locomotion, change the physics recipe or establish physical
stop from a command ACK. The model-identity and solved-contact contracts remain
unchanged.

## Observations and positive evidence are different

A state returned by an in-flight read can arrive after the positive observation
deadline while still inside the existing read budget. A conservative capture
bound can also place a newly received state before the stop ACK; a duplicate
physics clock supplies no new duration. None of these observations may supply
positive sample count, a baseline or settling duration.

They must still be validated and retain veto power. Previously, a fault captured
before the deadline but delivered just afterward could be omitted, leaving an
earlier quiet window authoritative. Real TCP tests reproduced both stop commands
reporting `physical_stop_verified=true` and permitting `task_done.success` while
the producer already reported a fault.

The verifier now records bounded `observations` separately from eligible
`samples`. Identity, clocks, numeric bounds and support schema remain mandatory.
The gap to the last eligible sample is checked even across a terminal excluded
suffix. Only eligible advancing samples establish the rest window; every valid
observation within or after that physical window can veto it for activity,
residual motion, path drift or failed/unavailable support. A full newer eligible
quiet window is necessary for recovery. Episode-wide faults, posture failures
and forbidden support remain visible rather than expiring with a short suffix.

No sampling/read/wall budgets or physical tolerances were increased. The tests
use the current explicit synthetic support contract, retain matched healthy,
timely-fault and complete-recovery controls, and do not count software fixtures
as physical measurements.

## Signals during startup

The native owner defers interruption only while acquiring the `SimulationApp`
handle, so cleanup can retain a successfully returned handle. It checks the
first delivered signal before entering active native initialization. The caller
must not defer the entire `open()` call: that could dispatch SDK initialization
after cancellation even while returning the eventual correct signal exit code.

Signal handlers remain scalar-only. They perform no RPC, logging, lock-taking,
filesystem writes or physics operations. Existing inference/commit/upload
cancellation boundaries and their uncertainty records are unchanged.

## Receipts and SDK shutdown

The first observed SIGINT/SIGTERM takes precedence over normal completion and
later shutdown errors. After each persistence/returning-shutdown boundary the
owner reconciles the result and its receipt; signal checks follow the final
read/encode/equality work rather than trusting byte equality alone.

`receipt.json` starts with `receipt_phase=before_sdk_shutdown`. Each correction
preserves the exact earlier bytes in `receipt-history-NNN.json`, increments
`receipt_revision` and atomically replaces the current receipt. Publication has
a fixed three-pass budget, not an unbounded retry loop. Keep the history when
investigating an interrupted run.

`KitNewtonBackend.shutdown()` explicitly distinguishes an actual SDK.close
return (`True`) from no owned app handle (`False`). A legacy/test adapter that
returns no explicit attestation remains unknown; its return cannot certify SDK
cleanup. `teardown.json` reports these cases separately. Partial constructors do
not fabricate successful SDK cleanup.

A native close may terminate the interpreter without returning to Python. No
Python finalizer can then revise an exit code already consumed by the SDK.
Those runs retain a provisional pre-shutdown receipt and require external
process/owned-resource observation. A signal delivered inside such a nonreturning
close remains outside the returning-close guarantee; it is not hidden by the
passing CPU regressions.

Persistence is not the lifecycle. A failed receipt read/write or teardown write
is recorded under `persistence_errors` and never skips the mandatory SDK
shutdown, overrides the resolved exit outcome, or demotes a returned SDK.close
into `sdk_close_returned=false`. If the initial receipt could not be written,
the later reconciliation publishes the resolved outcome. A signal delivered after
SDK.close has already consumed its exit code still decides the process exit and
the final receipt, even though that consumed argument cannot be rewritten.

## Validation scope

The regressions exercise actual runtime/verifier/CLI methods with explicit CPU
state/model/SDK-acquisition doubles, real loopback TCP and real OS signals.
Independent exact-PID signal probes also exercise native `open`, `_initialize`,
`close` and `shutdown` without loading Kit or GPU. Tests cover late receipt writes,
returning/failing close, absent/partial app acquisition, stable receipt reads and
both stop commands' downstream `task_done` decisions.

These are source-bound software checks. CUDA/Kit lifecycle, physical contact
classification, braking from an established gait and locomotion acceptance
remain separate gates described in [MicroDuck](MICRODUCK.md).
