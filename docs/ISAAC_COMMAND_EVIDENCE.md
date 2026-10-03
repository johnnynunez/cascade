# Optional Isaac command evidence

This diagnostic separates a queued command, a successful SDK target setter,
and a later physical state. None of these alone proves motion, braking, rest,
grasp, placement or task success. It changes no motion budget, velocity limit,
post-ACK fresh-step wait or following pacing interval.

Both options default off. Enable `CASCADE_ISAAC_TARGET_RECEIPTS=1` in the bridge
process and set `CASCADE_MOTION_EVIDENCE_DIR` to a private output directory in
the runtime/MCP process. Tool arguments cannot activate or configure recording.
The feature introduces no SDK reads, RPC operations or physics updates. Added
copies and wire bytes do have overhead; no native timing improvement is claimed.

The TCP ACK describes **queued** asset-convention float64 q, producer instance,
epoch, robot, sequence and a client-generated command ID. The main thread copies
that sequence alongside the existing target snapshot. Only after the ordinary
setter returns does it record the exact little-endian float32 arm/full target
values and byte hashes. The first successful write stays immutable; repeated
writes have a separate last-write record and count. A later queued command or
stop cannot relabel an earlier snapshotted write. The conservative flag
`superseded_before_setter_receipt` means a newer command arrived before the old
setter's receipt; the old call may already be in flight and may still return.
It does not claim that the old target was never written. Setter failures, epoch
invalidation and bounded-retention loss remain visible.

For PhysX, the receipt uses the last completed update already sampled by frame
history, with its original epoch, step and time. This is the recorded boundary
preceding the setter, not proof of the next solve or readback of the drive.
Each update attempt clears that diagnostic boundary first; failed updates or
history capture cannot present the previous boundary as the latest update.
No timestamp is refreshed by a state request. Newton's independent authoritative
clock is not read again for telemetry: an unavailable write boundary stays null
and submission coverage stays incomplete. Existing physical state clocks retain
their normal validation. Remote and client monotonic clocks are distinct;
comparing their absolute values requires separate same-host clock provenance.

Ordinary motion-skill dispatch creates a host-owned recording with resolved arm,
stream IDs and a `home` phase covering every nested return-home segment. The
resolved goal preserves full float precision. Existing pre-send and post-ACK
anchors are copied alongside commands and driver states; nothing is read merely
to make a receipt. `IsaacArm.send_joint_target` still returns None. Old bridges
continue to work: missing optional fields produce incomplete evidence, with no
fallback command, retry or change of action verdict.

The client budgets 30,000 serialized events/16 MiB in total; the bridge retains
at most 4,096 commands. The client reserves 128 events/64 KiB **inside** those
limits for `lifecycle_journal`, leaving 29,872 events/16,711,680 bytes for the
existing detailed prefix. For smaller host-created budgets each reservation is
at most one quarter of its total, rounded down; if either is zero both are zero.
Byte accounting sums the UTF-8 serialized event rows, as before; the final JSON
document's metadata, array separators and Python container overhead are separate.

The journal retains the latest existing `skill_begin/end/raised`,
`phase_begin/end` and `stream_begin/returned/raised` events. Its rows reuse the
same client monotonic capture as the detailed row. They contain phase/stream
identifiers and, where applicable, `settled`, exception type and error text;
targets, physics clocks and state packets remain only in the detailed prefix.
Text limits are explicit (phase/type 96, stream 64, error 512 characters), with
per-row `truncated_fields` and aggregate truncation counts. An oversized compact
row is dropped without evicting the existing tail; otherwise the oldest rows
are evicted until both reserved limits hold. Retained byte/event counts,
observed events, evictions, drops and serialization errors are reported.

This separate tail addresses diagnostic closure loss after detailed-prefix
saturation, as observed in the retained trial-12 failure. It is always marked
`diagnostic_only: true` and `physical_acceptance: false`. A return/end record is
not evidence of braking, physical rest, successful task completion or clean
process teardown. Its `complete` flag refers only to loss-free retention of the
projected lifecycle records, not to whether a skill reached a terminal event.
Any truncation, eviction, drop or logging error leaves overall
`logging_complete` and `submission_coverage.complete` false. The unchanged
submission matcher uses only the detailed records: a retained end cannot fill
missing command/setter evidence. This change has not been replayed natively and
does not revise the earlier failed campaign or claim lower logging overhead.

Files are written after the action body and watcher hold exit.
Trace context links the JSON sidecar and SHA256. File/source-read failures,
overflow, mismatched or missing ACK/write bindings are diagnostic failures;
the original action result or exception is preserved. Source hashes describe
on-disk files at flush, not proof of remote/imported bytecode. A campaign still
needs its own before/after source and process binding. Read-only skills do not
create motion receipts. No postcondition, learned memory or task-completion
rule consumes this evidence as authority.

CPU tests exercise the real bridge Handler/target loop, real TCP compatibility,
the ACK/fresh-state-before-application interleaving, queue/stop races, first-write
retention, errors and invalidation. Deterministic SimulationMotion tests compare
all commands, reads, approvals, waits and terminal outcomes with recording on
and off, including slow ACK, jumps, frozen physics and cancellation. Runtime
tests cover nested home goals, exact signs, bounds and unwritable output. These
tests do not constitute native PhysX/Newton or hardware admission.

The lifecycle follow-up also checks saturation with both byte and event caps,
tiny budgets, nonfinite/invalid fields, independent serialization failures and
original `BaseException` propagation, including broken exception formatters.
Enabled/disabled comparisons include saturated logging and retain the complete
command/read/approval/wait sequence. Source-bound CPU results and retained red
controls are indexed in
[`motion_lifecycle_journal_20261003.json`](../benchmark/results/motion_lifecycle_journal_20261003.json).
