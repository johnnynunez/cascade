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

The client buffers at most 30,000 events/16 MiB; the bridge retains at most
4,096 commands. Files are written after the action body and watcher hold exit.
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
