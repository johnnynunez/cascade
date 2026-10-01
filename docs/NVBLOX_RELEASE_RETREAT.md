# Retained withdrawal after release

This contract is merged through PR #27. The retained NV06 camera failure below
and NV08 pre-release false-slip failure are separate historical results; the
latter never reached a release episode. See [held observations](HELD_OBJECT_OBSERVATION.md)
and [current acceptance](PROJECT_STATUS_20261001.md).

Before intentional release, the [carry attachment guard](NVBLOX_CARRY_ATTACHMENT.md)
retains the original NV attachment through lift and transport. A latched loss or
unavailable attachment blocks reset before movement. The explicit recovery
described here applies to an existing valid release episode; it cannot clear
that separate carry failure or infer that opening is safe.

The first normal five-object trial on `1efa8e9` physically placed the orange,
then aborted its withdrawal on a 500 ms masked-depth mapper timeout. The one
explicit reset refused home at 27 mm clearance against the unchanged 30 mm gate.
That failed trial remains preserved; this change is not a physical acceptance
receipt or an automatic recovery claim after restarting the runtime.

For payload-tracking Isaac arms, placement retains its object, original retreat
joint target and duration, contact cylinder, backend/harness/map identity,
producer epoch, camera streams and halt/scene generations before opening. A
confirmed release requires both individual fingers at least 98% open and empty
captured bilateral-contact paths on every map camera. Mean opening and
`held_object=None` alone cannot establish release. The camera snapshot uses the
existing physics epoch and same atomic joint read, including individual fingers;
old bridge snapshots without those fields fail closed.

The payload-tracking release path waits for both actual fingers and stable
measured arm positions under the profile's existing opening timeout (eight
seconds in `isaac_kitchen`). That deadline starts on entry to `wait_open`, after
the existing post-open dwell; it is not a total ACK-to-withdrawal budget.
Each poll uses one atomic state reply with finite exact-DOF q/dq, the retained
clock, joint convention, jaw limits, backend and halt guards. Invalid feedback
is terminal. The mean opening cannot substitute for both individual fingers
reaching 98%, and absent or malformed attachment metadata is not an empty hand.

After both fingers are open and the same-step snapshot reports no bilateral
attachment, at least three distinct physics samples must span the backend's
existing `settle_hold_s` window (0.1 physical seconds here). Every observed arm
position must stay within 1 mrad of one fixed copied anchor. Drift or an
observation gap larger than the window starts a new candidate window under the
same wall deadline. Gap and dwell comparisons use 1 ns of numerical slack to
avoid rejecting exact physical-step boundaries due to floating-point rounding.
Repeated steps cannot establish dwell; changed q, dq, jaws
or attachment paths at the same step fail. Once open/no-attachment eligibility
has been seen, jaw regression or reattachment is terminal even after a candidate
window restarts. No new velocity threshold, target, gripper command or recovery
is introduced. Empty bilateral paths do not establish zero unilateral contact.

This observation does not mark the episode released or supply geometry
authority. The geometry barrier still takes its own fresh before/after arm
states and rejects movement over the unchanged 1 mrad bound. NV15 demonstrated
2.304 mrad of measured joint-6 movement during that barrier after opening; its
failed retreat remains a failure. The pending stability change has CPU coverage;
it has not established a subsequent physical success within the existing limits.

The subsequent five-second geometry barrier can receive a valid camera packet
captured while the fingers were still opening. Before release is confirmed,
such a packet, or one still showing the original bilateral attachment, is
pending. Its stream identity is retained immediately; the next packet must
have a strictly newer producer timestamp. Only packets showing both fingers
at least 98% open and no bilateral attachment become map floors. Foreign
contacts, invalid metadata, changed identity and expired deadlines fail. Once
release is confirmed, reopening/contact regressions are terminal, including
during explicit recovery. This adds no gripper commands or opening retries.

Withdrawal waits at most five seconds for every source to commit a newer depth
integration and ESDF query beyond a shared post-open floor. The ordinary
attachment-to-empty transition retains measured background anchors while its
per-prop floors exclude old released-object poses. This barrier does not reset
props, clear history independently, infer unseen free space or reuse the separate
attached-payload refresh API. Mapper RPC remains 500 ms; state reads inside the
barrier use at most one second and share its remaining time.

A failed withdrawal always removes the active exemption and retains the episode.
Public movement and gripper commands remain blocked. Only explicit `reset_scene`
may validate the same open/empty state, camera configuration and map again,
verify measured failure feedback within 1 mrad, and finish the exact original
withdrawal under its original cylinder. Every ordinary joint, occupancy and
workspace check still runs, including both legacy 50 Hz safety edges and the
30 Hz target profile. Home follows with no contact exemption and with the same
halt generation; the ordinary post-place home also retains the place generation.

Scope is local to the executing thread and exact operation. Other threads cannot
borrow, enlarge or clear the cylinder. A real scene reset invalidates the episode.
A rejected retry cannot redefine its measured failure pose. After a partial
withdrawal, new failure feedback is retained only when that same backend has
acknowledged at least one target; lost acknowledgements remain ambiguous and
blocked. The counter proves transport acknowledgement, not physical motion or
settling. Unconfirmed release or missing/changed feedback also remains blocked.

The original `1efa8e9` trial did not retain this new episode. Its final state was
captured without commands, then only its owned bridge and mapper were terminated
normally. That administrative close did not finish withdrawal or home and is not
a restorable PhysX checkpoint. No release authority was reconstructed from the
old receipt or from `held_object=None`. The five-object campaign remains 0/5
complete, with one physically placed orange and four objects unattempted.

The release implementation pin `224b2eb` passed 2,880 tests (43 skipped, 3 deselected) in
318.97 seconds after independent review. Earlier full runs are preserved:
`3564e0b` found ten optional-harness compatibility failures; `ee7e4cf` found one
launcher `/proc` disappearance race. That pin fixes both without weakening
release guards. The subsequent mapper candidate `7bdaf5a` preserves those guards
and passed 2,883 tests; see [mapper reset validation](NVBLOX_MAPPER_RESET.md).
Native physical validation of the release episode uses a fresh scene and remains
separate from these offline results.

A separate mapper-only replay of saved measured frames reproduced the 500 ms
limit without any robot RPC. After the third clear, the first integration took
718.718 ms on the server (client timeout at 500.644 ms), with host spans of
278.697 ms in depth integration and 439.299 ms in ESDF update. All 29 requests have
matching server timing records; six queries took 25.043–37.045 ms. This later
stationary workload is not the lost pregrasp history and does not establish the
cause of the original timeout. Mapper optimization remains separate from this
release-recovery change; no deadline, clearance or publication rule was relaxed.

## Native release diagnostic: withdrawal passed, global camera audit failed

A single run on `7bdaf5a` in a new scene exercised the episode in its original
runtime. The diagnostic first observed three new map commits, performed one
setup reset and picked and placed an orange. It injected labelled mapper
integration refusals only after the native opening command was acknowledged.
The post-release barrier failed as intended, retained measured failure feedback
and the original episode, removed the global contact exemption and rejected an
unrelated command. No actuator call occurred after the injected fault until the
one explicit recovery reset.

After restoring the mapper, that reset required three source commits newer than
the common recovery floor, with the original epoch and empty bilateral attachment
paths. It sent exactly 60 targets for the original two-second withdrawal, with
no jaw command or prop reset in that phase. The measured endpoint error was
0.004086 rad against the unchanged 0.045 rad settling tolerance; the orange's
before/after position changed by 0.000291 mm. This position comparison does not
establish orientation invariance. Home followed without the contact exemption,
then the five props were reset and three fresh map commits were observed. The
withdrawal and home requests took 15.560 s and 24.782 s; the longest motion request in
the run took 73.795 s, below the unchanged 120 s budget. Cleanup sent no commands.
The release contract proves individual opening and absence of a bilateral
attachment, not zero contact at each finger.

**The diagnostic's overall result is FAIL.** Its physical pickup/placement
subcheck passed, but the aggregate physical audit also includes cameras and
failed. Each of the three cameras had two stale snapshots, sequences 170 and
171 out of 1,947, with maximum server age 3.310713 s (client-finished age
3.362640 s). All failed samples remain part of the verdict. These samples
preceded the first pick actuator call and were outside every recorded joint
motion request interval. The largest sampled camera age inside those individual
intervals was 0.774842 s; that narrower result is not continuous-camera acceptance.

The saved timeline places a 3.214 s gap between the pick request and the first
in-skill note. In the stalled observer call, simulation advanced four physics
steps across 3.295 s of server wall time. Recorded grasp localization and GGX
planning follow that interval. The runtime calls its lazy truth-pose snapshot
before entering the skill; its initial `RigidPrim` construction on the bridge
main thread is the leading stall hypothesis. Exact truth-stage timing and cache
contents were not recorded, so exclusive causality is not established. Offline
vendor inspection also found that the legacy constructor defaults can author
physics settings and transforms. An initial scene audit and unchanged source
hashes do not establish an unchanged runtime scene profile after that call.
The next validation therefore requires the corrected observation path and a
fresh scene; this scene has received no further actuator commands.

A separate 4.550 s gap between captures observed by the witness overlaps the
setup prop reset. It does not establish that the producer emitted no intermediate
frames. The diagnostic recorded 56 synthetic refusals, 1,035 actual joint targets,
four jaw commands and two prop resets across its authorized phases. All 27
artifact hashes, 299 diagnostic source hashes and 259 startup/source/model hashes
matched, including the model fixed before launch. The
[compact receipt](../benchmark/results/nvblox-release-recovery-camera-failure-20261001.json)
binds the full evidence, independent review and offline timing analysis. This
single recovery exercise does not complete any normal five-object campaign;
the ordinary acceptance result remains pending.
