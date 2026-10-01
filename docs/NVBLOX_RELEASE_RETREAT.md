# Retained withdrawal after release

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

The original live scene still runs `1efa8e9`; these source changes are prepared
and tested in a separate worktree. The old runtime's episode was not retained,
so any recovery of that scene would require a separately reviewed diagnostic
rehydration with explicit historical and live evidence. No such actuation is
performed by the offline tests.
