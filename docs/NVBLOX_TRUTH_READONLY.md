# Read-only construction of truth-pose views

Candidate `fe2007cbbf1812803f41a5930690822cb9bcd493` changes the lazy
`TruthPoseReader` probe to construct `RigidPrim` with
`reset_xform_properties=False` and `prepare_contact_sensors=False`. The installed
Isaac API enables both by default. Those defaults can normalize USD transform
operations and author a zero sleep threshold while constructing a view that the
caller intended to use only for observation. The change preserves the existing
pose API, per-path view cache, label matching and error handling.

The NV06 release diagnostic on the preceding `7bdaf5a` runtime observed a 3.214 s
interval between starting the pick request and its first in-skill note. Its two
stale snapshots per camera occurred before the first pick actuator call. The
runtime takes its lazy truth snapshot before entering the skill, making first-use
view construction a leading stall hypothesis. That run did not record truth
stage timings or cache contents. The constructor's authoring behavior is a
separate, concrete code finding; neither it nor the later timing control proves
exclusive causality for the earlier camera stall. The original diagnostic remains
a global FAIL despite its successful withdrawal and reset subchecks.

Focused validation recorded 197 passing tests in 18.27 s. The complete command
for that initial selection was not retained, so its log is evidence of the
reported selection only, not a fully reproducible suite manifest. An independent
rerun on the clean, immutable candidate recorded its command, source hashes and
log: 84 tests passed in 0.20 s across the probe and existing truth-channel tests.
Two new tests execute the actual probe against a synthetic API with Isaac's
mutating constructor defaults. They cover observation without authoring, view
reuse, and refusal to use a direct authored fallback or expired pose when the
view call raises.

The same two tests against the preceding `7bdaf5a` implementation failed on the
synthetic authoring assertions. The corrected baseline invocation isolated the
copied test from repository `conftest.py` and selected the baseline with
`PYTHONPATH`. An earlier invocation accidentally imported the candidate and
passed; it is preserved but is not a valid negative control. The initial broad
focused run also encountered missing ignored kitchen assets; its setup-failure
receipt is retained, and the 197-pass log is the subsequent run after verifying
those assets. A combined full suite and fresh-scene acceptance remain separate.

A single live control then executed the candidate probe twice in a new private
namespace on the preserved NV06 scene. It did not replace or modify the bridge's
existing truth cache. The diagnostic required a valid live physics handle at
each construction and pose read, recorded the loaded vendor file and hash,
and observed five new views and ten finite physics-pose reads. It traversed
1,131 prims, with five dynamic props. Both constructor options were explicitly
false for all five views.

| Observed phase | Wall time |
| --- | ---: |
| Separate stage inventory | 2.984 ms |
| First instrumented probe, five new views and five pose reads | 24.664 ms |
| Second instrumented probe, same private cache and five pose reads | 5.001 ms |
| Entire server control, including its checks | 34.561 ms |
| Client RPC, including queueing and transport | 52.894 ms |

The first constructor took 15.828 ms; the other four took 1.019–1.112 ms.
The diagnostic saw no USD object-change notices and no changes in the recorded
transform operations, sleep-threshold attributes, property names or applied APIs.
The original cache and its five view identities were unchanged. The measured
robot and props remained within the control's 1 mrad / 1 mm comparison bounds;
owner, listeners, epoch, protected processes and 259 source/model hashes passed
the final checks. The control sent no actuator or mapper commands. Its one
experimental `exec` allowed five seconds; state reads remained at one second,
and no native mapper, motion or capture-barrier deadline changed.

This was an already running session with existing imports and prior truth-view
initialization. It measures new private views in that warmed context, not a
full cold start or the production probe's total latency. It did not undo any
previously authored scene settings. The observed no-change result is bounded to
that control and the attributes/notices it recorded.

There is also an unchanged SDK boundary: `RigidPrim.get_world_poses()` falls back
to `XFormPrim.get_world_poses(..., usd=True)` when its physics handle is invalid.
The production change does not add a handle-validity check or remove that SDK
fallback. The synthetic exception test does not exercise this internal fallback.
Only the live diagnostic explicitly rejected invalid handles, so its ten reads
are evidence of live physics poses in that interval, not a guarantee for every
future production read.

The [compact receipt](../benchmark/results/nvblox-truth-readonly-20261001.json)
links the source, positive and negative logs, independent review, full control
receipt and plan. Deployment requires the combined reviewed pin and a new scene;
the NV06 source and its original failed receipt remain unchanged.
