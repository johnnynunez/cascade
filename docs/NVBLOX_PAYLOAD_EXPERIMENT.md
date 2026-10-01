# Experimental nvblox manipulation

This follow-up is separate from the validated CUDA backend/build change in
PR #26. It is opt-in, supports one Isaac arm, and does not change presenter
defaults. The hardware mapping measurements remain in [NVBLOX.md](NVBLOX.md).

## Problem and implementation

The first live Spark trial reconstructed the cube while it rested on the
table. After grasping, that same cube remained a world obstacle and rejected
its own carry path. Simply masking future pixels did not remove that history.
Clearing the entire map and integrating only the latest masked image instead
discarded earlier observations behind the gripper.

The experiment requires measured force and contact on both actual jaws before
classifying a prop as attached. Capture-time instance segmentation separates
that exact prop from robot pixels and neighboring objects. Its measured depth
surface becomes geometry transformed with the candidate TCP pose, and is
checked against the observed world distance field during motion approval.
This does not infer an attachment from the runtime's intended target.

On each attach/release transition, the world map is reconstructed from an
anchor and the newest captured view per camera. Exact per-prop masks exclude
the held object and its obsolete historical poses, while retaining background
and neighboring objects from those captures. No synthetic depth ray is inserted
behind an occluded object. Two retained frames per camera bound memory and
replay work. Frames older than newer confirmed contact evidence cannot undo
an attachment transition.

The nvblox query grid now samples native voxel centers, at
`(index + 0.5) * voxel_size`, and brackets the requested workspace. Sampling
voxel boundaries previously allowed float32 rounding to select a neighboring
voxel while labeling its distance with the wrong position. This correction
preserves the world registration of interpolated clearance when query bounds
or voxel size change; it does not turn an unobserved voxel into free space.

Masking also preserves the measured free part of an excluded ray. The client
first requires a probe identifying `nvblox` with `masked_depth: true`, then
sends measured depth and a binary active mask through the distinct
`integrate_masked_depth` operation. Native inactive pixels integrate free
space only before the positive TSDF truncation band. They integrate neither
the excluded surface nor geometry behind it. The same operation is used for
current captures and historical replay. Other backends and older bridges
continue to receive zero depth for excluded pixels. Rejection of a negotiated
native operation blocks clearance instead of silently dropping its mask.

The existing 30 mm clearance and intentional grasp-contact cylinder remain
unchanged. Measured carried surfaces additionally require observed ESDF
support outside that cylinder: unknown does not count as free space. Only
corners with nonzero interpolation weight determine whether a sampled
distance is known. These checks cover observed surface samples, not the full
unseen payload volume.

Scene reset invalidates the map, payload samples and replay history under the
refresh lock. Integration remains fenced until every participating camera has
supplied captures newer than its post-reset delivery floor. A reset freshness
timeout leaves the map invalidated. A subsequent `reset_scene` retry now
reacquires fresh camera geometry while stationary, before requesting home
motion or releasing an object. If capture or integration still fails, the
retry reports `stage: reset_recovery` and preserves the held object, props,
beliefs and episode memory. A rejected home motion similarly returns at
`stage: home` before releasing or resetting anything. A `map_depth` camera
option separates depth mapping from semantic belief fusion, so the proof
camera can contribute depth without changing object localization or
GraspGen-X's input.

## Reproduce the diagnostic

Install the isolated nvblox environment following [NVBLOX.md](NVBLOX.md),
and start its bridge in a separate terminal with explicit backend and voxel
size. The 1 cm setting below matches trials 08 and 09:

```bash
LD_LIBRARY_PATH="$PWD/.nvblox/cuda-13.2.2/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
.nvblox/venv/bin/python scripts/serve_occupancy_bridge.py \
  --backend nvblox --port 25558 --voxel-size 0.01
```

Trial 10 used `--voxel-size 0.005` and also failed physical acceptance.

With Isaac installed and its EULA already accepted, start the simulator through
the repository launcher in another terminal:

```bash
CASCADE_REQUIRE_CUDA=1 CASCADE_PROOF_CAMERA=1 \
CASCADE_ISAAC_PIXEL_MASK=1 CASCADE_ISAAC_CONTACT_MASK=1 \
CASCADE_ISAAC_DT=0.008333333333333333 \
.venv/bin/python scripts/isaac_launch.py --python .isaacsim/bin/python -- \
  scripts/isaac_bridge.py \
  --port 8611 --engine physx --scene-config demo/scene/kitchen_config.json
```

The wrapper sanitizes the inherited Python environment, retains both masking
variables and supplies the aarch64 `libgomp.so.1` preload needed on Spark.
Use the same checked-out sources for the simulator, mapping bridge and
application. The mapping probe must advertise `masked_depth: true` to exercise
native measured-ray masking; backend identity alone does not establish this.

Run the existing acceptance harness using the application environment, with
the real GraspGen-X service available:

```bash
.venv/bin/python benchmark/diagnostics/kitchen_acceptance.py \
  --port 8611 --engine physx --occupancy nvblox --occupancy-port 25558 \
  --pregrasp-offset 0.08 --map-cameras 3 --rounds 1 \
  --objects orange green_cube pink_cube lemon tomato_can --fail-fast \
  --output runs/nvblox-spark/physical-new
```

The harness enables payload tracking and an explicit allowlist of the five
kitchen props. It requires the nvblox CUDA backend to answer, records source
hashes, and requires a fresh error-free map alongside the existing passive
physics audit. That audit checks actual GPU contacts, destination containment,
release, settling, scene identity, reset and advancing camera images. The
skill's center-only placement message is not the acceptance oracle.

## Spark physical results, 2026-09-30

Trial 07 used three map cameras. Four of five objects passed the existing
passive physics audit and scene reset. This is one round per object, not a
reliability estimate; the orange needed two grasp attempts.

| Object | Result | Final center error | Pick / reset wall time |
| --- | --- | --- | --- |
| Orange | PASS | 6.7 mm | 116.96 / 28.75 s |
| Green cube | PASS | 12.4 mm | 92.44 / 27.90 s |
| Pink cube | PASS | 6.6 mm | 75.01 / 28.03 s |
| Lemon | PASS | 13.4 mm | 72.23 / 27.99 s |
| Tomato can | FAIL, carry rejected | Not placed | 45.38 / 10.46 s |

The can was grasped and lifted, but a carried-surface point at
`[0.2465, 0.1987, 0.1179]` m had unobserved clearance. The recorded query had
460 unobserved samples out of 2900; the minimum among observed samples was
72.22 mm. The map was fresh and error-free. The home motion during reset was
also rejected while holding the can, and no prop reset was reported. This is
a coverage limitation, not evidence of a known obstacle at that point.

The following targeted can trials retained three map cameras:

| Trial | Change under test | Voxel size | Result | Unobserved carried-surface samples | Recorded rejected point (m) |
| --- | --- | --- | --- | --- | --- |
| 08 | Native voxel-center alignment; excluded depth still zeroed | 10 mm | FAIL, pre-carry lift rejected | 263 / 2694 | `[0.2621, 0.1909, 0.1250]` |
| 09 | Center alignment plus native inactive measured rays | 10 mm | FAIL, pre-carry lift rejected | 484 / 2967 | `[0.2416, 0.2155, 0.1045]` |
| 10 | Same native masking with finer voxels | 5 mm | FAIL, placement rejected after two attempts | 211 / 2907 | `[0.2431, 0.2030, 0.0925]` |

Trials 09 and 10 also rejected home during reset and correctly preserved the
held object. Trial 10's reset query had 220 unobserved samples out of 2920,
with a rejected point at `[0.2419, 0.2018, 0.0921]` m. No props were reset and
no beliefs were forgotten. This verifies that a failed home no longer clears
held state; it does not establish successful recovery or placement. The
physical implementation is recorded by commit `63dee51`. Different captures
and stochastic grasp samples mean these live sample counts are not a
controlled comparison of masking methods or voxel sizes. The can has not
passed this physical acceptance test at either resolution.

Earlier trial 06, with two map cameras, passed the green cube at 2.7 mm and
failed the orange on unobserved payload clearance. Trial 07 also included
watermark/reset fixes and a new stochastic grasp sample, so it does not isolate
the causal effect of the third camera. Trials 01–05 document the carried-object
world ghost, lost map history, socket race and initial unknown-check ordering.
All ten trials, including failures, are preserved in the
[compact receipts](../benchmark/results/nvblox-spark-payload-experiment-20260930.json),
with original campaign and receipt hashes and tested application source
hashes. The original seven trial records are unchanged. These application
hashes do not independently attest every external service file; bridge
probes retain the reported backend, voxel size and masking capability.

## RTX physical continuation, 2026-10-01

The complete-surface replay below motivated a higher initial lift. Trials
11–13 used a separate RTX simulator on port 8691, native masked nvblox at
5 mm, GPU PhysX at 120 Hz and a separate real GraspGen-X CUDA service. They
used a 100 mm initial approach/lift. The 30 mm clearance, unknown-space rule,
workspace bounds, contact cylinder and physical acceptance criteria were
unchanged. All three campaigns failed overall; the
[compact receipts](../benchmark/results/nvblox-rtx-payload-experiment-20261001.json)
retain the original source, campaign, receipt and witness hashes.

| Trial | Object | Requested support gap | Outcome |
| --- | --- | --- | --- |
| 11 | Tomato can | 15 mm | Carry reached the destination. Descent stopped at 29.63 mm observed clearance; 0 / 2,605 samples were unknown. Release and reset failed. |
| 12 | Tomato can | 40 mm | All physical placement checks passed, including bilateral support during entry, actual release, complete footprint containment and upright settling; center error 8.7 mm. Overall acceptance failed on home/reset and camera freshness. |
| 13 | Orange | 40 mm | Grasp failed after two attempts on a TCP clearance of 19 mm; no attached payload was recorded. Reset also failed. Camera checks passed. |

Trial 12's last payload query had 0 / 2,582 unknown samples and at least
49.66 mm observed clearance. The subsequent empty-gripper home trajectory
crossed the configured workspace's y = -0.30 m boundary, so both return-home
and reset stopped. The maximum recorded camera age was 3.10 s, above the
unchanged 2 s limit. Thus physical placement evidence is positive, but this
is not a passed task-and-reset acceptance. The runtime's center-only placement
postcondition also remained explicitly unverified.

Trial 13 reduced render resolution from 1280×720 to 960×540 to test camera
freshness on the shared RTX host. It passed those camera checks but stopped
at the orange, before the requested green cube, pink cube and lemon runs.
Trial 12 similarly stopped before its remaining four objects. There is no
complete five-object regression for the 100 mm / 40 mm configuration.

The support gap is a requested controller setting, not a measured release
height. The held object's support offset and TCP height ceiling determine
the effective command. The diagnostic rejects a requested gap above the
configured ceiling, records the setting and preserves the runtime's ceiling.
It does not widen the intentional-contact cylinder to lower the can farther.
The independently observed release and settling, rather than the requested
gap, establish the positive placement evidence in trial 12.

The trial-12 application command was:

```bash
CASCADE_GRASPGENX_PORT=25556 .venv/bin/python benchmark/diagnostics/kitchen_acceptance.py \
  --port 8691 --engine physx --occupancy nvblox --occupancy-port 25559 \
  --pregrasp-offset 0.10 --place-support-clearance 0.04 --map-cameras 3 \
  --rounds 1 --objects tomato_can orange green_cube pink_cube lemon --fail-fast \
  --output runs/nvblox-rtx/physical-new
```

These results do not isolate a causal benefit from either setting: the GPU,
grasp samples and, in trial 13, image resolution differ from the earlier Spark
trials. Local GraspGen-X used PyTorch 2.7.0+cu128, while nvblox retained its
separate 2.14.1+cu132 environment. Wall times came from a host running other
GPU services and are not performance comparisons. Presenter defaults remain
unchanged; robust home planning, safe failed-grasp recovery, fresh camera
delivery and repeated complete campaigns remain blockers.

## Home planning and retained recovery failures

The [follow-up receipt](../benchmark/results/nvblox-rtx-recovery-progress-20261001.json)
preserves source manifests and artifact hashes for trials 15–16, including
failures. The manifests match commits `9fb8121` and `bfffa80`, respectively.
Both campaigns requested two rounds of all five objects and stopped at the
first can case; neither is a completed five-object campaign.

Home motion now preflights the complete bounded joint route and checks each
streamed waypoint again. Planning finishes before the streaming clock starts.
Feedback that continues settling is measured and re-vetted within five seconds
and eight checks, retaining the 1 mrad drift limit and stop/watchdog checks.
A pre-close descent failure withdraws to its vetted pregrasp within the
original contact exemption. This does not establish post-close held recovery.

| Trial | Recorded result |
| --- | --- |
| 15 | Can placement physics and the subsequent explicit reset passed. Overall **FAIL**: two early camera samples were 2.903/2.998 seconds old, above the unchanged 2-second limit. The remaining samples were at most 0.599 seconds old. The return-home inside the pick also refused feedback drift during preflight; the harness now checks this separately. |
| 16 | Application perception moved to GPU 1, while Isaac, mapping and GraspGen-X stayed on GPU 0. Camera checks passed, maximum age 0.582 seconds. **FAIL**: the first lift stopped after a 500 ms mapping timeout during attachment; reset then refused unknown attached clearance. |

Trial 15's application reported a completed placement despite its nested
return-home failure. Its physical placement evidence remains valid, but that
result cannot establish a complete task. The diagnostic now requires
`return_home.ok=true` separately. The recorded feedback changed by 8.22 mrad
over approximately 0.46 seconds; bounded rebinding addresses that measured
settling rather than widening the drift tolerance.

After trial 16, a separate read-only observation retained 21 physics samples
and three RGB-D captures. Both jaws still contacted the can near its original
support surface. This is retained-state evidence, not successful recovery.
The historical harness also attempted to park after its observer closed;
unknown attached clearance rejected that attempt. Subsequent diagnostics close
their consumers and sockets without a park command. Neither a simulator
restart nor an unobserved release may be counted as recovery of this episode.

The full regression at `d0efb1b` passed 2,569 tests, with 43 skipped and three
deselected. A later independent run of home, contact recovery and camera-probe
tests passed 47 tests after feedback rebinding. These scopes precede subsequent
attachment-barrier work and do not replace live acceptance. Frozen-camera and
labelled descent-failure probes are implemented; live PASS receipts remain
required.

The post-close barrier and retained episode implementation at `72bf778` passed
2,605 tests, with 43 skipped and three deselected. The first full run had one
ownership-test failure because its subprocess imported another checkout's
editable installation; reproducing that mismatch and setting `PYTHONPATH` to
this checkout resolved it without changing production code. The new retained
recovery diagnostic at `ca4c037` separately passed 28 tests and independent review.

One explicitly rehydrated recovery of the untouched trial-16 scene then
completed the original contact withdrawal. All three producer captures had
successfully entered the map before motion. Independent physics measured a
97.549 mm can lift, continuing bilateral jaw contact and a 0.036792 rad maximum
pregrasp error, within the unchanged profile's 0.045 rad tolerance. The command
trace contains 100 actual joint targets and no jaw or prop-reset command. The
original pregrasp and contact cylinder were preserved throughout the withdrawal.

**The complete reset still failed.** Home preflight rejected unknown attached
clearance at `[0.2140, 0.2089, 0.1826]` m before sending any home target. The can
remained held at the elevated pregrasp, with no opening, prop reset or cleanup
motion. The [recovery receipt](../benchmark/results/nvblox-retained-contact-recovery-20261001.json)
retains source/input/witness hashes and both the successful withdrawal and
failed home. Software state was reconstructed explicitly from the historical
commands and a later stationary capture; this is not evidence of ordinary
automatic recovery after a process restart or a successful full scene reset.

## Replay and regression evidence

An RTX PRO 6000 Blackwell CUDA replay used identical archived idle and held
RGB-D captures for each masking variant, including capture hashes. At 1 cm,
zeroing excluded depth left two of three recorded failure positions unknown
after all three cameras. Native inactive measured rays made all three
positions observed, with clearances of 90.87–97.44 mm. The
[1 cm replay receipt](../benchmark/results/nvblox-rtx-mask-replay-1cm-20260930.json)
contains the per-camera results and capture hashes.

The separate [5 mm replay receipt](../benchmark/results/nvblox-rtx-mask-replay-5mm-20260930.json) checked four
positions from trials 07–09 against identical idle and trial-09 held captures.
Native masking observed all four after the first camera. Zeroed depth observed
all four only after the third camera; both variants ended with clearances of
92.64–100.41 mm at those positions. This isolates fixed-capture behavior and
the effect of retaining measured rays. It does not test every carried sample,
the swept motion, or physical acceptance at 5 mm. Trial 10 demonstrates that
those resolved replay positions were insufficient for a complete live task.

A follow-up [complete visible-surface replay](../benchmark/results/nvblox-rtx-payload-surface-replay-20261001.json)
queries 2,860 measured can surface cells from the same archived trial-09
captures. It bins one real point per 3 mm cell in the base frame, per camera;
the live runtime bins in captured TCP coordinates, so its sample indices and
counts differ. Both the original depth stride of two and full image resolution
are tested against identical anchor and held captures, using native masking:

| Voxel size | Depth stride | Unknown at captured pose | Unknown with +20 mm vertical translation | Unknown with +30 mm translation |
| --- | --- | --- | --- | --- |
| 10 mm | 2 | 383 / 2,860 | 43 / 2,860 | 0 / 2,860 |
| 10 mm | 1 | 383 / 2,860 | 43 / 2,860 | 0 / 2,860 |
| 5 mm | 2 | 300 / 2,860 | 0 / 2,860 | 0 / 2,860 |
| 5 mm | 1 | 290 / 2,860 | 0 / 2,860 | 0 / 2,860 |

At 5 mm, the unobserved points occupy the can's lower band, from about
77.8 to 92.5 mm above the table. Full-resolution depth does not close that
gap. The +20 mm candidate has at least 80.3 mm observed clearance, while
the +30 mm candidate at 10 mm voxels has at least 87.9 mm. These translations
move only the query geometry; they add no map observations. They support
testing a higher initial lift but do not establish its reachability, swept
clearance, grasp retention or physical acceptance. The recorded surface is
still only the visible portion of the can.

To reproduce this coverage check with contact-mask captures:

```bash
# Capture once before pickup and once while the held object is stationary.
# The simulator must have CASCADE_ISAAC_CONTACT_MASK=1 for exact prop masks.
.venv/bin/python benchmark/diagnostics/capture_nvblox_scene.py \
  --port 8611 --output runs/can-anchor
.venv/bin/python benchmark/diagnostics/capture_nvblox_scene.py \
  --port 8611 --output runs/can-held
.nvblox/venv/bin/python benchmark/diagnostics/nvblox_payload_replay.py \
  --anchor runs/can-anchor --held runs/can-held --output runs/can-surface-replay.json
```

The capture tool never moves the robot. A real grasp must separate the two
captures; the replay rejects a held capture without confirmed contact. Its
receipt records input checksums, mapping source hashes, native backend identity
and every candidate's unknown count and minimum observed clearance. Original
trial-09 captures, rather than a newly captured pair, produced the table above.

The [native mask CUDA probe](../benchmark/results/nvblox-rtx-native-mask-proof-20260930.json)
uses an analytic depth image with an excluded foreground object. Native
masking observes the front query at 0.2 m; the object surface at 0.4 m and the
occluded query at 0.6 m remain unknown. The neighboring query is unchanged.
The probe retains the native unknown sentinel `100.0` alongside explicit
`observed` flags; that sentinel is not an observed 100 m clearance. All three
small evidence files are exact copies of their original JSON, with SHA256
hashes recorded in the compact receipt's `replay_evidence` field.

The focused regressions cover native voxel registration, mask negotiation and
binary shape/type validation, current and historical masking, preservation of
neighbor pixels, and rejection by a bridge lacking native masking. An actual
CUDA test confirms observed free space in front of an excluded object,
unknown cells at and behind its surface, and unchanged neighboring geometry.
The reset tests exercise the production barrier and map logic with synthetic
camera/service I/O: both direct and watched-camera retries recover before
home, while camera or integration failures preserve the pending barrier and
held state. Live Spark fault injection for a stalled camera followed by a
successful retry remains unvalidated.

The continuation's focused suite passed 70 tests, with one optional native
CUDA test deselected. Seven new regressions reject invalid depth units,
non-rigid camera calibration, mismatched capture identities and a changed
anchor/held robot. The real CUDA surface replay was rerun after that input
validation was added; its receipt matches the committed replay source hashes.

## Scope and remaining work

This is a diagnostic using a mock language model; it is not a full native-agent
`launch.sh` READY proof. The initial lift/approach offset is 80 mm, versus the
40 mm initial diagnostic; the later RTX trials used 100 mm. Therefore absolute placement errors from these runs
cannot isolate a causal accuracy benefit from nvblox or from a third camera.
No improvement in grasp or placement accuracy has been demonstrated.

The private nvblox service uses CUDA 13.2.2 and PyTorch 2.14.1+cu132 on Spark.
The separate GraspGen-X service retains its already validated
PyTorch 2.14.0+cu130 environment. The simulator uses GPU PhysX at 120 Hz;
Newton manipulation with this payload integration has not been validated.

Repeated rounds and end-to-end native-agent acceptance remain necessary before
presenter activation. The retained anchor assumes background geometry remains
valid within an episode; arbitrary manual movement of unheld props needs a
separate dynamic-map test. Captured visible surfaces are not a complete object
model. Multiple arms and real-camera attachment sensing are unsupported here.

Recovery while still holding an object also needs dedicated validation. After
any pending geometry barrier is recovered, scene reset attempts a
collision-checked move home before resetting props. If unobserved payload
clearance blocks home, it reports a failed reset; these experiments do not
bypass that check or claim recovery success. Earlier failed trials used clean
simulator restarts. Trial 16's retained-state recovery above did not restart
the simulator or reset its props.
