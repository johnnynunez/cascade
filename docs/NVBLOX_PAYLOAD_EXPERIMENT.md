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

The existing 30 mm clearance and intentional grasp-contact cylinder remain
unchanged. Measured carried surfaces additionally require observed ESDF
support outside that cylinder: unknown does not count as free space. Only
corners with nonzero interpolation weight determine whether a sampled
distance is known. These checks cover observed surface samples, not the full
unseen payload volume.

Scene reset invalidates the map, payload samples and replay history under the
refresh lock. Integration remains fenced until every participating camera has
supplied captures newer than its post-reset delivery floor. A reset freshness
timeout leaves the map invalidated. A `map_depth` camera option separates depth
mapping from semantic belief fusion, so the proof camera can contribute depth
without changing object localization or GraspGen-X's input.

## Reproduce the diagnostic

Use the isolated nvblox environment from PR #26 to run a bridge with explicit
`--backend nvblox`. Start an Isaac bridge with these additional variables:

```bash
CASCADE_REQUIRE_CUDA=1 CASCADE_PROOF_CAMERA=1 \
CASCADE_ISAAC_PIXEL_MASK=1 CASCADE_ISAAC_CONTACT_MASK=1 \
CASCADE_ISAAC_DT=0.008333333333333333 \
.isaacsim/bin/python scripts/isaac_bridge.py \
  --port 8611 --engine physx --scene-config demo/scene/kitchen_config.json
```

Run the existing acceptance harness using the application environment, with
the real GraspGen-X service available:

```bash
.venv/bin/python benchmark/diagnostics/kitchen_acceptance.py \
  --port 8611 --engine physx --occupancy nvblox --occupancy-port 25558 \
  --pregrasp-offset 0.08 --map-cameras 3 --rounds 1 \
  --objects orange green_cube pink_cube lemon tomato_can --fail-fast \
  --output runs/nvblox-spark/physical-07
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

Earlier trial 06, with two map cameras, passed the green cube at 2.7 mm and
failed the orange on unobserved payload clearance. Trial 07 also included
watermark/reset fixes and a new stochastic grasp sample, so it does not isolate
the causal effect of the third camera. Trials 01–05 document the carried-object
world ghost, lost map history, socket race and initial unknown-check ordering.
All seven trials, including failures, are preserved in the
[compact receipts](../benchmark/results/nvblox-spark-payload-experiment-20260930.json),
with original receipt hashes and tested source hashes. The application and
deployment regression passed **2288 tests**, with 46 skipped and 2 deselected.

## Scope and remaining work

This is a diagnostic using a mock language model; it is not a full native-agent
`launch.sh` READY proof. The initial lift/approach offset is 80 mm, versus the
40 mm initial diagnostic. Therefore absolute placement errors from these runs
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

Recovery while still holding an object also needs dedicated validation. The
existing scene-reset skill attempts a collision-checked move home before
resetting props. If unobserved payload clearance blocks home, it reports a
failed reset; these experiments do not bypass that check or claim recovery
success. A clean simulator restart was used between failed trials.

There is also a specific recovery gap after a camera freshness timeout:
`begin_scene_reset()` deliberately leaves a pending barrier that blocks
motion, but a later `reset_scene` currently attempts home before reopening
that barrier. It cannot recover by retrying the same skill even if the camera
returns. A recovery operation must reacquire fresh geometry before requesting
motion; this is a blocker for presenter activation, and is not implemented
in this draft.
