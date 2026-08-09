# Layer attribution on LIBERO: which layer is actually failing?

Status: measured 2026-08-09. Suite `libero_spatial`, 10 tasks x 10 seeds = 100
episodes per condition, on this machine, with `benchmark/libero/run_wrc.py`.

Every agentic-manipulation paper surveyed in `SOTA_CONTRIBUTION_ANALYSIS.md`
reports a single end-to-end success rate. None reports a no-op floor, and none
separates perception failure from orchestration failure. This is that
separation, run on the same suite those papers use.

---

## The table

| condition | perception | success | self-claimed | FALSE CLAIMS |
|---|---|---|---|---|
| no-op floor | n/a | **0/50** | n/a | n/a |
| skill_only | oracle | 0/100 | 48 | **48** |
| verified | oracle | 0/100 | 5 | **5** |
| skill_only | camera | 0/100 | 0 | 0 |
| verified | camera | 0/100 | 0 | 0 |

Artifacts: `proto_spatial_oracle.json`, `proto_spatial_camera.json`,
`noop_libero_spatial_ep5.json`. Oracle rows are stamped `perception: oracle`
and are NOT comparable to systems that perceive their own scene; ASPIRE
forbids the API they use (see `SOTA_PERCEPTION_AND_EVALUATION.md`).

## What each row says

**The no-op floor is 0/50.** A zero-action policy scores nothing on this
suite, so every number above is earned, and the success column is not being
inflated by a lax predicate. This is the control none of the surveyed papers
publish.

**Verification removes 90% of false claims (48 -> 5).** With identical
perception and identical skills, the only difference between those two rows is
whether the postcondition channel runs. That is the headline result here, and
it is a direct measurement of the failure mode LIBERO-PRO was written to
expose: a system that believes it succeeded. Bare, this orchestrator lies
about half its episodes. Verified, one in twenty.

**Camera perception never reaches the arm.** 0 claims, not 0 successes after
trying: the episode ends at
`localize 'black bowl' failed: no detections for ['black bowl', 'bowl']`.
The open-vocabulary detector emits 38-44 detections per LIBERO frame and never
names the bowl. So the camera rows measure a perception domain gap, and say
nothing at all about orchestration. Reporting them as a task score would be
the same category error the oracle rows would be in the other direction.

## Why success is 0 everywhere, stated precisely

Read from LIBERO's own predicate
(`libero/envs/object_states/base_object_states.py:78`):

```python
return ((this_object_position[2] <= other_object_position[2])
        and self.check_contact(other)
        and np.linalg.norm(this_xy - other_xy) < 0.03)
```

Three conditions: the bowl must not be above the plate's centre, they must be
in contact, and they must be within **3 cm** horizontally.

Traced on one episode, with physics settled (verified: the object stops moving
within 5 sim steps of release and does not drift over the next 160):

| quantity | value |
|---|---|
| bowl displacement | 27.2 cm (the robot really does the task) |
| placement error vs its own aim point | **6.2 cm** |
| plate shoved by the arm during place | **9.2 cm** |
| final bowl-to-plate horizontal | 7.7 cm (limit: 3 cm) |
| final bowl z vs plate z | +1.1 cm (must be <= 0) |

So with perfect perception the bottleneck is neither planning nor grounding:
it is **place precision**, and it misses by roughly 2x the tolerance. The arm
also disturbs the destination while placing, which moves the goalposts after
the belief was seeded.

That is an interpretable 0%. An undecomposed 0% would be indistinguishable
from broken perception, a broken harness, or an unreachable workspace, and in
fact this repo has already been all three (see the harness bugs fixed in
`f2d4c77`).

## Where the place error comes from, measured

The 6.2 cm above was one episode. Over 8 tasks (`scripts/measure_place_error.py`):

| quantity | median | min | max |
|---|---|---|---|
| placement error vs own aim | 6.4 cm | 5.5 | 29.8 |
| destination displaced by the arm | 0.6 cm | 0.0 | 9.2 |
| final object-to-destination | 12.8 cm | 6.6 | 36.5 |

**0 of 8 episodes land inside LIBERO's 3 cm tolerance.**

The aim error is not one distribution, it is two regimes:

- a tight cluster at **5.5 / 6.1 / 6.4 cm**, too consistent to be noise
- gross failures at **15.6 / 29.8 cm**, a different cause
- three episodes never released at all (the skill aborted mid-place)

Tracing the tight regime on task 0 by sampling the object's pose relative to
the TCP at every gripper command:

```
event      pos          TCP                    object            obj - TCP
gripper   0.50   [-0.178 0.312 0.935]   [-0.179 0.323 0.899]   [-0.001  0.011 -0.036]   <- grasp
gripper   0.00   [ 0.060 0.231 0.950]   [ 0.087 0.274 0.928]   [ 0.027  0.043 -0.022]   <- release
```

Two independent, additive defects:

1. **The object shifts inside the gripper during transport.** Horizontal
   offset from the TCP is 1.1 cm at grasp and 5.1 cm at release: it slid
   **4.3 cm** while being carried. `place_at` solves IK so the *TCP* reaches
   the target, and nothing accounts for where the object actually sits in the
   jaws, so that slip transfers straight into placement error.
2. **The TCP itself misses by 2.3 cm.** Commanded `[0.073, 0.212]`, reached
   `[0.060, 0.231]`. That is the same settle-tolerance family as the pregrasp
   bug fixed in `f2d4c77`, now visible at the place stage.

1.1 cm of grasp offset plus 4.3 cm of slip plus 2.3 cm of tracking error is
the observed ~6 cm, and it is systematic rather than random. Neither defect is
about perception or planning: with an oracle handing over exact poses, the arm
still cannot put the object where it decided to put it.

That makes place precision the concrete next target, with two separable fixes
(re-observe the held object's pose before releasing; tighten place-stage
settling) rather than a vague "improve manipulation".

## Honest limits

- One suite (`libero_spatial`), 100 episodes per condition. ASPIRE's protocol
  is 10 tasks x 50 held-out seeds with a learn/eval split; this is the
  Harness-VLA scale (10x10) and has no learn split, so it is not yet a
  like-for-like comparison to either.
- All ten tasks in this suite are the same manipulation (bowl onto plate) with
  different distractor layouts, so task diversity is low.
- The false-claim asymmetry is measured with oracle perception. Whether
  verification helps as much when perception is also wrong is untested,
  because camera mode never gets far enough to claim anything.
- Place precision was traced on a single episode. The 6.2 cm figure needs a
  distribution before it is quoted as a number.

## What this changes about the plan

The next bottleneck is now identified and it is not the one the roadmap
assumed. Before more benchmark runs:

1. **Place precision** is the single blocking defect for LIBERO success. 6.2
   cm against a 3 cm predicate. Worth a distribution over episodes and then a
   root cause, the same way the pregrasp settle bug was found.
2. **The arm disturbs the destination** during placement. A place that shoves
   its own target invalidates the belief it was aiming at.
3. Camera mode needs either pixel addressing driven by a real agent (the path
   built in `760cdf9`, currently unused because the harness calls skills
   directly with `llm=mock`) or a text-promptable segmenter. Neither is a
   detector-vocabulary problem that more tuning fixes.
