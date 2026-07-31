# Bridge degradation under sustained verification polling

Found while chasing three false success claims in the wrc_demo ablation
(2026-07-31). Two independent defects came out of it; one is fixed in code,
the other is an operational hazard that needs a guard.

## What was observed

The ablation's `verify_retry` condition produced three episodes that claimed
success while the cube had moved 1.7-3.0 cm. All three ended at
`[0.17, 0.15, 0.04]` -- the cube's **default spawn pose**, not where the
episode had placed it.

## What it was NOT

Each hypothesis was tested and killed:

| hypothesis | test | result |
|---|---|---|
| the retry loop resets the scene | `repro_retry_reset.py` | cube jumps with **no skill running** |
| something calls `set_world_poses` | `catch_teleport.py` (hooked the setter) | **zero calls** logged |
| the cube slides on the table | `trace_slide.py` (0.25 s trace) | never moves |
| `RigidPrim` construction resets xforms | `confirm_mechanism.py` | stable, both APIs |
| the probe body walks the stage badly | `narrow_probe.py` A/B | stable |
| the arm knocks it while homing | `test_arm_knocks.py` | 6x `move_home`, **0.0 cm** |

## What it actually was

**Accumulated bridge state.** The reset only reproduced on a bridge process
that had been up for hours and had served the entire 30-episode ablation. On
a freshly started bridge:

```
verbatim truth.py probe   x15   cube stable
TruthPoseReader (ttl=0)   x15   cube stable
stress test               x400  cube stable
6x move_home                    cube untouched (0.0 cm)
full pick_and_place             SUCCESS, physics-confirmed, 0.4 cm error
```

Corroborating evidence from the degraded process:

* probe latency grew **20 ms -> ~85 ms** over 400 calls on an idle scene
* variant C (`isaacsim.core.prims.RigidPrim` per prop, the pattern
  `truth.py:90` uses) eventually **killed the bridge's TCP thread**: the
  process stayed alive while `:8611` stopped listening
* the same run produced a `[-11.8, -10.6, -122.1]` pose -- a body read
  mid-solver

`truth.py` builds a **new** `RigidPrim` view per prop on **every** probe and
never releases it. Over an hour of verification polling that is thousands of
live views against the same prims.

## Fixes applied

**1. Reject impossible poses (`sim/truth.py`)**

```python
_SANE_RADIUS_M = 5.0
def _is_sane(xyz) -> bool:   # rejects NaN, inf, |v| > 5 m
```

Insane poses are now dropped from the reading instead of cached, and
`TruthPoseReader.rejected` counts them. The checker already degrades to the
next channel on a missing pose, so silence is the correct answer -- handing a
123-metre displacement to the verifier turns a numerical fault into a
confident verdict.

**2. A single shared token is not identification (`sim/truth.py`)**

Writing the test for fix 1 exposed a second, worse bug. With `pink_cube`
dropped from the reading, `pose("pink cube")` scored `green_cube` on the
shared token `{cube}` and **returned the green cube's position**. The
postcondition would then confirm a placement against the wrong object.

Now an exact key match or **at least two** shared tokens is required.

**3. Regression tests (`tests/test_truth_channel.py`, 16 tests)**

`TruthPoseReader` had no test coverage at all despite being the only
independent verification channel in simulation.

## Still open: the operational hazard

Nothing yet stops a long-lived bridge from degrading again. Options, cheapest
first:

1. **Cache the `RigidPrim` views** in `truth.py` instead of rebuilding per
   probe — addresses the suspected leak directly.
2. **Use `isaacsim.core.experimental.prims`** rather than the deprecated
   wrapper (the experimental API was stable in every test here).
3. **Watchdog**: if probe latency exceeds a threshold, log loudly. Silent
   4x degradation is what made this take a dozen experiments to find.

Until then: **restart the bridge between benchmark runs.** Any result from a
bridge that has served hundreds of episodes should be treated as suspect.
