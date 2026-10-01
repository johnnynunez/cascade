# Preserve the planned grasp order

Outcome memory already ranks model proposals by similarity to past successful
geometry, then applies its existing height correction. `select_grasp` sorted
that result again by the unchanged model quality, erasing the memory ranking.
The runtime now establishes quality order before applying memory and passes
`preserve_order=True` to the selector. Direct callers retain quality sorting by
default. Width limits, IK, geometric vetoes, cancellation, deadlines, model
qualities, candidate poses and the stored learning rules are unchanged.

The saved orange attempt on MAIN `6b5dad6` demonstrates the defect. All five
first-batch candidates remain infeasible. In the second batch, memory ranked
the viable 38.559 mm-high proposal first, but the second sort selected a
45.350 mm-high proposal. Replaying the corrected selection on the exact saved
cloud, target mask, prior and finger geometry selects the former. The observed
obstacle at pixel `[185, 688]` remains outside the target mask and still vetoes
closure where it did before. This is a planning counterfactual, not a physical
success claim. Input hashes, candidate results and validation are in the
[replay receipt](evidence/grasp-memory-ranking/native-final-main-replay.json).

The physical failure had good descent tracking (0.613 mm TCP position error),
then bilateral contacts and lateral target displacement during closure, with
no observed lift. Comparing authored finger hulls and observed target surfaces
supports shallow contact as a contributing mechanism, but does not establish
an insertion, contact-count or symmetry threshold that reliably separates all
successful grasps. This fix adds none of those heuristics.

The failed outcome was retained: the matching orange profile changed from
3 wins/2 losses to 3 wins/3 losses, and its existing height correction changed
from +3 mm to -2 mm. No memory was reset. The replay uses the historical prior;
the next physical run will use the current preserved memory and needs its own
acceptance. The green physical audit on that run passed the existing server
camera-age criterion; one reset sample reached 2.122 seconds at client delivery,
which is recorded separately from that criterion.

Validation: 109 focused tests passed, including nine ranking regressions and
the existing planning cancellation/deadline and observed-finger tests. The two
runtime ranking regressions fail against the explicitly imported unmodified
`6b5dad6` source. The combined suite with the held-object observation correction passed 3,205
tests on immutable `3914360`, merged as MAIN `477c88f`. The follow-up optional
backend test checks learned-candidate identity and quality order rather than
the obsolete list position. [Current physical acceptance](PROJECT_STATUS_20261001.md)
uses preserved memory and remains a separate result.
