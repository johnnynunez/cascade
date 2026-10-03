# Placement verification after execution refusal

A retained SO-101 MuJoCo episode refused the second placement before opening
and reported `ok: false`, `holding: "blue cube"` and `home_skipped: true`.
The legacy position-only checker nevertheless confirmed placement because the
held cube was 25.3 mm from the requested XY point. That is not evidence of
release. The original physical failure and trace remain unchanged.

For `pick_and_place`, `place_at` and `place_on_object`, a failed execution or
an explicitly held object now vetoes a positive placement postcondition.
The verdict becomes `unverified`, retaining the independent channel and
measurements. An independent negative observation remains `refuted`, including
an object outside the scene. Execution reports cannot supply positive proof.
A separately reported failed return home does not erase an otherwise observed
successful placement.

`annotate_result` also derives `verified` from the current postcondition even
when execution already failed, replacing any stale positive flag. It preserves
the actor's existing failure and error. Calls with no postcondition retain their
existing behavior. No geometric threshold, release motion or admission policy
is relaxed by this change.

The [receipt](../benchmark/results/placement_refusal_20261003.json) binds the
retained trace, rounded replay fixture, source hashes and checks. The original
12-test causal replay fails nine expected positive-verdict cases and passes
three negative/success controls on the unchanged source. The corrected
production passes 199 focused tests in 19.17 seconds. A subsequent test-local
variable rename to satisfy Ruff passes all 14 new regression cases in 0.71
seconds, with identical production and fixture hashes. This exercises the real
checker and runtime trace path with recorded or controlled observations; it is
not a new physical placement trial. The remaining two-cube motion/occupied
destination failure requires its separate physical correction.
