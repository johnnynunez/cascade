# Replan the empty-tool escape after actual opening

Base e2d89c8 and its only original CPU two-pick remain FAIL and frozen.
The retained plan passed before opening at43.5s, but the released cube changed
pose during the normal0.6s opening. Its revalidation correctly rejected the
old joint0-first escape before sending any retreat waypoint. Detached replay
at44.1s finds a different existing joint3-last candidate with a complete home
path; that geometric result is not physical success or an episode retry.

Feature-to-skills: manipulation-ik preserves contact-only object motion,
measured joints/jaws/payload, the intended-contact escape rule and complete
retreat/home clearance. Physics-simulation preserves all solver/material/
actuator settings and independent release/support/rest gates. The optional
MuJoCo adapter owns geometry; SafetyHarness remains backend independent.

Implement a post-opening scratch replan under the already-retained fence,
same generation and cancellation state, using current measured coordinates
and the same finite candidate families and3s computation budget. Validate the
entire escape/home before its first command; reject state/binding drift or
missing opening. Preserve the old debt on every failure. No automatic home,
world stepping, object pose writes, or changed physical admission thresholds.
Tests must cover captured pre/post-release geometry, cancellation, opening
failure, snapshot drift and exact command scope. Native work needs a separate
review and authorization. Failure-journal persistence will be considered
separately and cannot turn diagnostic rows into a successful verdict.

CPU controls are retained per revision. Initial01 caught two unqualified names
after extracting the shared candidate method;02 passed10 controls. Combined03
passed77 and exposed an old transport-fault fixture that had not modeled the
new empty-tool/opening prerequisite; that fixture now explicitly sets measured
open jaws before injecting its transport failure. No production gate changed
for it. Final01 passes78 tests in14.45s, including13 new captured-state and
adversarial cases, with sources and protected stores unchanged. RuffF/E9 and
diff checks pass. No new physical episode or integration call was executed.

Review repair (freeze02): preserve the seven freeze01 files and binary diff
outside the checkout before editing. OV's three guard-only failures establish
that actuator indices, velocity addresses and jaw affine control were missing
from the binding. Bind those plus all existing joint/jaw names and addresses,
runtime jaw units, native engine/model/data/lock and the control buffer. Check
the inexpensive mapping in the retained common fence, including scopes that
were already entered; keep model hashing at planning/preflight boundaries.
No physics constants, trajectory candidates, clearance, timing or verifier
criteria change. Diagnostic journal persistence remains a separate follow-up.

The first mapping group passed41 controls. The first old-regression group
passed64 and exposed one partial `_admit_model` fixture without its `runtime`;
the fixture now supplies that existing dependency. This failure is retained.
These are CPU contract/static-geometry controls, not a new two-pick episode.
Existing withdrawal fault tests exercise their normal priority-stop path;
their suite is not described as a zero-solver-invocation measurement.

Final freeze02:106 PASS in19.19s and the exact external three-case review
probe now3 PASS in0.88s (old3 RED retained). Source and protected-store hashes
remained unchanged within both runs. RuffF/E9 and diff checks pass. Two NumPy
deprecation warnings come only from intentionally changing/restoring the
control-array shape in an adversary. No physical task episode was run.

Separate diagnostic follow-up from0b73a48: failed placement must retain the
already-produced bounded batch journal even if its independent verifier was
not reached. Copy at most256 existing rows and recorded identity/error/goal
ledger under world.lock, then serialize outside it. No SDK, FK, state read,
guard, audit or additional integration is allowed in this path. It cannot
alter the physical verdict, pending withdrawal or prefix obligations. Missing
history and persistence failure remain explicit diagnostics; neither enables
a backend or retries an action. Test copying/isolation/locking, failure-path
trace attachment, serialization faults and no lazy-backend activation.

Diagnostic final01:32 PASS in5.79s, including12 new cases and ordinary
placement/refusal/verifier regressions. The exact executor AST from0b73a48
fails the same retained-failure test at missing `placement_diagnostic` (1 RED,
0.64s); the action remains refuted in both versions. Initial journal01 kept
7 PASS/3 failures caused by an incorrect new test expectation of unverified
for an unmoved object; the checker correctly refutes it, and only that test
expectation changed. Source/protected stores remained unchanged within all
controls. Empty and already-materialized LazyArm, unverified place_at, actor
exceptions and storage failure are covered. No new robot episode or GPU run.
