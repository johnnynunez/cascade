# MicroDuck locomotion composition — 3 October 2026

## Publication boundary

The publication branch is based on
`7e15cfca6201051d69d698c747c2fabb543a2799` (conversation and passive generalized
joint observations). It imports only the 14 locomotion checkpoints and the
composition tests/documentation. It does **not** depend on the later RGB-D
pipeline or other unpublished aggregate features. All six modified production
files and two added modules are byte-identical to the reviewed locomotion
checkpoint `15382f3`.
The newer conversation, generation and shutdown implementations from this base
remain unchanged. Conversation-specific distance profiles/tools are not added.

The aggregate validation described below is a separate retained checkpoint,
`499d50af29c76e2d90d782f72751cb65878a841b`, and is not a test result for this
publication branch. Its paths/hashes remain intact. Both source combinations
require their own newly prepared native model identity; neither receives an old
physical admission pin.

The [publication receipt](evidence/robot-modularity/microduck-locomotion-publication.json)
binds this branch separately: 1,279 CPU checks passed in 117.47 seconds, with five
OpenUSD checks skipped by the general interpreter. The pinned optional CPU
environment passed 112 checks in 27.67 seconds, including those five cases and
the existing BAM tests. Deduplicating the JUnit case identities gives 1,382
distinct passing cases, no remaining failures or skips. This is an affected-file
selection, not a full-suite result. Its 1,602 input files and protected stores
were unchanged during the main run; the optional run also preserved source and
stores. Ruff F/E9 and syntax checks passed. An initial invocation named a
nonexistent test file and collected no tests; the invocation and correction are
retained. No production or test semantics changed to obtain these results.

## Merge onto current main

The publication branch subsequently merged GitHub main
`075c08c32eec235e3c06e8c7ff8e27c3b74174fa` through merge commit
`d9aa5a6474dc9d86a0f493935be52e89cfee7b59`, without conflicts or source
changes to the locomotion implementation. The previous publication freeze and
all 44 referenced artifacts were retained. The
[main-composition receipt](evidence/robot-modularity/microduck-locomotion-main-composition.json)
records 332 passing CPU checks in 40.25 seconds across runtime/MCP, distance,
sensing/generalized observations and the newly merged kitchen observer code.
All 1,612 source inputs and protected stores remained unchanged. Factory was
not yet present on this main revision and is not covered by this selection.
No native episode or physical admission was added.

## Retained aggregate composition

The aggregate at `9b97b65486e4b48d3e7e7a5fe5bb3970ac5c30e8` lacked
`SafeBase.walk_distance`, its mobile tool registration and the distance
profiles. This increment imports the reviewed controller, independent verifier
and native recipe dependencies from the final local locomotion checkpoint
`15382f3e9d29b32197f450035f15f3902927e7aa`. The
[composition receipt](evidence/robot-modularity/microduck-locomotion-composition.json)
records each original/cherry-picked commit, source hashes and CPU checks.

The four mobile controller/verifier/runtime files are byte-identical to that
reviewed checkpoint. The two native producer files also retain the aggregate's
optional RGB-D capture and calibration path. Existing signal fences, canonical
identity builder, transport, task admission, teardown, sensor hub and conversation
implementations remain unchanged. Only append conflicts in `WORKLOG.md` required
manual resolution; no production code conflict was resolved by replacing a file.

## Contract and identity

`walk_distance` remains opt-in. Its signed measured target is bounded to 0.1 m,
with a 5 mm controller tolerance and the existing physical duration, support,
posture, lateral/heading, freshness, cancellation and independent rest gates.
The first completed post-ACK sample starts measurement. Commanded velocity or
elapsed time cannot replace independently measured displacement. `turn` retains
its existing separate angular and translation-path criteria.

The original profiles retain their defaults. The imported slow validation
profile is an explicit alternative with the previously reviewed host budgets;
it does not alter the three-second physical command cap. Both physical distance
profiles have **null model and support pins** and remain pending admission.
Construction refuses these missing bindings before opening transport.

The optional rough policy pin, official inference BAM recipe, exact SDK solver
graph contract and same-solve detached observation reuse arrive together with
their controls. Their selectors remain explicit. The native source inventory
includes the policy admission module, candidate manifest and solver graph module,
as well as the existing RGB-D module. This source composition therefore requires
a newly prepared native identity. No earlier model digest is assigned to it.

Historical results remain bound to their original sources: repeated forward and
reverse objectives were **30 mm**, while the 100 mm ceiling is a software bound.
The separate negative turn and failed larger-turn/transition cases remain in the
[distance report](MICRODUCK_DISTANCE_CANDIDATE.md). They do not establish long
walking, continuous chained commands or physical acceptance of this composition.
Conversation-specific changes from `d09489b` are not part of this increment.

## CPU validation

- Focused controller/verifier/native-contract selection: 243 passed, 96 optional
  BAM cases skipped in the general interpreter.
- Mobile runtime/MCP, temporal and support checks, identity/RGB-D, composition
  and teardown: 1,061 passed, six OpenUSD cases skipped in that interpreter.
- Existing optional BAM suite under Newton 1.6.0, MuJoCo/MJW 3.12.0 and Warp
  1.17.0: 96 passed initially; two subprocess imports failed because the private
  parent-only PyYAML preload was not inherited. A private directory exposing
  only the existing YAML package resolved those two cases. The same short run
  passed all six previously skipped OpenUSD checks: eight passed.

Across those explicitly recorded runs, the final outcome map contains **1,406
distinct passing cases** and no remaining skips or failures. This is a combined
selection, not a full-suite claim. The earlier failures remain in the receipt,
including two corrected expectations in the new integration test. No production
limits were changed to address them. Source inventories and protected stores
remained unchanged during each run; Ruff F/E9 and syntax checks passed.

The new integration checks traverse real MCP stdio and composed runtime dispatch:
an opted-in kinematic fixture can complete execution, but its postcondition stays
unverified and cannot produce task success. Over-limit requests are refused.
Optional tests use explicit CPU devices and private caches. Two existing BAM
fixtures each execute 12 CPU solver steps on a free-root/16-hinge test model;
these are numerical adapter checks, not a MicroDuck episode. No GPU, Kit, native
mobile service, real policy rollout or new physical admission was performed.
