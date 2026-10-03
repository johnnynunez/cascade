# Passive RGB-D publication composition

This topic starts at main `075c08c32eec235e3c06e8c7ff8e27c3b74174fa` and
imports 14 reviewed RGB-D checkpoints. It adds the opt-in calibrated producer,
passive cache/TCP/sensor path, retained pixel-to-surface annotations, and separate
planar/binary/checker reference benchmarks. The unrelated MuJoCo placement-cost
commit `891d6c7` is excluded. Locomotion, Factory, conversation and actuator
changes from the former aggregate are not imported by this topic.

All 14 patches preserve their functional added/deleted lines. Only append
conflicts in WORKLOG were resolved, retaining the current log plus each selected
commit's own additions. Of 38 changed source/benchmark files, 37 are byte-exact
to the retained aggregate `9b97b65`. The composed runtime keeps this main's
lifecycle and task boundaries while adding only the sensor-to-spatial hooks.
The [publication receipt](../benchmark/results/rgbd_publication_20261003.json)
records original and picked commits, those differences, and raw artifact hashes.

The general CPU selection passed 588 cases in 45.77 seconds, with 18 OpenUSD
checks skipped. All 18 passed in 8.89 seconds under the existing OpenUSD
interpreter (USD 25.5), with only the existing OpenCV 5.0 and YAML packages loaded
by exact file path. This gives 606 distinct passing cases, not a full-suite
result. Source, protected stores and the audited USD package remained unchanged.
Ruff F/E9 and syntax checks passed for the 47 changed Python files.

Two optional-environment failures are retained: a pxr-only symlink overlay in
the general interpreter segfaulted at its first in-memory Stage creation; the
initial full-USD bootstrap then lacked the repository root on PYTHONPATH and
failed collection. The successful run used the original USD interpreter and
added the repository root to its explicit path. No production code or test
expectation changed to address either failure. No shared environment was edited.

Legacy RGB remains the default. RGB-D can omit extrinsics; spatial consumption
requires bound capture/calibration/frame provenance. Product read tokens do not
prove independently synchronized per-AOV exposure. Pixel-center conventions are
explicit and versioned. Annotations do not establish semantic truth or quantified
uncertainty. Benchmark authoring, image visibility, held-out geometry and runtime
admission remain separate judgments. Retained native successes and failures are
historical, source-bound evidence; this composition has no new native identity,
rendering result, physical admission, localization, map-free-space claim or motion
tool. Previously reported detector/geometry failures are not converted to passes.

See [observed RGB-D](OBSERVED_RGBD.md), [spatial observations](RGBD_SPATIAL_OBSERVATIONS.md),
and [checker accuracy](RGBD_CHECKER_ACCURACY.md) for each explicit contract.

## Teardown follow-up from main

The branch subsequently merged main `70d22deab7f4abb5e5e691386b4b5f952647abca`
(PR83) through `43928e6c01b438c02ebb5d417c21a40713ce2fe3`. Only the append log conflicted;
both histories were preserved. The [follow-up receipt](../benchmark/results/rgbd_teardown_composition_20261003.json)
records 103 passing runtime/MCP/teardown composition checks in 13.11 seconds,
with all 1662 inputs and protected stores unchanged. Earlier broad and
optional checks above remain historical receipts for their explicitly named
source revisions; they were not rerun or relabeled. The latest runtime shutdown
semantics are retained, and no new native/physical acceptance is claimed.
