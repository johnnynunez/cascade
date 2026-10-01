# Detector preparation and model reuse

The open-world watcher and prompted localization share one locked detector.
Previously, switching between their modes constructed a new YOLO checkpoint;
changing a query also rebuilt its text embeddings. Localization could spend
part of an image's five-second lifetime on that preparation.

[PR #60](https://github.com/johnnynunez/cascade/pull/60), tested candidate
`ff8d58be1ae8e000bbecb19cd8db1551ac54be73`, retains at most two model instances
and eight ordered query vocabularies. Only successfully applied embeddings are
cached. Restoring the same query after an open-world pass also preserves its
predictor. Model or embedding CUDA checks still apply; failed preparation is
not admitted into the cache.

Localization prepares its actual query vocabulary under the existing detector
lock before selecting an image. This does not infer from, replace or retimestamp
an image. Detection, depth grounding and the
[analysis freshness checks](LOCALIZATION_FRESHNESS.md) still use the actual
selected frame. Slow analysis and preparation failures remain terminal, without
automatic home or retry. The normal outer request budget includes preparation.
The watcher heartbeat, physical checks and motion deadlines are unchanged.

## Software validation

The complete suite on the tested candidate passed **3,643 tests**, with
43 skipped and four deselected in 342.90 seconds. All 1,229 source and release
files remained unchanged. Independent review and 45 focused tests covered mode
restoration, ordered vocabulary keys, eviction, CUDA rejection, legacy detector
interfaces, lock serialization and image expiry. All five PR and all five merged-source CI jobs passed. It merged as `c9147db8`
with the exact tested tree.
The [retained records](evidence/detector-reuse/retained-inputs.json) bind the
software suite, CI, merge and separate detector comparison.

## Local GPU comparison

The comparison uses three archived 960 × 540 RGB images from nvblox environment
14 and identical local model files. Each version runs in a fresh process on
the second RTX Pro, with the existing unrelated GPU workload left running.
The mandatory sequence is open world → `orange fruit` → open world →
`orange fruit`, applied to every image. Three additional queries on the front
image exercise changes to `yellow object`, `green cup` and back to `orange fruit`.
This does not exercise eviction at the eight-vocabulary boundary on hardware.

Both GPU children completed within their 120-second limits and exited normally.
YOLO construction fell from four instances to two; text-encoder construction
fell from five to three across the complete sequence. Each orange query block
returned eleven detections across the three images in both versions.

| Measured preparation | Previous implementation | Candidate |
| --- | ---: | ---: |
| First orange query | 0.516 s | 0.902 s |
| Return to open world | 46.5 ms | 0.012 ms |
| Return to orange query | 294.0 ms | 0.011 ms |
| Orange after two other vocabularies | 219.9 ms | 0.172 ms |

These are individual host measurements with diagnostic instrumentation. The
fixed baseline-first order does not control the OS or driver cache. The initial
open-world inference still took about 14–15 seconds in both fresh processes.
Keeping models resident avoids repeated setup; it does not eliminate cold
startup or establish end-to-end manipulation latency. Archived frames have no
live freshness authority, and this comparison cannot attribute all of NV14's
9.5-second analysis delay to model preparation.

The original supervisor failed during its final CPU comparison because the
system Python lacked NumPy. Both GPU results were retained unchanged; their
saved outputs were compared separately with the pinned environment, then
rechecked independently by root. All **15 comparisons and 182 detections**
passed: labels, confidence scores, bounding boxes and masks were identical.
No GPU inference was repeated to repair that reporting error.

Physical proof, campaign and normal restart acceptance remain separate in
[local validation](LOCAL_RTX_VALIDATION.md).
