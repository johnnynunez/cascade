# Real-host observer composition — 2026-10-03

Owned clone: `REALHOST_OBSERVER/cascade`, branch `perf/realhost-observer`.
Base: `a1f739ed1fe300e68791b3214ff0931f9bd6b3df`.

This composition restores the two passive observer changes used by retained
MCP30004/06 evidence and absent from real-host source 6e7d299. It applies the
original patches from be894967f2f45b14c8cb49ded6466f8a88c185bc and
b77cd23e953cbc033f0f57a9be12538c4bab90ac, plus the SDK row-order evidence update
c12c64185a3ad7da21587611629e379aabb76437. The live affine transform is read anew
for every sample; non-affine/nonfinite transforms remain rejected. Tensor rows
bind to the actual native view paths, with exact dimensions and inventory
checks; individual views remain available for convex mass/inertia observations.

The application retains the base composition's structured teardown and pending
cleanup retry behavior. No controller, task deadline, camera-age gate, geometry,
solver, launcher or proof admission is changed. This is a software composition,
not a new native validation. The retained earlier measurements belong to their
recorded sources and cannot establish performance under current shared GPU load.

CPU selection covers the added affine/batch/profile controls, kitchen acceptance,
placement/camera negatives, actual MCP cancellation/preemption, Isaac motion
cancellation and teardown receipts. Initial run: 163 passed, 25 failed in 15.30 s;
all 25 failures rejected the new clone's missing gitignored Cocina Asier release
files. The failure log and before/after source hashes are retained externally.
Eight already pinned release files (49,741,716 bytes) were copied from the
untouched real-host checkout after matching its frozen manifest, then verified
byte-for-byte. No assets were generated and no SDK was imported or launched.

The final selection and source/asset hashes are reported in the external
`REALHOST_OBSERVER/validation` receipt. No new native plan is admitted; the next
plan must explicitly bind hashes of both modified observer files and identify
the difference from source 6e7d299. GitHub publication and merges remain frozen.

Final targeted selection: **188 passed, 0 skipped, 16.12 s**. Ruff F/E9 and
`git diff --check` pass. All 1652 snapshotted source/asset/protected inputs matched
before/after. This final documentation paragraph was added after the test run;
functional code and tests remained unchanged. No native launch was performed.
