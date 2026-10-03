# Trial12 observer latency checkpoint

Owned branch `perf/trial12-observer-zones`, base `origin/main`f736862; separate
worktree `TRIAL12_LATENCY/cascade`. The previous b31f2df kitchen acceptance,
bridge and demo/kitchen code is byte-identical at this base. Old evidence and
all foreign processes/environments remain read-only.

The user authorized completion of manipulation, assembly, trial12 and docs;
root authorized the concrete profile/fix/checkpoint. Applied
`profile-isaac-sim` for measured zones, unchanged arguments and retained
reference comparison. This is a custom passive-observer CPU-wall investigation,
not a generic standalone benchmark or GPU-kernel timing claim. Existing
profile09 already isolates exec_job.code; do not repeat the broad benchmark.

Retained640 evidence, native02: actual pick_and_place483.722s orange and438.827s
green; receipt499.684/454.270s includes independent post-action settling.
During those phases there are2982/2742 samples; request roundtrip medians
144.20/141.51ms. The post-base-snapshot→client-return portion is59.78/55.91ms
median, including remaining readback/geometry/codec/transport. These are not
CPU self-times or causal attribution. Details live in the task's external
`evidence/retained-timing-inventory.json`.

Read-only source review: five separate cached RigidPrim views, GPU articulation
readback, contacts, live scene/material/topology checks, actual GPU attestation,
and (orange) native convex representation plus compression happen each sample.
GPU attestation performs actual SDK scene/view reads; it does not run nvidia-smi
or a subprocess. Similar orange/green roundtrip means convex-only cost cannot
be assumed dominant.

The optional benchmark-only instrumenter wraps these existing call sites in
bounded nested wall zones. It preserves arguments/results/exceptions and the
original snapshot stdout. Up to1024 samples per source, four source IDs, finite
aggregate counters and one original native thread are admitted. A separate
exec request fetches counters so original witness wire limits remain unchanged.
Nine software tests pass: original read order/results/stdout, nested accounting,
argument expansion, exception propagation, bounded samples and rejected scopes.
No production optimization or threshold/cadence change has been made.

## Affine-world checkpoint

The preceding text records preparation. Native01/02 each completed500samples
with original observer source hashes matched byte-for-byte to retained trial12.
Their fine zones identified eight mesh `_world` calls (60vertices) taking22ms
of25ms scene geometry. GPU identity cost0.24ms, not the suspected bottleneck.

Implemented only `_world` vectorization in gpu_proof_audit.py, with explicit
finite/affine refusal and a live transform read on every invocation. Native03
completed700samples, source unchanged; original/candidate medianRPCgreen
79.19/55.55ms andorange85.89/62.26ms. Scene geometry dropped to2.6ms.
Twenty old/new pairs in the same native completed step had identical canonical
JSON for all fields except actual server_monotonic timestamps, including
contacts and convex mass/support. Real Gf standalone affine parity128matrices,
12,800vertices: maxdifference1.776e-15m. Forty-five targeted tests passed.

Source and compact artifacts: docs/KITCHEN_OBSERVER_LATENCY.md and
benchmark/results/kitchen_observer_affine_20261002.json. Initial preparation
selected the base observer rather than Spark wrapper; assertion caught this
before launch. All successful measurements remain source-bound separately.

The owned8741 bridge was stopped after measurement. Both recorded births are
gone, its port is closed and GPU process absent. Kit logged shutdown; the
control-group TERM yielded launcher143/unitfailed, so strict closure remains
false (no zero-exit lifecycle claim). No foreign process was touched. Full
pick/place/MCP host300s remains untested for this candidate. Do not combine
future batched-pose work with this measured geometric checkpoint. GitHub
push/PR/merge frozen while Hermes coordinates integration.

## Tensor-batch checkpoint after be89496

Observer poses/velocities now use one tensor batch; individual views remain
for convex mass/inertia. Root review correctly identified that requested
RigidPrim.paths and reshape(N,3) were insufficient. Final code binds native
prim_paths/count, remaps named rows and rejects missing/duplicate/extra
identities and nonexactNx3/Nx4 shapes. Native04 is retained as preliminary,
not promoted to final row/shape acceptance.

Native05 final:700samples/sourceunchanged,20complete same-step pairs exact
after excluding onlyserver_monotonic,15pairs across3requested permutations
with5distinct prop positions against the old individual reads. This SDK
preserves requested order; synthetic backend-reorder regression exercises
the independent remap.62targetedtestsPASS15.54s. State collection7.87/9.30ms
green/orange; snapshot20.81/25.20ms. Native05 uninstrumented medians56.72→47.56ms
green and68.05→60.21msorange; passive RTF.2705→.2808/.2516→.2738. No full-task
or300-second host inference follows from these passive results.

Secondbridge8742 closed cleanly0: signal only birth-bound final interpreter,
then shell/launcher naturally reap. BothPIDs/port/GPU absent; no fallbackstop.
Rawruns native04/05 and compactbatchreceipt retain source and SDK hashes.
Next: actual normal MCP JSON-RPC client with deadline300 for the completecall,
prioritycancel/stop, isolated profile ports/stores, real ownedGGX and standard
independentwitness. Root reviews that harness before native actuation.
Absolute GitHubpush/PR/mergefreeze remains; no foreign checkout/service changes.
