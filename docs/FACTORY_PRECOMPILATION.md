# Finite Factory compilation preparation

`factory_m20_precompile_explicit_v2` is an opt-in compiler preparation variant
of the mounted margin-v2 fixture. Device and model identity remain `null` until
independent preparation. It does not enable a tool, reset the stop latch, change
fixture or motor limits, advance physics, or establish physical readiness.
The earlier `factory_m20_precompile_v1` profile is withdrawn; the
`factory_nv19_dense_compile_v1` selector is rejected, not reinterpreted.

The first ordinary readiness attempt on `7851e474` remains failed: its cold
first advance took 35.0895 s against the unchanged 10 s readiness deadline.
Zero solves had completed at the deadline; one late solve drained after
cancellation. The retained log attributed 22.72875 s to Newton module loads and
11.79410 s to MuJoCoWarp loads on a shared GPU. These are observed costs, not
isolated performance estimates.

A subsequent preparation on `a2c40c83` also failed, before any finite-plan
module load: the guard wrongly required `nxn`, whereas the original
`CollisionPipeline(model, reduce_contacts=True, rigid_contact_max=2048,
verify_buffers=True)` selects `BroadPhaseExplicit`. Construction took
40.6348 s; scene and native counters remained zero. Python and the scope exited
naturally with failure, all owned processes closed, and the six pre-existing
GPU processes kept their original births. Its caches and failure remain
preserved. No owner/readiness/turn/reset ran.

This correction keeps the original physics route. It checks the exact pinned
`BroadPhaseExplicit` class, the same `model.shape_contact_pairs` object, int32
N×2 values, unique in-range non-self pairs, exact candidate capacity and absent
exclusions. That implementation lives in `broad_phase_nxn.py` but launches
`_nxn_broadphase_precomputed_pairs`; the filename does not select all-pairs
physics. The compiler takes that owned kernel's module explicitly.

All 22 admission predicates are evaluated before any factory/load, including
solver flags, clocks, collision features, device/block dimensions and mass
layout. The receipt retains each value or attribute-read error, expected
condition and verdict, even when several predicates fail. Array metadata and
hashes survive rejection; small values are included. Nonfinite clocks remain
rejected and serializable. The normal preparation path persists this report
in `precompile.json` on rejection. An external constructor observer must also
retain it if construction fails before returning a model.

The finite inventory is still 39 variants: 11 Newton owned/static variants,
eight static MuJoCoWarp variants and 20 cached factories. The pinned SDK is
Newton 1.6 / MuJoCoWarp 3.12 / Warp 1.17, one world, nv19/nv_pad20, dense
Newton/elliptic/implicitfast, no sleeping/flex/plugins/tendons or callbacks,
the original flags/iterations, six- and thirteen-DOF mass tiles, and CSR
width90. Twenty reviewed source files now include the MJWarp layout builder.
Other branches are rejected before factory construction/loading.

The helper calls Warp `load_module` only for the concrete module, device and
block, with `recursive=False` and serial loading. Factories run inside the
original solver's compiler-option/determinism scope. A loader return must
retain the executable for the exact device/block and module hash. The helper
never calls a solve, forward/collision pass, graph capture or global module
scan. Ordinary scene construction remains separate and may initialize arrays
and execute builder/FK kernels.

Before and after loading it hashes all exposed physical arrays in both Newton
states, controls, contacts/collision scratch, model, mappings, MJWarp and CPU
MuJoCo model/data. Scene authoring, IK arrays, epoch, operation flags and class
constants are included; only the earlier precompile receipt is excluded from
scene fields. Nested structures, texture and volume bytes are included.
Unknown representations fail closed. Compiler infrastructure is excluded.
Raw equality includes dtype/shape, including allocated native scratch without
calling it solved evidence. All clocks remain zero and the model fingerprint
must match. Failures retain partial loads and audit errors; persistence
failure cannot return a constructed model.

The new model identity binds the v2 selector, admission report, helper/pin
sources, options, module hashes, device architecture and blocks. Wall times
and raw scratch bytes remain evidence outside reproducible identity. No old
model pin is transferred. Readiness remains 10 s, and the unchanged owner and
independent checker govern any later physical episode.

The new real SDK CPU control constructs a mesh/floor `CollisionPipeline` using
those exact source-extracted keywords and reproduces the old guard rejection.
It also converts the actual arm/socket XML plus a separate free-body topology
with `put_model`. All 39 definitions are constructed with identical physical
snapshots before/after; no solver, collision/FK call, finite module load or
physical step is run. CPU builder allocation/inertia/BVH kernels do compile
and execute. This is not the Factory SDF geometry. Its mass CSR width is75,
versus90 in retained native generated CUDA; CUDA device/block predicates also
correctly reject this CPU model. No array is padded or rewritten to pass.

The [explicit-route CPU receipt](../benchmark/results/factory_precompile_explicit_cpu_20261003.json)
links the complete 22-predicate source audit, retained failed native attempt,
CPU branch control and 466 passing tests (four optional skips). Missing
attributes, malformed/foreign pairs, nonfinite clocks and multiple simultaneous
rejections have direct controls. The
[earlier CPU receipt](../benchmark/results/factory_precompile_cpu_20261003.json)
remains historical evidence for the first inventory and state-fencing work.

The first v2 CUDA preparation completed all 39 finite module loads and passed
all 22 recipe predicates, but **failed** the unchanged-physical-state check.
Nine CPU MuJoCo arena descriptor hashes differed; the model fingerprint stayed
identical, all native/scene steps and clocks remained zero, no owner/readiness
started, and the process closed naturally with exit1. The preparation took
69.883 s in its process, including 29.404 s inside finite compilation. This
failure and its original receipts remain unchanged.

A separate CPU audit against the actual installed MuJoCo3.12 headers inspected
157 numeric pointer fields in fresh and contact-forwarded data. Several arena
pointers were NULL, while their Python getters returned new owning arrays of
nonzero shape. Those uninitialized Python allocations are not native physical
storage. Repeated getter reads (and writing the owning return allocation) left
the complete native struct, buffer and arena bytes unchanged. The forwarded
fixture used one CPU `mj_forward` for setup; neither fixture advanced physics.
This explains a mechanism consistent with the nine differing fields; it does
not reconstruct their pointer values inside the closed CUDA process.

The revised snapshot records all 64 numeric arena-pointer descriptors, including
native pointer presence/address, dtype and shape. Nonempty backed descriptors
must point exactly to native storage and still have their full bytes hashed.
NULL descriptors must be independent Python-owned allocations and are recorded
as absent native storage, with no hash of uninitialized return memory. Empty
descriptors retain their native pointer even though the Python array is empty.
Pointer allocation, removal, relocation and backed-byte mutation remain visible.
The traversal also follows nested MuJoCo contact lists, warning/solver statistics
and option structs. A CPU mutation control exposed their omission in the prior
auditor; changed contact position/friction, gravity and statistics are now
detected. Other physical arrays retain their full-byte comparison.

Native offsets come from `offsetof(mjData, field)` against the installed
headers, never a guessed ctypes structure. Before any pointer read, the finite
recipe checks the manifest digest, exact MjData type, `_structs` binary digest,
header digests, pointer width/endianness and actual mapped `libmujoco` digest.
The last check detects loader substitution despite a matching Python binding.
The layout is explicitly restricted to this reviewed Linux CPython3.12 wheel;
other builds are refused and require their own audited ABI. Ordinary profiles
and platforms do not import or execute this optional preparation path. No C
compiler runs in the runtime. The manifest is packaged and bound into the new
model source identity. This snapshot is for the owned preparation interval
before an owner starts, not for concurrent arbitrary simulation reads.

The revised snapshot has not run in CUDA. Neither preparation failure admits
readiness, physical fastening or a reusable model pin. Readiness remains10s;
later contact branches may still compile. No real-time or task-success claim
is made.
