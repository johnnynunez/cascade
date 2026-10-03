# Finite Factory compilation preparation

`factory_m20_precompile_v1` is an opt-in compiler preparation variant of the
mounted margin-v2 fixture. Its device and model identity are unprepared (`null`).
It does not enable a tool, reset the stop latch, change the fixture or motor
limits, advance the simulation, or establish physical readiness.

The retained first readiness attempt on source `7851e474` failed correctly:
the first cold `advance` took 35.0895 s, while the ordinary readiness deadline
remained 10 s. There were zero completed solves at that deadline and one late
solve after cancellation. The owner drained and the process exited naturally
with failure. That attempt remains failed. Its first-advance compilation log
contained 22.72875 s of Newton loads and 11.79410 s of MuJoCoWarp loads, on a
shared GPU; these are retained log costs, not an isolated performance estimate.

The new preparation selects explicit module handles and block dimensions from
the constructed model. The inventory contains 11 Newton variants, eight static
MuJoCoWarp variants and 20 cached kernel factories. It uses Warp's public
`load_module` with a concrete module and device, `recursive=False` and serial
loading. It never invokes a solve, forward pass, collision pass, graph capture,
kernel launch, or global module scan. MuJoCoWarp factories run inside the
original Newton solver's compiler-option/determinism scope. A returned loader
call must have retained the executable for the exact device/block and hash.

This is deliberately restricted to the reviewed SDK sources and recipe:
Newton 1.6 / MuJoCoWarp 3.12 / Warp 1.17, one world, nv19/nv_pad20, dense
Newton/elliptic/implicitfast, no sleeping/flex/plugins/tendons or custom
callbacks, the existing solver flags and iterations, and the existing collision
route. The mass-tile branches are six- and thirteen-DOF blocks, with CSR width90.
Other branches fail before factory construction/loading. The reviewed source
hashes are shipped in `factory_precompile_pins.json`; no SDK file is patched.

Before and after loading, the helper hashes all exposed arrays in both Newton
states, controls, contacts/collision scratch, model, solver mappings, MuJoCoWarp
model/data and CPU MuJoCo model/data. It also observes the scene’s authoring,
IK model/data, epoch, operation flags and effective class constants. Only the
prior preparation receipt is excluded from the scene fields. Nested dataclasses, Warp structs, texture
bytes and NanoVDB volume bytes are included. Device/compiler infrastructure is
excluded from the physical traversal. Unknown representations are rejected.
Raw byte equality includes dtype and shape; uninitialized scratch is compared
without pretending it is a finite solved observation. Scene/solver/native
clocks must remain zero. The existing model fingerprint must remain identical.
Any compiler failure, missing executable, changed array/model or unreadable
after-snapshot is a failure. `precompile.json` retains partial compilation and
audit errors; receipt persistence failure cannot return a constructed model.

The model document binds the selected preparation recipe, helper/pin-file
sources, actual compiler options, module hashes, device architecture and block
dimensions. Timing and raw scratch bytes stay in evidence, outside the
reproducible identity. A new independent model preparation/pin is required;
the old `7b64…` identity is not transferred. The unchanged ordinary owner and
readiness still decide whether a fresh physical episode is admissible.

CPU controls cover branch refusal, exact launch selection, silent loader
failure, compilation exceptions, immutable-model changes, mutations in every
physical root, audit failure and passive profile wiring. A separate real SDK
CPU control traverses 1,627 arrays in a minimal model, constructs the 20
factories, and loads one generated CPU kernel without executing it. All
physical snapshots are identical. Twenty-seven of 28 MJWarp module hash prefixes (the seven hex digits retained
by the native log) match the retained native variants directly. The minimal CPU model has CSR
width97; a separate pure factory call with the retained width90 matches the
remaining prefix. This does not compare full native module hashes or prove
identical geometry; the new receipt records full module hashes for later checks.

No new Factory CUDA preparation or readiness has been executed. Complete
snapshot compatibility with its native allocations, all 39 CUDA executables,
new model identity, cold-cache timings, and the subsequent 10 s readiness remain
unvalidated. Later contact branches may still incur compilation; this inventory
does not promise a hard real-time bound or successful fastening. See the
[CPU receipt](../benchmark/results/factory_precompile_cpu_20261003.json).
