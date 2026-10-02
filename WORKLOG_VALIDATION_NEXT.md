# Native benchmark validation checkpoint

Base: origin/main `563b615fd255cc7ff4bd934cbbf3d496672accef` (PR69).
Owned worktree: `VALIDATION_NEXT/cascade`, branch
`feat/native-benchmark-validation`. No other worktree or shared learned store
is modified.

Scope: execute the pinned real VAB framework through CASCADE's adapter and
ordinary safety-gated manipulation skills; retain an independent observation
of the same episode. A native interface preflight is not a pick/place score
or deployment admission. Arena remains separate and needs its declared Isaac
stack. GPU resources are not used by this CPU/offscreen increment.

Preparation: private Python 3.11 environment, robosuite 1.4.0 (upstream pin),
MuJoCo 2.3.7 (explicit compatibility recipe), NumPy 1.26.4. OSMesa/LLVM packages
are unpacked under the task directory; no system package or shared venv edits.
The actual renderer reports `llvmpipe (LLVM 20.1.8, 256 bits)`.

Completed: full upstream checkout is clean; native movement `runs/native07`
passes two independently measured relative moves (42 control / 1,050 solver
steps, 2.10 physical seconds). `runs/native08-cancel` passes cancellation after
8 controls / 200 solves, .400 seconds; next command denied with unchanged
qpos/time. Runtime/environment close return successfully; both source
inventories remain identical. Upstream task success remains false, and no
placement or physical braking is claimed. All evidence is indexed in
`docs/evidence/robot-modularity/vab-native-preflight-20261002.json`.

Tests: `test_vab_native_preflight.py`, `test_vab_adapter.py`,
`test_arena_adapters.py`: 60 passed in .38 s using the private Python 3.11
environment and normal conftest. An initial test invocation failed to create
the missing basetemp parent; the next exposed the MuJoCo 2.3 XML requirement
for explicit `limited=true`; both logs remain retained. A final corrected
run passed. Compile and `git diff --check` pass; Ruff is not installed in the
available read-only review environment.

Native developmental failures retained: missing upstream termcolor/easydict;
camera utility h5py dependency; attempting to serialize a live simulator
handle; IK rejection before motion due to scratch Jacobian prerequisites.
No shared dependency environment, GPU, hardware, upstream source or learned
store was modified. Arena execution and full VAB task evaluation remain open.

Parent reviewed runner/tests/docs: scratch/physics separation, independent postconditions, clocks and scope accepted. Ready as a logical validated checkpoint; no push or PR requested.
