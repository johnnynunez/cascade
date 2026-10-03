# Kitchen observer latency: affine vertex transforms

The passive kitchen observer now transforms mesh vertices in one NumPy matrix
operation. It still reads each live USD transform and checks topology,
materials, visibility, collision geometry and identity on every sample.
Non-finite or non-affine matrices are refused. Physics, controller commands,
observer cadence and task deadlines are unchanged.

This addresses measured observer work; it **does not close trial 12's
300-second host/MCP acceptance**. The retained full skills took approximately
484 seconds for orange and 439 seconds for green. The new measurements below
are passive, with the arm holding its initial pose.

## Evidence, 2 October 2026

Source base: `f73686216deafbcb219be7cbc214d6dfbc0c0734`. The original generated
observers exactly matched the retained trial's SHA-256 values: green
`42ed39655a14e56861c99ac100814340b4296c58d7e8e823e711b5c4f8348341`, orange
`43a0031e46cd0a1a07db43f44a9db9b26a132824590dcf551f41ce38e489972a`.
One owned PhysX CUDA kitchen bridge used the existing 640×360 recipe,
`dt=1/120`, `cam_every=6` and 0.15-second observer scheduling interval.

Coarse and fine profiling collected 500 samples each. The candidate comparison
then collected 700 samples in seven original/candidate/instrumented blocks,
100 samples per block. All blocks passed the unchanged geometry checks and
had advancing physical steps; each run's source inventory was unchanged.

| Candidate comparison, uninstrumented | Original | Affine candidate |
| --- | ---: | ---: |
| Green median snapshot RPC | 79.19 ms | 55.55 ms |
| Green sampled simulation/wall ratio | 0.2094 | 0.2729 |
| Orange median snapshot RPC | 85.89 ms | 62.26 ms |
| Orange sampled simulation/wall ratio | 0.1898 | 0.2494 |

The final original green control returned to 80.39 ms median. Nested wall
zones isolated approximately 22 ms in eight per-vertex `Gf.Vec3d`/`Transform`
loops. Vectorization reduced that zone to 0.29 ms and complete scene geometry
from approximately 25 ms to 2.6 ms. GPU identity checks cost about 0.24 ms;
they were retained. Wall zones include synchronization and exclude the outer
exec queue, compilation and transport; these are not GPU kernel measurements.
Other GPU workloads were resident, so these sequential blocks do not establish
isolated machine performance or performance during a complete manipulation.

Twenty native old/new pairs ran inside the same main-loop exec, with no
physics step between them. Every canonical JSON payload was byte-identical
after excluding only the two actual `server_monotonic` read timestamps. This
includes contacts, GPU attestation, joints, props, cameras, geometry and orange
convex mass/support data. The comparison added no rounding or numeric tolerance; the existing snapshot
still rounds open-box bounds to 1 nm as before. Numeric differences in the
compared payloads are therefore zero. A separate real Gf
USD 0.25.5 comparison over 128 arbitrary affine matrices and 12,800 vertices
had maximum absolute difference `1.7763568394002505e-15 m`.

Software verification: **45 tests passed** with normal repository conftest,
including transport/placement auditors, invalid transforms, fresh transform
reads, row-vector conventions and bounded profiling. An initial preparation
assertion caught selection of the base observer instead of its Spark wrapper;
it was corrected before launching the simulator.

The bridge PID and launcher PID are gone, the port is closed and its GPU
process is absent. Kit logged application shutdown, but the systemd
control-group TERM also terminated the shell/launcher with status 143.
The strict closure receipt remains **false**; a clean zero-exit lifecycle is
not claimed. Completed measurement receipts are separate from that outcome.

The [compact receipt](../benchmark/results/kitchen_observer_affine_20261002.json)
binds source, raw samples, generated code, scripts, zones, parity and closure
artifacts. Raw artifacts remain in the task's `TRIAL12_LATENCY/native01`,
`native02` and `native03` directories. Reproduction must use a new owned bridge
and the pinned original and candidate code; it must retain original sampling
and validation. The optional
[instrumenter](../benchmark/diagnostics/kitchen_observer_profile.py) changes
only generated diagnostic code, with bounded counters fetched separately.

## Separate tensor-batch checkpoint

The next candidate reads the bounded prop set's poses and velocities in one
tensor batch. It retains individual views for the convex mass/inertia witness.
`RigidPrim.paths` describes requested paths, so it is insufficient evidence of
row ownership: the candidate reads the native view's effective `prim_paths`
and count, rejects missing/duplicate/extra identities and maps rows by path.
Arrays must be exactly `N×3` or `N×4`; transposed arrays are rejected even if
their element count matches. Missing SDK identity metadata is an error. This
uses an experimental SDK internal view reference with explicit refusal, not
a fallback or a claim of compatibility with untested releases/backends.

Native04 retains the preliminary measurements before those identity and shape
checks were tightened. Native05 is the final candidate: **700 samples**,
unchanged source, **20 complete same-step payload pairs** identical excluding
only `server_monotonic`, and **15 native permuted-order pairs** against the
original individual-view reader. Three requested orders each included five
distinct actual prop positions. All corresponding named values were identical;
the requested and native paths are retained. This SDK preserved requested
order; a separate synthetic test deliberately permutes effective backend rows
while leaving requested paths fixed and verifies the remap.

| Native05, uninstrumented | Affine-only baseline | Affine + batch |
| --- | ---: | ---: |
| Green median snapshot RPC | 56.72 ms | 47.56 ms |
| Green sampled simulation/wall ratio | 0.2705 | 0.2808 |
| Orange median snapshot RPC | 68.05 ms | 60.21 ms |
| Orange sampled simulation/wall ratio | 0.2516 | 0.2738 |

The final affine-only green control was 60.92 ms. Instrumented batch state
collection took 7.87 ms green / 9.30 ms orange. Total snapshot bodies took
20.81 / 25.20 ms. These passive gains still do not establish a complete kitchen
task within the host budget. **62 targeted tests passed** with normal conftest.
The [batch receipt](../benchmark/results/kitchen_observer_batch_20261002.json)
retains final/preliminary source hashes, all comparisons, SDK binding sources
and limitations. Validation is for the tested PhysX recipe; Newton and hardware
were not run for this change.

This second owned bridge closed with exit 0 after SIGTERM to its verified final
interpreter PID, letting its shell and launcher reap normally. Both process
births are gone, the port is closed and its GPU process is absent. The first
run's nonzero closure remains recorded above. No shared process was signalled.
