# nvblox map reset and allocation history

Merged through PR #27. Hardware/RPC measurements in this report certify their
stated clear/replay cases, not the complete manipulation campaign. Current
source and physical stages are tracked in [project status](PROJECT_STATUS_20261001.md).

Repeated map clears made a saved-frame workload exceed the existing 500 ms
mapper RPC deadline. Candidate `7bdaf5aacc581652cbb1c7704702170fe09d6bdc`
replaces the native mapper on clear, using the same voxel size and integration
parameters. It publishes the replacement only after CUDA synchronization.
Construction or synchronization failure leaves mapping unavailable until a
later explicit clear succeeds; queries cannot return the previous map.

In the inspected nvblox source (`c457c3fc01003bec6eba3ec1c61e6bf84bc3f51f`),
the full layer clear drops live blocks and resets its GPU hash view without
returning those blocks to the pool. The pool's historical allocation count
continues to influence subsequent growth. The candidate removes that allocation
history by replacing the mapper. No vendor source, installed library, sensor
mask, unknown-voxel rule, safety threshold, or RPC deadline was changed.
Three pre-existing vendor build-file changes were retained.

## Recorded checks

- The original isolated mapper replay reached its first timeout on request 28,
  the first integration after the third clear: 500.644 ms at the client and
  718.718 ms in the instrumented server. Requests and server records matched.
- A direct CUDA control compared three native-clear cycles with three fresh
  mapper cycles on the same saved views. All six final grids were bitwise
  identical, with 1,333,710 known voxels. Each map was unknown immediately
  after clearing. Native-clear first integrations took 97.743, 304.072 and
  591.764 ms; fresh-mapper first integrations took 81.446, 92.997 and
  88.564 ms. The largest full replacement took 264.995 ms.
- The uninstrumented candidate then completed all 43 RPCs across three cycles
  with the unchanged 500 ms client timeout. The slowest request took
  260.960 ms; clears took 107.672, 147.274 and 166.366 ms. Each clear produced
  an unknown map and each rebuilt grid, origin and occupied-point array matched
  the native control bit for bit. These requests used only a separately owned
  mapper on port 25562; there were no robot connections or robot commands.
- Focused tests passed: 24 offline tests and one CUDA test. The CUDA test checks
  three unknown-after-clear and masked-reintegration cycles. Offline failures
  cover construction and synchronization errors, blocked reuse, explicit retry
  and preservation of nondefault geometry and integration parameters.

The accompanying JSON receipt contains the complete-suite result, input and
library hashes, raw receipt hashes, and source revision. Independent review
checked the replay artifacts and compared all three candidate grids with the
native reference.

These are three-cycle measurements under the recorded workload, not a general
latency guarantee. The saved frames were later stationary captures from the
failed orange trial, not its missing historical pre-grasp anchor sequence.
The direct control ran native-clear cycles before replacement cycles, and the
earlier RPC profile had a different background load. This evidence demonstrates
an allocation-history latency problem and a correction for this workload; it
does not establish the sole cause of the historical withdrawal timeout or
constitute a successful physical five-object campaign.

Run the focused checks with the candidate's `src` on `PYTHONPATH`:

```sh
python -m pytest -q tests/test_nvblox_clear.py tests/test_nvblox_masking.py tests/test_nvblox_query.py -m 'not hardware'
python -m pytest -q tests/test_nvblox_clear.py -m hardware
```

The hardware check requires the installed nvblox CUDA backend. Raw saved-frame
replays and process-ownership receipts are retained with the local run archive;
they are not replaced with synthetic physical acceptance claims.
