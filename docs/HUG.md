# HUG — an opt-in second grasp backend

[HUG (Human Universal Grasping)](https://grasping.io/) predicts how a **human
right hand** would grasp the object under one query pixel of one RGB-D frame.
CASCADE can use it as a second learned grasp backend next to GraspGen-X:
`grasp.backend: hug`, selected only by the `isaac_kitchen_hug` arm profile or
an explicit `CASCADE_GRASP_BACKEND=hug`. Every other profile keeps its
GraspGen-X / OBB behaviour unchanged, and tests pin that.

| | pinned release |
|---|---|
| paper | Wu et al., *Human Universal Grasping*, arXiv:2606.17054v1 (CC BY 4.0) |
| code | `github.com/KevinyWu/hug` @ `8d1c52d4c24bfae5a369e32e3f134f5601a02630`, MIT |
| weights | HF `kevinywu/hug` @ `1415c9eaab4408cf2defdf8cb98e4d0d06b9b063`, `hug_full.safetensors`, 241,359,400 B, sha256 `515b5c3bc7987739aec019e754c15df5fbf3eff9daefb93924da098ae4bd1eae`, MIT |
| also needed | DINOv2 `facebook/dinov2-with-registers-base` (Apache-2.0, fetched by HUG on first start); the **MANO** right-hand model (licensed per user, see below) |

## What HUG provides and what CASCADE adds

HUG ships inference code and a click-to-grasp app. Per sample it returns a
99-D MANO vector, which its own `mano_params_to_grasp_dict` turns into 21
camera-frame hand landmarks plus `T_camera_wrist`. It does **not** provide a
server, a confidence score or a parallel-jaw retargeting. These parts are
CASCADE's, and the code labels them as such:

| CASCADE addition | where | note |
|---|---|---|
| REQ/REP msgpack server shaped like GraspGen-X's | `scripts/serve_hug.py` | wraps HUG's documented path: `prepare_pkl` → `GraspDataset.get_inference_data` → the `hug.app.handle_click` body → `model.sample` → `mano_params_to_grasp_dict`; asserts HUG's own `K_224` equals ours |
| N samples of one query per request | server | one batched `model.sample`; the app samples one per click (the paper names many-candidates as the natural extension) |
| `crop: query` | server | opt-in; pre-crops the square containing the query before HUG's centre crop. Default `center` = HUG's own preprocessing; a query outside the crop is **refused**, never clamped |
| hand → parallel-jaw **pinch** | `grasping/hug_backend.py::pinch_from_landmarks` | **our assumption**: jaws close along thumb tip (4) → index tip (8) (`index_middle`: mean of index + middle tips); TCP = their midpoint; approach = palm centre (wrist + 4 MCPs) → pinch, orthogonalised to the jaw axis; `pinch_approach: vertical` keeps the contacts and approaches top-down |
| `Grasp.quality` | `HugPlanner.plan` | **CASCADE's geometric score**, `1 / (1 + lateral/1 cm + off-centre/5 cm)`: how far the pinch centre sits, across the approach, from the observed object and from its middle. Not a HUG output |
| filters | `HugPlanner.plan` | `approach_z_max` (same tabletop rule as GraspGen-X), `max_lateral_offset_m` (jaws would close beside the object), `surface_tolerance_m` (pinch in front of the first surface closes on air); counts recorded as `hug_filtered_counts` evidence |
| ranking | runtime | the same `GraspOutcomeMemory` re-rank and z-nudge every backend gets; then `select_grasp` (width ▸ IK ▸ harness pre-vet). The SafetyHarness remains the sole motion authority: a HUG pinch is a proposal |

## Selection and failure contract (same as GraspGen-X)

- **Required** (`grasp.hug.required: true`, the default and the profile's
  setting): a missing server, the analytic protocol stub, a malformed batch
  or a failed request is an error at startup and per grasp. It is **never**
  replaced by OBB or GraspGen-X. Required profiles retry on the next command.
- **Optional** (`required: false`): a failure is reported as
  `obb (hug down)` and retried after a 5 s cooldown. When HUG answers, OBB
  candidates are appended (`learned + obb`, as for GraspGen-X) and the
  pre-existing quality sort orders the mix. That sort compares CASCADE's
  geometric score with OBB's analytic quality, so an OBB candidate can
  outrank HUG pinches. Use the required profile for an A/B.
- Inside `grasp_object`'s bounded search, the search deadline and
  cancellation check reach the probe and the request. Transport, malformed
  or cancellation errors propagate unlatched. An all-filtered batch is
  GraspGen-X's `NoEligibleGrasps` (an empty batch), so the search may draw
  fresh hands; HUG sampling is stochastic.
- `backends()` / the capability matrix name what answered:
  `hug (learned human hand -> cascade pinch, cuda:0)`, `hug-stub (analytic
  protocol double)` (never reported as learned), or `obb (hug down)`.
- Only HUG is handed the RGB-D frame. Other backends are called exactly as
  before, and a test pins the call shape.

## Inputs and frames

The client sends what HUG's app expects:

- RGB, not CASCADE's BGR;
- depth as uint16 **millimetres**, where NaN, inf, non-positive and
  ≥ 65.535 m all become 0 (= invalid);
- `K` at the RGB resolution;
- a query pixel on the object: the pixel of the 3×3-eroded detection mask
  with valid depth that is nearest the mask centroid. HUG trains on query
  points from a 3×3-eroded mask. A mug's centroid lies in its hole.

Only measured depth (`sensor` / `mono`) is accepted. Plane-cast depth would
feed fabricated geometry.

HUG answers in the OpenCV frame of that camera. The 224 px crop changes `K`,
not the 3-D frame. Landmarks therefore map to the base frame with the same
`T_base_cam` that lifted the object's mask points. Before any request, the
planner checks that this transform back-projects the query pixel onto the
fix's own points (`max_query_offset_m`). This refuses a secondary camera's
frame paired with the primary extrinsic.

## Install and run (operator; CUDA host)

HUG needs Python 3.10, torch 2.9.1 (cu128), torch-cluster, chumpy and
manotorch. That stack is incompatible with CASCADE's Python 3.12 venv, so it
lives in its own environment (`.hug/`, `.hug-src/`, both git-ignored), like
`.graspgenx`.

```bash
# from the cascade checkout
git clone https://github.com/KevinyWu/hug .hug-src
git -C .hug-src checkout 8d1c52d4c24bfae5a369e32e3f134f5601a02630
conda env create -f .hug-src/environment.yaml -p ./.hug && conda activate ./.hug
pip install torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1 --index-url https://download.pytorch.org/whl/cu128
pip install torch-cluster -f https://data.pyg.org/whl/torch-2.9.1+cu128.html
pip install --no-build-isolation git+https://github.com/mattloper/chumpy.git@580566e
pip install -e .hug-src pyzmq msgpack-numpy

# MANO: licensed per user (registration, non-commercial research licence),
# NOT redistributable. Register at https://mano.is.tue.mpg.de/, download and
# unzip it yourself, then copy the CONTENTS of mano_v*_*/ into
# .hug-src/assets/mano_models/ (HUG reads assets/mano_models/models/MANO_RIGHT.pkl).
# CASCADE never downloads, vendors or commits MANO.

# weights, pinned to the published revision and digest
hf download kevinywu/hug hug_full.safetensors \
    --revision 1415c9eaab4408cf2defdf8cb98e4d0d06b9b063 --local-dir .hug-src/checkpoints/
echo "515b5c3bc7987739aec019e754c15df5fbf3eff9daefb93924da098ae4bd1eae  .hug-src/checkpoints/hug_full.safetensors" | sha256sum -c

# serve (CUDA required; --device cpu only when you mean it)
python scripts/serve_hug.py --checkpoint .hug-src/checkpoints/hug_full.safetensors --port 5558
```

The server refuses to start, with exit code 2, in any of these cases:

- CUDA is absent and `--device cpu` was not passed (HUG's own app would
  silently use the CPU). This check runs before HUG or the weights are
  touched.
- MANO is missing.
- The checkpoint digest differs from the pinned one. `--checkpoint-sha256 ''`
  disables the pin, and `health` then reports `pinned: false`.

`health` reports `stub`, `device`, `gpu`, `checkpoint_sha256`, `pinned`,
`hug_commit` (vs the expected one) and `sampling_steps`. The default is 1,
HUG's app default since `8d1c52d`.

`python scripts/serve_hug.py --stub` is an analytic protocol double. It needs
no torch, weights or MANO, so the whole client path runs on any machine. A
required profile rejects it.

Selecting HUG:

- `CASCADE_HUG_PORT` / `CASCADE_HUG_HOST` override `grasp.hug.port/host`.
  The launcher forwards both to the MCP server.
- `scripts/launch.sh` has no HUG sidecar; start the server yourself.
- With `--arm isaac_kitchen_hug`, the launcher refuses `--graspgenx
  local|external|none` and any non-`hug` `CASCADE_GRASP_BACKEND`, because
  those export a backend override that would replace HUG. Use the default
  `auto` or `stub`.
- The Spark delivery path (`CASCADE_INSTALL_PROFILE=spark`) accepts only its
  delivery arms. No kitchen admission is claimed for HUG.

## Validation status

- **CPU-tested against the stub, no weights:** `tests/test_hug_backend.py`
  and `tests/test_hug_runtime.py`. They cover:
  - the wire protocol and bounded requests;
  - BGR→RGB and the mm-depth conversion;
  - centre/query crop arithmetic against HUG's `prepare_inputs`;
  - pinch mapping and the tool-axis conventions;
  - geometric ranking and its filters;
  - refusals: crop, extrinsics, plane depth, malformed batches;
  - the CUDA/MANO/sha256 server policy;
  - the real-engine wrapper, driven through a fake `hug` package;
  - config and profile wiring and the launcher guard;
  - required/optional failure paths and the memory re-rank;
  - golden backends for every existing profile;
  - a mock-stack `grasp_object` that executes a HUG pinch through
    `select_grasp` and the harness.
- **Measured on the mock SO-101 (5-DoF):** pregrasp IK fails for every stub
  pinch that keeps the palm's ~8° tilted approach. `pinch_approach: vertical`
  makes them reachable, and the end-to-end test uses it. Whether real HUG
  hands clear a given arm's IK envelope is exactly what the live run must
  measure.
- **Not claimed:**
  - grasp quality or success rates of real HUG hands;
  - latency on any GPU;
  - that the thumb/index pinch is the right retargeting for a parallel jaw;
  - any comparison with GraspGen-X.

  The real-engine wrapper has never run against real HUG, real weights or
  MANO here.
- **Open (parent):** real weights + MANO on a CUDA host, then a live Isaac
  A/B: `isaac_kitchen_hug` against `isaac_kitchen_gpu`. The two profiles
  differ only in `grasp.backend` and the `grasp.hug` block, and a test pins
  that.
