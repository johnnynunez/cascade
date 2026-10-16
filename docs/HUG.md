# HUG — an opt-in second grasp backend

**Next action:** start the analytic protocol double (no torch, weights or MANO; seconds; the
`grasping` extra brings its zmq/msgpack wire):

```bash
uv run --extra grasping python scripts/serve_hug.py --stub
```

It only starts a stub server on `127.0.0.1:5558` (or `$CASCADE_HUG_PORT`): no grasp happens until
a client dials it, and only a profile with `grasp.hug.required: false` accepts it. The default
(`required: true`) and the `isaac_kitchen_hug` profile reject the stub (`real HUG required;
analytic protocol stub rejected`); the CPU tests below drive it.

[HUG (Human Universal Grasping)](https://grasping.io/) predicts how a **human right hand** would
grasp the object under one query pixel of one RGB-D frame. CASCADE can use it as a second learned
grasp backend next to GraspGen-X: `grasp.backend: hug`, selected only by the `isaac_kitchen_hug`
arm profile or an explicit `CASCADE_GRASP_BACKEND=hug`. Every other profile keeps its
GraspGen-X / OBB behaviour (tests pin it). Status: CPU-tested against the stub only; the real
engine has **never run** against real HUG, weights or MANO here.

| | pinned release |
|---|---|
| paper | Wu et al., *Human Universal Grasping*, arXiv:2606.17054v1 (CC BY 4.0) |
| code | `github.com/KevinyWu/hug` @ `8d1c52d4c24bfae5a369e32e3f134f5601a02630`, MIT |
| weights | HF `kevinywu/hug` @ `1415c9eaab4408cf2defdf8cb98e4d0d06b9b063`, `hug_full.safetensors`, 241,359,400 B, sha256 `515b5c3bc7987739aec019e754c15df5fbf3eff9daefb93924da098ae4bd1eae`, MIT |
| also needed | DINOv2 `facebook/dinov2-with-registers-base` (Apache-2.0, fetched by HUG on first start); the **MANO** right-hand model (licensed per user, see below) |

## What HUG provides and what CASCADE adds

HUG returns per sample a 99-D MANO vector that its `mano_params_to_grasp_dict` turns into 21
camera-frame hand landmarks plus `T_camera_wrist`. It has no server, confidence score or
parallel-jaw retargeting; those are CASCADE's:

| CASCADE addition | where | note |
|---|---|---|
| REQ/REP msgpack server shaped like GraspGen-X's | `scripts/serve_hug.py` | HUG's documented path (`prepare_pkl` → `GraspDataset.get_inference_data` → `hug.app.handle_click` body → `model.sample` → `mano_params_to_grasp_dict`); asserts HUG's `K_224` equals ours; N samples per request in one batched `model.sample` |
| `crop: query` | server | opt-in pre-crop around the query before HUG's centre crop. Default `center` = HUG's preprocessing; a query outside the crop is **refused**, never clamped |
| hand → parallel-jaw **pinch** | `grasping/hug_backend.py::pinch_from_landmarks` | **our assumption**: jaws close along thumb tip (4) → index tip (8) (`index_middle`: mean of index + middle tips); TCP = midpoint; approach = palm centre (wrist + 4 MCPs) → pinch, orthogonalised; `pinch_approach: vertical` keeps the contacts and approaches top-down |
| `Grasp.quality`, filters | `HugPlanner.plan` | **CASCADE's geometric score** `1 / (1 + lateral/1 cm + off-centre/5 cm)`, not a HUG output; filters `approach_z_max`, `max_lateral_offset_m`, `surface_tolerance_m`, counts in `hug_filtered_counts` |
| ranking | runtime | same `GraspOutcomeMemory` re-rank and z-nudge, then `select_grasp` (width ▸ IK ▸ harness pre-vet). The SafetyHarness is the sole motion authority: a HUG pinch is a proposal |

## Selection and failure contract (same as GraspGen-X)

1. **Required** (`grasp.hug.required: true`, default and the profile's setting): a missing
   server, the stub, a malformed batch or a failed request is an error at startup and per grasp,
   **never** replaced by OBB or GraspGen-X; retried on the next command.
2. **Optional** (`required: false`): a failure reports `obb (hug down)` and retries after a 5 s
   cooldown. When HUG answers, OBB candidates are appended (`learned + obb`) and the quality sort
   compares CASCADE's geometric score with OBB's, so OBB can outrank HUG: use the required
   profile for an A/B.
3. `grasp_object`'s search deadline and cancellation reach the probe and the request; transport,
   malformed or cancellation errors propagate unlatched; an all-filtered batch is
   `NoEligibleGrasps`, so the search may draw fresh (stochastic) hands.
4. `backends()` / the capability matrix name what answered: `hug (learned human hand -> cascade
   pinch, cuda:0)`, `hug-stub (analytic protocol double)` (never reported as learned) or
   `obb (hug down)`. Only HUG receives the RGB-D frame (a test pins the other backends' calls).

## Inputs and frames

1. The client sends RGB (not CASCADE's BGR); depth as uint16 **millimetres** (NaN, inf,
   non-positive and ≥ 65.535 m become 0 = invalid); `K` at the RGB resolution; and a query pixel:
   the valid-depth pixel of the 3×3-eroded mask nearest its centroid (a mug's centroid lies in its
   hole).
2. Only measured depth (`sensor` / `mono`); plane-cast depth would feed fabricated geometry.
3. HUG answers in that camera's OpenCV frame (the 224 px crop changes `K`, not the frame), so
   landmarks map to base with the same `T_base_cam` that lifted the mask points. Before any
   request the planner checks the transform back-projects the query onto the fix's own points
   (`max_query_offset_m`), refusing a secondary camera paired with the primary extrinsic.

## Install and run (operator; CUDA host, ~30 min)

HUG needs Python 3.10, torch 2.9.1 (cu128), torch-cluster, chumpy and manotorch, incompatible
with CASCADE's 3.12 venv, so it gets its own env (`.hug/`, `.hug-src/`, git-ignored):

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

# serve from the checkout with the HUG env's interpreter (CUDA required; --device cpu only
# when you mean it)
./.hug/bin/python scripts/serve_hug.py --checkpoint .hug-src/checkpoints/hug_full.safetensors --port 5558
```

1. The server exits 2 when CUDA is absent without `--device cpu` (checked before HUG or weights
   load), when MANO is missing, or when the checkpoint digest differs (`--checkpoint-sha256 ''`
   disables the pin; `health` then reports `pinned: false`).
2. `health` reports `stub`, `device`, `gpu`, `checkpoint_sha256`, `pinned`, `hug_commit` and
   `sampling_steps` (default 1, HUG's app default since `8d1c52d`).
3. `CASCADE_HUG_PORT` / `CASCADE_HUG_HOST` override `grasp.hug.port/host` (applied last by
   `load_demo_config` to every arm's view; empty = unset; malformed is refused naming the
   variable); `serve_hug.py` reads `CASCADE_HUG_PORT` when `--port` is absent. The launcher and
   `scripts/setup_agents.py` forward both to the MCP server.
4. `scripts/launch.sh` has no HUG sidecar: start the server yourself. With
   `--arm isaac_kitchen_hug` it refuses `--graspgenx local|external|none` and any non-`hug`
   `CASCADE_GRASP_BACKEND` (use the default `auto` or `stub`).
5. The Spark delivery path (`CASCADE_INSTALL_PROFILE=spark`) accepts only its delivery arms; no
   kitchen admission is claimed for HUG.

## Validation status

1. **CPU-tested against the stub, no weights** (`tests/test_hug_backend.py`,
   `tests/test_hug_runtime.py`): wire protocol and bounded requests, BGR→RGB and mm depth,
   crop arithmetic against HUG's `prepare_inputs`, pinch mapping and tool axes, ranking and
   filters, every refusal, the CUDA/MANO/sha256 policy, the real-engine wrapper through a fake
   `hug` package, config/launcher wiring, failure paths, golden backends for every profile and a
   mock `grasp_object` executing a HUG pinch through `select_grasp` and the harness.
2. **Measured on the mock SO-101 (5-DoF):** pregrasp IK fails for every stub pinch with the
   palm's ~8° tilted approach; `pinch_approach: vertical` makes them reachable (the end-to-end
   test uses it).
3. **Not claimed:** real HUG grasp quality or success rates, GPU latency, that thumb/index is the
   right parallel-jaw retargeting, any comparison with GraspGen-X.
4. **Open:** real weights + MANO on a CUDA host, then a live Isaac A/B of `isaac_kitchen_hug`
   vs `isaac_kitchen_gpu` (they differ only in `grasp.backend` and the `grasp.hug` block; a test
   pins that).
