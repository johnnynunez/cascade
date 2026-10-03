# Ground texture prototype — 2026-10-03

Scope authorized by the parent: CPU prototype in a new clone; no SDK edits,
solver, GPU, publication or mutation of `RGBD_XY_NATIVE/native-01` / source
`1918f6d1f903f386019b36fdea7b6dc8f81c4e01`.

The earlier native trial correctly rejected 53 extra Newton shape labels.
Purpose changes do not prevent import. A texture on the existing Ground avoids
that additional geometry, without introducing an importer override. The new
GroundTextureBoard is explicit and keeps the historical mesh default intact.

Implementation binds deterministic PNG/metric texel geometry, explicit vertex
UVs, visual shader and untouched original scene. The MCP comparison helper now
accepts the actual board descriptor. CPU controls exercise the retained capture
through the existing handler and spatial consumer; no physical authority is
created. Native wiring remains a separate reviewed step.

External evidence root: sibling directory `RGBD_GROUND_TEXTURE/` (outside this
checkout). `cpu-import-01` is an initial successful import comparison;
`cpu-import-02` additionally tests complete bindings and six mutations. Both
use CUDA-hidden CPU imports without a solver. A development C++ probe first
failed for missing headers/link dependencies; the final installed-library
probe reports OpenUSD2511 and the four expected vertices. The raw compile
logs are retained. Initial synthetic-crop controls exposed clipped markers
and aliasing; the clipped fixture was corrected and the aliased case is kept
as a rejection under unchanged thresholds.

The import probe's legacy diagnostic assignment `wp.config.enable_cuda=False`
is not a supported Warp1.17 configuration field and is not relied on as a
disable mechanism. Isolation is evidenced by `CUDA_VISIBLE_DEVICES=''`, explicit
`ScopedDevice('cpu')`, the observed device list containing only CPU, and no
finalize/solver/Kit calls. Warp's failed CUDA availability query is retained
in the log. The final receipt distinguishes that observation from a GPU run.
