# Ground texture native benchmark entrypoint

`benchmark/rgbd/ground_native_bridge.py` is an opt-in benchmark entrypoint. It
preserves the existing mesh entrypoint and the failed mesh episode. It authors
the reviewed [Ground texture](RGBD_GROUND_TEXTURE.md) before the ordinary camera,
runtime-scene export and bootstrap. The parent backend owns physics; no extra
app update, motion command, reset, pose write or importer override is introduced.

The fixed `GroundTextureBoard` has world origin `(0.08, 0.01, 0)` and a 4000²
RGB8 PNG over the existing 2 m Ground. The committed lossless PNG is compared
with its deterministic generator before admission and after bootstrap. Its
SHA-256 is `2638e36175318b42f2b477f3e8539b9afb441738f4737fb4f4177ba9dab05253`.
Source admission includes the PNG, descriptor implementation, new entrypoint,
ordinary image oracle and consumer. The model identity builder rehashes each.

The material's `cascade:rgbdGroundReference` custom data stores canonical JSON
containing the exact board, board digest, PNG digest and quantized-rectangle
digest. Thus the consumed runtime-scene hash also binds the descriptor, shader
connections and explicit ST mapping. The existing physics-purpose material is
preserved. No geometry subtree is exempted from scene checks. The unchanged
native comparator requires every baseline native field, all shape/body labels,
and BAM parameters to match; failure prevents identity/cache/listener admission.
The new sources and scene must produce a new model identity. An older identity
is a comparison reference, never permission to reuse its physical admission.

The external campaign in the sibling `RGBD_GROUND_NATIVE/` directory passes
only the producer's checked Ground descriptor to the ordinary `annotate_capture`
consumer. It preserves 18 RGB-only fit corners, 17 fixed holdouts, integer-pixel
depth sampling and all existing thresholds. The oracle receives no K, extrinsic
matrix or depth. Live inputs still pass through native TCP, SensorHub and the
in-process MCP handler; fan-out must issue no additional frame RPC. This is not
an external LLM/stdio host or semantic-object detection. Capture age and read
timeout remain **2 s**, with 80 recorded solves, camera every 20, inner 180 s and
outer 240 s bounds. At least one complete fresh comparison is required for the
campaign geometry gate; partial capture/decode success is reported separately.

CPU controls cover source/PNG tampering, producer/consumer descriptor mismatch,
unchanged lifecycle ordering, rejection of an extra native shape, passive
check-only, and actual OpenUSD authoring without extra Gprims or loss of the
physics binding. Ordinary environments without OpenUSD visibly skip its test;
the existing installed OpenUSD interpreter separately exercises that control.
These checks do not establish that RTX honors the ST or texture graph. Native
rendering, planar X/Y accuracy and a new model pin remain unvalidated until a
separately reviewed episode runs and closes. Even a planar pass will not prove
general 3D calibration or independent per-AOV acquisition timestamps.
