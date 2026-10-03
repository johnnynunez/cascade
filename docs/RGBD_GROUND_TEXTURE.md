# Existing-ground planar RGB-D reference (CPU prototype)

`benchmark.rgbd.ground_texture` supplies an explicit `GroundTextureBoard` and
deterministic PNG authoring for the existing `/World/Ground` plane. It avoids
adding visual meshes to the physics importer. It is not wired into the old
`native_bridge` entrypoint: that entrypoint and its failed 53-mesh episode retain
their original meaning. No RTX render, new native model identity or planar
physical acceptance is claimed here.

The texture is 4000×4000 RGB8 over the unchanged 2×2 m Z plane, at world Z=0.
Each texel spans 0.5 mm. All 53 rectangles use integer texel edges: checker
squares are 60×60 texels, marker half-width is 15 texels and marker centers are
48 texels beyond the checker bounds. The 80/10 mm origin, 30 mm squares and
42 mm background margin are exactly representable. An off-grid metric, wrong
height, incompatible reference type or out-of-plane rectangle is rejected
before bitmap allocation. The historical `Board()` still describes its raised
visual meshes at Z=0.002.

Vertex `st` is explicitly [(1,1),(0,1),(0,0),(1,0)], corresponding to the
installed plane-generator point order (+X,+Y), (−X,+Y), (−X,−Y), (+X,−Y).
The shader reads that primvar via `UsdPrimvarReader_float2` and a raw-color,
clamped `UsdUVTexture`, connected to `UsdPreviewSurface.emissiveColor`.
Diffuse color remains the schema's 0.18 default, roughness and opacity are 1;
the surface is emissive plus diffuse, not an asserted unlit material.
Filtering is selected by the renderer and remains unmeasured. PNG row zero
maps to positive world Y. No implicit UV or camera K/pose enters the metric
bitmap or RGB-only homography oracle.

`author_ground_texture(stage, png, board=GroundTextureBoard())` requires the
original static Ground geometry and physics-only material binding. It adds
four Material/Shader prims and no geometry, body or collider. Physics binding
to `/World/GroundMaterial` remains separate. `validate_ground_texture` binds
the exact PNG, descriptor, full shader/UV opinions and all original stage
opinions; changed physics, an extra mesh (including the former board path),
time samples or appearance edits reject. The underlying snapshot's new
`exclude_path=None` mode includes every prim; its historical default is
unchanged. A future native recipe must bind PNG, descriptor, shader/st, composed
scene and source hashes and establish a new model identity. This prototype
does not rename old captures or reuse their admission.

The live comparison helper accepts an explicit `board=`. CPU tests route both
the historical mesh fixture and the new texture-derived synthetic RGB-D through
ordinary in-process MCP, retained SensorHub capture and spatial annotation.
Correct projection passes; altered X/Y focal scale, stale capture and an extra
reader operation reject. The fit and metric thresholds are unchanged. An
aliased CPU resampling case is retained as a rejection; CPU image controls are
not renderer evidence.

Local validation uses two distinct USD installations: OpenUSD 0.25.5 in the
existing converter interpreter for real Newton 1.6.0/Warp 1.17.0 public import,
and the installed Kit OpenUSD 0.25.11 C++ plane-generator library for its actual
point order. CUDA devices are hidden for import; Warp reports only `cpu`.
The comparator covers all 95 builder fields whose names begin `body_`,
`joint_`, `shape_` or `articulation_`, including mesh content and appearance
fields. Both imports have 148 labels and identical compared data. Changing
friction produces a real `shape_material_mu` difference; an extra visual cube
changes labels and shape arrays. No model finalization, solver construction,
physics step, Kit application or renderer was invoked. SDK and source hashes,
import options and all negative controls are retained in the external CPU
receipts; this is import equivalence, not completed native-model equivalence.

The standard [OpenUSD plane generator](https://github.com/PixarAnimationStudios/OpenUSD/blob/v25.11/pxr/usdImaging/usdImaging/implicitSurfaceMeshUtils.cpp)
defines the referenced order. Actual RTX handling of this implicit Plane's
authored vertex primvar still needs a newly reviewed native episode with the
original full invariance guard. A material that is ignored or mapped differently
must not be repaired by fitting an undocumented world transform afterwards.
