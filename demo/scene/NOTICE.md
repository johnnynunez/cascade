# Cocina Asier kitchen and original props

The room is adapted from `Cocina_Asier_01`, a 3ds Max USD export supplied to
Asier by a friend. On 30 September 2026 the owner authorized its geometry
for this demo and distribution as a release asset. That authorization is
recorded separately from the licenses of textures, code and original props;
the supplied geometry is not represented as CC0 or covered by the code MIT
license. The source archive SHA-256 is
`d087d4c536e26a79f9eb867dfbdcad09a6fb4b03e5ed461dd20eba43a6609555`.

`cocina_asier_audit.json` inventories every input prim, material and file.
`cocina_asier_sources.json` records the source and texture hashes.
The converted release includes `geometry-audit.json`, mapping retained meshes
to their source prims and documenting every omitted mesh.

The conversion bakes millimetres into metre vertices, preserves Z up, normals,
UV coordinates and material subsets, and removes exporter helpers, empty
shapes, food, island dressing and stools. The island is fitted locally to the
calibrated countertop; the room retains its human scale. No source kitchen
physics is loaded. The demo keeps its original independent task colliders.

| Component | Provenance | Rights |
| --- | --- | --- |
| Room, cabinet, island, appliance and background furnishing geometry | Owner-supplied Cocina_Asier_01 | Owner-authorized adaptation and distribution |
| Wood044 worktop color and roughness; WoodFloor051 floor color and roughness | [ambientCG Wood044](https://ambientcg.com/view?id=Wood044), [WoodFloor051](https://ambientcg.com/view?id=WoodFloor051) | CC0-1.0 |
| Replacement paint, metal, glass and plain appliance surfaces; opaque window treatment and analytic lights | `scripts/build_cocina_asier.py`, `demo/cocina_asier.py` | Original artwork CC0-1.0; code MIT |
| Orange, platter and decorative citrus | `demo/own_kitchen_props.py` | Original artwork CC0-1.0; code MIT |
| Tomato can and lemon | `demo/scene/props/*.usda` | Original CC0-1.0; see `props/LICENSE.txt` |
| Green cube, pink cube, open box and green square | `demo/isaac_scene.py` | Original artwork CC0-1.0; code MIT |

All six supplied wood maps were byte-compared with the official ambientCG
2K JPG downloads; four color/roughness images are used. The two normal maps
are omitted. ambientCG grants CC0 for its downloadable assets under its
[official license](https://docs.ambientcg.com/license/). The named ambientCG
fruit/bread/cake families also exist under CC0, but their local image bytes
were not needed or reused: all supplied food dressing is removed.

Every other bundled bitmap is excluded. This includes all `ikea_*`, `gloss.*`,
`steel_*`, marker and perforation maps, the generated wall picture, `calav`,
`beet`, `dooob-top`, `IMG_5904` and `oliveschen_maser_color`. Their supplied
license provenance was unclear. No pixels or texture-derived data from them
are embedded in the replacements. All material networks are rebuilt, so even
inactive texture references from the source export are absent.

No Lightwheel geometry, photograph, map, material, botanical decoration or
composed film layer is included. The earlier film supplied conversion
knowledge only. Original procedural citrus uses deterministic mathematics;
its old visual references are not redistributed or sampled as textures.

The separate reBot robot retains its existing MIT notice and provenance in
`assets/REBOT_UPSTREAM_LICENSE.txt` and `assets/REBOT_PROVENANCE.md`. Model
weights and software retain their own licenses.

`own_assets.json` pins the authoring code and notices. The bundle manifest
pins the release archive and every extracted asset. Startup verifies source
and asset bytes; the live scene identity and physics proof include both.
After an intentional reviewed change, regenerate the manifests using the
maintainer options in `scripts/kitchen_assets.py` before recording a new proof.
