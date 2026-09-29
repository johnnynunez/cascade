# PAAI original kitchen asset notice

The kitchen, orange and decorative fruit platter are original procedural
artwork authored for this repository. Their geometry, colors, terrazzo chips,
wood grain, fruit dimples, citrus sections and lighting are generated from
mathematical shapes and deterministic seeds. No downloaded mesh, photograph,
material, texture or environment map is used by these generators.

The generated visual artwork is dedicated to the public domain under
[CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/). The generator
source code remains under the repository's MIT license.

| Scene component | Authoring source | Visual asset license |
| --- | --- | --- |
| Floor, walls, cabinets, countertop, appliances and lights | `demo/own_kitchen.py` | Original, CC0-1.0 |
| Orange, platter and decorative citrus | `demo/own_kitchen_props.py` | Original, CC0-1.0 |
| Tomato can and lemon | `demo/scene/props/tomato_can.usda`, `lemon.usda` | Original, CC0-1.0; see `LICENSE.txt` |
| Green stone cube, pink puzzle cube, open box and green square | `demo/isaac_scene.py` | Original, CC0-1.0; see `props/LICENSE.txt` |
| Surface and contact materials | The same authoring sources | Original, CC0-1.0 |
| Cameras and analytic fill lighting | `scripts/isaac_bridge.py` | Repository source; no external image assets |

The robot is a separate existing reBot asset. Its MIT notice and recorded
provenance are in `assets/REBOT_UPSTREAM_LICENSE.txt` and
`assets/REBOT_PROVENANCE.md`; conversion notes remain in
`assets/usd/RS-rebot-dev-arm/docs/`.
Perception model weights and simulation/software dependencies retain their
own licenses. This dedication does not change those licenses.

`own_assets.json` pins the source bytes used to generate the kitchen. Startup
verifies every entry. The scene identity includes both the configuration and
this verified inventory, so an artwork change requires a new proof. After
reviewing intentional source changes, maintainers can regenerate the inventory
with `python scripts/kitchen_assets.py --write-manifest`.
