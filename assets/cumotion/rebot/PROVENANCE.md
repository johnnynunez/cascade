# reBot cuMotion candidate collision model

This XRDF adapts the existing local Cascade reBot model prepared with
`scripts/build_cumotion_xrdf.py` on the same robot meshes. Geometry comes from
cuMotion's sphere generator with a 3 mm radius buffer and growth to include its
sampled mesh surface. The original URDF is retained byte for byte:
`assets/urdf/00-arm-rs_asm-v3/urdf/00-arm-rs_asm-v3.urdf`, SHA256
`72b4a40aeb78f1229c782e7501c3be26ea093562cf960a0ca1c2e4a966b4e4e3`.
The robot asset's attribution and MIT licence remain in `assets/REBOT_PROVENANCE.md`
and `assets/REBOT_UPSTREAM_LICENSE.txt`.

The six revolute defaults were negated from the previous local-convention XRDF
to match the original asset URDF. The runtime adapter separately maps local
positions and velocities with the profile's six negative joint signs. Sphere
centres are link-local and do not change with that convention conversion.

The structural self-collision ignore pairs and base-link self-only geometry
are retained explicitly in the file. Both prismatic fingers have a static
0.026 m opening in this candidate model. It is not a measured jaw, an attached
payload model, a synchronized kitchen scene, or a complete continuous mesh
collision certificate. The normal live harness, observed-finger, retained
attachment and release checks still veto the actual execution curve. Any
physical acceptance must report its separately measured outcomes.
