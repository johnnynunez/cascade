# Ground inherited material guard — 2026-10-03

Small separate follow-up to frozen `f48fb851556d5bfaaf07ea7d97af032e895b4b99`.
Root identified that filtering ancestor properties omitted material-binding
relationships and strength metadata. A `/World` binding marked
`strongerThanDescendants` can therefore change the effective Ground appearance.

The benchmark now compares each ancestor's complete composed prim snapshot,
without a property filter. Child prim additions by the ordinary camera are
separate and remain permitted. Native physics/shape/BAM comparisons, parent
lifecycle ordering, shader/PNG bytes and all time/freshness limits are unchanged.

Real OpenUSD CPU causal evidence is kept outside Git in the sibling
`RGBD_GROUND_ANCESTOR/causal-{before,after}` directories. The parent applies a
stronger ancestor material with the same diffuse color (0.18), and the probe
verifies that `ComputeBoundMaterial` resolves the new material and relationship.
Both camera-time and post-bootstrap mutations escape f48 (two failing controls,
two clean controls pass). The candidate rejects both and preserves the two clean
cases. No Kit, renderer, native solver, physical episode or GPU is used.
