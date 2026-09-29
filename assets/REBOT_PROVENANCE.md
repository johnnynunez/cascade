# reBot DevArm source notice

The selected robot URDF and structured USD derive from
[Seeed-Projects/reBot-Isaacsim](https://github.com/Seeed-Projects/reBot-Isaacsim).
Seeed added an explicit MIT notice for the reBot DevArm assets in
[commit 9f5a26a51d0f64899a984102cae841a2b74ca81a](https://github.com/Seeed-Projects/reBot-Isaacsim/commit/9f5a26a51d0f64899a984102cae841a2b74ca81a)
on July 30, 2026. The original notice is preserved verbatim in
[REBOT_UPSTREAM_LICENSE.txt](REBOT_UPSTREAM_LICENSE.txt).

The selected URDF, root USD layer and four subsidiary USDA files match vendor
Git objects. The URDF/root-layer revision is
`6a18beabefa9564d9b66c93fd671a8c7a4548601` (August 2, 2026).
Selected mesh and binary USD payloads include different exported versions;
this package does not claim a single byte-identical upstream snapshot for the
complete robot asset closure. Preserve the recorded hashes and existing
conversion/modification documentation with any data distribution.

The original ROS `package.xml` still states BSD and has TODO author metadata.
Those fields and existing contributor history are retained unchanged. The
later MIT notice does not resolve that upstream metadata inconsistency or
establish individual CAD authorship. Keep any separately embedded third-party
notices with their files; this notice does not relabel unrelated assets.
