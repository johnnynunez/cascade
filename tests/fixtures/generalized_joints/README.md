# Version 1 compatibility fixture

`v1-wire.json` was captured with the unmodified CASCADE source at
`5f50b8b67600123f02488b6c721e685ac051b10e` before adding version 2. It records
normalized embodiment declarations, their canonical SHA-256 digests and scalar
joint payload wire dictionaries for the shipped `fixed_so101_mock` and
`wheeled_lift_sensors` profiles. Source/configuration hashes are included.

Capture used `load_robot_config`, `EmbodimentDescriptor`, `JointStatePayload`
and `wire` directly: no runtime, driver, simulation or sensor was launched.
The values are synthetic software fixtures. Tests compare the full dictionaries
and digests, rather than recomputing a baseline with the new implementation.
