# reBot gripper collision hulls

The 28 hulls in `rebot_gripper_hulls.usda` are exported from the vendor
MJCF in [reBot-Isaacsim](https://github.com/johnnynunez/reBot-Isaacsim/tree/199a0fa587ef15408432c8c135212d81304de1e3/mjcf/rebot_devarm).
The source checkout at this revision includes the Seeed Studio MIT license,
reproduced in `LICENSE`.

- Input: `mjcf/rebot_devarm/rebot_devarm.xml` and its referenced mesh assets.
- XML SHA-256: `28d69120da31f6ca3bb138e6a0805e972bcc879416373e89229e98c4b2122731`.
- Conversion: `scripts/build_newton_gripper_hulls.py --menagerie <mjcf directory>`.
- Modification: exported compiled collision vertices into each USD body's
  frame, retaining 12 palm hulls and 8 hulls per finger; no visual meshes.

The original generated header incorrectly attributed this XML hash to a
MuJoCo Menagerie commit. The hash matches the vendor checkout above.
