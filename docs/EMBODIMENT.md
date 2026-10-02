# Declared embodiment and mixed-unit observations

An optional `embodiment` section in `configs/robots/*.yaml` describes a robot's
structure independently of its drivers and policies. The same composed MCP
`list_resources` response includes this declaration and its SHA-256 digest.
Discovery remains passive. An absent section preserves the existing profiles.

The descriptor validates:

- A single fixed or floating root and a connected primary tree of named links.
  Additional joints marked `loop_closure: true` declare loop constraints. They
  do not provide a constraint solver or make that mechanism controllable.
- Fixed, revolute, continuous and prismatic joints. Movable axes are unit vectors
  in the **joint-local frame**, following the model convention; this descriptor
  contains no joint origins, poses or measured transforms. Revolute/prismatic
  coordinates have finite lower/upper bounds; continuous coordinates have none.
- Explicit position, velocity and effort units: `rad`, `rad/s`, `N*m` for angular
  coordinates; `m`, `m/s`, `N` for linear coordinates. Fixed joints have none.
- Actuated coordinates with one declared transmission per coordinate. An actuator
  may couple several joints; each finite nonzero ratio specifies joint position
  per actuator position, with dimensions such as `m/rad`. This is a declaration,
  not a runtime conversion or a calibrated motor mapping.
- Effector and sensor attachments to existing links and resources. Transmissions
  require an existing controller/writer; effectors cannot invent capabilities.
  Existing exclusive-controller ownership checks still apply. Sensor attachments
  identify their capture frame and, for joint sensors, their exact ordered joints.

All nested values are immutable and finite. Unknown fields, unresolved references,
disconnected graphs, unmarked cycles, mismatched dimensions, duplicate ownership
and incomplete actuator bindings fail before a domain opens device IO. `passive`
means **no declared actuator binding**; it is not proof of mechanical passivity.

`embodiment_sha256` hashes the normalized declaration, including attachment and
joint order. It is separate from the sensor envelope's `model_identity_sha256`,
which binds the physical producer. A description neither authenticates a robot nor
admits a controller, supplies transforms, certifies contact, or proves task success.
A floating-root physical arm is refused while its legacy static-frame control lacks
a validated dynamic-frame/shared-control path. The existing refusal of mixed
physical manipulation/locomotion remains in place.

## Joint observations

The new `joint_state` payload contains typed `joints` entries with `joint_id`,
`joint_type`, `position`, `velocity`, `position_unit`, `velocity_unit`, `effort_unit`
and optional `effort`. Missing effort remains `null`. Its `embodiment_sha256` must
match the declared structure; the sensor binding also checks exact ordered joint
identities, types and units. Values are never silently converted or clipped.
Out-of-range measurements remain visible to health/safety consumers.

The existing angular `proprioception` payload and mobile-state adapter are
unchanged. With an embodiment present, angular observations must match its joint
names/order and cannot stand in for a prismatic coordinate. The binding returns
the original immutable capture: it does not advance simulation, generate a new
epoch/sequence, reset producer age, or replace the physical model identity.
Stale readings remain subject to the existing sensor-hub gate.

## Exercised profiles

`fixed_so101_mock` composes the existing mock SO-101 arm with synthetic angular
joint observations. Its graph, joint-local axes and limits match the vendored
SO-101 URDF, including the **revolute** hinged jaw. Direct transmissions describe
model coordinates, not measured Feetech motor calibration. The synthetic sensor
values are explicit fixtures, **not feedback from the mock arm**.

`wheeled_lift_sensors` is an observation-only fixture with two continuous wheel
coordinates and a prismatic lift. It has no configured wheel/lift controller and
exposes no locomotion or lift tools. It demonstrates meter/radian observations
through the actual composed runtime and MCP without fabricating a driver.

```bash
python -m cascade.apps.demo --robot wheeled_lift_sensors --llm mock --no-view --no-serve
CASCADE_ROBOT=wheeled_lift_sensors python -m cascade.apps.mcp_server
```

Call `list_resources`, then `sensing.read_sensor` with `{"sensor_id":"joints"}`.
The response labels observations `synthetic`; the physical model identity is
unknown. Use task-specific `CASCADE_RUN_DIR`, `CASCADE_BELIEFS_PATH`,
`CASCADE_GRASP_MEMORY_PATH` and `CASCADE_ENVELOPE_PATH` for isolated diagnostics.

CPU tests exercise immutable topology/unit/resource failures, frame/joint/digest
mismatch, stale capture preservation, backward-compatible angular packets,
passive MCP discovery, runtime dispatch and actual stdio MCP reads. These checks
provide software evidence only. General multi-DOF internal joints, multiple
actuators driving one coordinate, control allocation, whole-body dynamics and
hardware admission need separate implementations and validation; no universal
robot-control claim follows from this contract.
