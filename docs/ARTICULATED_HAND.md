# Fixed LEAP hand runtime

`leap_hand_right_mujoco` exposes a fixed right hand with sixteen independently
actuated joints through the ordinary robot/MCP runtime. Its two tools are
`hand.get_hand_state` and `hand.move_fingers`. Discovery reads the profile
without importing MuJoCo, constructing a model or moving a joint.

This first recipe supports bounded free finger motion. Grasp acquisition,
contact-rich manipulation, slip recovery, calibrated tactile sensing, a moving
wrist and physical hardware remain outside its admission. Every enabled contact
with positive solved normal force vetoes this recipe. Contacts disabled by the
source model are not collision evidence.

## Prepare and connect

Use MuJoCo **3.10.0** for this explicitly versioned recipe. Fetching assets does
not construct or run a simulator:

```bash
python scripts/fetch_robot_assets.py leap_hand
PYTHONPATH=src python -m cascade.apps.hand_runtime \
  --asset-root "$PWD/assets/mjcf/leap_hand" --output-dir /absolute/new/preparation
```

The asset manifest pins Menagerie's right-hand XML, ten meshes and MIT license
at commit `c1a4eeb85694ae1dffe33ff1797d4e528928a133`. Cached and downloaded assets
are checked by SHA-256. Preparation constructs the model but starts no owner
and performs zero physics steps. Its `prepared-profile.json` is the domain
profile: place those fields under `domains.hand` in a robot configuration.
The checked-in robot profile intentionally has no model identity and refuses
runtime construction until this explicit preparation is performed.

The identity includes the compiled model, derived XML, loaded MuJoCo libraries,
controller/observer sources and limits. Changing any of these requires a new
preparation. Private runtime output directories must be empty. Starting the
runtime establishes 0.2 simulated seconds of observed rest with the stop latch
set; an explicit `reset_stop` is required before a motion command.

Joint order is index, middle and ring fingers (`mcp`, `rot`, `pip`, `dip` for
each), followed by thumb (`cmc`, `axl`, `mcp`, `ipl`). `move_fingers` requires all
sixteen absolute target angles in radians. The recipe retains source geometry,
inertia, contacts and position-servo gains; it adds a 2 ms timestep and a
±0.5 Nm actuator effort cap. This is a simulation recipe, not a calibrated
Dynamixel motor model.

## Authority and verification

One owner thread alone uploads targets and advances physics. Reads consume
immutable completed-solve samples; neither polling nor verification advances
the simulation. Each sample binds the model identity, epoch, command generation,
step, joint state, commanded targets, effort and complete contact ledger.

Targets retain a 0.02 rad joint margin, change at most 0.5 rad/s and move no more
than 0.4 rad from the measured starting position. Observed speed is capped at
2 rad/s. Approach has fixed budgets of 2 simulated seconds and 10 wall seconds.
Success requires target error at most 0.02 rad and speed at most 0.05 rad/s for
0.2 simulated seconds, followed by a stop ACK and another 0.2-second retained
target/rest window. Rest has fixed budgets of 1 simulated second and 3 wall
seconds. Observations older than 0.2 seconds are rejected. The owner itself has
a 30-second wall lifetime and a 16,000-sample archive bound.

Stop revokes the command generation and holds the last uploaded position
targets. It is **not motor power-off**. A solve already in flight retains its
old generation and cannot count as post-ACK rest. Reset waits for a completed
hold in its new generation; it neither resets the world nor replays a command.
Final verification checks live authority again, so a buffered good window
cannot hide a later stop or owner fault. Synthetic tests never earn physical
task credit.

MuJoCo's contact forces and actuator effort come from dynamics evaluation before
integration. The sample records that constraint time separately from the
advanced joint-state time; it does not present them as cotemporal tactile data.
World contact normals point from geom A to geom B, and recorded world force acts
on geom B. See MuJoCo's [step API](https://mujoco.readthedocs.io/en/3.2.1/APIreference/APIfunctions.html)
and [contact fields](https://mujoco.readthedocs.io/en/stable/APIreference/APItypes.html).

Closure stops the local simulation, joins its owner and writes `solves.jsonl`
plus `closure.json`. A solved safety veto remains in the archive. A failed
startup attempts closure and preserves the original failure even if saving or
closing also fails. Thread closure alone does not certify physical rest.

## Evidence boundary

Synthetic control tests cover normal retained rest, write ownership, in-flight
stop/reset, stale and malformed observations, loaded contact, terminal target
loss, stream gaps, late verification, revocation and cleanup failures. Native
model preparation and motion evidence must be recorded separately; the source
adapter and a passing software suite alone do not establish a native grasp or
hardware capability.
