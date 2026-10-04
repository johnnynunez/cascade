# Fixed LEAP hand runtime

`leap_hand_right_mujoco` exposes a fixed right hand with sixteen independently
actuated joints through the ordinary robot/MCP runtime. Its tools are
`hand.get_hand_state`, `hand.move_fingers` and `hand.set_hand_posture`. Discovery reads the profile
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

`hand.set_hand_posture(posture=...)` selects a fixed recipe target: `neutral`
means zero radians on all sixteen joints; `index_flex` means 0.1 rad on the
index MCP and zero elsewhere. It uses the same controller, motion budgets,
contact vetoes, stop latch and measured-rest verifier as `move_fingers`.
The conversation gateway permits this named operation only when explicitly
listed with motion enabled; raw `move_fingers` vectors remain excluded.
Changing the recipe's source requires a new model preparation and identity.
The [separate semantic validation](evidence/robot-modularity/hand-semantic-postures-20261004.json)
used source `a2bcf04` and fresh model identity `43bb8067…eaee2`. One ordinary
MCP episode completed `index_flex`, an explicit reset, and return to `neutral`.
Both postures established the original 0.2-second approach-rest window and a
further 0.2 seconds after their stop ACKs. Independent audit retained all 823
solves, zero contacts and normal owned closure with unchanged inputs. The first
posture remained within target/rest limits until the next admission; neutral
remained within them through the final closing solve. This paired sequence is
separate from the earlier raw-vector captures and does not establish semantic
repeatability or a completed physical voice chain.

Joint order is index, middle and ring fingers (`mcp`, `rot`, `pip`, `dip` for
each), followed by thumb (`cmc`, `axl`, `mcp`, `ipl`). `move_fingers` requires all
sixteen absolute target angles in radians. Recipe `leap_right_bounded_free_motion_v2`
retains source geometry, inertia, contacts and velocity gain (0.01 Nm·s/rad);
it uses a 6 Nm/rad position gain, a 2 ms timestep and a ±0.5 Nm actuator effort
cap. This is a simulation recipe, not a calibrated Dynamixel motor model.

The first native motion with source position gain 3 Nm/rad failed its unchanged
2-second approach budget: gravity left several joints 0.022–0.026 rad from their
targets, beyond the 0.02 rad tolerance, despite observed rest. All 1,164 solves
are retained, with 1,000 command-generation solves, no contacts, maximum effort
0.101305 Nm and maximum speed 0.927493 rad/s. Recorded effort reproduces the
source servo equation within 5.6e-17 Nm. Version 2 doubles only the position gain
to reduce that steady offset; it does not change any verification threshold or
deadline. Its native outcome requires separate evidence.

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
loss, stream gaps, late verification, revocation and cleanup failures.

On 2026-10-04, source `788b56620ab5425ff9764f924e4edc17f3d41973` passed
zero-step preparation and one ordinary in-process production MCP episode using
MuJoCo 3.10.0 on CPU. It commanded index MCP to 0.1 rad and the other fifteen
joints to zero. Of 490 completed solves, 227 carried the motion generation and
101 carried the post-ACK hold generation (steps 390–490, 0.2 simulated seconds).
Maximum final target error was 0.012958 rad; maximum observed effort was
0.109979 Nm and speed 0.743115 rad/s. All enabled contact ledgers were empty.
The runtime owner and exact process scope closed normally, with no signals or
remaining processes, and the 7,637 pinned inputs remained unchanged.

The [compact evidence](evidence/robot-modularity/leap-hand-free-motion-native-20261004.json)
binds source, model, plans, full solve journals, MCP responses and an independent
JSON-only replay of joint limits, target slew, servo effort and retained rest.
It also retains the source-gain negative and both setup failures (missing
OpenCV dependency and missing isolated mock profile, before their respective
native task construction). The positive model pin is
`cf8cbc1599fe954d5ddf3422f4a0f078b13eaeb724883a10f46a25e0f7d4d358`.

Three subsequent predeclared repetitions of the same target also passed, each
in a fresh process with a new model epoch and the same recipe, limits and
deadlines. They retained 494, 493 and 491 solves respectively; each confirmed
0.2 simulated seconds of post-ACK rest and normal closure. All three JSON-only
audits passed. The compact evidence includes every attempt and their launch
binding; the original four successful episodes do not establish robustness to
different targets, objects or disturbances.

Grasps, calibrated tactile measurements and hardware acceptance remain pending.
The MCP handler was exercised in process, without an LLM or stdio transport.

The subsequent [bounded voice-to-hand attempts](evidence/robot-modularity/voice-hand-negative-20261004.json)
dispatched no hand action: one hit a harness file-opening error and one reached
the original owner deadline after correct internal ASR and an invalid provider
tool name. They do not extend the named-posture motion evidence to speech or
spoken confirmation of rest.
