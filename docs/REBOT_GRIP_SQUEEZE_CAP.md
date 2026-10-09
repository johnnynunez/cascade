# reBot RS gripper: contact-relative squeeze cap (B38, opt-in)

Seeed feedback (2026-10-08): "no gripper constraint on object size; the paper
cup should not have been crushed; make it adapt to the object size."

## Why the real pick close squeezes wide objects harder

Both RS transports (`control/rebot_rs_arm.py`, `control/rebot_rs_mb_arm.py`)
close the jaws with `close_gripper_two_stage`. The runtime calls it from
`skill_grasp_object` → `_close_two_stage` with the grip profile from
`grasping/force.py::GRIP_PROFILES`. Stage 1 sends the jaws to
`close_frac_stage1` of travel at 0.7 × effort, and stage 2 sends them to
`close_frac_stage2` at the profile's effort. A stall ends a stage, and the jaws
are then **left pushing at the stage-2 target**.

The RobStride motor runs in MIT mode, where tau = kp·(target − pos) − kd·vel.
A jaw stopped on an object at angle `c` therefore holds it with

    torque = kp · effort · (c − stage-2 target)

That torque grows with the object's width: a 7.5 cm cup is squeezed harder than
the 5 cm cube. The park close (`close_gripper_torque`) drops to `hold_kp` after
contact; the pick close never did. PR #265 (WRC control port) added the travel
clamp, the fault clear and the measured polarity, but it did not change this.

## The cap

`gripper.max_contact_squeeze_rad` in `configs/arms/rebot_rs.yaml` and
`rebot_rs_mb.yaml` is the number of radians of jaw travel allowed past the first
contact. The default is `null`, which runs the fixed close unchanged, byte for
byte (pinned by golden traces). When the value is set, both drivers run
`robstride.close_two_stage_capped`. It uses the same grace period (0.3 s),
poll period (0.15 s), reach tolerance (0.1 rad) and per-stage deadline as the
fixed close, with these differences:

- **Contact is a mechPos stall.** That means two consecutive polls that each
  advance less than 0.03 rad, after at least 0.05 rad of travel and short of
  the target. This is the stall rule `close_gripper_torque` already uses.
  mechVel (0x701A) is never read, because it is not in rad/s on this firmware
  ([WRC_CONTROL_PORT.md](WRC_CONTROL_PORT.md)).
- **Every jaw target after the first contact is at most `cap` past it.** On the
  RS jaw, which closes toward smaller angles, the target is
  `max(stage target, contact − cap)`. The steady torque is therefore
  **min(today's, kp·effort·cap)**. An object whose squeeze already fits inside
  the cap gets the same targets and kp as today. A wider or deformable object
  is bounded. The contact is the *first* stall and is never moved, so a
  deformable object cannot ratchet the jaw further in.
- **A jaw that reaches a capped target met nothing.** If the jaw comes within
  0.1 rad of a capped target, the stall was a stick-slip or was misread in free
  air. The contact is dropped and the stage target is restored, so in free air
  the close never ends weaker than today, and the runtime's post-lift air-grasp
  check still sees the jaws fully closed.
- **Validation fails closed.** The cap must be a finite number greater than
  0.1 rad, the reach tolerance; with a cap that small, a real contact would
  read as "reached". Any other value raises `ValueError` while the driver is
  built, before any jaw command can be sent.
- **Commands still go through `set_gripper`.** The travel clamp still applies,
  and after `stop()` every further command is refused, including the cap's
  re-command. `_grip_contact_pos` keeps the contact angle.

## Torque table (offline, simulated jaw)

The numbers come from `tests/test_rebot_grip_squeeze_cap.py::_SimJaw`, an MIT
jaw with a rigid stop at each object's angle. Object widths are mapped to
angles with the profile's **linear** map (`open_pos` 6.2 rad ≙ `max_width_m`
0.09 m), which has **not been verified on hardware**
([WRC_CONTROL_PORT.md](WRC_CONTROL_PORT.md) lists the 5.0 vs 6.2 rad
disagreement).

The cap is 2.52 rad and kp = 2.0 (`rebot_rs.yaml`). The motorbridge profile
uses kp 6.0, so every torque is 3× larger there. "Torque" is the MIT proxy
kp_eff·(pos − target), in N·m if kp is in N·m/rad.

| profile (stage-2 frac, effort) | object | contact rad | stage-2 target rad | squeeze today rad | torque today | torque capped | ratio |
|---|---|---|---|---|---|---|---|
| rigid (0.85, 0.70) | 3 cm | 2.067 | 0.930 | 1.137 | 1.591 | 1.591 | 1.00 |
| rigid (0.85, 0.70) | 5 cm | 3.444 | 0.930 | 2.514 | 3.520 | 3.520 | 1.00 |
| rigid (0.85, 0.70) | 7.5 cm | 5.167 | 0.930 | 4.237 | 5.931 | 3.528 | 0.59 |
| rigid (0.85, 0.70) | 8.5 cm | 5.856 | 0.930 | 4.926 | 6.896 | 3.528 | 0.51 |
| fragile (0.60, 0.30) | 3 cm | 2.067 | 2.480 | no contact | 0.000 | 0.000 | - |
| fragile (0.60, 0.30) | 5 cm | 3.444 | 2.480 | 0.964 | 0.579 | 0.579 | 1.00 |
| fragile (0.60, 0.30) | 7.5 cm | 5.167 | 2.480 | 2.687 | 1.612 | 1.512 | 0.94 |
| fragile (0.60, 0.30) | 8.5 cm | 5.856 | 2.480 | 3.376 | 2.025 | 1.512 | 0.75 |
| soft / deformable (0.95, 0.45) | 3 cm | 2.067 | 0.310 | 1.757 | 1.581 | 1.581 | 1.00 |
| soft / deformable (0.95, 0.45) | 5 cm | 3.444 | 0.310 | 3.134 | 2.821 | 2.268 | 0.80 |
| soft / deformable (0.95, 0.45) | 7.5 cm | 5.167 | 0.310 | 4.857 | 4.371 | 2.268 | 0.52 |
| soft / deformable (0.95, 0.45) | 8.5 cm | 5.856 | 0.310 | 5.546 | 4.991 | 2.268 | 0.45 |
| slippery (0.90, 0.85) | 3 cm | 2.067 | 0.620 | 1.447 | 2.459 | 2.459 | 1.00 |
| slippery (0.90, 0.85) | 5 cm | 3.444 | 0.620 | 2.824 | 4.802 | 4.284 | 0.89 |
| slippery (0.90, 0.85) | 7.5 cm | 5.167 | 0.620 | 4.547 | 7.729 | 4.284 | 0.55 |
| slippery (0.90, 0.85) | 8.5 cm | 5.856 | 0.620 | 5.236 | 8.900 | 4.284 | 0.48 |
| heavy (0.90, 1.00) | 3 cm | 2.067 | 0.620 | 1.447 | 2.893 | 2.893 | 1.00 |
| heavy (0.90, 1.00) | 5 cm | 3.444 | 0.620 | 2.824 | 5.649 | 5.040 | 0.89 |
| heavy (0.90, 1.00) | 7.5 cm | 5.167 | 0.620 | 4.547 | 9.093 | 5.040 | 0.55 |
| heavy (0.90, 1.00) | 8.5 cm | 5.856 | 0.620 | 5.236 | 10.471 | 5.040 | 0.48 |

Notes on the table:

- In the fragile 3 cm row, the fragile stage-2 opening (3.6 cm) is wider than
  the object. The jaws never touch it, today or with the cap.
- Illustrative compliant case: a 7.5 cm cup under the deformable profile,
  modelled as a 1.0 N·m/rad spring. Today it is compressed by 2.301 rad
  (≈ 33 mm on the linear map, torque 2.30). With the cap it is compressed by
  1.677 rad (≈ 24 mm, torque 1.68). The capped squeeze is 2.52 rad past the
  first stall. The stage-1 (scout) compression before that stall is not
  reduced.

## Suggested starting cap (it ships `null`)

The suggested start is the squeeze that the 5 cm booth cube gets today under the
rigid profile (the profile a "cube" label selects). On the linear map:

- contact 0.05 / 0.09 × 6.2 = 3.444 rad;
- stage-2 target 6.2 × (1 − 0.85) = 0.930 rad;
- squeeze 2.514 rad, rounded **up** to **2.52 rad**.

With that cap:

- The cube and everything narrower keep today's exact targets and kp under the
  rigid profile. The Isaac lesson from B36 is that a grip weaker than today's
  dropped cubes in carry.
- At 3 cm, every profile is unchanged.
- At 5 cm, soft and deformable objects get 0.80 of today's torque, and
  slippery and heavy objects get 0.89.
- If 5 cm slippery or heavy objects (cans) must keep today's force, use
  **3.14 rad** instead. That is the deformable squeeze at 5 cm, the largest of
  any profile. The 7.5 cm cup then still drops to 0.65 of today's torque, and
  8.5 cm objects to 0.57.

The cap is relative to the measured contact, so the kp·effort·cap bound does
not depend on the width map. Which objects are bounded does depend on the map:
if the real cube's squeeze is larger than the cap, the cube gets weaker. Measure
the cube's squeeze first (protocol below), then set the cap at or above it.

## Hardware protocol (the user runs this; it cannot run here)

**Supervised only.** Someone at the arm must have the e-stop or power cut
within reach. Complete WRC_CONTROL_PORT.md onsite steps 1–4 first: fault clear,
gripper travel 0 → +6.39 rad, and **the width map measured at a few angles**.
Park the arm at `park_q`, supported. Only the gripper moves. Keep fingers out of
the jaws: place each object, take your hand away, then press Enter.

Objects:

- 3 cm prop (rigid);
- the 5 cm booth cube (rigid reference);
- an empty 7.5 cm paper cup (deformable);
- an ≈ 8.5 cm rigid box or bottle that fits the 9 cm opening;
- a ≈ 6.6 cm drink can (slippery).

Run a Python session from the repo root. It uses the motorbridge transport,
whose `connect()` holds the live pose before enabling torque:

```python
import time
from cascade.config import load_profile
from cascade.control.arm_base import make_arm
from cascade.control.robstride import capped_squeeze_target
from cascade.grasping.force import GRIP_PROFILES

cfg = load_profile("arms", "rebot_rs_mb")
g = cfg._data["gripper"]
g["kp"] = 2.0                        # the production rebot_rs gripper kp (rebot_rs_mb ships 6.0)
g["max_contact_squeeze_rad"] = None  # phase A; phase B: the measured cap
arm = make_arm(cfg)
arm.connect()

def trial(obj, profile="rigid", effort_scale=1.0):
    p = GRIP_PROFILES[profile]
    effort = p.effort * effort_scale
    arm.open_gripper(); time.sleep(2.0)
    input(f"place {obj} centred between the jaws, hands clear, Enter to close ")
    final = arm.close_gripper_two_stage(width_frac_stage1=p.close_frac_stage1,
                                        width_frac_stage2=p.close_frac_stage2, effort=effort)
    time.sleep(1.0)
    held = arm._gripper_pos()
    stage2 = arm._grip_open + (arm._grip_closed - arm._grip_open) * p.close_frac_stage2
    contact = arm._grip_contact_pos
    cap = arm._grip_squeeze_cap
    target = stage2 if cap is None else capped_squeeze_target(
        stage2, contact, cap, arm._grip_open, arm._grip_closed)
    kp_eff = arm._grip_kp * min(max(effort, 0.05), 1.0)
    print(dict(object=obj, profile=profile, effort=round(effort, 3), cap=cap,
               contact=contact, final=final, held=held, stage2_target=round(stage2, 3),
               last_target=round(target, 3), squeeze_today=None if held is None else held - stage2,
               torque_proxy=None if held is None else kp_eff * (held - target)))
    input("inspect: dent / crush in mm? does it slip when tugged gently? Enter to open ")
    arm.open_gripper()
```

Steps:

1. **Phase A: today's close.** Run each object with `trial(obj, profile,
   0.4)` first, at low effort, and then with `trial(obj, profile)`. Use the
   rigid profile for the cube, the prop and the box, deformable for the cup,
   and slippery for the can. Record every printed dict, plus a caliper reading
   of the jaw gap while it holds. The cube's `squeeze_today` under the rigid
   profile at full effort is the measured reference.
2. **Choose the cap.** Set it to the cube's measured squeeze rounded up to
   0.05 rad, or to the largest 5 cm squeeze across profiles if the can must
   keep today's force (see above). Set
   `g["max_contact_squeeze_rad"] = <cap>`, then `arm.disconnect()`,
   `arm = make_arm(cfg)` and `arm.connect()`.
3. **Phase B: the same trials with the cap.** Check each of these:
   - The cube's `last_target` equals its `stage2_target`, and its `held` matches
     phase A (never weaker).
   - The cup's caliper gap is larger than in phase A, and the cup is less
     dented.
   - No `[rebot] jaw reached the capped target` line appears for an object.
     That line means the cap gave the full squeeze back; note which object
     caused it.
4. **Free close with the cap.** Run `trial("nothing")` with the jaws empty. The
   jaw must end at `stage2_target`.
5. **Slip check.** For the cube and the can, give each a gentle tug while it is
   held in phases A and B. Report any difference.
6. Report the dicts, the caliper gaps, the dent and slip notes, and the chosen
   cap. Only then put the cap in `configs/arms/rebot_rs.yaml`. A full
   supervised pick of the cube and the cup with the cap set comes after the
   first-motions checklist (ROADMAP "Real rig first motions").

## Not claimed

- No hardware measurement. Every number above comes from the simulated jaw and
  the unverified linear width map.
- The torque is the MIT stiffness proxy, not a calibrated force (ARCHITECTURE
  Known limitations).
- Whether the cup still crushes at the capped torque is unknown until the
  protocol has been run.
- The squeeze from the stage-1 scout before the first stall is not capped.
- A contact that the mechPos rule never sees as a stall (an object that keeps
  creeping faster than 0.03 rad per poll) gets today's close.
- With a small cap, a jaw that creeps slowly near a stage target in free air
  (the fragile profile's 0.21-effort scout) can be taken for a contact. The
  reach guard then restores today's target, so the close ends at the same
  target but sends extra commands. If a real object then stops the jaw within
  0.1 rad of that capped target, the guard also restores today's full squeeze,
  which is the fail-safe direction. With the suggested 2.52 rad, the simulated
  free close sends exactly today's commands for every profile.
- Isaac and MuJoCo grippers do not use this driver path. The B36 Isaac
  post-contact hold is a separate item.
