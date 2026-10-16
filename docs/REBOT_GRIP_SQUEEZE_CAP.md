# reBot RS gripper: contact-relative squeeze cap (B38, opt-in)

**Next action:** before setting any value, run the supervised
[hardware protocol](#hardware-protocol-supervised-the-user-runs-this) (about 45 min with five
objects); until then `gripper.max_contact_squeeze_rad` ships `null` in `configs/arms/rebot_rs.yaml`
and `rebot_rs_mb.yaml`. Offline check, no robot (seconds):

```bash
uv run pytest tests/test_rebot_grip_squeeze_cap.py -q
```

Status: measured only on a simulated jaw with the **unverified** linear width map; no hardware
measurement. Origin: Seeed feedback (2026-10-08), "the paper cup should not have been crushed; make
it adapt to the object size."

## Why the pick close squeezes wide objects harder

Both RS transports (`control/rebot_rs_arm.py`, `control/rebot_rs_mb_arm.py`) close with
`close_gripper_two_stage`, called from `skill_grasp_object` → `_close_two_stage` with a profile
from `grasping/force.py::GRIP_PROFILES`: stage 1 to `close_frac_stage1` at 0.7 × effort, stage 2 to
`close_frac_stage2` at the profile's effort; a stall ends a stage and the jaws are **left pushing
at the stage-2 target**. In MIT mode (tau = kp·(target − pos) − kd·vel) a jaw stopped at angle `c`
holds with

    torque = kp · effort · (c − stage-2 target)

so a 7.5 cm cup is squeezed harder than the 5 cm cube. The park close (`close_gripper_torque`)
drops to `hold_kp` after contact; the pick close never did.

## The cap

`gripper.max_contact_squeeze_rad` is the jaw travel allowed past the first contact. `null` runs the
fixed close unchanged, byte for byte (golden traces). When set, both drivers run
`robstride.close_two_stage_capped` with the same grace (0.3 s), poll (0.15 s), reach tolerance
(0.1 rad) and per-stage deadline:

1. **Contact is a mechPos stall:** two consecutive polls each advancing less than 0.03 rad, after
   ≥ 0.05 rad of travel and short of the target (`close_gripper_torque`'s rule). mechVel (0x701A)
   is never read: it is not in rad/s on this firmware ([WRC_CONTROL_PORT.md](WRC_CONTROL_PORT.md)).
2. **Every target after the first contact is at most `cap` past it:** `max(stage target, contact −
   cap)` (the RS jaw closes toward smaller angles), so steady torque is **min(today's,
   kp·effort·cap)**. The contact is the *first* stall and never moves, so a deformable object cannot
   ratchet the jaw in.
3. **A jaw that reaches a capped target met nothing:** within 0.1 rad of it, the contact is dropped
   and the stage target restored, so a free-air close never ends weaker than today and the post-lift
   air-grasp check still sees closed jaws.
4. **Validation fails closed:** the cap must be finite and greater than 0.1 rad (the reach
   tolerance); anything else raises `ValueError` while the driver is built, before any jaw command.
5. **Commands still go through `set_gripper`:** the travel clamp applies, and after `stop()` every
   command (the cap's re-command included) is refused. `_grip_contact_pos` keeps the contact angle.

## Torque table (offline, simulated jaw)

From `tests/test_rebot_grip_squeeze_cap.py::_SimJaw` (MIT jaw, rigid stop per object). Widths map
to angles **linearly** (`open_pos` 6.2 rad ≙ `max_width_m` 0.09 m), **not verified on hardware**
(WRC_CONTROL_PORT lists the 5.0 vs 6.2 rad disagreement). Cap 2.52 rad, kp 2.0 (`rebot_rs.yaml`);
the motorbridge profile's kp 6.0 makes every torque 3× larger. "Torque" is the MIT proxy
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

The fragile 3 cm object is narrower than the fragile stage-2 opening (3.6 cm), so the jaws never
touch it. Compliant example: a 7.5 cm cup under the deformable profile as a 1.0 N·m/rad spring is
compressed 2.301 rad (≈ 33 mm) today and 1.677 rad (≈ 24 mm) capped; the stage-1 scout compression
before the first stall is not reduced.

## Suggested starting cap: 2.52 rad

It is the 5 cm booth cube's squeeze today under the rigid profile (contact 0.05 / 0.09 × 6.2 = 3.444
rad, stage-2 target 6.2 × (1 − 0.85) = 0.930 rad, squeeze 2.514 rounded **up**).

1. The cube and everything narrower keep today's exact targets and kp under rigid (B36's Isaac
   lesson: a weaker grip dropped cubes in carry); at 3 cm every profile is unchanged.
2. At 5 cm, soft/deformable get 0.80 of today's torque, slippery and heavy 0.89.
3. If 5 cm slippery or heavy objects (cans) must keep today's force, use **3.14 rad** (the largest
   5 cm squeeze); the 7.5 cm cup then gets 0.65 and 8.5 cm objects 0.57.
4. The kp·effort·cap bound is independent of the width map, but which objects are bounded is not:
   measure the real cube's squeeze first and set the cap at or above it.

## Hardware protocol (supervised; the user runs this)

**Supervised only:** someone at the arm has the e-stop or power cut within reach. First complete
WRC_CONTROL_PORT onsite steps 1–4 (fault clear, gripper travel 0 → +6.39 rad, **the width map
measured at a few angles**). Park the arm at `park_q`, supported; only the gripper moves. Keep
fingers out of the jaws: place each object, take your hand away, then press Enter.

Objects: a 3 cm prop (rigid), the 5 cm booth cube (rigid reference), an empty 7.5 cm paper cup
(deformable), an ≈ 8.5 cm rigid box or bottle that fits the 9 cm opening, an ≈ 6.6 cm drink can
(slippery). Open a Python session from the repo root (`uv run python`); it uses the motorbridge
transport, whose `connect()` holds the live pose before enabling torque:

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

1. **Phase A, today's close:** each object with `trial(obj, profile, 0.4)` (low effort) and then
   `trial(obj, profile)`; rigid for the cube, prop and box, deformable for the cup, slippery for the
   can. Record every dict plus a caliper reading of the held jaw gap. The cube's rigid
   `squeeze_today` at full effort is the reference.
2. **Choose the cap:** the cube's squeeze rounded up to 0.05 rad, or the largest 5 cm squeeze if
   the can must keep today's force. Set `g["max_contact_squeeze_rad"] = <cap>`, then
   `arm.disconnect()`, `arm = make_arm(cfg)`, `arm.connect()`.
3. **Phase B, same trials with the cap:** the cube's `last_target` equals its `stage2_target` and
   `held` matches phase A (never weaker); the cup's caliper gap is larger and it is less dented; no
   `[rebot] jaw reached the capped target` line for an object (if one appears, note the object: the
   cap gave the full squeeze back).
4. **Free close and slip:** `trial("nothing")` with empty jaws must end at `stage2_target`; tug the
   cube and the can gently in phases A and B and report any difference.
5. Report the dicts, caliper gaps, dent and slip notes and the chosen cap; only then put the cap in
   `configs/arms/rebot_rs.yaml`. A supervised pick of the cube and cup with the cap set comes after
   the ROADMAP "Real rig first motions" checklist.

## Not claimed

1. No hardware measurement; the torque is the MIT stiffness proxy, not a calibrated force; whether
   the cup still crushes at the capped torque is unknown until the protocol runs.
2. The stage-1 scout squeeze before the first stall is not capped.
3. An object creeping faster than 0.03 rad per poll never reads as a stall and gets today's close.
4. With a small cap a slowly creeping free-air jaw (the fragile profile's 0.21-effort scout) can
   read as a contact; the reach guard restores today's target (extra commands, same end), and a
   real object stopping within 0.1 rad of the capped target also gets today's full squeeze (the
   fail-safe direction). With 2.52 rad the simulated free close sends exactly today's commands.
5. Isaac and MuJoCo grippers do not use this driver path; the B36 Isaac post-contact hold is
   separate.
