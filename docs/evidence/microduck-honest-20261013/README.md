# MicroDuck agentic, ONBOARD-ONLY runs (2026-10-13)

Every onboard-only episode is listed, pass or fail (`honest_attempts.md` / `.json`). The `stack` column says which
estimator / prompt version ran, because both changed between attempts. Each change is logged with its reason in the
workspace STATUS (D16-D29). One run is not in the table: multiroomH1 crashed at start (the renderer clock did not
advance in per-robot isolated renders; fixed, then rerun as multiroomH2).

## What the robot and its agent had
- **Sensors:** OVRTX head camera 640x360 at 15 Hz, VL53L8CX ToF model, IMU model (delta-angle gyro, residual bias),
  joint encoders.
- **Localization:** own estimate from AHRS heading, leg odometry (walker), floor-flow visual odometry (skates) and
  cuVSLAM RGBD increments, gated.
- **Perception:** YOLOE detector on the camera image.
- **Map and memory:** its own ToF map and its own memory.
- **Not available:** simulator state never reaches the robot or the agent. It is used only by the offline checks in
  this table, including the localization error.

## Summary
| scenario | attempts | passed | typical failure |
|---|---|---|---|
| backstage (walker) | 7 | 0 | Reaches the pad (0.14-0.17 m in 4 runs) but grazes the cable reel on the way. bsH7 kept 0.073 m from the reel (radial), yet the box-based check still flags it. |
| multi-room fetch (3 walkers) | 1 (3 robots) | 2 of 3 robots | duck_a stopped 0.412 m from the red box (limit 0.4). |
| skate plaza | 8 | 0 | Cone B is never seen: the agent always swerves left of the bench. Early runs also had camera-odometry drift up to 3.7 m. |
| skate race (3 skaters, shared world) | 3 (9 skaters) | 1 skater | skateraceH3/duck_a: one lap in 25.3 s, 5 % off-track. The others: falls (no get-up on skates), off-track, out of steps. |

## Localization measured (offline, vs truth)
- **Walker:** 0.24-0.55 m over 5-15 m.
- **Skates, cuVSLAM only:** 1.2-3.7 m over ~9 m.
- **Skates, floor flow:** 0.35-0.49 m final (0.7-1.1 m peak) over 10-12 m.
