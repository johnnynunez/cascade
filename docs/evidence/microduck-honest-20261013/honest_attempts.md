| run | scenario | result | detail | own localization error final / max (path) | agent claimed | LLM calls | stack |
|---|---|---|---|---|---|---|---|
| bsH1 | backstage_obstacles | FAIL | goal 0.323 m (<= 0.35); possible contact with LowCase | 0.55 / 0.63 m (9.25 m) | failure | 30 | v1: scalar gyro heading, leg odometry, cuVSLAM with ToF-zone depth |
| bsH2 | backstage_obstacles | FAIL | goal 0.745 m (<= 0.35); possible contact with LowCase | 0.32 / 0.329 m (14.9 m) | failure | 41 | v2: + floor-plane depth for cuVSLAM, leg-odometry scale 1.19, contact-FK height, set dressing |
| bsH3 | backstage_obstacles | FAIL | goal 0.17 m (<= 0.35); possible contact with CableReel | 0.421 / 0.437 m (14.68 m) | failure | 41 | v3: + AHRS (SO(3) gyro + accel tilt), delta-angle gyro model, cone-aware ToF floor test |
| bsH4 | backstage_obstacles | FAIL | goal 0.14 m (<= 0.35); possible contact with CableReel | 0.314 / 0.333 m (5.43 m) | failure | 17 | v3 (repeat of bsH3) |
| bsH5 | backstage_obstacles | FAIL | goal 1.594 m (<= 0.35) | 0.25 / 0.253 m (4.57 m) | failure | 20 | v4: walker honest prompt updated (drift numbers, side-blind ToF, inflation 0.20-0.25 m advice) |
| bsH6 | backstage_obstacles | FAIL | goal 0.157 m (<= 0.35); possible contact with CableReel | 0.236 / 0.236 m (6.51 m) | success | 26 | v5: + walker 'do not give up while upright' (bsH5 quit at 1.6 m after 33 s) |
| bsH7 | backstage_obstacles | FAIL | goal 0.165 m (<= 0.35); possible contact with CableReel | 0.355 / 0.357 m (6.69 m) | success | 26 | v8: + walker map guard (own map, blind sides; bsH3/H4/H6 grazed the cable reel) + calibrated-IMU residual gyro bias |
| multiroomH2/duck_a | multi_room_fetch | FAIL | RedBox 0.412 m (<= 0.4) | 0.495 / 0.545 m (11.7 m) | success | 88 | v7 (multiroomH1 crashed at start: renderer clock, see runs/kit_dumps) |
| multiroomH2/duck_b | multi_room_fetch | PASS | stay upright / explore (no position check) | 1.539 / 1.786 m (13.93 m) | failure | 88 | v7 (multiroomH1 crashed at start: renderer clock, see runs/kit_dumps) |
| multiroomH2/duck_c | multi_room_fetch | PASS | BlueBox 0.282 m (<= 0.4) | 0.409 / 0.418 m (2.34 m) | success | 88 | v7 (multiroomH1 crashed at start: renderer clock, see runs/kit_dumps) |
| skateH1 | skate_plaza | FAIL | cones A:0.46(miss) B:1.93(miss) C:3.673(miss); stop 5.059 m | 0.189 / 0.189 m (1.72 m) | failure | 11 | v1 (centre-ray ToF floor test: phantom brakes) |
| skateH2 | skate_plaza | FAIL | cones A:0.192 B:0.96(miss) C:0.554(miss); stop 1.66 m | 1.218 / 1.218 m (6.78 m) | failure | 18 | v2: + zone-extent ToF floor test |
| skateH3 | skate_plaza | FAIL | cones A:0.22 B:1.244(miss) C:0.387; stop 1.755 m | 2.21 / 2.21 m (6.58 m) | failure | 20 | v3: + AHRS, delta-angle gyro, real traffic cones in the scene |
| skateH4 | skate_plaza | FAIL | cones A:0.307 B:1.889(miss) C:3.575(miss); stop 4.974 m | 0.134 / 0.134 m (1.76 m) | failure | 7 | v4: + constant-velocity fill over rejected VO frames, 40-step budget, 'navigate by fresh detections' prompt |
| skateH5 | skate_plaza | FAIL | cones A:0.359 B:1.284(miss) C:0.325; stop 0.228 m | 0.312 / 1.313 m (8.38 m) | failure | 41 | v5: + 'do not give up while upright' prompt (skateH4 quit after 6 calls) |
| skateH6 | skate_plaza | FAIL | cones A:0.246 B:1.35(miss) C:0.345; stop 1.873 m | 3.729 / 3.768 m (8.79 m) | failure | 41 | v5 (repeat of skateH5) |
| skateH7 | skate_plaza | FAIL | cones A:0.335 B:1.187(miss) C:0.534(miss); stop 1.656 m | 2.386 / 2.386 m (9.9 m) | failure | 41 | v6: + 'bring the expected target into the 82 deg view before detecting; pass furniture on the side toward the next target' (H5, H6 missed B) |
| skateH8 | skate_plaza | FAIL | cones A:0.204 B:1.216(miss) C:0.554(miss); stop 0.187 m | 0.352 / 1.102 m (10.44 m) | failure | 40 | v7: + floor-flow visual odometry on skates (cuVSLAM fallback) |
| skateraceH1/duck_a | skate_race | FAIL | laps 0.269, off-track 0.551 | 2.207 / 2.343 m (5.61 m) | failure | 63 | v6 + operator track map and start pose; skate_lane on the estimate |
| skateraceH1/duck_b | skate_race | FAIL | laps 0.762, off-track 0.0 | 1.222 / 1.222 m (10.35 m) | failure | 63 | v6 + operator track map and start pose; skate_lane on the estimate |
| skateraceH1/duck_c | skate_race | FAIL | laps 0.105, off-track 0.848; possible contact with BoardS; FELL | 0.382 / 1.279 m (2.44 m) | failure | 63 | v6 + operator track map and start pose; skate_lane on the estimate |
| skateraceH2/duck_a | skate_race | FAIL | laps 0.808, off-track 0.253 | 0.879 / 1.437 m (9.33 m) | failure | 63 | v7: + floor-flow odometry (skateraceH1 ran the VO-only v6 stack) |
| skateraceH2/duck_b | skate_race | FAIL | laps 0.424, off-track 0.42 | 1.727 / 1.756 m (6.55 m) | failure | 63 | v7: + floor-flow odometry (skateraceH1 ran the VO-only v6 stack) |
| skateraceH2/duck_c | skate_race | FAIL | laps 0.137, off-track 0.54; possible contact with BoardE10 | 2.268 / 2.268 m (5.93 m) | failure | 63 | v7: + floor-flow odometry (skateraceH1 ran the VO-only v6 stack) |
| skateraceH3/duck_a | skate_race | PASS | laps 1.043, off-track 0.054, time 25.28 s | 0.491 / 0.697 m (11.94 m) | success | 47 | v8: + calibrated-IMU residual gyro bias, 30-step budget |
| skateraceH3/duck_b | skate_race | FAIL | laps 0.695, off-track 0.194 | 1.744 / 2.195 m (10.86 m) | failure | 47 | v8: + calibrated-IMU residual gyro bias, 30-step budget |
| skateraceH3/duck_c | skate_race | FAIL | laps 0.186, off-track 0.0; FELL | 0.43 / 0.667 m (3.04 m) | failure | 47 | v8: + calibrated-IMU residual gyro bias, 30-step budget |
