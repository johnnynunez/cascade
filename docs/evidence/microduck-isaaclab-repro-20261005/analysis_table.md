### Walk-then-zero episodes (heading = root +x at the end of the 1 s settle)
| policy | env | variant | seed | case | cmd steps | reached | during along [mm] | during lat [mm] | during yaw [deg] | post-zero along [mm] | post-zero lat [mm] | post-zero path [mm] | post-zero yaw [deg] | still (0.2 s) at [s] | last-1 s disp [mm] | max tilt [deg] | fell |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| velocity_flat | flat-task(plane) | cascade_nominal | 0 | fwd_cl | 13 | yes | 25.8 | 16.8 | -0.0 | 10.4 | -12.7 | 28.7 | 10.9 | 0.60 | 0.35 | 2.7 | no |
| velocity_flat | flat-task(plane) | cascade_nominal | 0 | rev_cl | 14 | yes | -25.3 | -4.7 | 1.6 | -10.9 | -10.0 | 29.0 | -4.9 | 0.64 | 0.07 | 2.6 | no |
| velocity_flat | flat-task(plane) | cascade_nominal | 0 | fwd_ol | 15 | - | 31.9 | 11.5 | -0.1 | 10.7 | -10.7 | 29.3 | 10.2 | 0.58 | 0.24 | 2.8 | no |
| velocity_flat | flat-task(plane) | cascade_nominal | 0 | rev_ol | 15 | - | -29.0 | -9.0 | 1.9 | -34.8 | -2.8 | 58.9 | -16.5 | 1.34 | 0.11 | 5.7 | no |
| velocity_flat | flat-task(plane) | cascade_nominal | 0 | fwd_1s | 50 | - | 142.1 | 0.6 | 2.6 | 14.2 | 4.7 | 33.6 | -3.1 | 0.50 | 0.05 | 3.3 | no |
| velocity_flat | flat-task(plane) | cascade_nominal | 0 | rev_1s | 50 | - | -157.7 | -3.7 | 4.2 | -10.6 | -15.0 | 26.1 | -1.8 | 0.56 | 0.02 | 2.7 | no |
| velocity_flat | flat-task(plane) | lab_play | 0 | fwd_cl | 14 | yes | 26.1 | 26.7 | 1.0 | 27.9 | -5.0 | 44.9 | 0.7 | 0.60 | 0.13 | 2.4 | no |
| velocity_flat | flat-task(plane) | lab_play | 0 | rev_cl | 18 | yes | -27.9 | 5.5 | 1.0 | -18.2 | -12.7 | 42.6 | -0.5 | 0.68 | 0.03 | 2.8 | no |
| velocity_flat | flat-task(plane) | lab_play | 0 | fwd_ol | 15 | - | 28.7 | 21.8 | 1.2 | 24.4 | -6.5 | 43.3 | 1.6 | 0.56 | 0.15 | 3.0 | no |
| velocity_flat | flat-task(plane) | lab_play | 0 | rev_ol | 15 | - | -18.5 | 12.0 | 2.0 | -22.5 | -19.3 | 45.2 | -1.5 | 0.62 | 0.07 | 2.7 | no |
| velocity_flat | flat-task(plane) | lab_play | 0 | fwd_1s | 50 | - | 162.3 | -1.7 | -1.0 | 17.4 | 8.4 | 29.5 | -2.6 | 0.44 | 0.01 | 3.0 | no |
| velocity_flat | flat-task(plane) | lab_play | 0 | rev_1s | 50 | - | -149.3 | -20.1 | 4.2 | -18.1 | 14.9 | 29.9 | 8.2 | 0.44 | 0.02 | 2.5 | no |
| velocity_flat | flat-task(plane) | lab_play | 1 | fwd_cl | 14 | yes | 25.7 | 26.8 | 1.3 | 16.7 | -15.7 | 39.7 | -1.0 | 0.50 | 0.08 | 1.5 | no |
| velocity_flat | flat-task(plane) | lab_play | 1 | rev_cl | 16 | yes | -27.0 | 9.8 | 2.2 | -33.1 | -21.1 | 53.8 | -3.0 | 0.48 | 0.05 | 1.8 | no |
| velocity_flat | flat-task(plane) | lab_play | 1 | fwd_ol | 15 | - | 28.3 | 21.2 | 1.2 | 15.8 | -17.0 | 39.7 | -1.1 | 0.50 | 0.08 | 1.7 | no |
| velocity_flat | flat-task(plane) | lab_play | 1 | rev_ol | 15 | - | -24.5 | 9.4 | 2.3 | -34.3 | -21.4 | 51.3 | -1.9 | 0.54 | 0.08 | 1.4 | no |
| velocity_flat | flat-task(plane) | norand | 0 | fwd_cl | 14 | yes | 27.5 | 26.9 | 1.4 | 31.2 | -6.7 | 45.8 | 1.1 | 0.50 | 0.07 | 2.7 | no |
| velocity_flat | flat-task(plane) | norand | 0 | rev_cl | 17 | yes | -27.1 | -1.7 | 0.8 | -23.9 | -11.6 | 45.9 | 2.4 | 0.66 | 0.09 | 2.6 | no |
| velocity_flat | flat-task(plane) | norand | 0 | fwd_ol | 15 | - | 29.3 | 20.6 | 1.4 | 23.1 | -10.9 | 43.3 | 1.4 | 0.60 | 0.03 | 2.9 | no |
| velocity_flat | flat-task(plane) | norand | 0 | rev_ol | 15 | - | -20.5 | 2.1 | 0.9 | -21.5 | -20.3 | 43.1 | -0.4 | 0.50 | 0.15 | 2.3 | no |
| velocity_rough | flat-task(plane) | lab_play | 0 | fwd_cl | 15 | yes | 27.9 | 18.8 | 0.9 | 17.6 | -10.9 | 41.2 | 2.3 | 0.52 | 0.04 | 3.1 | no |
| velocity_rough | flat-task(plane) | lab_play | 0 | rev_cl | 17 | yes | -25.1 | 10.0 | -1.8 | -28.8 | -17.9 | 53.7 | -3.5 | 0.60 | 0.12 | 3.7 | no |
| velocity_rough | flat-task(plane) | lab_play | 0 | fwd_ol | 15 | - | 30.1 | 19.2 | 1.1 | 21.8 | -9.5 | 37.8 | 2.2 | 0.46 | 0.05 | 3.6 | no |
| velocity_rough | flat-task(plane) | lab_play | 0 | rev_ol | 15 | - | -18.8 | 14.3 | -0.9 | -27.8 | -22.3 | 56.7 | -7.0 | 0.60 | 0.12 | 3.6 | no |
| velocity_rough | rough-task(plane) | cascade_nominal | 0 | fwd_cl | 14 | yes | 25.4 | -0.6 | 0.0 | 5.0 | -7.7 | 24.1 | 2.9 | 0.54 | 0.13 | 2.5 | no |
| velocity_rough | rough-task(plane) | cascade_nominal | 0 | rev_cl | 14 | yes | -28.7 | -11.7 | 0.3 | -12.0 | -2.4 | 31.4 | -3.0 | 0.68 | 0.03 | 2.5 | no |
| velocity_rough | rough-task(plane) | cascade_nominal | 0 | fwd_ol | 15 | - | 27.5 | -4.5 | 0.6 | 4.1 | -3.5 | 20.4 | 1.5 | 0.52 | 0.14 | 2.5 | no |
| velocity_rough | rough-task(plane) | cascade_nominal | 0 | rev_ol | 15 | - | -32.9 | -15.0 | 0.3 | -22.9 | 6.7 | 40.1 | -3.2 | 0.68 | 0.03 | 2.5 | no |
| velocity_rough | rough-task(plane) | cascade_nominal | 0 | fwd_1s | 50 | - | 140.3 | 16.8 | 2.6 | 2.4 | -8.3 | 30.4 | 8.6 | 0.66 | 0.07 | 3.8 | no |
| velocity_rough | rough-task(plane) | cascade_nominal | 0 | rev_1s | 50 | - | -172.5 | 3.1 | -2.1 | -13.5 | -13.0 | 35.4 | -4.2 | 0.68 | 0.04 | 2.8 | no |
| velocity_rough | rough-task(plane) | lab_play | 0 | fwd_cl | 15 | yes | 27.9 | 18.8 | 0.9 | 17.6 | -10.9 | 41.2 | 2.3 | 0.52 | 0.04 | 3.1 | no |
| velocity_rough | rough-task(plane) | lab_play | 0 | rev_cl | 17 | yes | -25.1 | 10.0 | -1.8 | -28.8 | -17.9 | 53.6 | -3.5 | 0.60 | 0.13 | 3.7 | no |
| velocity_rough | rough-task(plane) | lab_play | 0 | fwd_ol | 15 | - | 30.1 | 19.2 | 1.1 | 21.8 | -9.5 | 37.9 | 2.2 | 0.46 | 0.05 | 3.6 | no |
| velocity_rough | rough-task(plane) | lab_play | 0 | rev_ol | 15 | - | -18.8 | 14.3 | -0.9 | -27.7 | -22.3 | 56.5 | -7.1 | 0.62 | 0.11 | 3.6 | no |
| velocity_rough | rough-task(plane) | lab_play | 0 | fwd_1s | 50 | - | 159.2 | 4.4 | -1.1 | 16.8 | 10.0 | 34.2 | -3.7 | 0.50 | 0.04 | 3.5 | no |
| velocity_rough | rough-task(plane) | lab_play | 0 | rev_1s | 50 | - | -165.7 | -10.1 | -1.4 | -21.1 | 20.1 | 34.6 | 4.0 | 0.42 | 0.06 | 3.6 | no |
| velocity_rough | rough-task(plane) | lab_play | 1 | fwd_cl | 15 | yes | 26.3 | 16.1 | 1.6 | 15.1 | -19.1 | 41.9 | -0.6 | 0.54 | 0.04 | 1.7 | no |
| velocity_rough | rough-task(plane) | lab_play | 1 | rev_cl | 16 | yes | -26.7 | 12.8 | -1.0 | -43.7 | -21.5 | 70.9 | -4.1 | 0.66 | 0.26 | 2.9 | no |
| velocity_rough | rough-task(plane) | lab_play | 1 | fwd_ol | 15 | - | 27.3 | 12.1 | 1.1 | 11.6 | -15.9 | 40.6 | -0.4 | 0.54 | 0.04 | 2.1 | no |
| velocity_rough | rough-task(plane) | lab_play | 1 | rev_ol | 15 | - | -24.7 | 10.7 | -0.1 | -45.8 | -21.7 | 73.3 | -3.8 | 0.74 | 0.23 | 2.6 | no |
| velocity_rough | rough-task(plane) | norand | 0 | fwd_cl | 14 | yes | 25.6 | 21.3 | 0.9 | 24.5 | -12.0 | 38.6 | 1.4 | 0.46 | 0.01 | 3.2 | no |
| velocity_rough | rough-task(plane) | norand | 0 | rev_cl | 16 | yes | -25.5 | 2.1 | -1.4 | -26.0 | -15.8 | 50.1 | -3.9 | 0.54 | 0.06 | 3.2 | no |
| velocity_rough | rough-task(plane) | norand | 0 | fwd_ol | 15 | - | 28.5 | 8.3 | 0.8 | 14.3 | -12.3 | 38.7 | 0.4 | 0.44 | 0.04 | 3.2 | no |
| velocity_rough | rough-task(plane) | norand | 0 | rev_ol | 15 | - | -23.1 | 2.5 | -1.6 | -26.4 | -17.2 | 52.7 | -3.7 | 0.50 | 0.05 | 2.9 | no |

### 10 s zero-command standing
| policy | env | variant | seed | drift [mm] | along [mm] | lateral [mm] | yaw change [deg] | max planar speed [m/s] | still fraction | max tilt [deg] | fell |
|---|---|---|---|---|---|---|---|---|---|---|---|
| velocity_flat | flat-task(plane) | cascade_nominal | 0 | 0.13 | -0.11 | 0.08 | 0.05 | 0.002 | 1.000 | 1.2 | no |
| velocity_flat | flat-task(plane) | lab_play | 0 | 0.27 | -0.26 | 0.08 | -0.00 | 0.001 | 1.000 | 1.4 | no |
| velocity_flat | flat-task(plane) | lab_play | 1 | 0.33 | -0.30 | 0.13 | -0.05 | 0.003 | 1.000 | 0.7 | no |
| velocity_flat | flat-task(plane) | norand | 0 | 0.30 | -0.30 | 0.04 | 0.06 | 0.002 | 1.000 | 1.5 | no |
| velocity_rough | flat-task(plane) | lab_play | 0 | 0.22 | 0.12 | 0.18 | -0.15 | 0.000 | 1.000 | 1.6 | no |
| velocity_rough | rough-task(plane) | cascade_nominal | 0 | 0.16 | 0.14 | 0.08 | -0.08 | 0.001 | 1.000 | 1.7 | no |
| velocity_rough | rough-task(plane) | lab_play | 0 | 0.22 | 0.12 | 0.18 | -0.15 | 0.000 | 1.000 | 1.6 | no |
| velocity_rough | rough-task(plane) | lab_play | 1 | 0.19 | 0.16 | 0.10 | -0.23 | 0.000 | 1.000 | 1.1 | no |
| velocity_rough | rough-task(plane) | norand | 0 | 0.10 | 0.02 | 0.10 | -0.09 | 0.001 | 1.000 | 1.8 | no |
