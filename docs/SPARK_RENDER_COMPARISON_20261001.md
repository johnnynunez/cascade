# Idle Spark rendering comparison

Headless Isaac now disables updates to the unused default viewport. The named
camera render products remain enabled; GUI runs retain their viewport. The
setting is supported by NVIDIA's [performance guidance](https://docs.isaacsim.omniverse.nvidia.com/latest/reference_material/sim_performance_optimization_handbook.html).

An isolated simulator on port 8612 ran the same kitchen at 1280×720 and a
1/120 s physics timestep. The driver only read the authoritative physics clock
and the `cam0`, `side` and `proof` images. It issued no joint, jaw or prop-reset
commands. Each measurement followed startup and ten seconds of warmup, then
sampled approximately 45 wall seconds. The native demo and its model services
were stopped during this comparison.

| Run | Default viewport | `--cam-every` | Configured camera tick rate | Simulation / wall time |
| --- | --- | ---: | ---: | ---: |
| A | Enabled | 2 | 60 Hz | 0.2414 |
| B2 | Disabled | 2 | 60 Hz | 0.2456 |
| C2 | Disabled | 12 | 10 Hz | 0.6053 |
| A2 | Enabled | 2 | 60 Hz | 0.2401 |
| D | Enabled | 12 | 10 Hz | 0.4015 |

Disabling the viewport alone made little difference at the original camera
rate. At the lower camera rate, removing the viewport also improved measured
throughput. The lower rate uses the existing `CASCADE_ISAAC_CAM_EVERY=12`
setting; this change does not alter the default camera rate or physics timestep.

All five completed runs returned three nonblank 720×1280 RGB images and valid
depth at the beginning and end. Producer stamps advanced on all three cameras.
The three final C2 images were also inspected visually. These are sampled idle
checks, not continuous freshness bounds or proof of correct masks during motion.
The original B attempt stopped before scene startup because its source manifest
had not been regenerated. Its failure receipt is retained; B2 used a corrected,
committed manifest.

The [receipt](../benchmark/results/spark_idle_render_comparison_20261001.json)
contains exact source pins, commands, measurements, camera receipts and SHA-256
hashes of the raw records and drivers. The runs were sequential and only A was
repeated. They establish neither statistical confidence nor loaded demo
throughput. Correct physical pacing, dynamic grasp performance, camera freshness
under load and normal restart acceptance require separate validation.
