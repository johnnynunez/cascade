# Evidence: MicroDuck OVRTX replay (2026-10-13)

Machine: workstation with 2x RTX PRO 6000 Blackwell. Isaac Lab venv: Python 3.12, ovrtx 0.5.1.385782,
ovstage 0.2.1.385922. Every job ran through `heavyrun` (memory-capped systemd units).

## First-write staleness probe

`probe_first_write.py` renders the MicroDuck rest pose four times (yaw 0/45/90/180 deg), moving 8 cm along +x each
time. It posts the 16 flat body prims world poses WITHOUT the sentinel and takes an argument that selects which prim
is listed first:

```
heavyrun -n cascade-dbg-yaw-first -m 16G -g auto -w -t 300 -- \
  bash tools/labpy.sh <worktree> probe_first_write.py <outdir> first   # trunk_base listed first
heavyrun -n cascade-dbg-yaw-last  -m 16G -g auto -w -t 300 -- \
  bash tools/labpy.sh <worktree> probe_first_write.py <outdir> last    # trunk_base listed last
```

`first_write_order_probe.jpg`:
* Top row: trunk_base listed first. The white trunk shell stays at its frame-1 pose while the robot moves and turns.
* Bottom row: trunk_base listed last. The trunk follows the robot, and the new first prim (a small hip link,
  `yaw2roll`) is the piece left behind at yaw 180.

## Recorded episode before and after the fix

Frames 0, 100, 200, 300, 400 and 500 of the agentic stage_edge episode stage1 are rendered sequentially by one
renderer at 480x360:
* `recorded_frames_before_fix.jpg`: without the sentinel, a trunk piece drifts off the body from frame 200 on.
* `recorded_frames_after_fix.jpg`: with `/World/ReplayWriteSentinel` written first, all six frames are assembled.

Recorded data check: the local transforms recomputed from the recorded body world poses match the USD rest
translation offsets to <= 0.2 mm at frames 0-500. The remaining rotations are the joint angles.

## Throughput (1080p review videos, RealTimePathTracing)

Each composite frame contains one 1280x720 follow/overview view, plus a 640x360 overview view for single-robot videos.

| Run | Robots | Frames | Wall time |
| --- | --- | --- | --- |
| stage_edge (stage2) | 1 | 487 | 28.4 s |
| backstage (backstage9) | 1 | 1276 | 58.1 s |
| creative gym (gym1) | 1 | 1093 | 64.8 s |
| multi-room fetch (multiroom2) | 3 | 4913 | 260.0 s |

## Tests

`python -m pytest tests/test_microduck_ovrtx_replay.py tests/test_ovrtx_renderer.py tests/test_ovrtx_runtime.py -q`
gives 64 passed, 2 skipped. The two new tests cover the transform math, the sentinel being written first, and the SDK
pair gate. The existing OVRTX tests pass unchanged, so the default pin is untouched. The full suite result is in the
PR description.
