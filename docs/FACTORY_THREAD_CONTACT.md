# Robot-driven Factory thread contact

`scripts/validate_screw_contact.py` runs an isolated SO-101 with a mounted,
motor-driven hex socket. The nut is a free dynamic body; the bolt is a static
fixture. The original NVIDIA Factory M20 loose meshes form both threads, with
Newton SDF collision at resolution 512. Nut motion results from contact forces.
There is no helical constraint, fastener motor, injected fastener torque,
attachment or runtime object-pose assignment.

The initial state has the socket around the nut. This is an assembly fixture
experiment, not a pick, vision-localization, seating or hardware claim. It does
not silently replace a CASCADE arm backend or the ordinary `turn_screw` skill.
The controller moves only SO-101 position actuators and the socket spindle's
torque actuator. After one measured nut turn it commands a velocity brake.

The independently read body poses and solved nut/bolt and nut/socket contact
forces feed `sim/threading_verification.py`. The spindle speed servo targets
1.5 rad/s with an illustrative 0.05 N m torque limit. Every physical substep is
checked for finite state, solver overflow and the nut's angular speed. A result
above the declared sampling bound remains unverified. Seating and preload are
always unverified by this one-turn experiment.

## Reproduce

Use an isolated Python 3.12 environment with `newton[sim]==1.6.0`,
`warp-lang==1.17.0`, `mujoco==3.12.0`, `mujoco-warp==3.12.0`,
`numpy==2.5.3`, `scipy==1.18.1`, and `trimesh==4.12.2`.

```bash
python scripts/fetch_factory_assets.py
python scripts/fetch_robot_assets.py so101
CUDA_VISIBLE_DEVICES=0 python scripts/validate_screw_contact.py \
  --robot-asset "$PWD/assets/mjcf/so101/so101.xml" \
  --output "$PWD/runs/thread-contact/positive" \
  --cache "$PWD/runs/thread-contact/sdf-cache"
```

Use a new output directory for each run. `--no-drive` commands exactly zero
socket actuator torque (the authored spindle viscous damping remains).
`--misaligned` moves the socket 60 mm off the bolt axis. `--substeps 20` halves
the 1/600-second physics timestep. Negative controls intentionally exit nonzero
when they do not verify the requested threading task. Failures preserve raw
observations; constructor errors produce a terminal error report.

For a replay of measured physical states, install `pyglet==2.1.16` and Pillow,
and use `scripts/render_screw_contact.py` with `--evidence`, `--robot-asset`,
`--cache`, and optionally `--video` (requires ffmpeg). Its three PNG frames and
MP4 render saved solver poses through Newton's viewer; the render buffer is
never passed to a simulation step.

## Local evidence, 2026-10-02

On one RTX PRO 6000 Blackwell GPU, the source-bound positive run measured
1.004155 clockwise turns and 2.508275 mm axial advance. Maximum pitch residual
was 0.009285 mm, radial offset 0.141841 mm, and axis tilt 0.006581 rad.
Both solved contact types witnessed 96.751% of angular travel. The verifier
confirmed threading and left seating unverified.

The exact-zero-drive control measured only 0.197470 turns / 0.490479 mm over
the same 7.2-second observation window and correctly failed the one-turn task.
All recorded motor torque values were zero. This small passive motion is
expected in a gravity-loaded low-friction thread, so rotation alone is not
evidence of commanded work. The displaced-tool control had zero tool contacts
and was refused (its free nut also exceeded the sampling speed bound).

A separate half-timestep run measured 1.004315 turns / 2.509058 mm, with a
0.007221 mm maximum pitch residual. It used the same physical geometry and
controller before terminal-report hardening and visual-only color changes;
its source hashes are retained separately. This is timestep sensitivity
evidence, not an assertion of identical source.

Compact summaries and source/asset bindings are under
`docs/evidence/factory-thread-contact/`. Full raw body states, the final video,
and start/middle/end frames are retained in the evidence locations listed in
that directory's artifact manifest. The original reduced Newton screw demo
remains an explicitly different mechanism using an ideal helix.
