"""Isolated, contact-only Factory mesh experiment; no object actuation."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import trimesh
import warp as wp
import newton


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--assets', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seconds', type=float, default=3.0)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    wp.init()
    wp.set_device('cuda:0')
    builder = newton.ModelBuilder()
    cfg = builder.ShapeConfig(margin=0.0, mu=0.01, ke=1e7, kd=1e4,
                              gap=0.005, density=8000., mu_torsional=0., mu_rolling=0.)
    for name, file in [('bolt', 'factory_bolt_m20_loose.obj'),
                       ('nut', 'factory_nut_m20_loose_subdiv_3x.obj')]:
        source = trimesh.load(args.assets / file, force='mesh')
        vertices = np.asarray(source.vertices, dtype=np.float32)
        center = (vertices.min(0) + vertices.max(0)) / 2
        mesh = newton.Mesh(vertices - center, np.asarray(source.faces.flatten(), dtype=np.int32))
        print('SDF', name, flush=True)
        mesh.build_sdf(max_resolution=512, narrow_band_range=(-.005, .005),
                       margin=.005, cache_dir=args.output.parent / 'sdf-cache')
        rotation = wp.quat_identity() if name == 'bolt' else wp.quat_from_axis_angle(wp.vec3(0, 0, 1), np.pi / 8)
        position = np.array([0., 0., .041 if name == 'nut' else 0.]) + np.asarray(wp.quat_rotate(rotation, wp.vec3(*center)))
        if name == 'bolt':
            body = -1
        else:
            body = builder.add_body(xform=wp.transform(wp.vec3(*position), rotation), label=name)
        shape_pose = wp.transform(wp.vec3(*position), rotation) if body == -1 else wp.transform_identity()
        builder.add_shape_mesh(body, xform=shape_pose, mesh=mesh, cfg=cfg, label=name)
    model = builder.finalize(device='cuda:0')
    state, nxt = model.state(), model.state()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state)
    solver = newton.solvers.SolverMuJoCo(model, use_mujoco_contacts=False, solver='newton',
                                      integrator='implicitfast', cone='elliptic', njmax=1024,
                                      nconmax=1024, iterations=30, ls_iterations=100, impratio=1.)
    pipeline = newton.CollisionPipeline(model, reduce_contacts=True, rigid_contact_max=1024)
    contacts, control = pipeline.contacts(), model.control()
    rows = []
    dt = 1 / 600
    started = time.monotonic()
    with (args.output / 'samples.jsonl').open('w') as output:
        for step in range(int(args.seconds / dt)):
            state.clear_forces()
            pipeline.collide(state, contacts)
            solver.step(state, nxt, control, contacts, dt)
            state, nxt = nxt, state
            if step % 10 == 0:
                row = {'step':step + 1, 'time_s':(step + 1) * dt,
                       'body_q':state.body_q.numpy().tolist(),
                       'contact_count':int(contacts.rigid_contact_count.numpy()[0])}
                rows.append(row);output.write(json.dumps(row)+'\n');output.flush()
            if step % 600 == 0: print('step', step, 'pose', state.body_q.numpy().tolist(), flush=True)
    print(json.dumps({'wall_s':time.monotonic()-started,'first':rows[0],'last':rows[-1]}),flush=True)


if __name__ == '__main__':
    main()
