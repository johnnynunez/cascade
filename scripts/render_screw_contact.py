"""Render saved solver poses; never advance or modify the physical simulation."""
import argparse
import json
import math
from pathlib import Path
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from cascade.sim.newton_screw_contact import ThreadingScene


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evidence', type=Path, required=True)
    parser.add_argument('--robot-asset', type=Path, required=True)
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--assets', type=Path, default=ROOT / 'assets/factory/nut_bolt')
    parser.add_argument('--video', action='store_true')
    args = parser.parse_args()
    import pyglet
    pyglet.options['headless'] = True
    import warp as wp
    from newton.viewer import ViewerGL
    from PIL import Image
    rows = [json.loads(line) for line in (args.evidence/'samples.jsonl').read_text().splitlines()]
    scene = ThreadingScene(args.assets, args.robot_asset, args.cache)
    # MJCF imports hide collision-only geometry when separate arm visuals are
    # present. Show the actual socket collision solids in this display model.
    # No solver is stepped and no collision or geometry flags are removed.
    import newton
    flags = scene.model.shape_flags.numpy()
    for index, label in enumerate(scene.model.shape_label):
        if label.rsplit('/', 1)[-1].startswith(('socket_wall_', 'socket_case')):
            flags[index] |= int(newton.ShapeFlags.VISIBLE)
    scene.model.shape_flags.assign(flags)
    viewer = ViewerGL(width=1280, height=720, headless=True,
                      enable_cuda_interop=ViewerGL.CudaInterop.NONE)
    viewer.set_model(scene.model)
    viewer.show_joints = False
    target = np.array([.18, 0., .12])
    position = target + [.42, -.46, .28]
    delta = target-position
    viewer.set_camera(wp.vec3(*position), math.degrees(math.atan2(delta[2], np.linalg.norm(delta[:2]))),
                      math.degrees(math.atan2(delta[1], delta[0])))
    display = scene.model.state()  # separate buffer, never supplied to solver.step
    encoder = None
    if args.video:
        encoder = subprocess.Popen(['ffmpeg','-y','-loglevel','error','-f','rawvideo',
            '-pixel_format','rgb24','-video_size','1280x720','-framerate','60','-i','-',
            '-an','-c:v','libx264','-pix_fmt','yuv420p',str(args.evidence/'threading.mp4')],
            stdin=subprocess.PIPE)
    try:
        captures = {0:'start',len(rows)//2:'middle',len(rows)-1:'end'}
        indices = range(len(rows)) if args.video else captures
        for index in indices:
            row = rows[index]
            display.body_q.assign(np.asarray(row['body_poses_xyzw'], dtype=np.float32))
            for _ in range(3):
                viewer.begin_frame(row['time_s'])
                viewer.log_state(display)
                viewer.end_frame()
            pixels = viewer.get_frame().numpy()
            if index in captures:
                Image.fromarray(pixels).save(args.evidence/f'{captures[index]}.png')
            if encoder is not None:
                encoder.stdin.write(pixels.tobytes())
    finally:
        viewer.close()
        if encoder is not None:
            encoder.stdin.close()
            if encoder.wait() != 0:
                raise RuntimeError('video encoder failed')


if __name__ == '__main__':
    main()
