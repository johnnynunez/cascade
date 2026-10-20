"""Render recorded G1 three-finger driver fastening evidence; never advances or modifies physics.

Replays saved solver body poses (``body_poses_xyzw``) of the runtime-route
episodes (``validate_humanoid_fastening.py``), the standalone run
(``validate_humanoid_screw_contact.py``) and the fingers-open negative control
through Newton's headless viewer and burns English labels (title, grasp,
simplifications, live nut rotation/advance, thumb/index/middle normal forces,
handle rotation in the hand, contacts, the verifier outcome) into an H.264 MP4
plus PNG stills. The close-up camera follows the hand and looks at the grip
from the index/middle side, as in a photo of a person holding a screwdriver.
Run with the Newton environment and Pillow; ffmpeg must be on PATH.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys

import numpy as np

import pyglet

pyglet.options['headless'] = True  # before any viewer/window import

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from cascade.sim.newton_humanoid_screw import (
    GRIP_X_M, HANDLE_TOP_M, SIMPLIFICATIONS, HumanoidThreadingScene, rot_xyzw,
)

W, H = 1920, 1080
MAIN = (1280, 720)
CLOSE = (640, 480)
FONT = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
BOLD = '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'
TITLE = ('Unitree G1 humanoid tightens a bolt with a hand-held screwdriver - '
         'Newton physics, CASCADE fastening verifier')
GRASP_LABEL = 'three-finger contact grasp (thumb, index, middle), no weld: held by finger contact friction only'
SIMPLE = ('Simplified: pelvis fixed (no balance), gravity-compensation feed-forward, only the right hand collides, '
          'driver starts engaged on the nut, arm turn follows an IK table')
PHASES = {'idle': 'holding the driver, wrist-roll torque zero',
          'stroke': 'STROKE: arm turns the hand about the driver axis',
          'settle_stroke': 'end of arm range: settle', 'release': 'RE-GRIP: fingers open (driver stands in socket)',
          'return': 'RE-GRIP: open hand rotates back around the handle', 'settle_return': 'RE-GRIP: settle',
          'regrip': 'RE-GRIP: thumb, index, middle close until contact',
          'stopped': 'STOP: wrist-roll torque exactly zero, rest observed'}


def font(size, bold=False):
    from PIL import ImageFont
    return ImageFont.truetype(BOLD if bold else FONT, size)


def look(viewer, wp, position, target):
    position, target = np.asarray(position, float), np.asarray(target, float)
    d = target - position
    viewer.set_camera(wp.vec3(*position), math.degrees(math.atan2(d[2], np.linalg.norm(d[:2]))),
                      math.degrees(math.atan2(d[1], d[0])))


def display_scene(assets, robot_asset, cache):
    """The recorded model plus ONE render-only marker: a green tab fixed under the nut.

    The hex socket hides the nut, so its rotation/advance would be invisible.
    The tab is added to the display copy only (no collision, never stepped);
    body order and every recorded pose are unchanged.
    """
    import newton
    import warp as wp
    original = newton.ModelBuilder.finalize

    def finalize(builder, *args, **kwargs):
        nut = builder.body_label.index('nut') if 'nut' in builder.body_label else next(
            i for i, label in enumerate(builder.body_label) if label.rsplit('/', 1)[-1] == 'nut')
        visual = builder.ShapeConfig(has_shape_collision=False, has_particle_collision=False, density=0.)
        builder.add_shape_box(nut, xform=wp.transform(wp.vec3(.026, 0., -.0115), wp.quat_identity()),
                              hx=.009, hy=.003, hz=.0012, cfg=visual, label='render_marker', color=(.2, .95, .3))
        return original(builder, *args, **kwargs)
    newton.ModelBuilder.finalize = finalize
    try:
        return HumanoidThreadingScene(assets, robot_asset, cache)
    finally:
        newton.ModelBuilder.finalize = original


MAIN_CAMERA = ((1.05, -.75, .62), (.14, .08, .20))
# Close-up in the HAND frame (x: thumb -> fingers, y: palm normal = down the driver), relative
# to the top of the handle: from the index/middle side, above, slightly toward the index.
CLOSE_OFFSET_HAND = (.20, -.11, .07)


class Renderer:
    """One headless viewer; the close-up is a second camera pose cropped from the same render size."""

    def __init__(self, scene, main_camera=MAIN_CAMERA):
        import warp as wp
        import newton
        from newton.viewer import ViewerGL
        self.wp, self.scene = wp, scene
        self.main_camera = main_camera
        flags = scene.model.shape_flags.numpy()
        for index, label in enumerate(scene.model.shape_label):
            if label.rsplit('/', 1)[-1].startswith(('socket_wall_', 'socket_floor', 'driver_handle')):
                flags[index] |= int(newton.ShapeFlags.VISIBLE)
        scene.model.shape_flags.assign(flags)
        self.viewer = ViewerGL(width=MAIN[0], height=MAIN[1], headless=True,
                               enable_cuda_interop=ViewerGL.CudaInterop.NONE)
        self.viewer.set_model(scene.model)
        self.viewer.show_joints = False
        self.display = scene.model.state()

    def close_camera(self, poses):
        """Camera fixed to the hand, aimed at the grip (top of the handle)."""
        hand = np.asarray(poses[self.scene.hand_body], dtype=float)
        r = rot_xyzw(hand[3:])
        grip = hand[:3] + r @ np.array([GRIP_X_M, HANDLE_TOP_M + .025, 0.])
        return grip + r @ np.array(CLOSE_OFFSET_HAND), grip

    def render(self, poses, t):
        from PIL import Image
        self.display.body_q.assign(np.asarray(poses, dtype=np.float32))
        images = []
        for position, target in (self.main_camera, self.close_camera(poses)):
            look(self.viewer, self.wp, position, target)
            for _ in range(2):
                self.viewer.begin_frame(t)
                self.viewer.log_state(self.display)
                self.viewer.end_frame()
            images.append(self.viewer.get_frame().numpy())
        # Close-up: same 16:9 render, centre crop to 4:3 and scale to the side panel.
        crop = images[1][:, (MAIN[0] - MAIN[1] * 4 // 3) // 2:(MAIN[0] + MAIN[1] * 4 // 3) // 2]
        images[1] = np.asarray(Image.fromarray(crop).resize(CLOSE))
        return images

    def close(self):
        self.viewer.close()


def frames_from_route(episode_dir, step_stride):
    """(t, poses, info) from a runtime-route episode's retained raw solves."""
    rows = []
    with (episode_dir / 'run/domains/fastening/solves.jsonl').open() as stream:
        for line in stream:
            raw = json.loads(line)
            if raw['body_poses_xyzw'] is None or raw['solve']['step'] % step_stride:
                continue
            s = raw['solve']
            rows.append({'t': s['simulation_time_s'], 'step': s['step'], 'poses': raw['body_poses_xyzw'],
                         'phase': raw['program']['phase'], 'grip': raw['grip_normal_force_n'],
                         'grip_torque': raw['grip_torque_nm'], 'relative': raw['hand_driver_angle_rad'],
                         'phi': raw['hand_phi_rad'], 'torque': s['spindle_effort_nm'],
                         'nut_q': s['fastener_quaternion_xyzw'], 'nut_z': s['fastener_position_m'][2],
                         'thread': s['thread_contacts'], 'tool': s['tool_contacts'], 'generation': s['generation']})
    return rows


def frames_from_standalone(directory, stride):
    rows = []
    for i, line in enumerate((directory / 'samples.jsonl').read_text().splitlines()):
        if i % stride:
            continue
        r = json.loads(line)
        rows.append({'t': r['time_s'], 'step': r['step'], 'poses': r['body_poses_xyzw'], 'phase': r['phase'],
                     'grip': r['grip_normal_force_n'], 'grip_torque': r['grip_torque_nm'],
                     'relative': r['hand_driver_angle_rad'], 'phi': r['hand_phi_rad'],
                     'torque': r['wrist_roll_torque_nm'],
                     'nut_q': r['fastener_quaternion_xyzw'], 'nut_z': r['fastener_position_m'][2],
                     'thread': r['thread_contacts'], 'tool': r['tool_contacts'], 'generation': None})
    return rows


def yaw(q):
    x, y, z, w = q
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def annotate_progress(rows):
    unwrapped, last = 0., None
    z0 = rows[0]['nut_z']
    for r in rows:
        a = yaw(r['nut_q'])
        if last is not None:
            unwrapped += math.atan2(math.sin(a - last), math.cos(a - last))
        last = a
        r['turns'] = -unwrapped / (2 * math.pi)
        r['advance_mm'] = (z0 - r['nut_z']) * 1000


def fit(draw, text, size, width, bold=False):
    """Largest font <= size whose rendering of text fits width pixels."""
    while size > 12 and draw.textlength(text, font=font(size, bold)) > width:
        size -= 1
    return font(size, bold)


def compose(images, header, lines, footer, verdict=None, speed='1x (real time)', clip_turns=None, requested=1.):
    """One 1920x1080 frame: main view, close-up, live telemetry, episode footer, verdict box."""
    from PIL import Image, ImageDraw
    frame = Image.new('RGB', (W, H), (18, 20, 24))
    frame.paste(Image.fromarray(images[0]), (0, 120))
    frame.paste(Image.fromarray(images[1]), (MAIN[0], 120))
    d = ImageDraw.Draw(frame)
    d.text((24, 8), TITLE, font=fit(d, TITLE, 31, W - 48, True), fill=(118, 185, 0))
    d.text((24, 50), GRASP_LABEL, font=fit(d, GRASP_LABEL, 24, W - 48, True), fill=(255, 210, 90))
    d.text((24, 86), SIMPLE, font=fit(d, SIMPLE, 20, W - 48), fill=(190, 190, 190))
    d.rectangle((MAIN[0] + 4, 124, W - 4, 156), fill=(18, 20, 24))
    d.text((MAIN[0] + 14, 128), 'close-up (camera follows the hand): the grip', font=font(20), fill=(230, 230, 230))
    d.rectangle((14, 132, 210, 166), fill=(40, 44, 52))
    d.text((24, 137), speed, font=font(20, True), fill=(255, 210, 90))
    y = 606
    for text, color in lines:
        d.text((MAIN[0] + 18, y), text, font=fit(d, text, 21, W - MAIN[0] - 30), fill=color)
        y += 31
    d.text((24, 856), header, font=fit(d, header, 28, 1240, True), fill=(255, 255, 255))
    for i, text in enumerate(footer):
        d.text((24, 900 + 34 * i), text, font=fit(d, text, 22, 1240), fill=(200, 200, 200))
    if clip_turns is not None:
        x0, x1, yb = 24, 1000, 1040
        d.rectangle((x0, yb, x1, yb + 18), outline=(120, 120, 120))
        frac = max(0., min(1., clip_turns / requested))
        d.rectangle((x0, yb, x0 + (x1 - x0) * frac, yb + 18), fill=(118, 185, 0))
        d.text((x1 + 14, yb - 6), f'nut {clip_turns:.3f} / {requested:g} turn', font=font(22), fill=(230, 230, 230))
    if verdict is not None:
        text, color = verdict
        d.rectangle((1290, 860, 1906, 1066), fill=(30, 34, 40), outline=color, width=4)
        for i, line in enumerate(text):
            d.text((1306, 872 + 38 * i), line, font=fit(d, line, 26 if i == 0 else 22, 586, i == 0),
                   fill=color if i == 0 else (235, 235, 235))
    return frame


def card(lines, sizes=None):
    from PIL import Image, ImageDraw
    frame = Image.new('RGB', (W, H), (18, 20, 24))
    d = ImageDraw.Draw(frame)
    y = 90
    for i, (text, size, color, bold) in enumerate(lines):
        d.text((90, y), text, font=font(size, bold), fill=color)
        y += int(size * 1.55)
    return frame


class Encoder:
    def __init__(self, path, fps):
        self.process = subprocess.Popen(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pixel_format', 'rgb24',
            '-video_size', f'{W}x{H}', '-framerate', str(fps), '-i', '-', '-an', '-c:v', 'libx264', '-preset', 'medium',
            '-crf', '23', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(path)], stdin=subprocess.PIPE)
        self.count = 0

    def write(self, image, repeat=1):
        data = np.asarray(image, dtype=np.uint8).tobytes()
        for _ in range(repeat):
            self.process.stdin.write(data)
            self.count += 1

    def close(self):
        self.process.stdin.close()
        if self.process.wait() != 0:
            raise RuntimeError('ffmpeg failed')


def failure_reason(command):
    """Human-readable cause of a non-verified command, from its recorded result."""
    error = str(command.get('error') or command.get('reason') or '')
    if 'observation_age' in error:
        age = json.loads(error.split('observation_age=', 1)[1])
        return [f"observation age {age['age_s']:.4f} s > {age['max_observation_age_s']} s at solve {age['step']}",
                'host GC pause; the domain rejected the stale reading']
    if command.get('rotation_before_stop_turns') is not None and command['rotation_before_stop_turns'] < .5:
        return [f"nut rotation {command['rotation_before_stop_turns']:.4f} turn: pre-stop refuted",
                'command lease expired; no turn, no advance']
    return [error[:60]]


def verdict_lines(command, label):
    ok = command.get('verified')
    color = (118, 185, 0) if ok else (235, 80, 70)
    lines = [f'{label}: ' + ('CONFIRMED' if ok else 'NOT VERIFIED')]
    if ok:
        lines.append(f"rotation before stop {command['rotation_before_stop_turns']:.3f} turn (>= 0.95)")
        lines.append(f"final {command['final_rotation_turns']:.3f} turn, advance {command['axial_advance_m'] * 1000:.2f} mm")
        lines.append(f"contact-supported travel {command['contact_rotation_fraction'] * 100:.0f}%, rest "
                     f"{(command.get('rest') or {}).get('status')}")
    else:
        lines += failure_reason(command)
    return lines, color


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--route', type=Path, required=True, help='validate_humanoid_fastening.py output')
    parser.add_argument('--two-turn', type=Path, help='validate_humanoid_screw_contact.py --turns 2 output')
    parser.add_argument('--no-drive', type=Path, help='validate_humanoid_fastening.py --no-drive output')
    parser.add_argument('--fingers-open', type=Path, help='validate_humanoid_fastening.py --fingers-open output')
    parser.add_argument('--robot-asset', type=Path, required=True)
    parser.add_argument('--assets', type=Path, default=ROOT / 'assets/factory/nut_bolt')
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--stills', type=Path, required=True)
    parser.add_argument('--fps', type=int, default=30)
    args = parser.parse_args()
    args.stills.mkdir(parents=True, exist_ok=True)
    scene = display_scene(args.assets, args.robot_asset, args.cache)  # display model only, never stepped
    episodes = sorted(p for p in args.route.glob('episode-*') if (p / 'summary.json').exists())
    renderer = Renderer(scene)
    encoder = Encoder(args.output, args.fps)
    stills, table = [], []
    try:
        intro = card([(TITLE[:76], 40, (118, 185, 0), True), (TITLE[76:], 40, (118, 185, 0), True),
                      ('', 20, (0, 0, 0), False),
                      (GRASP_LABEL, 27, (255, 210, 90), True),
                      ('Real Newton 1.6 run (MJWarp, SDF thread contact). The Dex3 thumb opposes index + middle on the',
                       27, (235, 235, 235), False),
                      ('handle; the arm turns the hand about the driver axis (re-grips between strokes). The nut turns',
                       27, (235, 235, 235), False),
                      ('and advances only through solved finger/handle, socket/nut and thread contacts.',
                       27, (235, 235, 235), False),
                      ('Each command goes through CASCADE: robot profile -> build_robot_runtime -> fastening.turn_screw',
                       27, (235, 235, 235), False),
                      ('-> unchanged FasteningDomain verifier (rotation before stop, axial advance, observed rest).',
                       27, (235, 235, 235), False),
                      ('', 20, (0, 0, 0), False), ('Simplifications:', 27, (255, 210, 90), True),
                      *[('- ' + s, 25, (210, 210, 210), False) for s in SIMPLIFICATIONS],
                      ('', 20, (0, 0, 0), False),
                      ('Asset: newton-assets unitree_g1 g1_29dof_with_hand_rev_1_0 (29 DoF + Dex3 hands). '
                       'Bolt/nut: Factory M20, 2.5 mm pitch.', 23, (170, 170, 170), False)])
        encoder.write(intro, args.fps * 10)
        intro.save(args.stills / '00_intro.png')
        segments = []
        for index, episode in enumerate(episodes, 1):
            summary = json.loads((episode / 'summary.json').read_text())
            n = summary['commands_requested']
            label = f'Episode {index}/{len(episodes)}' + (' (two consecutive one-turn commands)' if n == 2 else '')
            segments.append(('route', episode, summary, label, 1 if index == 1 else 2 if index == 2 else 3))
        if args.two_turn:
            segments.append(('two', args.two_turn, json.loads((args.two_turn / 'summary.json').read_text()),
                             'Standalone two-turn request (verify_threading, requested 2 turns)', 3))
        if args.no_drive:
            segments.append(('neg', sorted(args.no_drive.glob('episode-*'))[0],
                             json.loads((sorted(args.no_drive.glob('episode-*'))[0] / 'summary.json').read_text()),
                             'Negative control: same command, wrist torque forced to zero', 8))
        if args.fingers_open:
            first = sorted(args.fingers_open.glob('episode-*'))[0]
            segments.append(('open', first, json.loads((first / 'summary.json').read_text()),
                             'Negative control: same turn with the fingers OPEN (no grip)', 6))
        for kind, directory, summary, label, speed in segments:
            # 600 Hz solves (route poses every 5th) / 60 Hz standalone frames -> 30 fps at 1x.
            rows = (frames_from_standalone(directory, 2 * speed) if kind == 'two'
                    else frames_from_route(directory, 20 * speed))
            start = next((i for i, r in enumerate(rows) if r['phase'] != 'idle'), 0)
            rows = rows[max(0, start - 15 // speed):]
            annotate_progress(rows)
            requested = 2. if kind == 'two' or summary.get('commands_requested') == 2 else 1.
            if kind == 'two':
                m = summary['measured']
                ok = summary['threading_verified']
                verdicts = [([f"Two-turn request: {'CONFIRMED (verify_threading)' if ok else 'REFUTED'}",
                              f"rotation {m['turns']:.3f} turn (>= 1.95), advance {m['axial_advance_m'] * 1000:.2f} mm",
                              f"pitch error {m['max_pitch_error_m'] * 1000:.3f} mm, contact travel "
                              f"{m['contact_rotation_fraction'] * 100:.0f}%",
                              f"{summary['regrips']} re-grips"], (118, 185, 0) if ok else (235, 80, 70))]
            else:
                commands = summary.get('commands', [])
                verdicts = [verdict_lines(c, f'Command {i}' if len(commands) > 1 else 'turn_screw')
                            for i, c in enumerate(commands, 1)]
            footer = [('fastening.turn_screw(turns=1, direction="tighten") via RobotRuntime.execute; '
                       'judged by the unchanged FasteningDomain') if kind != 'two' else
                      'scripts/validate_humanoid_screw_contact.py --turns 2; judged by the unchanged verify_threading',
                      'green tab under the nut: render-only marker (not simulated) so the hidden nut rotation is visible']
            shown, stops, frame = None, 0, None
            for i, r in enumerate(rows):
                if r['phase'] == 'stopped' and (i == 0 or rows[i - 1]['phase'] != 'stopped'):
                    stops += 1   # one verdict per observed stop, in command order
                shown = verdicts[min(stops, len(verdicts)) - 1] if r['phase'] == 'stopped' and stops else None
                phase = PHASES.get(r['phase'], r['phase'])
                g = r['grip']
                lines = [(f"t = {r['t']:.2f} s (sim)   {phase}", (255, 255, 255)),
                         (f"nut rotation {r['turns']:+.3f} turn, advance {r['advance_mm']:+.2f} mm", (118, 185, 0)),
                         (f"grip normal force: thumb {g['thumb']:.1f} N, index {g['index']:.1f} N, "
                          f"middle {g['middle']:.1f} N", (255, 210, 90)),
                         (f"torque from fingers to driver {r['grip_torque']:+.3f} N m", (200, 200, 200)),
                         (f"hand rotation {r['phi']:+.2f} rad, wrist-roll torque {r['torque']:+.3f} N m",
                          (200, 200, 200)),
                         (f"handle angle in the hand {r['relative']:+.3f} rad", (200, 200, 200)),
                         (f"solved contacts: thread {r['thread']}, hex walls {r['tool']}", (200, 200, 200))]
                frame = compose(renderer.render(r['poses'], r['t']), label, lines, footer, shown,
                                speed='1x (real time)' if speed == 1 else f'{speed}x speed',
                                clip_turns=r['turns'], requested=requested)
                encoder.write(frame)
                if kind == 'route' and directory.name == 'episode-01':
                    name = f'episode-01_{r["phase"]}.png'
                    if r['phase'] in ('stroke', 'release', 'return', 'regrip') and name not in stills \
                            and i > 2 and all(rows[i - k]['phase'] == r['phase'] for k in (1, 2, 3)):
                        frame.save(args.stills / name)
                        frame.crop((MAIN[0], 120, W, 120 + CLOSE[1])).save(
                            args.stills / name.replace('.png', '_closeup.png'))
                        stills.append(name)
            if frame is not None:
                # Hold the last frame with every verdict this segment has not shown yet.
                pending = verdicts[stops - 1:] if stops and shown is not None else verdicts[stops:]
                for index, verdict in enumerate(pending or [shown]):
                    if verdict is None:
                        continue
                    r = rows[-1]
                    held = compose(renderer.render(r['poses'], r['t']), label, lines, footer, verdict,
                                   speed='1x (real time)' if speed == 1 else f'{speed}x speed',
                                   clip_turns=r['turns'], requested=requested)
                    encoder.write(held, args.fps * 3)
                    tag = directory.name if kind == 'route' else {'two': 'two_turn', 'neg': 'no_drive',
                                                                      'open': 'fingers_open'}[kind]
                    name = f'{tag}_verdict{"" if len(pending) < 2 else index + 1}.png'
                    held.save(args.stills / name)
                    stills.append(name)
            for c_index, (vlines, _) in enumerate(verdicts, 1):
                table.append((label if len(verdicts) == 1 else f'{label} - command {c_index}', vlines))
            strokes = summary.get('strokes_grip') or (summary.get('stream') or {}).get('strokes_grip') or []
            if kind != 'open' and strokes:
                parts = []
                for st in strokes:
                    f = st['fingers']
                    parts.append(f"S{st['stroke']} T{f['thumb']['mean_n']:.0f}/I{f['index']['mean_n']:.0f}/"
                                 f"M{f['middle']['mean_n']:.0f} N {'OK' if st['grasp_ok'] else 'FAILED'}")
                table.append(('     grip per stroke (mean normal force)', [': ' + ', '.join(parts)]))
        rows_text = [('Per-episode verifier outcome (this run)', 36, (118, 185, 0), True), ('', 10, (0, 0, 0), False)]
        for label, vlines in table:
            if label.startswith('     grip'):
                rows_text.append((label + vlines[0], 19, (255, 210, 90), False))
                continue
            ok = 'CONFIRMED' in vlines[0]
            rows_text.append((label + ': ' + vlines[0].split(': ', 1)[-1], 23, (118, 185, 0) if ok else (235, 80, 70), True))
            rows_text.append(('     ' + ' | '.join(vlines[1:]), 19, (215, 215, 215), False))
        rows_text += [('', 12, (0, 0, 0), False),
                      ('Grasp check per stroke: thumb, index and middle each >= 15 % of the grip normal force and in '
                       'contact >= 95 % of the stroke.', 20, (255, 210, 90), False),
                      ('Not shown: standing balance, tool pickup/engagement, seating/preload, hardware.',
                       20, (255, 210, 90), False)]
        outro = card(rows_text)
        encoder.write(outro, args.fps * 12)
        outro.save(args.stills / '99_summary.png')
    finally:
        encoder.close()
    provenance = {'mode': 'saved_solver_body_poses_only_no_solver_steps', 'renderer_sha256':
                  hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), 'frames': encoder.count, 'fps': args.fps,
                  'resolution': [W, H], 'route_summary_sha256': hashlib.sha256((args.route / 'summary.json').read_bytes()).hexdigest(),
                  'stills': stills}
    (args.output.parent / (args.output.stem + '-provenance.json')).write_text(json.dumps(provenance, indent=2) + '\n')
    print(json.dumps(provenance, indent=2))


if __name__ == '__main__':
    main()
