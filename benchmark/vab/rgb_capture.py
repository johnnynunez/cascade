"""Record public RGB plus native camera calibration, without object state.

Camera poses may belong only to the fixed world or the robot's kinematic tree.
An instance-local hook observes original render calls exactly once. The caller
later joins their exact RGB bytes to the public observation. No render, solve
or forward call is added and no depth is read.
"""
from dataclasses import asdict
import base64
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from cascade.perception.rgb_triangulation import RgbView


class PublicRgbRecorder:
    def __init__(self, env, directory, *, model_identity_sha256, epoch, max_captures=256):
        if type(max_captures) is not int or not 1 <= max_captures <= 4096:
            raise ValueError('explicit bounded RGB capture count required')
        self.env, self.directory = env, Path(directory)
        self.model_identity_sha256, self.epoch = model_identity_sha256, epoch
        self.max_captures, self.count, self.last_time = max_captures, 0, None
        self.closed, self.failure = False, None
        self.cameras = ('agentview', 'robot0_eye_in_hand')
        self._sim = env.sim
        self._renders = {}
        self.recipe = self._recipe()
        self.calibration_sha256 = hashlib.sha256(json.dumps(self.recipe, sort_keys=True,
            separators=(',', ':'), allow_nan=False).encode()).hexdigest()
        self.directory.mkdir(exist_ok=False)
        (self.directory/'calibration.json').write_text(json.dumps({
            'sha256': self.calibration_sha256, 'recipe': self.recipe,
            'position_error_m': None, 'angular_error_rad': None,
            'scope': 'Nominal native camera declaration; not calibrated physical uncertainty.'}, indent=2)+'\n')
        self._had_render = 'render' in vars(self._sim)
        self._original_render = self._sim.render
        def observed_render(*args, **kwargs):
            name = kwargs.get('camera_name')
            if name not in self.cameras:
                return self._original_render(*args, **kwargs)
            try:
                if self.closed or self.failure is not None or self.env.sim is not self._sim:
                    raise ValueError('RGB render observer is not admitted')
                if args or kwargs != {'camera_name': name, 'width': 128, 'height': 128, 'depth': False}:
                    raise ValueError('original RGB render arguments changed')
                before = self._camera_state(name)
                image = self._original_render(*args, **kwargs)
                after = self._camera_state(name)
                if after != before:
                    raise ValueError('camera state changed inside original render')
                rgb = np.asarray(image)
                if rgb.shape != (128, 128, 3) or rgb.dtype != np.uint8:
                    raise ValueError('original render is not 128x128 RGB')
                self._renders[name] = (before, rgb.tobytes())
                return image
            except BaseException as exc:
                self.failure = f'{type(exc).__name__}: {exc}'
                raise
        self._hook = observed_render
        self._sim.render = self._hook

    def _camera_state(self, name):
        if self._recipe() != self.recipe:
            raise ValueError('camera calibration or robot ancestry changed')
        spec = self.recipe['cameras'][name]
        data, dt = self._sim.data, float(self._sim.model.opt.timestep)
        time = float(data.time)
        if not math.isfinite(time) or time < 0 or not math.isfinite(dt) or dt <= 0:
            raise ValueError('invalid native render clock')
        index = spec['id']
        transform = np.eye(4)
        transform[:3, :3] = np.asarray(data.cam_xmat[index]).reshape(3, 3) @ np.diag([1., -1., -1.])
        transform[:3, 3] = data.cam_xpos[index]
        return {'solver_step': round(time/dt), 'simulation_time_s': time,
                'world_from_camera': tuple(map(float, transform.flat))}

    def _recipe(self):
        model = self.env.sim.model
        root = model.body_name2id(self.env.robots[0].robot_model.root_body)
        if root <= 0:
            raise ValueError('explicit non-world robot root required')
        parents = np.asarray(model.body_parentid)
        def robot_body(body):
            seen = set()
            while body:
                if body in seen or not 0 <= body < len(parents):
                    raise ValueError('invalid body ancestry')
                if body == root:
                    return True
                seen.add(body); body = int(parents[body])
            return False
        cameras = {}
        for name in self.cameras:
            camera = model.camera_name2id(name)
            body = int(model.cam_bodyid[camera])
            # Fixed local camera transforms only. Track/target modes can expose
            # unobserved object motion through the camera orientation.
            if int(model.cam_mode[camera]) != 0 or not (body == 0 or robot_body(body)):
                raise ValueError('camera must be fixed to world or robot, without tracking targets')
            fovy = float(model.cam_fovy[camera])
            if not math.isfinite(fovy) or not 1 < fovy < 179:
                raise ValueError('finite perspective field of view required')
            cameras[name] = {'id': int(camera), 'body_id': body, 'mode': 0, 'fovy_deg': fovy,
                'local_position_m': np.asarray(model.cam_pos[camera]).tolist(),
                'local_quaternion_wxyz': np.asarray(model.cam_quat[camera]).tolist()}
        return {'schema': 'cascade.vab-public-rgb-calibration.v1',
            'model_identity_sha256': self.model_identity_sha256,
            'robot_root': int(root), 'body_parent_ids': parents.tolist(), 'cameras': cameras,
            'image_layout': 'top-down RGB, optical x right / y down / z forward',
            'pixel_center_offset_uv': [.5, .5]}

    def capture(self, observation, *, solver_step, simulation_time_s):
        if self.closed or self.failure is not None:
            raise RuntimeError(self.failure or 'RGB recorder closed')
        try:
            if self.count >= self.max_captures:
                raise ValueError('RGB capture archive exhausted')
            native_time = float(self.env.sim.data.time)
            dt = float(self.env.sim.model.opt.timestep)
            if (not math.isfinite(dt) or dt <= 0 or simulation_time_s != native_time
                    or solver_step != round(native_time/dt)):
                raise ValueError('RGB capture must bind the current native clock')
            if self._recipe() != self.recipe:
                raise ValueError('camera calibration or robot ancestry changed')
            if self._sim.render is not self._hook or self.env.sim is not self._sim:
                raise ValueError('RGB render hook or simulator changed')
            if self.count == 0 and not self._renders:
                # Reset rendered before this observer could attach to the new
                # simulator. Preserve that absence, never invent its pose/time.
                value = {'schema': 'cascade.vab-public-rgb-pair.v1', 'status': 'unverified',
                    'reason': 'reset images precede render observer', 'views': [],
                    'returned_solver_step': solver_step, 'returned_simulation_time_s': simulation_time_s,
                    'public_rgb_sha256': {name: hashlib.sha256(np.asarray(observation['images'][name]).tobytes()).hexdigest()
                                          for name in self.cameras},
                    'physical_admission': False, 'position_error_m': None}
                self._write(value)
                return
            if set(self._renders) != set(self.cameras):
                raise ValueError('both original camera renders are required')
            views = []
            for name in self.cameras:
                rgb = np.asarray(observation['images'][name])
                if rgb.shape != (128, 128, 3) or rgb.dtype != np.uint8:
                    raise ValueError('original 128x128 public RGB required')
                state, rendered = self._renders[name]
                if rgb.tobytes() != rendered:
                    raise ValueError('public RGB differs from the original observed render')
                step, time = state['solver_step'], state['simulation_time_s']
                if step > solver_step or time > simulation_time_s:
                    raise ValueError('RGB render comes from a future native clock')
                if self.last_time is not None and (step <= self.last_time[0] or time <= self.last_time[1]):
                    raise ValueError('RGB render clock did not advance')
                f = 64/math.tan(math.radians(self.recipe['cameras'][name]['fovy_deg'])/2)
                view = RgbView(name, 'vab_world', self.model_identity_sha256, self.calibration_sha256,
                    self.epoch, step, time, 128, 128,
                    (f, 0., 64., 0., f, 64., 0., 0., 1.), state['world_from_camera'],
                    (.5, .5), np.ascontiguousarray(rgb[::-1]).tobytes())
                value = asdict(view); value['rgb'] = base64.b64encode(view.rgb).decode('ascii')
                views.append({'capture_sha256': view.sha256, 'rgb_encoding': 'base64', 'view': value})
            first, second = [item['view'] for item in views]
            if any(first[k] != second[k] for k in ('solver_step', 'simulation_time_s')):
                raise ValueError('camera renders are not simultaneous')
            if float(self.env.sim.data.time) != native_time or self._recipe() != self.recipe:
                raise ValueError('native clock or calibration changed during RGB capture')
            # One bounded file per simultaneous pair; failed writes retain a
            # failed recorder and cannot later become a successful close.
            value = {'schema': 'cascade.vab-public-rgb-pair.v1', 'views': views,
                     'physical_admission': False, 'position_error_m': None}
            value.update(status='observed', returned_solver_step=solver_step,
                         returned_simulation_time_s=simulation_time_s)
            self._write(value)
            self.last_time = (first['solver_step'], first['simulation_time_s'])
            self._renders.clear()
        except BaseException as exc:
            self.failure = f'{type(exc).__name__}: {exc}'
            raise

    def _write(self, value):
        path = self.directory/f'{self.count:04d}.json'
        with path.open('x') as stream:
            stream.write(json.dumps(value, allow_nan=False, separators=(',', ':'))+'\n')
        self.count += 1

    def close(self):
        if not self.closed:
            if self._sim.render is not self._hook:
                self.failure = self.failure or 'RGB render hook was replaced'
            elif self._had_render:
                self._sim.render = self._original_render
            else:
                del self._sim.render
        self.closed = True
        return {'ok': self.failure is None, 'captures': self.count, 'error': self.failure,
                'physical_admission': False, 'position_error_m': None}
