"""Showcase choreography for the shared MicroDuck owner: a presenter proxy walks a path, robots follow.

Opt-in, in-process, no network admission. The presenter is a kinematic visual proxy (no physics
body) whose pose is an analytic function of simulation time; each robot's twist is computed from
its own completed physics state and the presenter pose (oracle pose from the simulation, no
perception) and handed to its controller through ``MobileBridgeController.script``. Speed
limits, freshness checks, falls, stop/latch and the policy/BAM path are the unchanged ones.
Nothing here is physical admission; the choreography receipt names every parameter.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field


def _finite(value, name):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(f'{name} must be a finite number')
    return float(value)


def _wrap(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


def _yaw_wxyz(q):
    w, x, y, z = q
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


@dataclass(frozen=True)
class PresenterPath:
    """Piecewise-linear path walked at constant speed after ``start_delay_s``; holds at the end.

    ``dwell_s[i]`` pauses the walk at intermediate waypoint ``i`` (a presenter stopping to talk);
    the followers use those pauses to close up, since the MicroDuck gait averages well under
    its commanded 0.3 m/s.
    """
    waypoints: tuple
    speed_m_s: float
    start_delay_s: float = 0.
    heading_end_rad: float | None = None
    dwell_s: tuple = ()

    @classmethod
    def from_dict(cls, data):
        pts = data.get('waypoints')
        if not isinstance(pts, list) or len(pts) < 2:
            raise ValueError('presenter path needs at least two waypoints')
        waypoints = tuple((_finite(p[0], 'waypoint x'), _finite(p[1], 'waypoint y')) for p in pts)
        speed = _finite(data.get('speed_m_s'), 'speed_m_s')
        if not 0 < speed <= 0.5:
            raise ValueError('presenter speed must be in (0, 0.5] m/s so the robots can follow')
        delay = _finite(data.get('start_delay_s', 0.), 'start_delay_s')
        if delay < 0:
            raise ValueError('start_delay_s must be nonnegative')
        dwell = data.get('dwell_s', [])
        if not isinstance(dwell, list) or len(dwell) > len(waypoints) - 2:
            raise ValueError('dwell_s lists at most one pause per intermediate waypoint')
        dwell = tuple(_finite(v, 'dwell_s') for v in dwell) + (0.,) * (len(waypoints) - 2 - len(dwell))
        if any(v < 0 for v in dwell):
            raise ValueError('dwell_s must be nonnegative')
        end = data.get('heading_end_rad')
        return cls(waypoints, speed, delay, None if end is None else _finite(end, 'heading_end_rad'), dwell)

    def length(self):
        return sum(math.dist(a, b) for a, b in zip(self.waypoints, self.waypoints[1:]))

    def duration_s(self):
        return self.start_delay_s + self.length() / self.speed_m_s + sum(self.dwell_s)

    def travelled_at(self, t):
        """Distance along the path at time ``t``: walking legs at ``speed_m_s``, pauses at waypoints."""
        clock = t - self.start_delay_s
        if clock <= 0:
            return 0.
        travelled = 0.
        for i, (a, b) in enumerate(zip(self.waypoints, self.waypoints[1:])):
            seg = math.dist(a, b)
            leg = seg / self.speed_m_s
            if clock < leg:
                return travelled + clock * self.speed_m_s
            clock -= leg
            travelled += seg
            dwell = self.dwell_s[i] if i < len(self.dwell_s) else 0.
            if clock < dwell:
                return travelled
            clock -= dwell
        return travelled

    def pose_at(self, t):
        """(x, y, heading, travelled_m, walking) at simulation time ``t``."""
        travelled = self.travelled_at(t)
        remaining = travelled
        for a, b in zip(self.waypoints, self.waypoints[1:]):
            seg = math.dist(a, b)
            heading = math.atan2(b[1] - a[1], b[0] - a[0])
            if remaining <= seg or (a, b) == (self.waypoints[-2], self.waypoints[-1]):
                u = min(1., remaining / seg) if seg > 0 else 1.
                x, y = a[0] + (b[0] - a[0]) * u, a[1] + (b[1] - a[1]) * u
                # walking only while the distance is actually changing (not during a pause)
                walking = 0. < travelled < self.length() and self.travelled_at(t + 1e-3) > travelled
                if not walking and travelled >= self.length() and self.heading_end_rad is not None:
                    heading = self.heading_end_rad
                return x, y, heading, min(travelled, self.length()), walking
            remaining -= seg
        raise AssertionError('unreachable')


@dataclass(frozen=True)
class WalkCycle:
    """Joint angles of the presenter proxy as a function of distance travelled (radians, metres)."""
    stride_m: float = 0.65
    hip_swing_rad: float = math.radians(25.)
    knee_flex_rad: float = math.radians(35.)
    arm_swing_rad: float = math.radians(20.)
    bob_m: float = 0.02

    def pose(self, travelled_m, walking):
        if not walking:
            return {'HipL': 0., 'HipR': 0., 'KneeL': 0., 'KneeR': 0., 'ShoulderL': 0., 'ShoulderR': 0., 'bob': 0.}
        phase = 2 * math.pi * travelled_m / (2 * self.stride_m)  # one full cycle = two steps
        hip_l = self.hip_swing_rad * math.sin(phase)
        hip_r = -hip_l
        # the knee flexes while the leg swings behind (hip angle negative = behind for +X forward)
        knee_l = self.knee_flex_rad * max(0., -math.sin(phase))
        knee_r = self.knee_flex_rad * max(0., math.sin(phase))
        return {'HipL': hip_l, 'HipR': hip_r, 'KneeL': knee_l, 'KneeR': knee_r,
                'ShoulderL': -self.arm_swing_rad * math.sin(phase), 'ShoulderR': self.arm_swing_rad * math.sin(phase),
                'bob': self.bob_m * abs(math.sin(phase))}


@dataclass(frozen=True)
class Formation:
    """Slots behind the presenter in its frame (x back is negative, y left is positive)."""
    slots: tuple
    gain_per_s: float = 0.6
    max_speed_m_s: float = 0.3
    max_turn_rad_s: float = 1.0
    heading_gain: float = 2.0
    deadband_m: float = 0.10
    turn_only_above_rad: float = math.radians(60.)

    @classmethod
    def from_dict(cls, data, count):
        slots = data.get('slots')
        if not isinstance(slots, list) or len(slots) < count:
            raise ValueError(f'formation needs at least {count} slots')
        slots = tuple((_finite(s[0], 'slot x'), _finite(s[1], 'slot y')) for s in slots[:count])
        for i, a in enumerate(slots):
            for b in slots[i + 1:]:
                if math.dist(a, b) < 0.35:
                    raise ValueError('formation slots must be at least 0.35 m apart')
            if a[0] > -0.5:
                raise ValueError('formation slots must sit at least 0.5 m behind the presenter')
        kwargs = {k: _finite(data[k], k) for k in ('gain_per_s', 'max_speed_m_s', 'max_turn_rad_s', 'heading_gain',
                                                  'deadband_m', 'turn_only_above_rad') if k in data}
        return cls(slots, **kwargs)

    def slot_world(self, index, presenter_pose):
        px, py, heading = presenter_pose[0], presenter_pose[1], presenter_pose[2]
        sx, sy = self.slots[index]
        c, s = math.cos(heading), math.sin(heading)
        return px + c * sx - s * sy, py + s * sx + c * sy

    def twist(self, index, state, presenter_pose):
        """Body twist (vx, vy, wz) steering robot ``index`` to its slot; zero inside the deadband."""
        tx, ty = self.slot_world(index, presenter_pose)
        x, y = state['position'][0], state['position'][1]
        yaw = _yaw_wxyz(state['orientation_wxyz'])
        ex, ey = tx - x, ty - y
        distance = math.hypot(ex, ey)
        if distance < self.deadband_m:
            # hold the presenter's heading in place so the formation stays oriented
            err = _wrap(presenter_pose[2] - yaw)
            wz = 0. if abs(err) < math.radians(8.) else max(-self.max_turn_rad_s, min(self.max_turn_rad_s, self.heading_gain * err))
            return (0., 0., wz)
        err = _wrap(math.atan2(ey, ex) - yaw)
        wz = max(-self.max_turn_rad_s, min(self.max_turn_rad_s, self.heading_gain * err))
        if abs(err) > self.turn_only_above_rad:
            return (0., 0., wz)
        vx = min(self.max_speed_m_s, self.gain_per_s * distance) * math.cos(err)
        return (max(0., vx), 0., wz)


DEFAULT_LIGHTING = {'dome_intensity': 20., 'sun_intensity': 100.}
"""Owner dome/sun intensities while a stage asset (with its own lights) is dressed in."""


@dataclass
class Choreography:
    presenter: PresenterPath
    formation: Formation
    walk: WalkCycle = field(default_factory=WalkCycle)
    presenter_asset: str | None = None
    stage_asset: str | None = None
    camera: dict | None = None
    lighting: dict = field(default_factory=lambda: dict(DEFAULT_LIGHTING))
    source: dict = field(default_factory=dict)

    @classmethod
    def load(cls, path, robot_count):
        from pathlib import Path
        path = Path(path)
        data = json.loads(path.read_text())
        presenter = PresenterPath.from_dict(data['presenter'])
        formation = Formation.from_dict(data['formation'], robot_count)
        walk = WalkCycle(**{k: _finite(v, k) for k, v in data.get('walk_cycle', {}).items()})
        camera = data.get('camera')
        if camera is not None:
            camera = dict(camera)
            mode = camera.setdefault('mode', 'static')
            if mode not in ('static', 'follow'):
                raise ValueError("camera mode must be 'static' or 'follow'")
            keys = ('eye', 'target') if mode == 'static' else ('offset', 'target_offset')
            for key in keys:
                if not isinstance(camera.get(key), list) or len(camera[key]) != 3:
                    raise ValueError(f'camera needs {keys[0]} and {keys[1]} triples')
                camera[key] = [_finite(v, key) for v in camera[key]]
            if mode == 'follow':
                camera['smooth_s'] = _finite(camera.get('smooth_s', 1.5), 'smooth_s')
                if camera['smooth_s'] < 0:
                    raise ValueError('smooth_s must be nonnegative')
                camera['eye'], camera['target'] = cls._follow_pose(camera, presenter.pose_at(0.))
            if 'focal_length_mm' in camera:
                camera['focal_length_mm'] = _finite(camera['focal_length_mm'], 'focal_length_mm')
                if not 4. <= camera['focal_length_mm'] <= 200.:
                    raise ValueError('focal_length_mm must be in 4..200')
        assets = {}
        for key in ('presenter_asset', 'stage_asset'):
            value = data.get(key)
            if value is not None:
                asset = (path.parent / value).resolve() if not Path(value).is_absolute() else Path(value)
                if not asset.is_file():
                    raise ValueError(f'{key} does not exist: {asset}')
                value = str(asset)
            assets[key] = value
        lighting = dict(DEFAULT_LIGHTING)
        for key, value in data.get('lighting', {}).items():
            if key not in lighting:
                raise ValueError(f'unknown lighting key {key}')
            lighting[key] = _finite(value, key)
            if lighting[key] < 0:
                raise ValueError('light intensities must be nonnegative')
        choreo = cls(presenter, formation, walk, assets['presenter_asset'], assets['stage_asset'], camera, lighting, data)
        choreo.spawn_positions()  # validates the start heading
        return choreo

    @staticmethod
    def _follow_pose(camera, presenter_pose, anchor=None):
        ax, ay = (presenter_pose[0], presenter_pose[1]) if anchor is None else anchor
        ox, oy, oz = camera['offset']
        tx, ty, tz = camera['target_offset']
        return [ax + ox, ay + oy, oz], [ax + tx, ay + ty, tz]

    def camera_pose_at(self, sim_time, dt):
        """Follow camera: eye/target around a first-order-lagged presenter anchor (visual only)."""
        cam = self.camera
        if cam is None or cam.get('mode') != 'follow':
            return None
        pose = self.presenter.pose_at(sim_time)
        anchor = getattr(self, '_camera_anchor', None)
        if anchor is None or dt is None or dt <= 0:
            anchor = (pose[0], pose[1])
        else:
            alpha = 1. - math.exp(-dt / cam['smooth_s']) if cam['smooth_s'] > 0 else 1.
            anchor = (anchor[0] + alpha * (pose[0] - anchor[0]), anchor[1] + alpha * (pose[1] - anchor[1]))
        self._camera_anchor = anchor
        return self._follow_pose(cam, pose, anchor)

    def spawn_positions(self):
        """World [x, y] of every formation slot at the presenter's start pose (robots face +x)."""
        start = self.presenter.pose_at(0.)
        if abs(_wrap(start[2])) > 1e-6:
            raise ValueError('the presenter must start walking along +x so the robots spawn facing it')
        return [list(self.formation.slot_world(i, start)) for i in range(len(self.formation.slots))]

    def receipt(self):
        return {'presenter': {'waypoints': list(map(list, self.presenter.waypoints)), 'speed_m_s': self.presenter.speed_m_s,
                              'start_delay_s': self.presenter.start_delay_s, 'dwell_s': list(self.presenter.dwell_s),
                              'length_m': self.presenter.length(),
                              'duration_s': self.presenter.duration_s(), 'asset': self.presenter_asset,
                              'pose_source': 'simulation (oracle); no perception',
                              'physics': 'kinematic visual proxy; no collision with the robots'},
                'formation': {'slots': list(map(list, self.formation.slots)), 'gain_per_s': self.formation.gain_per_s,
                              'max_speed_m_s': self.formation.max_speed_m_s, 'max_turn_rad_s': self.formation.max_turn_rad_s,
                              'heading_gain': self.formation.heading_gain, 'deadband_m': self.formation.deadband_m,
                              'spawn': self.spawn_positions()},
                'walk_cycle': self.walk.__dict__, 'stage_asset': self.stage_asset, 'camera': self.camera,
                'lighting': self.lighting,
                'scope': 'showcase choreography; scripted twists, not agent tasks; not physical admission'}

    def script_for(self, index):
        """Twist source for robot ``index``: evaluated by its controller inside control_at."""
        def source(sim_time, state):
            return self.formation.twist(index, state, self.presenter.pose_at(sim_time))
        return source


class PresenterProxy:
    """Kinematic visual proxy authored from ``presenter.usda``; no physics APIs, pose written per capture."""
    JOINTS = ('HipL', 'HipR', 'ShoulderL', 'ShoulderR')
    KNEES = {'KneeL': 'HipL/Knee', 'KneeR': 'HipR/Knee'}

    def __init__(self, stage, asset_path, *, root='/World/Presenter', referenced=False):
        from pxr import Gf, Usd, UsdGeom
        self.Gf, self.UsdGeom = Gf, UsdGeom
        prim = stage.GetPrimAtPath(root) if referenced else stage.DefinePrim(root, 'Xform')
        if not referenced:
            prim.GetReferences().AddReference(asset_path)
        if not prim or not prim.IsValid() or not prim.GetChildren():
            raise ValueError('presenter asset referenced nothing')
        for p in Usd.PrimRange(prim):
            if any(api.startswith('Physics') for api in p.GetAppliedSchemas()):
                raise ValueError(f'presenter proxy must carry no physics API: {p.GetPath()}')
        self.root = UsdGeom.Xformable(prim)
        self._root_ops = self._ops(prim)
        self._joint_ops = {}
        for name in self.JOINTS:
            self._joint_ops[name] = self._ops(stage.GetPrimAtPath(f'{root}/{name}'))
        for name, rel in self.KNEES.items():
            self._joint_ops[name] = self._ops(stage.GetPrimAtPath(f'{root}/{rel}'))
        self.last = None

    def _ops(self, prim):
        if not prim or not prim.IsValid():
            raise ValueError('presenter asset lacks a required joint prim')
        xf = self.UsdGeom.Xformable(prim)
        ops = {op.GetOpName(): op for op in xf.GetOrderedXformOps()}
        if 'xformOp:translate' not in ops or 'xformOp:rotateXYZ' not in ops:
            raise ValueError(f'{prim.GetPath()} must author translate and rotateXYZ ops')
        return ops

    def write(self, pose, joints):
        """Root translate/yaw and joint swings for one captured frame (degrees in USD)."""
        Gf = self.Gf
        x, y, heading = pose[0], pose[1], pose[2]
        self._root_ops['xformOp:translate'].Set(Gf.Vec3d(x, y, joints.get('bob', 0.)))
        self._root_ops['xformOp:rotateXYZ'].Set(Gf.Vec3f(0., 0., math.degrees(heading)))
        for name in self.JOINTS:
            self._joint_ops[name]['xformOp:rotateXYZ'].Set(Gf.Vec3f(0., math.degrees(joints[name]), 0.))
        for name in self.KNEES:
            self._joint_ops[name]['xformOp:rotateXYZ'].Set(Gf.Vec3f(0., math.degrees(joints[name]), 0.))
        self.last = {'x': x, 'y': y, 'heading_rad': heading, 'joints': joints}
        return self.last
