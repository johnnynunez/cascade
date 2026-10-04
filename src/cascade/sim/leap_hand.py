"""Explicit CPU MuJoCo adapter for the source-pinned right LEAP hand.

The authored experiment uses 6 Nm/rad position gain, a 0.5 Nm effort cap and
a 2 ms timestep. Those
are simulation limits, not a hardware calibration or Dynamixel motor model.
"""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import threading
import uuid
import xml.etree.ElementTree as ET

from ..control.hand import HandContact, HandFault, HandLimits, HandSample


RECIPE = "leap_right_bounded_free_motion_v2"
JOINTS = tuple(f"{finger}_{joint}" for finger in ("if", "mf", "rf")
    for joint in ("mcp", "rot", "pip", "dip")) + ("th_cmc", "th_axl", "th_mcp", "th_ipl")


def sha(data):
    return hashlib.sha256(data).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def asset_manifest():
    return json.loads(Path(__file__).with_name("leap_hand_assets.json").read_text())


class LeapHandBackend:
    synthetic = False

    def __init__(self, asset_root):
        root = Path(asset_root).resolve()
        manifest = asset_manifest()
        assets = {}
        for name, expected in manifest["files"].items():
            path = (root/name).resolve()
            if not path.is_relative_to(root):
                raise HandFault("hand asset escaped the declared source root")
            data = path.read_bytes()
            if sha(data) != expected:
                raise HandFault("hand source asset differs: "+name)
            assets[name] = data
        # Imports and construction happen only after explicit asset verification.
        import mujoco
        import numpy as np
        if mujoco.__version__ != "3.10.0":
            raise HandFault("this hand adapter requires the declared MuJoCo 3.10.0 recipe")
        self._mj, self._np = mujoco, np
        xml = ET.fromstring(assets["right_hand.xml"])
        xml.find("option").set("timestep", ".002")
        for actuator in xml.findall("./actuator/position"):
            actuator.set("kp", "6")
            actuator.set("forcelimited", "true")
            actuator.set("forcerange", "-.5 .5")
        authored = ET.tostring(xml)
        model = mujoco.MjModel.from_xml_string(authored,
            assets={name: data for name, data in assets.items() if name.startswith("assets/")})
        if (model.nq != 16 or model.nv != 16 or model.nu != 16 or model.na != 0
                or model.njnt != 16 or model.nplugin != 0 or model.ntendon != 0
                or tuple(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(16)) != JOINTS
                or not np.all(model.jnt_type == mujoco.mjtJoint.mjJNT_HINGE)
                or not np.array_equal(model.jnt_qposadr, np.arange(16))
                or not np.array_equal(model.jnt_dofadr, np.arange(16))
                or not np.array_equal(model.actuator_trnid[:, 0], np.arange(16))
                or not np.all(model.actuator_trntype == mujoco.mjtTrn.mjTRN_JOINT)
                or not np.all(model.actuator_gear == [1., 0., 0., 0., 0., 0.])
                or not np.all(model.actuator_forcelimited)
                or not np.all(model.actuator_forcerange == [-.5, .5])
                or not np.all(model.actuator_gainprm[:, 0] == 6.)
                or not np.all(model.actuator_biasprm[:, 1] == -6.)
                or not np.all(model.actuator_biasprm[:, 2] == -.01)
                or not np.all(model.actuator_ctrllimited) or not np.all(model.jnt_limited)):
            raise HandFault("compiled hand transmission or bounded-actuator inventory differs")
        self.joint_names = JOINTS
        self.geom_names = tuple(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i)
                               for i in range(model.ngeom))
        if any(not name for name in self.geom_names) or len(set(self.geom_names)) != len(self.geom_names):
            raise HandFault("every hand collider must have a unique name")
        self.limits = HandLimits(tuple(model.jnt_range[:, 0]), tuple(model.jnt_range[:, 1]))
        if model.opt.timestep != self.limits.dt_s:
            raise HandFault("compiled hand timestep differs")
        # Portable model content, explicit limits and actual loaded SDK identity.
        packed = np.empty(mujoco.mj_sizeModel(model), dtype=np.uint8)
        mujoco.mj_saveModel(model, buffer=packed)
        library = Path(mujoco.__file__).parent
        sdk_files = sorted(path for path in library.iterdir() if path.is_file()
            and (path.suffix in {".py", ".so", ".dylib", ".dll", ".pyd"} or ".so." in path.name))
        src = Path(__file__).parents[1]
        self.document = {"recipe": RECIPE, "assets": manifest, "authored_xml_sha256": sha(authored),
            "compiled_model_sha256": sha(packed.tobytes()), "mujoco_version": mujoco.__version__,
            "sdk": {path.name: sha(path.read_bytes()) for path in sdk_files},
            "sources": {name: sha((src/name).read_bytes()) for name in
                ("control/hand.py", "sim/leap_hand.py", "skills/hand_runtime.py", "apps/hand_runtime.py",
                 "sensing/models.py", "robotics/contracts.py")},
            "limits": asdict(self.limits), "joint_names": JOINTS, "geom_names": self.geom_names,
            "position_servo": {"kp_nm_per_rad": 6., "kv_nm_s_per_rad": .01},
            "root": "fixed", "stop": "hold last position-servo targets",
            "contact_scope": "all enabled collision pairs of the pinned model",
            "constraint_phase": "dynamics evaluation before integration; separate from advanced joint state",
            "hardware_admission": False, "tactile_calibration": None}
        self.model_sha256, self.epoch = sha(canonical(self.document)), uuid.uuid4().hex
        self._model, self._data = model, mujoco.MjData(model)
        self.initial_targets = tuple(map(float, self._data.qpos))
        self._step, self._thread = 0, None
        self._command = self.initial_targets
        self._callbacks_clear()

    def _callbacks_clear(self):
        for name in ("control", "passive", "act_dyn", "act_gain", "act_bias", "sensor"):
            if getattr(self._mj, "get_mjcb_"+name)() is not None:
                raise HandFault("hand recipe refuses external MuJoCo callbacks")

    def upload(self, targets):
        ident = threading.get_ident()
        if self._thread is None:
            self._thread = ident
        if self._thread != ident:
            raise HandFault("hand native write escaped its single solver owner")
        self._callbacks_clear()
        if (abs(self._data.time-self._step*self.limits.dt_s) > 1e-9
                or self._model.opt.timestep != self.limits.dt_s
                or self._np.any(self._data.qfrc_applied) or self._np.any(self._data.xfrc_applied)):
            raise HandFault("hand physics clock, timestep or external wrench changed")
        self._data.ctrl[:] = targets
        self._command = tuple(targets)

    def advance(self, generation, clock):
        if self._thread != threading.get_ident():
            raise HandFault("hand solve escaped its writer")
        model, data, np, mj = self._model, self._data, self._np, self._mj
        before = float(data.time)
        mj.mj_step(model, data)
        self._step += 1
        captured = clock()
        if np.any(data.warning.number):
            raise HandFault("MuJoCo reported a hand warning or capacity/solver failure")
        if data.ncon > 4096 or data.nefc < 0:
            raise HandFault("hand contact ledger exceeds the bounded observation capacity")
        contacts = []
        for index in range(data.ncon):
            contact = data.contact[index]
            a, b = int(contact.geom[0]), int(contact.geom[1])
            address = int(contact.efc_address)
            if not 0 <= a < model.ngeom or not 0 <= b < model.ngeom or address >= data.nefc:
                raise HandFault("hand contact identity/constraint is incomplete")
            frame = np.asarray(contact.frame).reshape(3, 3)
            if not np.allclose(frame@frame.T, np.eye(3), rtol=0, atol=1e-8):
                raise HandFault("invalid hand contact frame")
            wrench = np.zeros(6)
            mj.mj_contactForce(model, data, index, wrench)
            contacts.append(HandContact(index, self.geom_names[a], self.geom_names[b], address,
                tuple(map(float, contact.pos)), tuple(map(float, frame[0])),
                tuple(map(float, frame.T@wrench[:3])), float(wrench[0])))
        return HandSample(self.model_sha256, self.epoch, generation, self._step,
            float(data.time), before, captured, tuple(map(float, data.qpos)), tuple(map(float, data.qvel)),
            tuple(map(float, data.qfrc_actuator)), self._command, tuple(contacts))
