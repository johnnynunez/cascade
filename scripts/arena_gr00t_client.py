"""GR00T remote-policy clients for the Arena humanoid bridge that are correct on Newton.

Runs INSIDE Isaac Lab-Arena's Python environment (``scripts/arena_locomanip_bridge.py`` puts this
directory on ``sys.path``); CASCADE never imports it. Two fixes on top of Arena 0.3.1's
``Gr00tRemoteClosedloopPolicy`` (observation building, joint remapping, action chunking and
everything else stay Arena's own code):

1. Simulator joint order. Arena's G1 GR00T policy YAML maps simulator joints with a hard-coded
   index table (``isaaclab_arena_gr00t/embodiments/g1/43dof_joint_space.yaml``) in PhysX's
   breadth-first articulation order (left_hip_pitch=0, right_hip_pitch=1, waist_yaw=2, ...).
   Newton orders the same articulation depth-first (left leg 0-5, right leg 6-11, waist 12-14,
   left arm 15-21, ...). With the PhysX table on Newton the policy reads scrambled
   proprioception and writes arm targets to the wrong joints (observed: arms drop, the robot
   freezes at the shelf, no error and no success). ``sync_joint_order`` rebuilds both
   name -> index tables from the live ``scene["robot"].data.joint_names`` (the order of the
   ``robot_joint_pos`` observation and of the whole-body action's joint targets). On PhysX it
   reproduces the YAML exactly; either way the result is reported in the bridge's hello.
2. GR00T N1.7 replies. An N1.7 ``run_gr00t_server.py`` accepts the N1.6 client's requests
   unchanged but answers with msgpack-numpy envelopes, which the N1.6 client in Arena's
   submodule leaves undecoded (same decoder as Arena's upstream N1.7 branch). Only numeric
   arrays are decoded; object payloads are refused, never unpickled.

Policy types (``--policy_type``):
  arena_gr00t_client.Gr00tN17SimOrderRemotePolicy   GR00T N1.7 server (policy YAML with
                                                    action_horizon / action_chunk_length 40)
  arena_gr00t_client.Gr00tN16SimOrderRemotePolicy   GR00T N1.6 server, Arena's stock YAML
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from isaaclab_arena.assets.register import register_policy
from isaaclab_arena.policy.policy_base import PolicyBase
from isaaclab_arena_gr00t.policy.gr00t_remote_closedloop_policy import (
    Gr00tRemoteClosedloopPolicy,
    Gr00tRemoteClosedloopPolicyCfg,
)

NUMPY_ARRAY_MARKER = b"nd"


def decode_n1d7_response(value: Any) -> Any:
    """Decode numeric msgpack-numpy arrays (and N1.7 ModalityConfig envelopes) recursively."""
    if isinstance(value, dict):
        if "__ModalityConfig__" in value:
            from gr00t.data.types import ModalityConfig

            return ModalityConfig(**value["as_json"])
        if NUMPY_ARRAY_MARKER in value:
            dtype = np.dtype(value[b"type"])
            if dtype.kind not in "biufc":  # numeric only; never unpickle object payloads
                raise ValueError(f"unsupported GR00T response dtype: {dtype}")
            array = np.frombuffer(value[b"data"], dtype=dtype).copy()
            return array.reshape(value[b"shape"]) if value[NUMPY_ARRAY_MARKER] else array[0]
        return {key: decode_n1d7_response(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(decode_n1d7_response(item) for item in value)
    return value


class _SimJointOrderMixin:
    """Name-based simulator joint tables (see the module docstring, fix 1)."""

    joint_order: dict | None = None

    def sync_joint_order(self, joint_names) -> dict:
        names = list(joint_names)
        sim = {name: i for i, name in enumerate(names)}
        table = dict(self.robot_state_joints_config)
        if set(sim) != set(table):
            raise RuntimeError(f"robot joints {sorted(set(sim) ^ set(table))} differ from the policy joint table")
        moved = sum(1 for name in sim if sim[name] != table[name])
        self.robot_state_joints_config = sim
        self.robot_action_joints_config = {name: sim[name] for name in self.robot_action_joints_config}
        self.joint_order = {"source": "live articulation joint names", "joints": len(sim),
                            "yaml_table_matches": moved == 0, "joints_at_other_indices": moved}
        print(f"[gr00t_client] sim joint order {'matches' if moved == 0 else 'DIFFERS FROM'} the policy YAML table "
              f"({moved}/{len(sim)} joints at other indices); using the live articulation order", flush=True)
        return self.joint_order

    def get_action(self, env, observation):
        if self.joint_order is None:
            self.sync_joint_order(env.unwrapped.scene["robot"].data.joint_names)
        return super().get_action(env, observation)


@dataclass
class Gr00tN16SimOrderRemotePolicyCfg(Gr00tRemoteClosedloopPolicyCfg):
    """Arena's GR00T remote policy fields (a separate type: Arena maps each config type to one policy)."""


@register_policy
class Gr00tN16SimOrderRemotePolicy(_SimJointOrderMixin, Gr00tRemoteClosedloopPolicy,
                                   PolicyBase[Gr00tN16SimOrderRemotePolicyCfg]):
    name = "cascade_gr00t_n16_simorder_remote"
    gr00t_release = "N1.6"


@dataclass
class Gr00tN17SimOrderRemotePolicyCfg(Gr00tRemoteClosedloopPolicyCfg):
    """Arena's GR00T remote policy fields (a separate type: Arena maps each config type to one policy)."""


@register_policy
class Gr00tN17SimOrderRemotePolicy(_SimJointOrderMixin, Gr00tRemoteClosedloopPolicy,
                                   PolicyBase[Gr00tN17SimOrderRemotePolicyCfg]):
    name = "cascade_gr00t_n17_simorder_remote"
    gr00t_release = "N1.7"

    def __init__(self, config: Gr00tN17SimOrderRemotePolicyCfg):
        super().__init__(config)
        client = self._client
        raw_get_action = client.get_action

        def get_action(*args, **kwargs):
            return decode_n1d7_response(raw_get_action(*args, **kwargs))

        client.get_action = get_action
        print(f"[gr00t_client] N1.7 reply decoding on; action_horizon={self.policy_config.action_horizon} "
              f"chunk={self.action_chunk_length} server={config.remote_host}:{config.remote_port}", flush=True)
