"""Portable service configuration; no provider connection or robot construction."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

from .provider import RealtimeConfig


DEFAULTS = {"robot": "conversation_mock", "provider_url": None, "token_env": None,
            "allow_tools": [], "allow_motion": False, "barge_in": "stop_robot",
            "port": 8780, "config_dir": None, "run_dir": None, "run_root": None,
            "start_stopped": False, "intent_timeout_s": 10, "execution_timeout_s": 30,
            "provider_release_contract": None, "robot_lifecycle": None,
            # Split deployment (B51): the robot runtime runs in its own
            # cascade-robot-service process. Absent = built in-process (default).
            "robot_endpoint": None, "robot_token_env": None}


def configuration(args):
    values, digest = dict(DEFAULTS), None
    if args.config is not None:
        path = Path(args.config).resolve()
        raw = path.read_bytes()
        if len(raw) > 65536:
            raise ValueError("conversation configuration exceeds 64 KiB")

        def unique(pairs):
            result = {}
            for key, value in pairs:
                if key in result:
                    raise ValueError("duplicate conversation configuration field")
                result[key] = value
            return result

        data = json.loads(raw, object_pairs_hook=unique)
        if (type(data) is not dict or type(data.get("version")) is not int
                or data["version"] != 1 or set(data) - (set(DEFAULTS) | {"version"})):
            raise ValueError("conversation configuration requires version 1 and known fields")
        values.update({key: value for key, value in data.items() if key != "version"})
        for key in ("config_dir", "run_dir", "run_root"):
            if values[key] is not None:
                if type(values[key]) is not str or not values[key]:
                    raise ValueError(f"{key} must be a nonempty path")
                values[key] = str(path.parent / values[key])
        digest = hashlib.sha256(raw).hexdigest()
    for key in DEFAULTS:
        override = getattr(args, "allow_tool" if key == "allow_tools" else key, None)
        if override is not None:
            values[key] = override
    for key in ("robot", "provider_url", "barge_in"):
        if type(values[key]) is not str or not values[key]:
            raise ValueError(f"{key} must be a nonempty string")
    if values["barge_in"] not in {"stop_robot", "speech_only"}:
        raise ValueError("invalid interruption policy")
    if values["robot_lifecycle"] is not None and values["robot_lifecycle"] != "bounded_hand":
        raise ValueError("unknown conversation robot lifecycle")
    if values["robot_lifecycle"] is not None and values["barge_in"] != "stop_robot":
        raise ValueError("bounded hand activation requires stop_robot interruption")
    if values["token_env"] is not None and type(values["token_env"]) is not str:
        raise ValueError("token_env must name an environment variable")
    if values["robot_endpoint"] is not None:
        from ..robotics.endpoint import endpoint_origin
        values["robot_endpoint"] = endpoint_origin(values["robot_endpoint"])
        if type(values["robot_token_env"]) is not str or not values["robot_token_env"].isidentifier():
            raise ValueError("robot_endpoint requires robot_token_env naming an environment variable")
        if values["robot_lifecycle"] is not None or values["config_dir"] is not None:
            raise ValueError("a remote robot runtime takes no local robot lifecycle or profile directory")
    elif values["robot_token_env"] is not None:
        raise ValueError("robot_token_env requires robot_endpoint")
    for key in ("allow_motion", "start_stopped"):
        if type(values[key]) is not bool:
            raise ValueError(f"{key} must be boolean")
    if values["allow_motion"] and values["barge_in"] != "stop_robot":
        raise ValueError("motion conversations require stop_robot interruption")
    if type(values["port"]) is not int or not 0 <= values["port"] <= 65535:
        raise ValueError("port must be an integer in 0..65535")
    for key, maximum in (("intent_timeout_s", 60), ("execution_timeout_s", 300)):
        value = values[key]
        if type(value) not in (int, float) or not math.isfinite(value) or not 0 < value <= maximum:
            raise ValueError(f"{key} must be finite and in (0, {maximum}]")
    tools = values["allow_tools"]
    if (type(tools) is not list or len(tools) > 256
            or any(type(t) is not str or not 0 < len(t) <= 128 for t in tools)
            or len(set(tools)) != len(tools)):
        raise ValueError("allow_tools must contain unique tool names")
    for key in ("config_dir", "run_dir", "run_root"):
        value = values[key]
        if value is not None:
            if not isinstance(value, (str, Path)) or not str(value):
                raise ValueError(f"{key} must be a nonempty path")
            values[key] = Path(value).resolve()
    if (values["run_dir"] is None) == (values["run_root"] is None):
        raise ValueError("select exactly one of run_dir and run_root")
    RealtimeConfig(values["provider_url"], values["token_env"],
                   release_contract=values["provider_release_contract"])
    values["allow_tool"] = values.pop("allow_tools")
    values["service_config_sha256"] = digest
    return values
