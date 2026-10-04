"""Opt-in observation of the pinned HF provider's actual pipeline release.

A WebSocket close acknowledges transport shutdown, not SESSION_END propagation.
This contract observes /v1/pool on the same provider authority. It never releases
a unit, reconnects, or promotes a stuck/unknown pool state to success.
"""
from __future__ import annotations

import asyncio
import json
import math
from urllib.parse import urlsplit, urlunsplit


def pool_url(websocket_url):
    url = urlsplit(websocket_url)
    if url.path != "/v1/realtime" or url.query:
        raise ValueError("HF pool release requires the explicit /v1/realtime endpoint without query")
    return urlunsplit(("https" if url.scheme == "wss" else "http", url.netloc, "/v1/pool", "", ""))


def remaining(deadline):
    value = deadline - asyncio.get_running_loop().time()
    if value <= 0:
        raise TimeoutError("provider pool observation deadline expired")
    return value


def validate_pool(value):
    if (type(value) is not dict or set(value) != {"size", "in_use", "units"}
            or type(value["size"]) is not int or not 1 <= value["size"] <= 256
            or type(value["in_use"]) is not int or type(value["units"]) is not list
            or len(value["units"]) != value["size"]):
        raise ValueError("invalid HF pool snapshot")
    indices, sessions = set(), set()
    for unit in value["units"]:
        if (type(unit) is not dict or type(unit.get("index")) is not int or unit["index"] < 0
                or unit["index"] in indices or unit.get("state") not in {"idle", "active", "draining", "stuck"}):
            raise ValueError("invalid HF pool unit")
        indices.add(unit["index"])
        extra = {"draining_for_s"} if unit["state"] == "draining" else (
            {"draining_for_s", "stuck_for_s"} if unit["state"] == "stuck" else set())
        if set(unit) != {"index", "state", "session_id"} | extra or any(
                type(unit[key]) not in (int, float) or not math.isfinite(unit[key]) or unit[key] < 0
                for key in extra):
            raise ValueError("invalid HF pool state fields")
        session = unit.get("session_id")
        if unit["state"] == "idle":
            if session is not None or "session_id" not in unit:
                raise ValueError("idle HF pool unit still has a session")
        elif type(session) is not str or not 0 < len(session) <= 256 or session in sessions:
            raise ValueError("invalid HF pool session identity")
        else:
            sessions.add(session)
    if value["in_use"] != len(sessions):
        raise ValueError("HF pool occupancy disagrees with its units")
    return value


def _unique(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate HF pool JSON field")
        result[key] = value
    return result


async def read_pool(client, url, headers, deadline, max_bytes):
    async def read():
        # Redirects would sever the same-authority relationship and could carry
        # provider credentials elsewhere. Bound the body before parsing it.
        async with client.get(url, headers=headers, allow_redirects=False) as response:
            if response.status != 200 or response.content_type != "application/json":
                raise ValueError("HF pool endpoint did not return a JSON snapshot")
            data = bytearray()
            async for chunk in response.content.iter_chunked(4096):
                data.extend(chunk)
                if len(data) > max_bytes:
                    raise ValueError("HF pool snapshot exceeds event bound")
            return validate_pool(json.loads(data, object_pairs_hook=_unique))
    timeout = remaining(deadline)
    result = await asyncio.wait_for(read(), timeout)
    remaining(deadline)
    return result


async def wait_released(client, url, headers, binding, deadline, max_bytes):
    observations = 0
    while True:
        snapshot = await read_pool(client, url, headers, deadline, max_bytes)
        observations += 1
        if (snapshot["size"] != binding["size"]
                or sorted(u["index"] for u in snapshot["units"]) != binding["indices"]):
            raise ValueError("HF pool layout changed before release")
        unit = next((u for u in snapshot["units"] if u["index"] == binding["index"]), None)
        if unit is None:
            raise ValueError("bound HF pool unit disappeared")
        if any(u["session_id"] == binding["session_id"] and u["index"] != binding["index"]
               for u in snapshot["units"]):
            raise ValueError("bound HF pool session moved to another unit")
        if unit["state"] == "idle":
            released = asyncio.get_running_loop().time()
            if released >= deadline:
                raise TimeoutError("provider pool release completed after its original deadline")
            return {"ok": True, "complete": True, "contract": "hf_pool",
                    "unit_index": binding["index"], "session_id": binding["session_id"],
                    "observations": observations, "deadline_monotonic_s": deadline,
                    "released_monotonic_s": released}
        if unit["session_id"] != binding["session_id"] or unit["state"] == "stuck":
            raise ValueError("bound HF pool session changed or became stuck")
        # Observation cadence, not a delay chosen to admit a reconnect. Every
        # observation and wait consumes the one pre-close I/O deadline.
        await asyncio.sleep(min(.05, remaining(deadline)))
