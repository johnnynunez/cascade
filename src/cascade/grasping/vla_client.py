"""Client side of the openpi / LingBot-VLA websocket policy protocol.

The wire (identical in `openpi_client.websocket_client_policy` /
`openpi.serving.websocket_policy_server` and LingBot-VLA-v2's
`deploy/websocket_policy_server.py`):

  connect   ws://host:port, no compression, no frame size limit
  server -> one msgpack frame: the server's metadata dict
  client -> msgpack obs dict  (images uint8 HxWx3, state float32, prompt str)
  server -> msgpack dict      ({<action key>: (H, D) ndarray, "server_timing": {...}})
            or a TEXT frame carrying the server traceback, then close 1011
  GET /healthz -> 200 "OK"

ndarrays travel in openpi's msgpack-numpy form ({b"__ndarray__": True,
b"data", b"dtype", b"shape"}; numpy scalars as b"__npgeneric__"), NOT the
`msgpack-numpy` package's form, which is why the codec is reimplemented
here (17 lines, no pickle fallback for object arrays -- same as openpi).

The `vla` extra (`websockets`, `msgpack`) is imported lazily: nothing here is
loaded unless `grasp.executor: vla` is configured, and without the extra the
route refuses with `VLAUnavailable` naming it -- never a silent fallback to
the analytic pipeline.
"""

from __future__ import annotations

import numpy as np

from ..types import SkillError

#: what to install when the client is requested without its dependencies
EXTRA_HINT = "install the `vla` extra: uv pip install -e '.[vla]' (websockets, msgpack)"


class VLAUnavailable(SkillError):
    """The policy server cannot be used: extra missing or nothing answered."""


class VLAServerError(SkillError):
    """The server answered with an error (a text frame) or a malformed reply."""


class VLATimeout(SkillError):
    """A reply did not arrive inside its deadline."""


def require_deps():
    """(websockets.sync.client, msgpack), or VLAUnavailable naming the extra."""
    missing = []
    try:
        from websockets.sync import client as ws_client
    except ImportError:
        ws_client = None
        missing.append("websockets")
    try:
        import msgpack
    except ImportError:
        msgpack = None
        missing.append("msgpack")
    if missing:
        raise VLAUnavailable(
            f"grasp.executor: vla needs {', '.join(missing)} ({EXTRA_HINT})")
    return ws_client, msgpack


def _pack_array(obj):
    if isinstance(obj, (np.ndarray, np.generic)) and obj.dtype.kind in ("V", "O", "c"):
        raise ValueError(f"unsupported dtype on the policy wire: {obj.dtype}")
    if isinstance(obj, np.ndarray):
        return {b"__ndarray__": True, b"data": obj.tobytes(), b"dtype": obj.dtype.str,
                b"shape": obj.shape}
    if isinstance(obj, np.generic):
        return {b"__npgeneric__": True, b"data": obj.item(), b"dtype": obj.dtype.str}
    return obj


def _unpack_array(obj):
    if b"__ndarray__" in obj:
        return np.ndarray(buffer=obj[b"data"], dtype=np.dtype(obj[b"dtype"]), shape=obj[b"shape"])
    if b"__npgeneric__" in obj:
        return np.dtype(obj[b"dtype"]).type(obj[b"data"])
    return obj


def _impl():
    """msgpack's own Packer/unpackb, never the module attributes:
    `msgpack_numpy.patch()` (called by the GraspGen-X, HUG and occupancy
    clients in this same process) REPLACES `msgpack.packb/Packer/unpackb`
    with versions that encode arrays in msgpack-numpy's layout (and pickle
    object arrays) -- which an openpi / LingBot server cannot decode."""
    try:
        from msgpack import _cmsgpack as impl
    except ImportError:  # pure-Python build
        from msgpack import fallback as impl
    return impl


def pack(obj) -> bytes:
    """msgpack with openpi's ndarray encoding."""
    return _impl().Packer(default=_pack_array).pack(obj)


def unpack(data: bytes):
    return _impl().unpackb(data, object_hook=_unpack_array)


class PolicyClient:
    """One synchronous websocket session with a policy server.

    `connect()` returns the server's metadata frame; `infer(obs, timeout_s)`
    is one request/reply with its own deadline. A timeout leaves the session
    in an unknown state, so it is closed: the caller ends the episode.
    """

    def __init__(self, host: str, port: int, *, connect_timeout_s: float = 1.0):
        self.host, self.port = str(host), int(port)
        self.connect_timeout_s = float(connect_timeout_s)
        self.metadata: dict | None = None
        self._ws = None
        self._stack = None

    @property
    def uri(self) -> str:
        return f"ws://{self.host}:{self.port}"

    def connect(self) -> dict:
        import contextlib

        ws_client, _ = require_deps()
        stack = contextlib.ExitStack()
        try:
            # entered as a context manager: what websockets >= 17 expects of a
            # direct connection, and a plain close-on-exit on 13..16
            ws = stack.enter_context(ws_client.connect(
                self.uri, compression=None, max_size=None,
                open_timeout=self.connect_timeout_s, close_timeout=self.connect_timeout_s))
        except Exception as exc:  # noqa: BLE001 -- refused, DNS, handshake: all "not answering"
            stack.close()
            raise VLAUnavailable(
                f"no VLA policy server answered at {self.host}:{self.port} "
                f"({type(exc).__name__}: {exc})") from exc
        self._ws, self._stack = ws, stack
        try:
            meta = self._recv(self.connect_timeout_s, what="metadata frame")
        except SkillError as exc:
            self.close()
            raise VLAUnavailable(f"{self.host}:{self.port} accepted the connection but is not a "
                                 f"policy server: {exc}") from exc
        if not isinstance(meta, dict):
            self.close()
            raise VLAUnavailable(f"{self.host}:{self.port} sent a non-dict metadata frame "
                                 f"({type(meta).__name__}); not an openpi/LingBot policy server")
        self.metadata = meta
        return meta

    def _recv(self, timeout_s: float, *, what: str):
        try:
            data = self._ws.recv(timeout=max(float(timeout_s), 0.0))
        except TimeoutError as exc:
            self.close()
            raise VLATimeout(f"no {what} within {float(timeout_s):.2f}s") from exc
        except Exception as exc:  # noqa: BLE001 -- connection closed mid-episode
            self.close()
            raise VLAServerError(f"policy server connection lost ({type(exc).__name__}: {exc})") from exc
        if isinstance(data, str):
            self.close()
            raise VLAServerError(f"policy server error:\n{data.strip()[-800:]}")
        try:
            return unpack(data)
        except Exception as exc:  # noqa: BLE001 -- garbage on the wire is a server fault
            self.close()
            raise VLAServerError(f"undecodable policy reply ({type(exc).__name__}: {exc})") from exc

    def infer(self, obs: dict, *, timeout_s: float) -> dict:
        if self._ws is None:
            raise VLAServerError("policy session is closed")
        data = pack(obs)  # an unsupported dtype is a caller bug: ValueError, not a server fault
        try:
            self._ws.send(data)
        except Exception as exc:  # noqa: BLE001
            self.close()
            raise VLAServerError(f"policy server connection lost ({type(exc).__name__}: {exc})") from exc
        reply = self._recv(timeout_s, what="action chunk")
        if not isinstance(reply, dict):
            raise VLAServerError(f"policy reply is not a dict ({type(reply).__name__})")
        return reply

    def close(self) -> None:
        stack, self._stack, self._ws = self._stack, None, None
        if stack is not None:
            try:
                stack.close()
            except Exception:  # noqa: BLE001 -- closing a dead socket
                pass


def probe(host: str, port: int, *, timeout_s: float = 1.0) -> dict:
    """{"answered": bool, "detail": str, "metadata": dict | None}. Never raises:
    a connect + metadata frame within `timeout_s` is the protocol's own
    liveness signal (openpi and LingBot both send it on every connection)."""
    client = PolicyClient(host, port, connect_timeout_s=timeout_s)
    try:
        meta = client.connect()
    except SkillError as exc:
        return {"answered": False, "detail": str(exc), "metadata": None}
    finally:
        client.close()
    return {"answered": True, "detail": describe_metadata(host, port, meta), "metadata": meta}


def describe_metadata(host: str, port: int, meta: dict | None) -> str:
    """One short line: where, and whatever the server says it is."""
    parts = []
    for key, value in (meta or {}).items():
        if isinstance(value, (str, int, float, bool)) and len(parts) < 4:
            parts.append(f"{key}={value}")
    return f"{host}:{port}" + (f" ({', '.join(parts)})" if parts else "")
