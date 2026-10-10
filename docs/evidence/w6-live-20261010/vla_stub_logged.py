#!/usr/bin/env python
"""live-w6: scripts/serve_vla_stub.py's own ScriptedPolicy + StubPolicyServer (imported, unchanged), plus a
request log: one JSON line per observation the cascade VLA client sent (call number, monotonic time, prompt,
state vector, image shape/dtype) and the chunk replied. Same wire, same scripted chunks."""
import json
import sys
import time
from pathlib import Path

import numpy as np

WT = Path("/home/johnny/Projects/demo/cascade-lab/HERMES_BACKLOG_20261009/live-w6/cascade")
sys.path.insert(0, str(WT / "scripts"))
sys.path.insert(0, str(WT / "src"))
from serve_vla_stub import ScriptedPolicy, StubPolicyServer  # noqa: E402

port, chunks_path, log_path = int(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
chunks = json.loads(chunks_path.read_text())
policy = ScriptedPolicy(chunks)
log = log_path.open("a", buffering=1)


def on_call(obs, n):
    img = obs.get("observation/image")
    log.write(json.dumps({
        "call": n, "t_mono": round(time.monotonic(), 3), "prompt": obs.get("prompt"),
        "state": [round(float(x), 4) for x in np.asarray(obs.get("observation/state")).ravel()],
        "image_shape": list(np.asarray(img).shape) if img is not None else None,
        "image_dtype": str(np.asarray(img).dtype) if img is not None else None,
        "chunk": chunks[min(n, len(chunks)) - 1]}) + "\n")


policy.on_call = on_call
server = StubPolicyServer(policy, "127.0.0.1", port, metadata={"policy": "scripted-stub", "chunks": len(chunks)}).start()
print(f"[vla-stub] ws://127.0.0.1:{server.port} ({len(chunks)} scripted chunk(s), action key 'actions') + request log {log_path}",
      flush=True)
while True:
    time.sleep(3600)
