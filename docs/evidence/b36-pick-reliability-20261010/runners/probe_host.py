#!/usr/bin/env python3
"""Host-side MCP probe of the HTTPS listener (same CA/token the sandbox uses)."""
import http.client
import json
import ssl
import sys
import time
from pathlib import Path

D = Path.home() / ".cascade" / "nemoclaw"
ep = json.loads((D / "endpoint.json").read_text())
token = (D / "token").read_text().strip()
ctx = ssl.create_default_context(cafile=str(D / "ca.pem"))
session = None


def rpc(method, params=None, rid=[0]):
    global session
    rid[0] += 1
    frame = {"jsonrpc": "2.0", "id": rid[0], "method": method, **({"params": params} if params is not None else {})}
    h = {"Authorization": f"Bearer {token}", "Content-Type": "application/json", "Accept": "application/json"}
    if session:
        h["Mcp-Session-Id"] = session
    c = http.client.HTTPSConnection(ep["host"], 18740, timeout=400, context=ctx)
    t0 = time.monotonic()
    c.request("POST", "/mcp", body=json.dumps(frame), headers=h)
    r = c.getresponse()
    body = r.read()
    if method == "initialize":
        session = r.getheader("Mcp-Session-Id")
    return r.status, json.loads(body), time.monotonic() - t0


st, m, dt = rpc("initialize", {"protocolVersion": "2025-06-18", "capabilities": {}, "clientInfo": {"name": "probe"}})
print("initialize", st, m["result"]["serverInfo"], f"{dt:.2f}s")
st, m, dt = rpc("tools/list")
names = sorted(t["name"] for t in m["result"]["tools"])
print("tools", len(names), names[:8], "...")
for call in sys.argv[1:]:
    name, _, args = call.partition(":")
    st, m, dt = rpc("tools/call", {"name": name, "arguments": json.loads(args or "{}")})
    text = next(b["text"] for b in m["result"]["content"] if b["type"] == "text")
    print(f"{name} isError={m['result'].get('isError', False)} {dt:.1f}s {text[:600]}")
