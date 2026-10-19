import json
import time
from urllib.request import ProxyHandler, Request, build_opener


def send(base, payload):
    body = json.dumps(payload).encode()
    opener = build_opener(ProxyHandler({}))
    t0 = time.time()
    with opener.open(Request(base + "/chat/completions", data=body, headers={"Content-Type": "application/json"}),
                     timeout=300) as resp:
        wire = resp.read()
    dt = time.time() - t0
    usage = None
    if payload.get("stream"):
        calls, finish, text = {}, None, ""
        for line in wire.splitlines():
            if not line.startswith(b"data: ") or line == b"data: [DONE]":
                continue
            ev = json.loads(line[6:])
            usage = ev.get("usage") or usage
            for ch in ev.get("choices", []):
                finish = ch.get("finish_reason") or finish
                dl = ch.get("delta", {})
                text += dl.get("content") or ""
                for c in dl.get("tool_calls") or []:
                    cur = calls.setdefault(c["index"], {"name": "", "arguments": ""})
                    fn = c.get("function") or {}
                    cur["name"] += fn.get("name") or ""
                    cur["arguments"] += fn.get("arguments") or ""
        calls = list(calls.values())
    else:
        d = json.loads(wire)
        usage = d.get("usage")
        ch = d["choices"][0]
        finish, text = ch["finish_reason"], ch["message"].get("content") or ""
        calls = [{"name": c["function"]["name"], "arguments": c["function"]["arguments"]}
                 for c in ch["message"].get("tool_calls") or []]
    return {"finish": finish, "text": text, "calls": calls, "s": round(dt, 2),
            "prompt_tokens": (usage or {}).get("prompt_tokens")}


