"""Supplementary prefix / cold-prefill benchmark for the brain comparison (throwaway).

Why: agent_eval/bench_agent.py has two blind spots found while reading its results:
  1. Its "new_session" case never changed the session ids (`prep(..., session={})` skips the substitution because an
     empty dict is falsy), so that row measured an identical repeated prompt, not a new chat.
  2. Its unique nonce sits at the start of the system CONTENT. Bolt's chat template renders the system content
     before the 49 tools, so Bolt's "cold" calls were cold; Qwen3.8's template renders the tools FIRST, so Qwen's
     "cold" calls still hit the ~8.3k-token tool block (vLLM prefix hits 8,320 of 12,485 tokens).
This script measures, one request at a time, streamed, temperature 0, thinking off, max_tokens 1 (prefill + 1 token):
  cold_<k>   nonce at the start of the system content AND of the first tool's description -> nothing reusable for
             either template; k = 13k (request 41), 15k (43, +image), 18k (51, +image), 32k (41 + pasted log); 2 runs
  newchat    per trial: A = request 41 with session ids S1 under a fresh nonce N (system content only: the tools stay
             byte-identical, as in production), B = the same with NEW session ids S2 (what a new OpenClaw chat sends),
             C = A's session S1 with a different order ("Reset the scene."): 3 trials
  BRAIN_URL=... BRAIN_MODEL=... BRAIN_KEY_FILE=... COLD=1 python bench_prefix2.py out.json
"""
import copy
import json
import os
import pathlib
import statistics
import sys
import time
import uuid

sys.path.insert(0, "/home/spark/Projects/perception-bakeoff/agent_eval")
import bench_agent as B  # noqa: E402  (body(), call(), pad_body(), _SESS, MODEL)


def with_session(b, ids):
    b = copy.deepcopy(b)
    s = b["messages"][0]
    s["content"] = B._SESS.sub(lambda m: (m.group(1) or m.group(3)) + ids.setdefault(m.group(2) or m.group(4), str(uuid.uuid4())),
                               s["content"])
    return b


def nonce_sys(b, n):
    b = copy.deepcopy(b)
    s = b["messages"][0]
    s["content"] = s["content"].replace("cascade/nemotron-3.5-bolt", f"cascade/{B.MODEL}")
    s["content"] = f"Request {n}.\n" + s["content"]
    return b


def nonce_all(b, n):
    b = nonce_sys(b, n)
    f = b["tools"][0]["function"]
    f["description"] = f"[{n}] " + f.get("description", "")
    return b


def main(out):
    rep = {"url": B.URL, "model": B.MODEL, "date": time.strftime("%Y-%m-%d %H:%M"), "cold": {}, "newchat": []}
    B.call({"messages": [{"role": "user", "content": "Say OK."}]}, 4)  # warm-up
    nn = lambda: uuid.uuid4().hex[:12]  # noqa: E731
    if os.environ.get("COLD", "1") == "1":
        mk = {"cold_13k": lambda: B.body(41), "cold_15k": lambda: B.body(43), "cold_18k": lambda: B.body(51),
              "cold_32k": lambda: B.pad_body(B.body(41))}
        for name, f in mk.items():
            rows = [B.call(nonce_all(f(), nn()), 1) for _ in range(2)]
            rep["cold"][name] = {"runs": rows, "prefill_s": statistics.median([r["total_s"] for r in rows]),
                                 "prompt_tokens": rows[0].get("prompt_tokens"), "prefix_hit": [r.get("prefix_hit") for r in rows]}
            print(name, json.dumps({k: v for k, v in rep["cold"][name].items() if k != "runs"}), flush=True)
    b41 = B.body(41)
    for t in range(3):
        n = nn(); s1, s2 = {}, {}
        a = B.call(with_session(nonce_sys(b41, n), s1), 1)
        bb = B.call(with_session(nonce_sys(b41, n), s2), 1)
        c_body = with_session(nonce_sys(b41, n), s1)
        c_body["messages"][-1]["content"] = c_body["messages"][-1]["content"].replace("What do you see on the table?", "Reset the scene.")
        c = B.call(c_body, 1)
        assert s1 and s2 and s1 != s2, "session ids were not substituted"
        rep["newchat"].append({"first_call": a, "new_chat_same_system": bb, "same_chat_new_order": c})
        print("newchat", t, json.dumps({k: (v["total_s"], v["prompt_tokens"], v["prefix_hit"]) for k, v in rep["newchat"][-1].items()}),
              flush=True)
    pathlib.Path(out).write_text(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main(sys.argv[1])
