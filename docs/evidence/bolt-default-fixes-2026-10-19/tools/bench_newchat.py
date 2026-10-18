"""New-chat and same-chat prefix reuse of the agent prompt (throwaway, PR perf/bolt-default-fixes).

  BRAIN_URL=... BRAIN_MODEL=nemotron-3.5-bolt BRAIN_KEY_FILE=... python bench_newchat.py OUT.json [trials] [replays]

Uses the comparison's agent_eval/bench_agent.py (OpenClaw request bodies 41..51 of the Bolt take: system text with
OpenClaw's STABLE/DYNAMIC prompt-cache markers, 49 tools) and the session-id substitution of
out/brain-comparison-2026-10-18/scripts/bench_prefix2.py. One request at a time, streamed, temperature 0, thinking off.
  newchat     per trial: A = body 41 under a fresh nonce (start of the system text) and session ids S1 (cold),
              B = the same with NEW session ids S2 (what the first call of a new OpenClaw chat sends),
              C = A's session S1 with another order ("Reset the scene.": same chat, next order): max_tokens 1
  replay      the take's six agent calls (41, 43, 44, 47, 49, 51) of one chat in order, twice (fresh nonce each):
              follow-up calls of one chat; max_tokens 1 (2 passes unless [replays] says more)
  turn        body 41 streamed to completion (max 512 tokens), fresh nonce: the agent's tool call still parses
Each row: total_s (wall), ttft_s, prompt_tokens, prefix_hit (llama.cpp timings.cache_n), tool_calls.
"""
import json
import pathlib
import statistics
import sys
import time
import uuid

sys.path.insert(0, "/home/spark/Projects/perception-bakeoff/agent_eval")
sys.path.insert(0, "/home/spark/Projects/perception-bakeoff/out/brain-comparison-2026-10-18/scripts")
import bench_agent as B  # noqa: E402
from bench_prefix2 import nonce_sys, with_session  # noqa: E402


def row(r):
    return {k: r.get(k) for k in ("total_s", "ttft_s", "prompt_tokens", "prefix_hit", "tool_calls", "text_head", "error")}


def main(out, trials=3, replays=2):
    rep = {"url": B.URL, "model": B.MODEL, "date": time.strftime("%Y-%m-%d %H:%M"), "newchat": [], "replay": [], "turn": []}
    B.call({"messages": [{"role": "user", "content": "Say OK."}]}, 4)  # warm-up
    nn = lambda: uuid.uuid4().hex[:12]  # noqa: E731
    b41 = B.body(41)
    for t in range(trials):
        n = nn(); s1, s2 = {}, {}
        a = B.call(with_session(nonce_sys(b41, n), s1), 1)
        b = B.call(with_session(nonce_sys(b41, n), s2), 1)
        cb = with_session(nonce_sys(b41, n), s1)
        cb["messages"][-1]["content"] = cb["messages"][-1]["content"].replace("What do you see on the table?", "Reset the scene.")
        c = B.call(cb, 1)
        assert s1 and s2 and s1 != s2, "session ids were not substituted"
        rep["newchat"].append({"cold": row(a), "new_chat": row(b), "same_chat_new_order": row(c)})
        print("newchat", t, json.dumps({k: (v["total_s"], v["prompt_tokens"], v["prefix_hit"]) for k, v in rep["newchat"][-1].items()}), flush=True)
    for r in range(replays):
        n = nn(); sess = {}
        rows = []
        for seq in B.SESSION:
            x = row(B.call(with_session(nonce_sys(B.body(seq), n), sess), 1)); x["seq"] = seq
            rows.append(x)
        rep["replay"].append(rows)
        print("replay", r, [(x["seq"], x["prompt_tokens"], x["total_s"], x["prefix_hit"]) for x in rows], flush=True)
    for _ in range(2):
        x = row(B.call(nonce_sys(b41, nn()), 512))
        rep["turn"].append(x)
        print("turn", json.dumps(x), flush=True)
    med = lambda k, key: statistics.median([t[k][key] for t in rep["newchat"] if t[k].get(key) is not None])  # noqa: E731
    rep["summary"] = {"cold_s": med("cold", "total_s"), "new_chat_s": med("new_chat", "total_s"),
                      "new_chat_hit": [t["new_chat"]["prefix_hit"] for t in rep["newchat"]],
                      "same_chat_s": med("same_chat_new_order", "total_s"),
                      "replay_followups_median_s": statistics.median([x["total_s"] for rr in rep["replay"] for x in rr[1:]]),
                      "replay_followups_hit_share": round(sum(x["prefix_hit"] or 0 for rr in rep["replay"] for x in rr[1:]) /
                                                          sum(x["prompt_tokens"] or 1 for rr in rep["replay"] for x in rr[1:]), 3),
                      "turn_tool_calls": [x["tool_calls"] for x in rep["turn"]]}
    print("summary", json.dumps(rep["summary"]), flush=True)
    pathlib.Path(out).write_text(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 3, int(sys.argv[3]) if len(sys.argv) > 3 else 2)
