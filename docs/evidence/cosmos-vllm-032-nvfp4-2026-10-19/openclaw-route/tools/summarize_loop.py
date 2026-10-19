#!/usr/bin/env python3
"""Summarize replay_loop.mjs rows: usage summarize_loop.py <out.jsonl> [--detail]"""
import collections
import json
import sys

rows = [json.loads(line) for line in open(sys.argv[1])]
by = collections.defaultdict(list)
for r in rows:
    by[r["variant"]].append(r)


def bad_reply(r):
    f = (r["final"] or "").strip()
    return (not f or f.startswith("[TOOL") or f.startswith(("{", "<", "[")) or len(f) < 12
            or any(ord(ch) > 0x2000 for ch in f))


print(f"{'variant':18} {'single exact':>13} {'multi exact':>12} {'step2 ok':>9} {'extra':>6} {'repeats':>8} "
      f"{'hit cap':>8} {'bad reply':>10} {'requests':>9}")
for v, rs in by.items():
    single = [r for r in rs if r["steps"] == 1]
    multi = [r for r in rs if r["steps"] > 1]
    extra = sum(r["extra"] for r in rs)
    reps = sum(r["repeats"] for r in rs)
    cap = sum(1 for r in rs if r["n_calls"] >= 3 and r["ended"] in ("reply", "reply-call"))
    s2 = sum(1 for r in multi if r.get("step2_ok"))
    bad = sum(1 for r in rs if bad_reply(r))
    req = sum(len(r["turns"]) for r in rs)
    print(f"{v:18} {sum(r['exact'] for r in single):>6}/{len(single):<6} {sum(r['exact'] for r in multi):>5}/{len(multi):<6} "
          f"{s2:>4}/{len(multi):<4} {extra:>6} {reps:>8} {cap:>8} {bad:>5}/{len(rs):<4} {req:>9}")
if "--detail" in sys.argv:
    for v, rs in by.items():
        print("==", v)
        for r in rs:
            mark = "BAD" if bad_reply(r) else "   "
            print(f"  {r['id']:20} r{r['rep']} exact={str(r['exact']):5} {mark} "
                  f"{' -> '.join(r['calls'])[:140]} | {json.dumps(r['final'])[:110]}")
