import json, collections, sys
rows=[json.loads(l) for l in open(sys.argv[1])]
g=collections.defaultdict(list)
for r in rows: g[(r['variant'],r['kv'])].append(r)
print("| variant | KV | correct first call | native call |")
for k,rs in sorted(g.items()): print(f"| {k[0]} | {k[1]} | {sum(r['correct'] for r in rs)}/{len(rs)} | {sum(r['native_call'] for r in rs)}/{len(rs)} |")
if '--per-prompt' in sys.argv:
    pp=collections.defaultdict(list)
    for r in rows: pp[(r['prompt'],r['variant'],r['kv'])].append(r)
    for k,rs in sorted(pp.items()):
        c=collections.Counter((f"{x['calls'][0]['name']}({x['calls'][0]['arguments'][:70]})" if x.get('calls') else repr((x.get('text') or x.get('error',''))[:70]))+' '+str(x.get('finish')) for x in rs).most_common(2)
        print(f"{sum(x['correct'] for x in rs)}/{len(rs)} | {k[0][:45]} | {k[1]} | {k[2]} | " + " || ".join(f"{n}x {t}" for t,n in c))
