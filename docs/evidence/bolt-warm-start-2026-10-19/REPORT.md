# Bolt warm start: the first chat after a brain start is as fast as a new chat (2026-10-19)

After #391 a new OpenClaw chat reuses the system text + tools of the previous chat (13.74 s -> 0.79 s to the first
token). The first chat after a brain start had no previous chat: it re-prefilled all ~13.6k tokens (13.8 s). This PR
adds a warm start to `scripts/serve_nemotron_bolt_llamacpp.sh` (`scripts/brain_warm_start.py`, on by default,
`WARM_START=0` turns it off).

**Brain alone** (controller rule after three hard host resets, each with a brain beside Isaac Newton). llama.cpp v0.7.0
(`620d2225`), each tree's own `serve_nemotron_bolt_llamacpp.sh` from a `git archive` snapshot (2 slots x 64k, q8_0 KV,
LiLiCorr drafter, `--load-mode none`) on 127.0.0.1:18499 with a scratch key, one request at a time, temperature 0,
thinking off, streamed, `max_tokens` 1. Window 5 (`tools/warm5.sh`, 04:10-04:19) ran main and the final head in the
same window. 0 Xid in all windows.

| Window 5 | main `4f137d2` | this PR |
|---|---|---|
| **First chat after a brain start** (request 41, 13.6k tokens, new session ids) | **13.76 s** (0 reused) | **0.58, 0.58, 0.60 s** (13.29k reused, ~300 prefilled) |
| New chat after another chat, 3 trials | 0.77, 0.77, 0.76 s | 0.73, 0.77, 0.76 s |
| Same session again after another chat took its slot ("slot steal"), 3 trials | 0.75, 0.75, 0.74 s | 0.80, 0.73, 0.73 s |
| Follow-up calls (6-call take replayed 4x), median of 20 | 1.617 s | 1.640 s |
| Per call 43 / 44 / 47 / 49 / 51, median | 1.890 / 1.599 / 1.617 / 0.379 / 1.909 | 1.887 / 1.638 / 1.635 / 0.389 / 1.897 |
| Prefix reused, share of follow-up prompt tokens | 0.932 | 0.932 |
| Agent turn to completion: tool calls | 1, 1 | 1, 1 |

## 1. What the warm start does

`scripts/brain_warm_start.py` (standard library only) runs beside llama-server; the serve script starts it just before
`exec llama-server`, so the PID it watches is the server, and it exits with it.

1. **Readiness is the server's own.** The helper polls `/health` and does nothing before it answers 200; it never
   delays or gates readiness.
2. **Warm-up: two raw token prompts** (`/completion`, `n_cmpl` = slots, `n_predict` 1, `cache_prompt`), copied into
   every slot. They are raw prompts: no chat, no conversation, no user turn anywhere.
   - the recorded tokens before OpenClaw's cache boundary `<!-- openclaw:attempt:DYNAMIC -->`: system text + the
     49 tool schemas (12,555 tokens, 12.5-12.7 s);
   - then what the last two OpenClaw sessions shared after the boundary: date, output directives, up to the session
     ids (+737 tokens, 0.9 s). The first step's context checkpoints stay, so on another day the first chat parts at
     the date and resumes at the boundary (1.31 s, below).
   Each step is bounded by `--timeout` 120 s. A chat that arrives during the warm-up waits for a free slot; a chat
   that arrives before it is served cold, as on main (13.74 s vs 13.65 s, below).
3. **The prefix is the server's own tokens; nothing is rebuilt.** Every 10 s the helper reads `GET /slots`, which shows
   each slot's last prompt because the serve script sets `LLAMA_SERVER_SLOTS_DEBUG=1` (behind the API key when
   `API_KEY_FILE` is set). That text is detokenized with llama.cpp's `clean_spaces` (`" 'red"` -> `"'red"`), so it
   only says WHEN the prefix changed and where two sessions' system turns part. Re-tokenizing it drifts: window 2's
   first version did, diverged at token ~2.3k and gained nothing (`results/window2`: 13.6 s). So when the prefix
   changes and no slot is busy, the helper saves that slot (`POST /slots/<id>?action=save`; the serve script adds
   `--slot-save-path <state dir>/bolt-slots-PORT`, 0700), reads the token list from the slot file's header, deletes
   the file at once, and maps the cut points onto those tokens with `/detokenize` (plain concatenation, ~15 calls on
   llama-server's HTTP threads). The save costs ~0.1 s of server time (the log shows the 509 MiB of checkpoints
   appended 90 ms after the request ended); a prefix read before is kept in memory and not saved again.
4. **The record** (`WARM_PREFIX_FILE`, default `$XDG_STATE_HOME/cascade/bolt-openclaw-prefix-PORT.json`, mode 0600):
   the tokens, the template's sha256 and the model file (name + size), and the system turn of the latest session (to
   pair the first session after a restart). A record from another chat template or model is not prefilled. Before the
   first OpenClaw chat on a host there is nothing to prefill; the first chat records the prefix, the second session
   what sessions share.

## 2. Measurements

### First chat after a start (`bench_warm.py first`, the first request after the brain answers)

| | s | reused / prefilled tokens |
|---|---|---|
| main, windows 2-5 (6 starts) | 13.55, 13.60, 13.62, 13.66, 13.65, 13.76 | 0 / 13.6k |
| this PR, no record yet (first chat on a host; records the prefix), windows 3-5 | 13.69, 13.64, 13.77 | 0 / 13.6k |
| this PR, prefix only (one session recorded; this chat records what sessions share), windows 3-5 | 1.29, 1.31, 1.29, 1.27, 1.31 | 12,555 / ~1,035 |
| **this PR, prefix + shared text**, windows 4-5 | **0.58, 0.58, 0.59, 0.61, 0.58, 0.58, 0.60** | 13,292 / ~300 |
| this PR, after 12 perception inventories (window 4) | 0.65 | 13,292 / 298 |
| this PR, the next day (date line changed; window 4) | 1.31 | 12,551 / 1,040 |
| this PR, sent the moment `/health` answers (before the warm-up; window 4) | 13.74 | 0 / 13.6k (the warm-up waits for it: 26.1 s) |

The first chat after a start is now faster than a new chat (0.58 vs 0.76 s): the warmed slot ends exactly where
sessions part, while a new chat resumes from a context checkpoint 516 tokens before the end of the previous prompt.
Readiness is unchanged: the driver saw `/health` + `/v1/models` answer 3.1-5.1 s after the unit started on main and
3.1-4.2 s on the final head (`READY` lines in `results/*/status.txt`); the warm-up then took 13.4-13.6 s.

### Other latencies (bench `../bolt-default-fixes-2026-10-19/tools/bench_newchat.py 3 4`)

Window 5 is the table at the top. Earlier windows, head with each intermediate version: new chats 0.75-0.81 s,
slot steal 0.70-0.75 s, follow-up median 1.603-1.632 s, prefix share 0.932, tool calls 1, 1. In window 4 (before
`8a165f7`) the bench's nonce-prefixed system texts (every trial a new "agent") made the helper save a slot 10 times
mid-bench and new chats read 0.79 s; saves now wait for an idle server (3 saves, 0.2-0.3 s each, new chats
0.73-0.77 s). Perception inventories (12 bake-off frames, BrainDetector's request) are unchanged: median 4.68 s (2.79-6.38) in
window 4 vs 4.65 s (2.78-6.34) in window 2, whose first-version warm-up matched nothing (the no-warm baseline).

### Slot steal (0.33 -> 0.72 s after #391): investigated, not fixed

It is llama.cpp v0.7.0's slot choice (`get_available_slot`, `tools/server/server-context.cpp`): a request goes to the
idle slot with the longest common prefix relative to its own length, ties to the lowest slot id; empty slots are
skipped; the RAM prompt cache is consulted only when the chosen slot keeps less than half of its tokens. A new chat B
shares the 13.09k system text + tools with chat A's slot, so it takes that slot (log: `f_sim_best = 0.979, f_keep =
0.978`: A's state is not saved),
and A sent again reuses only the shared 13.09k from B's slot: 0.72 s instead of 0.33 s. The warm start cannot change
that choice: A's slot always shares more with B than a warmed slot does (the same day's date and directives). The
alternatives cost more than they save: `--slot-prompt-similarity 0` (pure LRU + prompt cache) sends a follow-up to
the other slot and restores an older state of its chat from RAM; a larger threshold cannot tell a new chat (0.98)
from follow-ups (0.91-0.99). A fix needs a llama.cpp change (prefer the slot that loses the fewest cached tokens among
near-equal prefixes) or a client-side `id_slot`; neither is cheap, so this PR leaves it and measures it (unchanged:
0.73-0.80 vs 0.74-0.75 s).

### `--brain qwen`

`git diff origin/main` touches only `scripts/serve_nemotron_bolt_llamacpp.sh`, the new helper, its tests and docs:
`serve_qwen_*.sh`, `launch.sh`, `openshell_rig.py`, `configs/` are untouched, so the Qwen command is byte-identical.

## 3. Tests

`tests/test_brain_warm_start.py` (34): a fake llama-server on an OS-assigned loopback port (character tokenizer,
`clean_spaces`-style `/slots` text, slot files in llama.cpp's format) and a child process standing in for the server
PID. Readiness first, both warm-up steps and their bodies, records from the server's own tokens (not the cleaned text),
what two sessions share (rendered by cascade's real template; a third session starts with exactly those tokens),
ids that start alike, follow-ups of one session, pairing across a restart, a new day, alternating agents (one save
each), a slot reused before its save, saves only while idle, long slots, failing saves (bounded), unusable or foreign
records, 0600 records, stale capture files, the warm-up bound, a zombie server PID. `tests/test_nemotron_bolt_llamacpp.py`:
the helper's argv (port, slots, server PID, template, model, key file, slot-save dir), `--slot-save-path` and
`LLAMA_SERVER_SLOTS_DEBUG` only with the warm start, `WARM_START=0` serves main's argv, a server without
`--slot-save-path` is refused unless `WARM_START=0`.

## 4. Reproduce

`tools/warm5.sh <main snapshot> <head snapshot>` as a systemd user unit while holding the brain-gpu lock (brain alone);
`tools/bench_warm.py` (first / nextday / slots / perception), `tools/warm4.sh` (window 4: next day, early, perception).
Commits measured: window 2 = `1d1487e` (text record), window 3 = `3de2d25` (tokens, prefix only), window 4 =
`d05555c` (+ shared text), window 5 = `8a165f7` (saves only while idle; the code of this PR). Raw JSON and the
driver's log per window in `results/`.
