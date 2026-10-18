# Bolt as the default brain, made better: no frame lost to a loop, fast first reply in a new chat (2026-10-19)

Owner decision (2026-10-18): NVIDIA Nemotron 3.5 Bolt 74B-A9B 24GB on llama.cpp v0.7.0 stays the default brain.
The 2026-10-18 brain comparison found two Bolt weaknesses; this PR fixes both:

1. **Inventory repetition loops.** Greedy Bolt repeated one label with a sliding box until the 1500-token cap on 6 of
   the 47 bake-off frames; production rejects a cut reply, so those frames showed nothing (40/47 frames usable).
2. **Every new chat re-prefilled the whole agent prompt.** OpenClaw puts a per-session block (date, directives, the
   Runtime line with the session ids) inside its system text, ~2.4k tokens in. Bolt's GGUF template renders the whole
   system text before the 49 tool schemas, so a new chat diverged there and re-prefilled ~13.6k tokens:
   13.7 s to the first token alone, 25.0 s beside Isaac Newton.

**Measured with the brain alone** (controller rule after three hard host resets, each with a brain beside Isaac
Newton: section 6). No end-to-end pick was run on this head.

| | main `3aa0a30` | this PR `6e9ffc5` | Qwen3.8 (comparison) |
|---|---|---|---|
| Inventory frames usable (bake-off scorer) | 40/47 (6 loops) | **46/47 (0 loops)** | 46/47 |
| Inventory localization / id strict / id lenient / precision | 0.78 / 0.73 / 0.74 / 0.75 | **0.87 / 0.80 / 0.82 / 0.82** | 0.89 / 0.79 / 0.83 / 0.82 |
| Inventory brain latency median / p90 | 3.65 / 13.36 s | 3.65 / **6.05 s** | 5.19 / 8.26 s |
| New chat, first token (13.6k-token OpenClaw request 41) | 13.74 s | **0.79 s** | - |
| Same-chat follow-up calls, median of 20 | 1.61 s (1.64 s rerun) | 1.60 s | - |

## 1. What changed

| Piece | Change | Tests |
|---|---|---|
| `scripts/chat_templates/nemotron-3.5-bolt-openclaw.jinja` (new), `serve_nemotron_bolt_llamacpp.sh` | The GGUF's own template (pinned verbatim beside it as `nemotron-3.5-bolt.gguf.jinja`, sha256 `6521afc7...`) with one change: when the system text holds OpenClaw's boundary `<!-- openclaw:attempt:DYNAMIC -->` and tools are present, the text after it is rendered AFTER the tool block, still inside the system turn. System text + tools become one byte-stable prefix that llama.cpp restores from a context checkpoint. Served with `--chat-template-file` (knob `CHAT_TEMPLATE_FILE`, empty = the GGUF's template); the script refuses a missing file and a server without the flag. | `tests/test_bolt_chat_template.py` renders both templates with jinja2 (new `dev` extra): only that block moves; prompts without the boundary or without tools (perception, readiness probe) render byte-identical; two chats share the whole system text and tool block. `tests/test_nemotron_bolt_llamacpp.py`: argv, refusals. |
| `perception/brain_detector.py` (`GUARDED_REPLY_MODELS = {"nemotron-3.5-bolt"}`), `configs/demo.yaml` `detector.reply_guards: auto` | Bolt's box requests ask for `BOX_LIST_SCHEMA` (a list of at most **12** `{label, bbox_2d}` objects, integer coordinates 0..1000, labels 1-64 chars; llama.cpp turns it into a grammar) within 1024 tokens. Replies go through `parse_guarded_boxes`: a reply cut at the token cap keeps its complete boxes; a reply that hit a bound keeps only the first box of each label listed 4+ times; exact repeats are dropped; inventory entries naming a category the prompt excludes are dropped (its list, plus the rest of that computer: computer mouse, mouse pad, computer tower / PC case). Refers keep the agent's own words ("find the keyboard" still answers). `auto` = by served id; `true` / `false` force it. | `tests/test_brain_detector_guarded.py` (62), `tests/test_make_detector_brain.py` (config key) |
| Every other brain (`--brain qwen`, Cosmos) | Request and parse unchanged. | `test_other_brains_keep_mains_request_and_reject_a_cut_reply` compares the request field by field with main's; `tools/qwen_identity_probe.py` builds the detector from `configs/demo.yaml` with `qwen3.8-27b` on main and on this head against a recording fake client: outputs byte-identical (`results/qwen-identity-*.json`). `git diff origin/main` of `serve_qwen_*.sh`, `launch.sh`, `configs/llm/`, `openshell_rig.py`: empty. |

## 2. Perception (47 rig frames, 157 certain GT objects, brain alone)

Bolt rows: the production request (`tools/perc_prod.py` builds it from the head's `brain_detector`: prompts, schema,
`max_tokens` 1024, temperature 0, thinking off) on the bake-off JPEGs, 2026-10-19 00:42-00:51 (window 3); main's row =
the comparison's request re-run 2026-10-18 23:25 parsed with main's parser; "schema 20" = the previous head `6e0a660`.
Replies parsed offline with each tree's parser (`tools/analyze_perc.py`), SAM2.1-L masks on the stored boxes
(`tools/sam2_stored.py`), scored by `tools/score_cmp.py` (the bake-off GT and matching, certain objects only). Tables
copied from the scorer output:

| Brain | frames ok | loc recall | id strict | id lenient | precision | wrong label | box IoU | centre err px (rel) | median s (brain) | p90 s |
|---|---|---|---|---|---|---|---|---|---|---|
| Bolt, main | 40/47 | 0.78 | 0.73 | 0.74 | 0.75 | 0.06 | 0.84 | 4.4 (0.03) | 3.65 | 13.36 |
| Bolt, schema 20 + guards (previous head 6e0a660) | 46/47 | 0.87 | 0.80 | 0.82 | 0.76 | 0.07 | 0.83 | 4.5 (0.03) | 3.67 | 8.20 |
| Bolt, this PR (schema 12 + guards, 6e9ffc5) | 46/47 | 0.87 | 0.80 | 0.82 | 0.82 | 0.07 | 0.83 | 4.5 (0.03) | 3.65 | 6.05 |
| Qwen3.8 (comparison) | 46/47 | 0.89 | 0.79 | 0.83 | 0.82 | 0.06 | 0.89 | 1.9 (0.01) | 5.19 | 8.26 |

| Brain | queries | hit | top-1 | empty | top-1 box IoU | IoU >= 0.5 | top-1 centre err px | median s | p90 s |
|---|---|---|---|---|---|---|---|---|---|
| Bolt, schema 20 + guards (previous head 6e0a660) | 157 | 0.90 | 0.90 | 0.07 | 0.82 | 0.92 | 4.56 | 1.92 | 2.02 |
| Bolt, this PR (schema 12 + guards, 6e9ffc5) | 157 | 0.90 | 0.90 | 0.07 | 0.82 | 0.92 | 4.56 | 1.96 | 2.06 |
| Qwen3.8 (comparison) | 157 | 0.94 | 0.94 | 0.01 | 0.84 | 0.91 | 2.29 | 1.90 | 2.16 |

Precision unrounded: Bolt 128/156 = 0.821, Qwen 130/159 = 0.818. The one frame without boxes (w07) answered `[]` for
every brain.

**The loops.** Greedy decoding still starts the repetition in the same 6 replies; the schema's bound ends it and the
guard keeps the label's first box (`results/perception/loops_*.json`):

| frame | repeated label | main: finish, tokens, s, boxes | schema 20: tokens, s, boxes | this PR: entries, tokens, s, boxes |
|---|---|---|---|---|
| s06 | black computer monitor | length, 1500, 13.36, 0 | 691, 8.01, 4 | 12, 423, 6.05, 1 |
| s10 | black rectangular object | length, 1500, 13.93, 0 | 725, 8.54, 10 | 12, 437, 6.02, 7 |
| s11 | black rectangular object | length, 1500, 14.21, 0 | 723, 8.77, 7 | 12, 435, 6.34, 7 |
| s14 | white cable | length, 1500, 13.69, 0 | 705, 8.24, 7 | 12, 425, 6.27, 7 |
| s15 | metal hex key | length, 1500, 13.98, 0 | 743, 8.51, 3 | 12, 447, 6.15, 3 |
| s16 | black cable | length, 1500, 13.62, 0 | 707, 8.20, 7 | 12, 427, 6.06, 7 |

All 47 live replies under the 12-box schema equal the first 12 entries of the 20-box replies (greedy decoding is
identical up to the bound), so the offline screen below predicted the live row exactly.

**Why 12 boxes and the computer parts** (`tools/screen_guards.py` on the stored 20-box replies, boxes only, so
precision differs slightly from the SAM table; `tools/fp_kinds.py` classifies false positives):

| bound | extra exclusion | loc | id lenient | precision |
|---|---|---|---|---|
| 20 | none (previous head) | 0.873 | 0.815 | 128/169 = 0.757 |
| 12 | none | 0.873 | 0.815 | 128/167 = 0.766 |
| 20 | computer mouse, mouse pad, computer tower / PC case | 0.873 | 0.815 | 128/158 = 0.810 |
| **12** | **computer mouse, mouse pad, computer tower / PC case** | **0.873** | **0.815** | **128/156 = 0.821** |
| 12 | + cable / cord / wire | 0.847 | 0.815 | 128/149 = 0.859 |

- Bolt's 41 false positives at the previous head: 31 objects the labelers did not list (12 computer parts, 11 of them
  named "computer mouse", "mouse pad" or "... computer tower"; 8 cables, lanyards, strings), 9 wrong labels (4 of them
  "red cable" on the lobster's lanyard), 1 near a GT object. Qwen listed 4 computer parts. The prompt already excludes
  the booth computer's screens, monitors, laptop and keyboard; the guard now also drops its other parts.
- Cables are NOT excluded: both brains call the plush lobster's lanyard "red cable" (s26-s29), so dropping cables cost
  localization (0.873 -> 0.847).
- No object Bolt found sat past the 12th entry of a reply (non-looping replies listed at most 15 entries; the busiest
  frame has 9 identifiable objects), and the loops now end ~40% sooner.
- DRY sampling was tried on the 20-box schema (2026-10-19 00:01-00:05, alone) and not shipped: default sequence
  breakers left 4 replies repeating and dropped id strict 0.80 -> 0.76; without breakers localization fell to 0.76.

## 3. New chat and same-chat follow-ups (brain alone)

`tools/window3.sh`, 2026-10-19 00:28-00:55: llama.cpp v0.7.0 (`620d2225`), each tree's own
`serve_nemotron_bolt_llamacpp.sh` from a `git archive` snapshot (2 slots x 64k, q8_0 KV, LiLiCorr drafter,
`--load-mode none`), one request at a time, temperature 0, thinking off, streamed, `max_tokens` 1 (the call ends at its
first token). Order: main, head, main again. Host load 2-3 (another agent's pytest), 0 Xid.

`tools/bench_prefix2.py` (the comparison's fixed new-chat bench: really new session ids) on OpenClaw request 41
(13.6k tokens, 49 tools), 3 trials each:

| Call | main: s (prefix reused / prompt tokens) | this PR: s (reused / prompt) |
|---|---|---|
| First call, fresh nonce (nothing reusable yet) | 13.69, 13.69, 13.76 (0 / 13.6k) | 13.76, 13.83, 13.80 (0 / 13.6k) |
| **New chat**: new session ids, same system text | **13.72, 13.74, 13.74 (0)** | **0.79, 0.78, 0.79 (13.09k)** |
| Same session, a different first order, right after the new chat | 0.33, 0.31, 0.33 (13.53k) | 0.72, 0.74, 0.71 (13.09k) |
| Cold prefill 13.6k / 15.0k / 18.2k / 32.0k tokens | 13.68 / 15.54 / 18.88 / 30.25 | 13.78 / 15.52 / 18.81 / 30.22 |

`tools/bench_newchat.py`: the take's 6 agent calls of one chat (41, 43, 44, 47, 49, 51) replayed 4 times, fresh nonce
per pass; the 5 follow-up calls per pass:

| | main | this PR | main, rerun |
|---|---|---|---|
| Follow-up calls, median of 20 (s) | 1.607 | 1.597 | 1.639 |
| Per call 43 / 44 / 47 / 49 / 51, median (s) | 1.873 / 1.583 / 1.607 / 0.378 / 1.842 | 1.888 / 1.591 / 1.597 / 0.370 / 1.848 | 1.940 / 1.618 / 1.637 / 0.387 / 1.885 |
| Prefix reused, share of follow-up prompt tokens | 0.932 | 0.932 | 0.932 |
| New chat (3 trials, median) | 13.72 | 0.75 | 14.08 |
| Agent turn to completion: tool calls | 1, 1 | 1, 1 | 1, 1 |

The one slower row: the same session sent again with a different first order right after another chat ran on its slot
(0.33 -> 0.72 s). The other chat now shares the system text + tools, so llama.cpp picks that slot for it, and the
first session's ~510-token per-session block is prefilled again from the 13.09k checkpoint. Real follow-ups append to
the history and are unchanged (table above).

Agent tool choice with the template (62 operator orders, `agent_eval/tool_eval.py`, 2026-10-18 window 1, same template
sha256 `0738f383...`): 52/62 passed (main 51/62, Qwen 52/62), tool-name accuracy 0.95, key-argument accuracy 0.98,
prefix hit rate 0.98, 0 malformed calls.

## 4. Beside Isaac Newton (2026-10-18, before the third host reset)

Window 1 (`tools/window1.sh`, 23:46-23:57, same template sha, Isaac Sim 6.2 NEWTON up and pinged at start and end,
`results/beside-isaac-2026-10-18/`, 0 Xid): new chat 24.97 s on main vs **1.30 s** with the template; follow-up calls
median 2.91 vs 3.07 s (10 calls each, one pass order main then head). The guarded perception request (12-box schema)
was never timed beside Isaac.

## 5. Not measured; needs the owner

- **End-to-end Spark kitchen pick on this head (Isaac Newton + Bolt + GraspGen-X + MCP): not run.** The host reset hard
  three times since 2026-10-18 13:56, each time with a brain and Isaac Newton on the GPU together (Bolt on the vLLM
  nightly, on vLLM 0.32.0, and on llama.cpp during this PR's e2e attempt at 00:11, CPU load low). The controller
  stopped every brain-beside-Isaac run until the owner decides. Still open: the pick itself, perception latency beside
  Isaac under the 5 s analysed-frame limit, and a soak.
- The first chat after a brain start is still a full prefill (13.8 s alone): nothing is cached yet.

## 6. Limits

- The computer-part exclusion goes beyond the prompt's words (it names screens, monitors, laptops and keyboards):
  an inventory no longer lists a computer mouse, mouse pad or computer tower; "find the mouse" (a refer) still does.
  Only exact head nouns go: "plush mouse", "pencil case" and "block tower" stay (tests).
- An inventory pass lists at most 12 objects; a table with more shows 12 per pass (refers are unaffected).
- Bolt's localization error stays ~2.4x Qwen's (4.5 vs 1.9 px centre error); refer hit 0.90 vs 0.94. Not addressed here.

## 7. Reproduce

Bake-off workspace `~/Projects/perception-bakeoff` (frames, GT and predictions stay there, never committed):

```bash
# brain window (holds the brain-gpu lock; brain alone)
systemd-run --user --collect --unit bf-driver3 ... bash tools/window3.sh <snap-main> <snap-head> prod12
PYTHONPATH=<snap-head>/src .venv/bin/python tools/analyze_perc.py <snap-head> prod12:bounded:bfp_prod12
.venv/bin/python tools/sam2_stored.py bfp_prod12
CERTAIN_ONLY=1 .venv/bin/python tools/score_cmp.py bfp_v0_main:"Bolt, main" bfp_js20:"..." bfp_prod12:"..." qwenv31:"Qwen3.8 (comparison)"
```

`tools/` holds the runner, parser, scorer and bench scripts byte-identical to the runs (`score_cmp.py` imports the
bake-off's `tools/score.py` and `tools/failures.py`; the bench scripts import `agent_eval/bench_agent.py`).
