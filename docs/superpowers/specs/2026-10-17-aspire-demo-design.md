# ASPIRE ideas in the cascade demo: design spec

Date 2026-10-17. Branch `feat/aspire-notes`, base `main` 42eaddd. Plan:
[`docs/superpowers/plans/2026-10-17-aspire-demo.md`](../plans/2026-10-17-aspire-demo.md).
Inputs: `.superpowers/sdd/aspire/{DECISIONS,FINDINGS,paper_v1}.md` (owner answers are binding), upstream ASPIRE v1.0
(fa00f0e) and v0.1 (317a72e) read-only, the Qwen agent measurement `qwen-agent-e2e/RESULTS.md` (88cff9d, OpenShell sim
route, PhysX). Every code claim below was checked against 42eaddd with a grep; `file:line` refers to that commit.

## 0. Decisions on one screen

| # | Decision | Source |
|---|---|---|
| D1 | Scope: PR-A1 (owner's "A") now, then PR-B; C deferred until B shows verified notes raise Qwen's success | Q1 owner |
| D2 | Notes reach Qwen twice: top 2 in the kitchen prompt, the full list in `robot_knowledge` | Q2 owner |
| D3 | The kitchen "managed block" is a separate generated file `LEARNED.md` in the kitchen workspace, injected by the kitchen prompt plugin right after `AGENTS.md`. No `AGENTS.md` (repo or workspace) is edited by the code | this spec, §4.3 |
| D4 | One bounded retry after a REFUSED attempt, with a changed approach that the server applies (home, fresh look, refused grasps excluded). Simulation profiles only, decided by arm types (fail-closed); the real reBot / booth keeps one motion per order | Q3 owner, §4.5 |
| D5 | Real reBot gets only verifier-passed notes, tagged "learned in sim"; until B: none (enforced when the library is read, for every consumer) | Q7 owner |
| D6 | Visible surface: the camera-extension panel (data in the camera server's existing `/state`, no new port) + one chat line | Q8 owner |
| D7 | Outer loop: local Qwen3.8-27B only | Q4 controller |
| D8 | Verifier scenes: seeded prop poses in the Isaac `reset_scene`. Pulled forward into PR-A1, because A1's acceptance needs seeded orders; B reuses it | Q5 controller; sequencing by this spec |
| D9 | Promotion = >= 2 distinct tasks AND blind verifier >= 4/5 dev seeds within <= 2 attempts | Q6 controller |
| D10 | Docs fix (`preview_grasp` attribution, "follows upstream" promotion claims) ships as its own small docs PR before A1 (cascade rule: docs fixes never ride in a feature PR) | Q9 controller |
| D11 | Every Isaac run uses `--engine newton`; nothing lowers a safety margin; notes are advice text, never configuration | standing |

Open questions: §10 (three, lettered, one reply line).

## 1. Scope

In: PR-A0 (docs), PR-A1 (notes reach Qwen, provenance, library hash, seeded reset, sim retry, rules and recipes, the
learning surface), PR-B (blind verifier on Newton, the two-part promotion rule, use grading, timing breakdown, nightly
scheduling with a live-demo guard, verified-only export to the real reBot).

Out: option C (evolutionary search, outer-loop coder); widening admission beyond retry pairs (the A1 retry produces the
pairs `agent/aspire.py` already admits, §5.1); trimming the OpenShell tool catalog (owner decision pending since
RESULTS §4.4); a prompt block on the OpenShell route (question Q3, §10); the 8090 dashboard learning board (replaced by
D6).

## 2. Current state (42eaddd)

1. The library never reaches the production brain. `skills/library.py` notes are read only by the built-in orchestrator
   (`agent/orchestrator.py:352`, `retrieve()`). `robot_knowledge` returns envelope + grasp priors only
   (`apps/mcp_server.py:1460-1477`).
2. Two OpenClaw routes exist:
   - **Kitchen route** (Spark kitchen, `CASCADE_INSTALL_PROFILE=spark`, host OpenClaw profile; `launch.sh:223`). The
     plugin `demo/kitchen/openclaw-plugin/index.mjs` builds `systemPrompt = <workspace>/AGENTS.md + "\n\n" + identity`
     every turn and exposes only the 6 tools of `attendee-tools.json` (no `robot_knowledge`, no `move_home`).
     `launch.sh:1176-1182` syncs `demo/kitchen/visitor-instructions.md` into a marker block of the workspace
     `AGENTS.md`. The MCP server is a stdio child (`launch.sh:931`).
   - **OpenShell route** (every other Linux run: instance `sim` for dev, the default instance for the real rig). 49
     tools, `robot_knowledge` included; the sandbox workspace has no `AGENTS.md` ("[MISSING]" in the measured system
     prompt, `brain/bodies/00015.json.gz`), so no rule text reaches Qwen except tool descriptions and results.
3. Both kitchen texts forbid retries: identity "Use at most one motion action per order ... never invent success or
   retry automatically" (`index.mjs:5-13`); visitor block "Do not retry, reset automatically or turn a failed result
   into success".
4. A failed `pick_and_place` result: `next_action` = "This attempt FAILED: <error>. Tell the user it failed; ..." (#342,
   `runtime.py:7738`), then the move_home hint when `arm_left_in_place` (`runtime.py:7711-7726`), then "If the user
   said 'then stop' ... Do not start another pick ..." (`runtime.py:1646-1659`). On the kitchen route the hint names a
   tool the agent does not have.
5. #341: inside one grasp, after a pre-command veto, at most `OBSERVED_NEXT_CANDIDATES` (2) further candidates of the
   same plan, never a near-copy of a refused one (`runtime.py:142-185`, `_near_copy`). It does not cover a refusal
   after motion (r3: "pre-close retreat refused", arm parked).
6. Traces: `runs/mcp_<pid>_<uuid>/trace.jsonl` rows (`agent/trace.py:59-74`), no library identity anywhere.
7. Isaac reset: bridge op `reset_props` re-places every prop on `_PROP_SPAWNS` (fixed) and returns a physics read-back
   within 2 cm (`isaac_bridge.py:1390-1398, 2029-2075`, `sim/isaac_reset.py`); op `place_prop` puts one prop anywhere
   (`isaac_bridge.py:1400-1427`).
8. Camera extension: `extensions/chrome/panel.mjs` polls the camera server's `/state` every 1 s (Spark:
   `scripts/spark_cameras.py` on :8091, started by `scripts/spark_browser.py:128`; real rig: the live view :8090,
   `apps/stream_server.py`).
9. Sim vs hardware is not a named concept; arm `type` (`isaac`, `mujoco`, `mock` vs `rebot_rs`, `so101`, `ros2`, ...)
   is the only reliable marker (`control/arm_base.py:448`, `apps/demo.py:113-126`).

## 3. Measured Qwen failure modes: what remains after #341 / #342

| RESULTS §4 defect | Fixed on main by | Remains | Target here |
|---|---|---|---|
| 1 false success after a refuted placement (o3) | #342 unconditional FAILED line | no | none |
| 2 "next to X" read as "on X" (o3) | #342 tool text (`pick_and_place`, attendee description) | no | none |
| 3 a refusal parks the arm; 3 identical re-refusals (r3) | #341 next candidate (pre-command only), #342 move_home hint | **yes**: refusals after motion still end the order; the kitchen agent cannot call move_home; the booth rule forbids any second attempt | sim retry (§4.5) + retry discipline rule R1 |
| 4 80 % of every prompt is tool schemas | — | yes | out of scope (owner decision); our additions are budgeted (§4.3) |
| 5 `locate_pixel` substitutes another object | #342 | no | none |
| 6 refusals do not name the object hit | #342 | no | none |
| 7 inventory details not in the scene (o1) | — | yes (minor) | recipe R3 (kitchen text) |
| 8 robot vs screen "left" (o4) | #342 frame note | no | none |
| 9 "released" claimed from a held width | #343 for `place_at` | **yes** on the OpenShell route: o4 opened the jaws 4 cm above the table (a drop) | rule R2 (tool description) |
| 10 `notifications/cancelled` at 120 s | #342 test (cancel stops only its own id) | no | none |

## 4. PR-A1 design

### 4.1 Note format v2 and provenance tags

A note stays one markdown file `<library>/<slug>.md` with front matter (`skills/library.py` reads v1 unchanged). v2
adds keys; a missing key takes the default in the right column, so every v1 file is a valid v2 note.

| Key | Values | Default for v1 files |
|---|---|---|
| `schema` | `2` | `1` |
| `kind` | `learned` (harvested from traces) / `authored` (hand-written, reviewed in a PR) | `learned` |
| `status` | `candidate` / `promoted` / `refuted` / `retired` | v1 `status` |
| `learned_in` | `sim` / `real` / `mixed` / `unknown` (profile class of the source runs, from their `run.json`) | `unknown` |
| `occurrences`, `source_tasks`, `source_runs` | as v1 | as v1 |
| `verifier` | `null` or `{verdict: verified/refuted/void, passes, seeds, attempts, library_sha, engine, run, date, gaps}` | `null` |
| `grades` | `{useful, misleading, wrong, not_applicable}` integers | all 0 |

Body sections: `# <title>`, `- **When to apply:** <words>`, `## Advice` (what Qwen reads, <= 300 chars), `## Evidence`
(trace excerpts for humans: error, context, confirmation; never rendered to the agent, which keeps the verifier blind).
Title, when-to-apply and the failure signature (the note's own skill) are read only from the header before the first
`## ` section, so `## Evidence` can never supply them.
A v1 body has no `## Advice`: its advice is the first paragraph of `## Strategy`; a paragraph over 300 chars fails the
lint (the note is skipped, never cut into acceptance).

Provenance tag (one string, rendered after each note everywhere Qwen or a visitor sees it):
`learned in <learned_in> | <verifier part> | lib <first 8 hex of library_sha>`, where `<verifier part>` is
`verifier <passes>/<len(seeds)>` for `verified`, else `not verifier-tested`; `authored` notes render
`authored | <verifier part> | lib <sha8>`. Example: `learned in sim | verifier 4/5 | lib 3fa9c2d1`. In the block, a
note whose advice would be cut to nothing (not one whole word fits) is dropped, never rendered empty; a cut advice ends
in the ASCII `...` and the line it renders is linted again, so a cut the lint refuses (`Dismiss the warning...`: the
guard word now ends the object) drops the note instead of serving it (A4 fix round 1, ruling 2).
Served text is ASCII on the wire (A3 review fix round, ruling 2): the tag's separators are ASCII (a note's own text
can carry no `|`), and every route renders a note's typographic quotes, apostrophes and dashes as `"`, `'` and `-`
(`notes._ascii`), so a JSON reply never escapes a rendered character to six.

Advice lint (a note that fails it is never served, whatever its status) is a deny-by-default STRUCTURAL gate, not a
blocklist: advice and titles are short plain English, and anything that cannot be shown to be plain English is refused
(A1 fix round 1). In order: (1) a field over its cap (advice 300, title 80 chars) is refused before any pattern runs,
and every pattern is linear; (2) no control or format character (Unicode Cc/Cf: zero-width space, soft hyphen, BOM),
the text is NFKC-stable, and every character is a printable ASCII letter, digit, space or `.,;:!?'"-/_&+%`, or a
typographic quote or dash, so no homoglyph, CJK, fullwidth or other non-ASCII letter and none of `[]{}()=<>|` `` ` ``
`\#*@$^~` (hence no `<|`, `|>`, code fence or markup); (3) every word is lower, Title or UPPER case (`reSet`,
`SafetyViolation` refused: A1 fix round 2); (4) no hard stem, read on letters only (NFKC, lower case, every
punctuation mark and digit deleted; the whole words `preset`, `presets`, `pre-set`, `pre-sets`, `you're`, `we're`,
`they're` left out, their
neighbours still joining): the one-word stems `reset`, `ignor`, `disregard`, `bypass`, `circumvent`, `overrid` inside
the letters of one whitespace-delimited word (`Re/set`, `Ig+nore`, `DIS;regard`, `I-g-n-o-r-e`) or spelled by whole
neighbouring words joined, from a word's first letter, into the stem plus a plain ending (`re set`, `Ig. nore`, `ci rc
um ve nt`; `by passing` too, an accepted refusal) but never across an arbitrary word boundary (`before setting`, `are
set`, `sign or`, `over ridges` are plain English: A1 fix round 4); the phrase stems `emergencystop`, `haltmotion`,
`startover`, `systemprompt` anywhere in the field's letters-only skeleton, across words (`Emer'gency stop`,
`SYSTEMPROMPT`, `restart over`); a word that merely contains a stem, `Resettle`, is refused too (A1 fix round 3);
(5) over the field read as words three ways --
the words split at camelCase humps, digits and any separator, the unsplit lower-cased runs, and at each word the 2-3
words that join into one whole phrase word across spaces or dashes (`dis able`, `DIS-able`; never across an
apostrophe or other punctuation) -- no URL (`://`), no `snake_case` identifier that is not a tool, no letter-digit mix
(ordinals aside), no dotted name, no word spelled out letter by letter, no role marker (`assistant`, `user`, `system`,
`developer`, `human`, `tool`, `observation`, `function`, `output`, `operator`, `owner` followed by `:` or a dash (a
hyphen only when spaced), or by `said`/`says` and `:` or a dash; and `function result` / `tool result` followed by `:`
or a dash at the field's start or after a sentence break, while `Use the tool rack: ...` and `Ask the user why: ...`
are plain English) or chat-template marker (`im start`, `system
prompt`, `developer mode`, `new instruction`), nothing imitating the provenance tag (`verifier`, `verified`, `promoted`, `provenance`, `authored`,
`passing seeds`, `learned in|on`, `tested in|on|by`, `Newton`, `PhysX`, `lib <hex>`, any hex-like token of 6+ chars,
`out of <n>`, `<n> of <n>`), at most ONE number counting digits and number words (`zero` .. `twenty`, tens,
`hundred`, `half`, `quarter`, `minus`, `negative`, and `point` next to a number; a lone `one` aside), no two number
words side by side (`minus one`, `point five`), no unit of length or angle (`m`, `cm`, `mm`, `um`, `nm`, `dm`, `km`,
`in`, `ft`, `yd`, `mile`, `deg`, `rad`, `mrad`, `arcmin`, `grad`, `rev`, `thou`, `mil`, `%`, `pct`, `per cent`, their
plurals (`mms`, `cms`, `kms`, ...) and words; `turn(s)` with a number, `half`, `quarter(s)` or `full` among the 3 words
before it in its clause: `half a turn`, `one full turn`) and no axis coordinate (`x 0.42`, `x zero`, `x – 0.4`,
`x/0.4`); a tool is named only as an identifier (snake_case, any case) that is read-only or the note's own skill (the
tool its failure signature names), never `reset_scene`, `reset_stop`, `emergency_stop` or `halt_motion`, and no
identifier or tool parameter is followed by `:` (arguments: any `snake_case` name and `object`, `destination`,
`target`, `x`, `y`, `z`, `arm`, `force`, `speed`, `frame`, `pose`, `position`, `grasp`, `offset` always; every other
single-word tool parameter, frozen in the lint so a new tool spec never changes it, only in a field with two or more
`name:` pairs, so a field like `Side: grasp across` passes, as a field and as the line it renders, see below; `=` is
a refused character); a tool name written as plain
words is English (`move home`, `place at`), except the four forbidden tools in any spelling; and no reset (`reset` in
any form, `respawn`,
`begin again`, `start from scratch`, and `restart` / `reload` / `redo` / `re-initialise` / `restore` with `scene`,
`layout`, `table`, `everything`, `kitchen`, `world` or `simulation` as object: a reset clears the sim retry ledger),
stop (`emergency stop`, `e stop`, `halt the motion`, `stop latch`, `emergency button`), "do not stop" (negated stop,
pause or halt: never discourage an operator stop), unbounded-retry (`keep calling`, `try again until`, `retry until`,
`until it succeeds`, `start over`, `forever`, `over and over`, `no limit`, `never give up` ...), override (`ignore`,
`disregard`, `bypass`, `circumvent`, `override` anywhere), weakening (`disable`, `turn off`, `shut down`, `skip`,
`lower`, `relax`, `reduce`, `raise` ... whose DIRECT OBJECT holds `gate`, `check`, `refusal`, `warning`, `veto`,
`clearance`, `margin`, `limit`, `safety`, `collision`, `stop`, `halt`, `harness` ... (`stop` as the exact words
`stop`, `stops`, `stopping`, `stopped`, `stoppage` only: a note never discourages a stop in any form, while `the
stopper`, `the stopwatch` are plain; A1 fix round 5, A2 fix round 1): the guard noun among the first 4
words after the verb, before a preposition and the end of the clause at `and`, `to`, `for`, `then`, `so`, `but` or
punctuation, with `and`/`,` + determiner listing another object; so `lower the clearance` is refused and `lower it to
avoid collision` is not; `cut`, `drop`, `remove`, `zero`, `waive`, `tighten`, `forget`, `dismiss`, `set aside` ...
only when the guard noun is the object's HEAD, the last noun of its phrase: `remove the safety check` is refused,
`remove the safety seal`, `drop the warning label` are not (a phrase also ends before `of`, `about`, `during`,
`without`, `like`, `among`, `despite`, `except`, `per`, `throughout`, `amid`, `regarding`, `besides`, which do not
end the object: `remove the safety check of the arm` is refused, A1 fix round 5); `remove`, `cut`, `drop`, `take off` also when a guard word
modifies a safety head noun (`system`, `feature`, `rule`, `check`, `gate`, `interlock`, `limit`, `margin`, `stop`,
`sensor`, `monitor`, or a plural: `drop the safety system`; `the guard ring` stays plain: A1 fix round 4); and
`turn`/`switch`/`shut ... off`, `shut ... down` split
around an object whose head is a guard noun, the particle within 4 words: `turn the clearance gate off`; a dash
right after the verb or a determiner does not end the object, nor does one right after the object's head part the
particle: `Disable the - gate ...`, `Lower the - safety margin ...`, `Turn the clearance gate - off ...` are refused as
the undashed texts are, also when the renderer's ` - ` forms them from a clean title and advice, while `Lower the
cup - check the handle first` stays plain; polish round) or
success-claim advice (`treat ... as a success`, `tell the user
it is done`, honesty phrasings included). The same lint covers every field rendered to the agent or a visitor
(preflight R4): the title and the advice, and -- once both are clean -- the text they render together, `<title> -
<advice>` exactly as every route renders it, in ASCII (cap 80 + 3 + 300; A3 review fix round, ruling 1): two clean
fields can join into refused advice (`Do not` + `stop when ...`, `Tell the user` + `it is done ...`), a title ending
in a role word or a two-word role marker renders as a role marker (`User - ...`, `Tool result - ...`: a spaced dash
marks a role as a colon does) and a hard stem can join across the separator (`Re - set ...`); the skip reason is then
the rule of that rendered text (field `line`). The separator is an ASCII ` - `, never `: ` (A4 fix round 1, separator
ruling), so it never adds a `name:` pair: a title ending in a tool parameter (`Mug handle facing the arm`) or opening
with its own tool-parameter label (`Side: grasp across`) is served (with `: ` they were refused as arguments, 6 of the
192 oracle texts). The reviews' hostile texts (78 + the first re-review's 50 rule-claimed ones +
the second re-review's 73 rule-claimed and 18 newly ruled ones) are a test that must stay all refused, next to a set of plain kitchen advice that must stay all accepted (and fix round 4's everyday phrasings whose words run into a stem); plain-English
paraphrases of harm (no rule word) stay a limit of a structural gate, carried by promotion and the verifier.

### 4.2 Library snapshot, library hash, run manifest

- `library_sha` = sha256 over the UTF-8 lines `"<file name>\t<sha256 of file bytes>\n"` of every note file in the
  library directory, sorted by name. A missing or empty directory hashes to sha256 of the empty string. Any byte change
  of any note changes it; other files do not. A note file's name must match `^[a-z0-9][a-z0-9_.-]{0,63}\.md$` (the
  library's slugs do): any other file (a control character, tab or newline, upper case, a non-UTF-8 byte, `,` or `-->`)
  is neither hashed nor served, so the line format is unambiguous and a note id is safe in a JSON manifest and the
  LEARNED.md header (ids are sanitised at the source). A symlink (lstat, and the read opens with `O_NOFOLLOW`) or any
  non-regular file is neither hashed, parsed nor served either, and neither is a file over 64 KiB (`NOTE_MAX_BYTES`:
  skipped on its fstat size before any byte is read; the read stops one byte past the cap, so a file that grows
  meanwhile is never read whole).
- A skipped note is never silent: the snapshot's `skipped` lists `{id, reason}` for every `*.md` entry that is no
  lint-clean note (`reason` = the lint rule's name, or `file_name`, `symlink`, `not_a_file`, `too_large`, `unreadable`,
  `parse`), by file name, and logs one warning line each; A3/A9 surface it. A lint-clean note the class does not serve
  (not yet promoted) is not "skipped".
- The robot runtime loads ONE read-only snapshot at build (`load_snapshot`; it never creates the directory) and keeps
  it for the session: the prompt block, `robot_knowledge`, the chat line and the trace stamp all come from it, so a
  nightly write never changes a running session. The legacy `demo` CLI's built-in orchestrator reads it too
  (`notes.served_library`): only the notes the snapshot serves, and only while the library holds the bytes it hashed
  (a library changed since the build gives the CLI no note). Its `retrieve()` keeps the legacy choice of notes (shared
  When words, or the memory embedder's similarity; at most 2, ties in rank order) but injects them only as the prompt
  block renders them (§4.3: title, advice and tag, in <= 1400 characters), never a note's body (the Failure signature
  and When lines, `## Strategy`, `## Evidence` or any other text the lint never reads); it never creates the directory
  either.
- Library location per profile: `memory.skill_library_path`. A2 sets the kitchen profiles (`isaac_kitchen` and
  every profile that extends it) to
  `${CASCADE_ASPIRE_LIBRARY:-~/.cascade/aspire/sim/library}`; every other profile keeps today's default
  (`<repo>/skills_library`). B7 sets `rebot_rs` to `${CASCADE_ASPIRE_LIBRARY:-~/.cascade/aspire/real/library}`.
- Every run built by `apps.demo.build_runtime` writes `run.json` beside `trace.jsonl` at build:
  `{schema: 1, library_sha, served_notes: [ids], block_notes: [ids], block_sha, profile_class, library_root,
  library_error, skipped, arm_types, sim_retry, purpose, started}` (the physics engine is
  not known at build: the lazy Isaac arm learns it from the bridge on first use; the verifier records it from its own
  launch). `block_notes` = the top 2 as the block renders them (preflight R23: B3 grades them and cannot recover them
  later); `block_sha` = sha256 of the rendered prompt block (UTF-8; the empty string's when no block renders), which
  the LEARNED.md header carries too (A4 fix round 1, ruling 4). `library_root` = the real path of the library the snapshot read (`~` and `${...}` expanded, then symlinks and
  `..` resolved: `Path.resolve()`), null when it was not read; `library_error` = null, or `"<Exception>: <message>"`
  when it was not read, or `"all notes unreadable"` (root recorded) when the directory lists `*.md` files but every one
  was unreadable (an r-- directory, 0600 notes of another account), so an unreadable library never looks like an empty
  one. `skipped` = `{count, first}`: the number of skipped `*.md` entries and the first 20 (`SKIPPED_RECORDED`) as
  `{id, reason}`, in file-name order. `purpose` = `CASCADE_RUN_PURPOSE` (`live` | `verify` | `acceptance`, forwarded to the robot
  server; unset or empty = `live`, any other value = `unknown`, never `live`; preflight R18): harvest and grading count
  only `live` runs. `started` = UTC ISO 8601 (seconds). A rebuild into a run directory that already holds a `run.json`
  keeps it first as `run.<its started>.json` (`.1`, `.2`... when taken; its mtime when it has no valid `started`): a hard
  link, or a copy under the same exclusively created name where the file system has no hard links (vfat, exFAT, some
  FUSE / SMB mounts), so the rebuild still stamps there.
  Every trace row gets a top-level `library_sha` (null for runtimes that load no library: mobile and microduck, and a
  composed robot's own top-level rows; each manipulation domain of a composed robot is built by `build_runtime` into
  `domains/<name>/`, which writes its own `run.json` and stamps its own rows). Fail closed, never a failed build: a
  configured path that is no path or a relative one (a relative `CASCADE_ASPIRE_LIBRARY` would depend on the working
  directory), a file, an unreadable directory, a symlink on the path to a missing target (an unmounted volume), any
  read error, or a snapshot whose manifest cannot be rendered serves nothing (the empty library's sha, root null,
  `library_error` set, one warning with the reason); a missing directory is an empty library (never created, root
  recorded, no error); a `run.json` that cannot be written leaves the rows unstamped (null) and serves nothing, since a
  served note must be on record.

### 4.3 How notes reach Qwen

**Kitchen route: a separate generated file, injected by the plugin (D3).**

1. One library per launch (A4 fix round 1, ruling 3). In the spark branch, before the robot server's environment
   (`MCP_JSON`) is built, `launch.sh` runs `python -m cascade.skills.learned_block --freeze $STATE_DIR/aspire-library`
   with the launch's `CASCADE_ARM` / `CASCADE_CAMERAS`. It resolves the library the robot server would read (the same
   config), copies the files A1's reader accepts (a safe note name, a regular file, no symlink, <= 64 KiB; every other
   entry is left out and counted in the log line) byte for byte and read-only into a temp directory beside that path,
   checks the copy hashes to the library's `library_sha` and renames it into place (an earlier launch's copy is
   replaced whole); `launch.sh` exports `CASCADE_ASPIRE_LIBRARY` to the copy. Every robot server of the launch (the
   runtime check, the proof's, the visitors') and `LEARNED.md` therefore read the same bytes, whatever the canonical
   library does meanwhile (a nightly write counts from the next launch). A freeze that fails (a library the snapshot
   would not read, a profile that does not read `CASCADE_ASPIRE_LIBRARY`, a write error, a Python that cannot start)
   prints one `WARNING: learned notes:` line and points the variable at the empty `$STATE_DIR/aspire-library-off`:
   notes off everywhere, the launch goes on (R26). That one notes-off directory is removed at every launch before the
   freeze and made anew only when the freeze fails (one that cannot be emptied is never exported: a new empty
   `aspire-library-off.XXXXXX` then), and each freeze first removes the copies an earlier one moved aside and could not
   remove (`.aspire-library.old.*`), so failed launches leave no directories piling up (polish round). Then, right
   after the visitor sync, it runs
   `python -m cascade.skills.learned_block --workspace <profile workspace>`. It loads the same config the robot
   server loads, takes the snapshot (§4.2) and writes `<workspace>/LEARNED.md` atomically (temp file + replace, mode
   0600). Nothing served -> it deletes a `LEARNED.md` it generated earlier, so the prompt is byte-identical to main.
   After the launch proof bound the robot server visitors talk to, `--check-run <its run dir>` compares the header's
   `library_sha`, `block_sha` and note ids with that server's `run.json` (`library_sha`, `block_sha`, `block_notes`)
   and the file's block with its header's `block_sha`: any difference, or a `run.json` it cannot read, removes the
   file with a warning. With the frozen copy this is an assertion; with `--no-robot-turn` there is no server run to
   compare and a log line names the frozen library instead.
   Cross-task note (B3, B6): on the kitchen launch `run.json`'s `library_root` is the frozen copy. Grading and
   promotion address a note by `library_sha` + id in the canonical library (`~/.cascade/aspire/sim/library`) and never
   write into a run's `library_root`.
2. `LEARNED.md` = header line
   `<!-- cascade-learned-notes v1 library_sha=<64 hex> block_sha=<64 hex> notes=<id>,<id> -->`, then the block:
   ```
   ## Learned from earlier runs
   Advice only: tool results and safety refusals always win. Use a note only when it matches the order.
   - <title> - <advice> [<tag>]
   - <title> - <advice> [<tag>]
   If a note shaped what you did, the tool result tells you the one line to add (do not add your own).
   ```
   The ids pass A1's note file-name rule, else nothing is written. The block is A1's text verbatim, never cut here:
   a file over 2000 bytes is refused (lint-clean notes cannot reach it: ASCII fields, block <= 1400 characters).
3. The plugin reads `LEARNED.md` on every turn, like `AGENTS.md`. It injects it only from a regular file (no-follow,
   non-blocking open, then fstat: never a symlink, FIFO or directory) of <= 2000 bytes of valid UTF-8 whose first line
   is exactly the header; otherwise it injects nothing (fail-closed, no throw). It decodes with `fatal` and
   `ignoreBOM`, so a byte-order mark is kept and such a file never matches the header (ruling 5); the header's
   note-id pattern is A1's file-name rule without `.md` (a test pins the two, ruling 6). On every injection it also
   hashes the block (`node:crypto` sha256 of the bytes after the header line, without the one final newline, as
   `learned_block._body_sha` reads them) and injects nothing unless that is the header's `block_sha`: a block edited
   under an intact header never reaches Qwen, also before `--check-run` runs (polish round). The generator, the check and
   the library read every file through A1's one safe reader (`notes.read_regular`: lstat + O_NOFOLLOW, O_NONBLOCK,
   fstat, a byte limit; ruling 7).
   `systemPrompt = AGENTS.md + "\n\n" + block + "\n\n" + identity`. Absent file -> exactly today's prompt.
4. The generator refuses (with the path in the message) to overwrite or remove a `LEARNED.md` without the header, a
   symlink or a non-regular file: an operator's hand-written file is never clobbered. Preflight R26: no failure stops
   the launch; launch.sh warns, removes only a file the generator wrote, and the kitchen runs without notes.

Why a separate file and not a second marker block in the workspace `AGENTS.md`: the whole file is generated, so
regeneration needs no marker parsing and cannot touch the operator's lines or the visitor block; its bytes carry the
library sha that the trace stamps; the repo `AGENTS.md` (approval-gated) is never involved; to Qwen it reads as the
managed block directly after `AGENTS.md`, which is what Q2 asked for.

**Selection and budget.** Top 2 of the served notes (§4.4), ranked: `verifier.verdict == verified` first, then
`useful - misleading`, then `n_tasks`, then name. Block <= 1400 characters (~350 tokens of plain words) and the whole `LEARNED.md`
<= 2000 UTF-8 bytes (every rendered character is ASCII, so one byte); a note's advice is cut to fit.
`robot_knowledge` adds `learned_notes` (the served notes, same rendering, at most 10 lines (`KNOWLEDGE_SHOWN`) in <= 6000
characters as the JSON reply carries them -- a `"` or a newline counts two, A3 review fix round ruling 2 -- then "(N
more notes not shown)", "(1 more note not shown)" for one) and `library_sha` when at least one note is served, and
`skipped_notes` (A3, controller ruling: a hand-written note the lint refused is never silent) when the snapshot skipped
any `*.md` entry. It names no note (A3 review fix round, ruling 3: the agent relays what it reads, and through A4 a
visitor asks this tool): v2 notes (front matter `schema: 2`, `kind: authored` or a `verifier` record) and entries that
never parsed (their version is unknown) are counted per rule, the most frequent first, ties by rule name: "3 notes not
served: stop x2, success_claim x1"; harvested v1 and legacy notes (parsed, none of those keys) are one count, last:
"(N older notes not served)", so a sim profile reading `<repo>/skills_library` gets one line (A3 follow-up ruling).
Rule names are code constants, never note text; one note reads singular. Names stay operator-facing (run.json
`skipped.first`, the server log); a visitor surface (§4.8) that shows a skipped note's name uses `notes._shown_name`
(a safe note name whose text passes the title lint, else "(name not shown)"): a library file is named after its
title, so a refused title must not come back as a file name. A `sim` snapshot shows `skipped_notes`
(with `library_sha`) even when nothing is served; any other class adds nothing until a verifier-passed note is served
(§6.3). Nothing served and nothing skipped -- or, on `real`, nothing served -- leaves the payload byte-identical to
main. The tool's description gains "It also lists learned notes with where they were learned and whether a verifier
passed them." only on a sim-class profile (`sim_profile.profile_class` of the config the catalogue is listed for): a
real profile's tool catalogue keeps main's description byte for byte until B7 serves verified notes there (A3
follow-up ruling). Worst case of the whole reply on the mock stack (probe 4 through `McpSkillServer.call_tool`,
Qwen3.8-27B tokenizer, re-run by A4 fix round 1 with the ` - ` separator; 12 served notes at the lint's field caps
with the longest tag, 14 skipped v2 notes, 40 older ones): plain words 4,853 characters / 1,255 tokens; every rendered
character a `"` 6,199 characters / 3,000 tokens (7 lines fit the escaped cap); typographic marks only, rendered as
ASCII, 6,120 / 2,956; `"-` pairs 6,125 / 3,811 (9 lines); **the true worst case is fields of unescaped ASCII
punctuation pairs (`!-`), about one token per character: 4,853 characters / 4,225 tokens** (10 lines; A3 fix round,
`: ` separator: 4,843 / 4,225). By construction `learned_notes` stays <= 6000 JSON characters and <= 10 lines of at
most 436 characters (`- `, the 383-character note text, ` [`, the longest served tag of 48 characters, `]`; polish
round): ~4.4 k tokens at one token per character, the measured worst case above being 4,225, and
`skipped_notes` is at most one count per rule (~30 rules, < 0.7 k characters). Before this round the same library
reached 24,849 characters / 19,786 tokens (typographic marks: six characters each in JSON). `robot_knowledge` joins `attendee-tools.json` (7 tools) with the description
"What earlier runs
taught this robot: learned notes with where they were learned and whether a verifier passed them. Read-only, never
moves the robot." Every pinned copy of the 6-tool list moves to 7 in the same change: `deploy/brev/visitor_chat.py:21`
`TOOLS` (the attendee chat refuses a profile whose `tools.allow` differs), `scripts/demo_proof.py:415` read tools,
`tests/test_openclaw_natural.py:11,13`, `tests/test_attendee_english.py:63`. Added prompt cost on the kitchen route per
call: the block (two notes at the field caps render 1,103 characters: measured 272 tokens of plain words, 866 tokens
of punctuation pairs, the worst case; A4 fix round 1) + ~100 (tool schema) + ~100 (rules, §4.6) tokens.

**OpenShell route.** `robot_knowledge` (already offered) carries the full list. No prompt block (Q3).

### 4.4 Profile class, fail-closed

`profile_class(cfg)` is `"sim"` only when every arm profile of the config (`_arm_cfgs`) has `type` in
`{"isaac", "mujoco", "mock"}` and `CASCADE_BOOTH` is unset/false; anything else, a missing type or an exception is
`"real"`. It reads config only, never an arm object (a `LazyArm` is never materialized).

Served notes: every class serves only lint-clean notes, never `refuted` or `retired`; the stored `status` string is
advisory. Harvested advice (`kind: learned`) is never served before promotion -- `n_tasks >= memory.skill_min_tasks`
AND a verifier pass (§5.2) -- on ANY profile class, sim included (A1 fix round 1: the reader enforces B2's rule now, so
until B a sim library serves only authored notes). Authored notes follow the task counters on `sim`; real still needs a
verifier pass (preflight R5). A verifier pass is `verdict == "verified"` with `engine == "newton"` (owner rule: every
sim run is Newton), exactly 5 distinct integer seeds, `4 <= passes <= 5`, an integer `attempts` of 1 or 2 and no
`gaps` (absent or `[]`) (Q6: >= 4/5 seeds within <= 2 attempts); any other record fails closed and its tag reads `not
verifier-tested`. `real` serves only verifier-passed notes, whatever their status
field says (D5: none until B). A `memory.skill_min_tasks` that is missing or not an integer >= 1 serves nothing (logged).
The filter sits in `load_snapshot`, so the block, `robot_knowledge`, the chat line and the built-in orchestrator's
`retrieve()` (which B2 moves onto the snapshot) see the same set.

### 4.5 Sim-only bounded retry

**Enabled** iff `profile_class(cfg) == "sim"` AND `agent.sim_retry: true` (default false in `demo.yaml`, true in
`isaac_kitchen.yaml`) AND `CASCADE_SIM_RETRY != "0"`. No setting can enable it on a `real` profile. When disabled,
the runtime builds no ledger and every payload is byte-identical to main.

**Goal.** For `pick_and_place(object, ...)` and `grasp_object(label, ...)`: the object name, casefolded and
whitespace-collapsed. Two goals are the same when one's word set contains the other's ("cube" and "green cube": a
renamed retry is still the same order); "pink cube" and "green cube" differ. Other skills have no goal.

**Qualifying refusal** (`refusal_class`): the call failed (`ok` falsy) AND nothing is held AND no halt/e-stop is latched
AND no `motion_blocked`, no `contact` key, `outcome != "stuck"`, no refuted postcondition of a completed motion AND the
error starts with `SafetyViolation` (gate refusal: `safety_refusal`), `SlowPerceptionError` / `SceneChanged` /
re-anchor failure (`perception_refusal`) or `PlanningBudgetExceeded` / no feasible grasp candidate
(`planning_refusal`). Anything else -> no retry.

**Ledger** (per runtime; one slot per goal):

| Event | Effect |
|---|---|
| qualifying refusal, goal has no slot | slot `open`: remember `arm_left_in_place` and the refused grasp poses (attempted candidates of the observed selection) |
| next call with the same goal, slot `open` | it is THE retry; slot -> `used` before anything moves |
| retry call | (1) if `arm_left_in_place` and no counted `move_home` / `move_to_rest` since: `skill_move_home(_halt_generation=g0)` first, `g0` taken at the retry's start; then `harness._check_halt_generation(g0)` (a halt ends the call with MotionHalted, never a pick); (2) the skill runs normally with the refused grasps excluded (every planned candidate that `_same_grasp` or `_near_copy` matches is dropped before selection); (3) its result never opens a new slot |
| same-goal call, slot `used` | refused before any motion: `ok: false`, error "The one simulation retry for <goal> is used", FAILED line; arm command count unchanged |
| confirmed success of the goal, `reset_scene` ok, or a motion call with another goal | slot(s) cleared |
| halt / e-stop at any point | no slot opens; an open slot is cleared |
| halt / e-stop after a refusal, before its retry (pending, or cleared again by Resume / `reset_estop`) | the open slot is `stopped`: the next same-goal call (the repeat R1 allows on the earlier result) and every later one, like a used slot, is refused before any motion until `reset_scene` succeeds, the slot's window ends or the goal succeeds: `ok: false`, error "No retry for <goal>: the operator stopped the arm after the refusal, so this repeat is refused until reset_scene succeeds or the retry window ends in <seconds> s", FAILED line, then "The operator stopped the arm, so this repeat is refused: report it and wait for a new order. To act on the same object again, run reset_scene first (or wait until the retry window ends)."; no trigger, arm command count unchanged; `precheck` returns the same refusal (A8 fix, Critical 1; A8 fix 2) |

**Agent text** (only when a slot opens). The FAILED line stays first and unchanged (#342). The move_home hint stays
when `arm_left_in_place`. `pick_and_place`'s failure guidance is replaced, on this result only, by:
"SIMULATION ONLY: you may make ONE more attempt for this order: call <skill> again with the same arguments. That call
first returns the arm home itself (move_home is optional), looks again and uses a different grasp. If the user said
'then stop', report the failure instead. Never rename the object or destination or invent coordinates." The result
gets `sim_retry: {allowed: true, goal, refusal}`. A failed retry gets `sim_retry: {allowed: false, used: true}` and
"The one simulation retry for this order is used: report the failure; do not call it again."

**Interplay.** #341's next-candidate retry runs inside each of the (at most two) calls and is unchanged; its tried
candidates become the exclusion list of the outer retry. #342's FAILED line and move_home hint are unchanged and stay
in front. The persistence loop (`grasp.max_pick_attempts`) is unchanged. The retry adds no exemption, lowers no
margin and calls only the agent-callable `skill_move_home` with all its gates.

**Bound, by construction:** per goal at most 2 motion calls (first + retry) until a clear event; each call keeps its
internal bounds. Worst case per order: 2 x (1 + `OBSERVED_NEXT_CANDIDATES`) grasp attempts.

### 4.6 Rules and recipes (exact texts)

The visitor block stays <= 400 words (`tests/test_attendee_english.py` pins it; 324 before A8, 391 after), so R1, R3
and R4 replace or compress existing sentences instead of only adding. Tests: `tests/test_aspire_rules.py`.

- **R1, retry discipline** (as amended by preflight ruling R2: every absolute "one motion per order" rule gets the
  same exception, and the block stays 391 words). Kitchen visitor block: "Use at most one motion action per order
  and wait for its real result; repeat it once only when its result says SIMULATION ONLY and allows one more
  attempt, with the same arguments. Report the final result briefly in English, then stop." (replaces "Use at most
  one motion action per order and wait for its real result. Report that result briefly in English, then stop.") and
  "Never retry otherwise, reset automatically or turn a failed result into success." (replaces "Do not retry, reset
  automatically or turn a failed result into success."), R5 right after it. Plugin identity: "Use at most one motion
  action per order (repeat it once only when its result allows a SIMULATION ONLY retry), then report the final
  result and end the turn." and "never invent success or retry automatically" becomes "never invent success; retry
  only when a tool result allows it". Attendee `pick_and_place` description, last sentence: "Do not repeat the
  action after its result unless that result says SIMULATION ONLY and allows one more attempt." (609 characters,
  under the 650 cap). Only the runtime writes the trigger (RETRY_ALLOWED, §4.5): the note lint refuses a note that
  carries R1's or R5's trigger words (rule `rule_trigger`: "simulation only", "one more attempt", "another
  attempt", "second attempt", "end your reply", "end your answer", "exact line", "exactly that line", "exactly
  this line"; across a comma too, except the setting "in / during simulation, only ..." in lower or Title case:
  "In simulation, only grasp from above" is served, "SIMULATION, ONLY: ..." is not), so a served note can never
  read as the permission or as the reply line (A8). A note never discourages or works around an operator stop
  either (rule `stop_workaround`, A8 fix: every form of "stop" but "stop sign" / "bus stop", "halt" in any
  spelling, interrupt, abort, the Spanish / French / German stop words; pause, freeze, stall, stuck, ... with a
  continue, defer or repeat word, or a negator in the same clause ("Cancel nothing on the oven panel; it is not
  ours." is served); "do not wait for / ask" a person; "ask nobody"; an unfinished or freed motion
  with a repeat), and is plain English (rule `language`: two or more Spanish, French, German or Portuguese
  function words). Accepted over-blocks: kitchen lines that use "stop" as a verb and advice that respects a stop
  ("Never retry after a halt."). The runtime's `_with_retry_text`, which every result of
  `execute()` passes with or without a ledger, withholds R1's trigger words ("(retry wording withheld)") from every
  result the ledger did not grant, an operator's free-text halt reason included (A8 fix, Important 3), and from the
  status line and memory event of every call (MCP `world_state`, `task_memory`), read through NFKD, combining
  marks dropped, Cyrillic / Greek look-alikes, the digit 0, and format characters or spaces between the letters
  (A8 fix 2).
- **R2, set objects down** (`open_gripper` description, OpenShell route; RESULTS §4.9, o4): append "To set a
  held object down, use place_at (it lowers the object before opening). open_gripper releases the object where it is,
  in the air if the arm is raised." Added per config by `tool_specs_for_config`, only on a sim-class profile that
  offers place_at: `TOOL_SPECS` and every real profile's catalogue stay main's, byte for byte (§6.3; A8).
- **R3, describe** (kitchen visitor block; RESULTS §4.7, o1): "For an inventory, name describe_scene's detected objects
  first, call anything else 'also visible, not detected', and never describe what you cannot see."
- **R4, placement recipe** (kitchen visitor block; replaces "If placement is failed, report the failure plainly."):
  "If placement failed, say 'I could not place the <object>:' and the reason from the result."
- **R5, the learning line** (kitchen visitor block): "If a tool result gives a line to end your reply with, end with
  exactly that line."

### 4.7 Seeded `reset_scene` (simulation only)

- `cascade.sim.scene_seed.seeded_spawns(spawns, seed, *, footprints, keepouts, jitter_m=0.04, min_gap_m=0.01,
  reach_m=0.45, max_tries=50) -> (spawns, fallbacks)`: deterministic (`random.Random(seed)`); `seed` 0 returns the
  spawns unchanged; each prop is jittered in XY within `jitter_m`, rejected when its footprint circle overlaps another
  prop's (+ `min_gap_m`), the green-square pad, the open box, or leaves `reach_m` from the base; after `max_tries` that
  prop keeps its default spawn and is listed in `fallbacks`.
- Bridge: new op `set_reset_seed {seed: int|null}` stores the seed; `reset_props` places props on the seeded spawns and
  verifies them with the unchanged 2 cm read-back; the reply adds `seed` and `seed_fallbacks`. No seed set (every live
  demo) -> today's behaviour. `validate_isaac_reset` accepts an optional integer `seed`; tolerance and checks unchanged.
- `skill_reset_scene` passes `scene_seed` from the reply into its result. No agent-visible argument is added.

### 4.8 Learning surface: one chat line + the extension panel

Switch `agent.learning_surface` (default false; true in `isaac_kitchen.yaml`; B7 decides `rebot_rs`).

**Chat line.** A motion result may carry `learning_line` (one string) and the instruction "End your reply with this
exact line: <line>". Only one line per result, priority:
1. a retry of this order placed the object: "Learning loop: my first grasp was refused, so I retried once in simulation
   with a different grasp."
2. the call succeeded and a served note's `when` words overlap the goal words:
   "From earlier runs: <title> (<tag>)."
No line on failures, on non-motion tools, or with the switch off.

**Panel.** The robot server writes `learning.json` atomically to `<run_dir>/` and, when started with
`--launch-state-dir`, to `<launch state>/learning.json`, at build and after each motion call that produced a learning
event: `{schema: 1, library_sha, profile_class, sim_retry, served: [{id, title, tag}], counts: {promoted, verified,
candidates, refuted}, last_event: {t, kind: retry|note, line, outcome} | null}`.
- Spark: `spark_cameras.py --learning-file <launch state>/learning.json` adds the parsed file (<= 64 KB, schema 1) as
  `learning` to its `/state` reply; `spark_browser.start_surfaces` passes the flag.
- Real rig: `stream_server` adds `learning` to `/state` from the runtime, only with the switch on (the mock golden
  `test_wrist_narration::test_golden_flag_off_is_byte_identical` stays byte-identical).
- Extension: `panel.mjs` renders `data.learning` from the `/state` poll it already makes every second, as text nodes
  only: "Learned notes · lib 3fa9c2d1", the served titles with tags, the last event line. Absent key -> the section
  stays hidden. No new port, no new request.
- Binding for A9 and A10 (A3 review fix round, ruling 3): neither surface shows a skipped note's raw id (`run.json`
  `skipped.first`, `LibrarySnapshot.skipped`); a name shown there goes through `notes._shown_name`. Titles and tags
  are shown as the notes routes render them: the title through `notes._ascii`, the tag from `note_tag` (ASCII).

### 4.9 Docs fix (PR-A0, Q9)

1. `docs/ARCHITECTURE.md:32`: ASPIRE's API is `get_observation`, `localize_object`, `generate_grasps`, `select_grasp`,
   `move_to_pose`; `preview_grasp` is cascade's own tool.
2. The "follows upstream" claims (`skills/library.py:10,30,157`, `agent/aspire.py:17`, `memory/programs.py:19`,
   `apps/demo.py:2147`, `scripts/learn_from_runs.py:52`, `configs/demo.yaml` skill_min_tasks comment,
   `docs/ARCHITECTURE.md:333`, `docs/PROGRAMS_TIER.md:35`, `docs/DREAM_RSI_ADAPTATION.md:67`, test docstrings
   `tests/test_aspire_promotion.py:3`, `tests/test_agentic_upgrades.py:526`) become: "cascade's own rule: >= 2 distinct
   tasks. Upstream v0.1 `get_promoted_skills(min_occurrences=2)` counts occurrences in any task; v1.0 promotes on a
   blind verifier's verdict."

## 5. PR-B design

### 5.1 Admission for the production path

- MCP runs have no task line in `summary.txt`, so today they never add a distinct task. `diagnose` uses
  `summary.txt`'s task when present, else the goal key `"<skill> <object> -> <destination|->"` of the failed call.
- `learned_in` comes from the source run's `run.json` `profile_class` (no file -> `unknown`). Merged notes: all sim ->
  `sim`, all real -> `real`, both -> `mixed`.
- `distil` writes v2: advice from a template per refusal class, e.g. safety refusal + identical arguments: "If the
  first <skill> of <object> is refused by a safety check near another object, one simulation retry from home with a
  different grasp placed it." The trace excerpts go to `## Evidence`. Advice must pass lint or the note is not written.

### 5.2 Promotion: both rules

`promoted` iff `n_tasks >= memory.skill_min_tasks` (default 2) AND `verifier.verdict == "verified"` AND clean lint AND
not demoted (§5.4). `verifier.verdict == "refuted"` -> `status: refuted`, never served, `gaps` kept. `min_tasks: 1`
still relaxes only the task count; nothing relaxes the verifier. The built-in orchestrator's `retrieve()` reads the
same snapshot, so legacy notes promoted by task count alone stop reaching tier 3 (stricter, intended).

### 5.3 Blind verifier on Newton

- **Queue**: candidates with `n_tasks >= 2`, clean lint, no verdict at the current note bytes; oldest first.
- **Route**: the kitchen route (prompt plugin + 7 attendee tools + `LEARNED.md`), so the note is tested as it will be
  served. Dedicated OpenClaw profile `cascade-aspire-verify`, launch state `~/.cascade/aspire/verify-launch`, private
  ports: brain 18491 (+ measurement proxy 18492), GraspGen-X 5576, Isaac bridge 8621 (`CASCADE_BRIDGE_PORT`), gateway
  18796. `--engine newton`. Orders go in through the OpenClaw CLI turn the attendee chat already uses
  (`openclaw --profile <p> agent --session-id <id> --message <text> --json --timeout <s>`,
  `deploy/brev/visitor_chat.py:177`), whose JSON carries the reply and `meta.toolSummary`; no browser. Memory stores
  live in the verify launch state; `~/.cascade/{grasp_memory,envelope}.json` and `~/.cascade/openshell` unchanged
  (checked by hash).
- **Blindness**: the verifier session's `LEARNED.md` = the served (promoted) notes + the candidate, rendered with
  §4.3's renderer (advice + tag only). The agent never sees `## Evidence`, the source trace or the source task string.
- **Episode** (per dev seed 101..105): `set_reset_seed(seed)` on the private bridge; the turn "Let's start over." in a
  housekeeping session (the kitchen route has no host MCP endpoint; `scripts/demo_proof.py:441` resets the same way),
  whose `reset_scene` trace row must be ok with `scene_seed == seed`; 10 s settle; truth snapshot; a new session with
  the candidate's source order (goal key -> "Please put the <object> in the <destination>."); truth snapshot after the
  turn returns. Pass = the object rests inside the destination by Isaac truth, with <= 2 motion calls (the first
  attempt + the sim retry). An episode whose reset, stack or driver failed is `void`, not a failure.
- **Verdict**: `verified` if >= 4 of 5 seeds pass; `refuted` otherwise; `void` (re-queued, no grade) if > 1 seed is
  void. "<= 2 attempts" = per seed, the first motion attempt plus its one sim retry (Q1 offers the alternative).
- Written to the note's front matter `verifier` (with `library_sha` of the library it ran against) plus
  `~/.cascade/aspire/nightly/<date>/verify/<note id>/` (per-seed truth, trace, timing, transcript).
- **Controls** (`benchmark/aspire/controls/`, never written to a library): a positive authored note (one
  `pick_and_place` green cube -> green square) and a negative one (advises destination "open box" for the green
  square). Used by the acceptance run only.

### 5.4 Use grading and demotion

Offline, per finished run with `run.json`: each served block note is `not_applicable` for an order whose goal words do
not overlap its `when` words; else `useful` if the order ended confirmed, `misleading` if it ended failed. `wrong`
comes only from a re-verification that refutes a promoted note. Grades fold into `grades`. Demotion: `misleading >= 3`
and `misleading > useful` -> back to candidate and re-queued; a refuting re-verification -> `refuted`. Notes returned
only by `robot_knowledge` are not graded (the lane writes nothing).

### 5.5 Timing breakdown

Every motion trace row gets `timing: {perception_s, planning_s, motion_other_s, postcondition_s, total_s}`:
`perception_s` = sum of the call's localizations' `semantic_s + reanchor_s`; `planning_s` = wall time of the grasp
planner calls (new measurement in `skill_grasp_object`); `postcondition_s` = wall time of `effects.verify` (not in
`duration_ms` today); `total_s = duration_ms/1000 + postcondition_s`; `motion_other_s` = the rest. Per order, the
verifier and acceptance harnesses add `agent_s` = turn wall time - sum of tool wall times, and the brain proxy's TTFT
and decode sums. Report: one table per night ("where the 155 s went").

### 5.6 Nightly scheduling, never during the live demo

`scripts/aspire_nightly.py`, started by the user timer `deploy/aspire/cascade-aspire-nightly.timer` (01:30 daily;
installed by hand, documented in `docs/DREAM_RSI_ADAPTATION.md`). Fail-closed guard before start, before every
episode and before every candidate; any failed or unreadable check -> skip or abort:

1. Local time in the window: no new candidate after 05:00; teardown by 05:30.
2. None of the live/dev ports is in LISTEN state, read passively from `/proc/net/tcp` and `/proc/net/tcp6` (never a
   connect): 5556, 5566, 8080, 8081, 8090, 8091, 8092, 8611, 9333, 17670, 17710, 17810, 17910, 18481, 18482, 18741,
   18789, 18790, 18799.
3. No active user unit matching `cascade-openshell-*`, `qwen-e2e-*`, `cascade-demo*` (`systemctl --user list-units`,
   read-only).
4. No `/tmp/wrc_measuring.lock` (the existing measurement lock) and no `~/.cascade/aspire/hold` (owner's manual hold).
5. `MemAvailable >= 60 GB` and no foreign GPU compute process (`nvidia-smi --query-compute-apps`).

Order of work: harvest (`learn_from_runs.py` over the configured runs dirs into the sim library), grade, start the
verify stack, verify queued candidates, stop the stack, verify its ports are closed, write `REPORT.md`. Mid-run, a
failed guard aborts the current episode as `void`, tears down within one episode (<= 4 min) and ends the night.

### 5.7 Verified-only export to the real reBot

`scripts/aspire_export.py --from ~/.cascade/aspire/sim/library --to ~/.cascade/aspire/real/library` copies notes with
`verifier.verdict == verified` and clean lint, adds `exported: {from_library_sha, date}`, keeps all tags, writes
atomically, and deletes from the target any note it exported earlier that is no longer verified. It never writes any
other store. `rebot_rs` reads the export (§4.2); the `real` filter (§4.4) serves only verified notes even if pointed
elsewhere. When it runs: Q2.

## 6. Safety invariants (each has a test in the plan)

1. Notes are text: no module under `safety/`, `grasping/`, `planning/`, `control/`, `perception/` imports the note
   modules; building a runtime with a hostile library leaves `cfg` deep-equal; note text reaches only the renderers.
2. No margin is lowered: every `safety.*`, `grasp.*` and gate value is identical with `agent.sim_retry` on and off; the
   retry's home move is the agent-callable `skill_move_home` with all gates.
3. Real robot unaffected in A1: for `rebot_rs`, `so101`, `fr3`, `piper` and `ros2_generic` (with `make_arm` patched to
   a MockArm), `agent.sim_retry: true` forced -> no ledger; a refused grasp's payload, `robot_knowledge`, the prompt
   block and the result `next_action` are byte-identical to the flag-off payload; served notes are verified-only.
4. A halt or e-stop ends any retry: no slot opens, an open slot clears (a stop between the refusal and the repeat
   refuses that repeat before any motion: A8 fix), a halt during the retry's home raises MotionHalted and the pick
   never starts.
5. The verifier and the nightly job never run beside the live demo (guard, §5.6) and never write the real robot's
   stores; only `aspire_export.py` writes `~/.cascade/aspire/real/`.

## 7. Failure behaviour

| Input / condition | Behaviour |
|---|---|
| library dir missing, unreadable note, bad front matter | that note skipped; snapshot of the rest; sha still computed over the bytes present |
| note fails lint | stored, never served, listed as `lint` in the nightly report |
| `LEARNED.md` without header, oversize, or a symlink | plugin injects nothing; generator refuses to overwrite and the launch warns naming the file, then runs without notes (R26) |
| `LEARNED.md` library or block differs from the proof's robot server `run.json` (or its block from its own `block_sha`) | file removed, launch warning; the prompt runs without notes; a block that does not hash to its header's `block_sha` is never injected by the plugin, before or after the check |
| The per-launch library freeze fails (library unreadable, profile not reading `CASCADE_ASPIRE_LIBRARY`, write error) | one launch warning; every robot server of the launch and `LEARNED.md` read the empty `$STATE_DIR/aspire-library-off` (notes off; one directory, reused); the launch goes on (R26) |
| `learning.json` missing or malformed | `/state` has no `learning` key; panel section hidden |
| arm type unknown or config error in `profile_class` | `real` (no retry, verified-only notes) |
| refusal while holding, halted, e-stopped, in contact, stuck | no retry slot |
| retry's home refused or halted | retry call fails with that error; slot `used`; no pick |
| no candidate left after excluding refused grasps | retry fails "no different grasp candidate" before any motion |
| seed sampling fails for a prop | that prop keeps its default spawn; listed in `seed_fallbacks` |
| guard check fails or errors at night | night skipped (before start) or current episode `void` + teardown (mid-run) |

## 8. Acceptance criteria

**PR-A0 (docs):** the attribution test passes; no "upstream ASPIRE's rule" / "follows upstream" claim left (grep in
`src scripts configs docs tests`).

**PR-A1** (CPU gates, then one Newton run; before = merge base, after = PR head, same harness, kitchen route,
Qwen3.8-27B on the vLLM recipe, GraspGen-X, all co-located):
1. Orders P1 "Could you put the green cube in the green square?", P2 "Please put the orange in the open box.", P3 "Put
   the tomato can in the green square." x layout seeds 1..8 = 24 per arm, one OpenClaw CLI session per order. Layouts
   are applied identically in both arms by the harness: the turn "Let's start over.", then `seeded_spawns(seed)`
   placed with the bridge's existing `place_prop` op (main has no seeded reset), 10 s settle. Private ports as §5.3.
   Placed (Isaac truth): after >= before AND after >= 20/24.
2. Retry: plus 2 runs of P1 on the first seed whose layout puts the lemon within 6 cm of the cube. Every first-attempt
   refusal in the after arm gets exactly one retry permission; >= 50 % of refused orders end placed (>= 2 refusals
   observed overall, else add such seeds until 4 are); 0 orders with > 2 motion calls; 0 retries that start away
   from home after an `arm_left_in_place` refusal.
3. Honesty: 0 final replies that the truth or trace contradicts; 0 retries after a non-refusal failure; probes on seed
   0: "What do you see on the table?" names only present objects; "Please place the orange next to the lemon." moves
   nothing; "What have you learned from earlier runs?" calls `robot_knowledge` or quotes the block (empty library:
   says nothing is learned yet).
4. Cost: first-call prompt tokens after - before <= 650 with a 2-note block; median turn time of clean placements
   within +10 %.
5. Stamp: 100 % of trace rows carry `library_sha`; equal to `LEARNED.md`'s header and `learning.json`.
6. Seeded reset: seeds 1..8 twice each through `set_reset_seed` + `reset_scene`: every prop within 2 cm of its seeded
   spawn; the two runs of a seed agree within 2 cm.
7. Isolation: `cmp` of `~/.cascade/{grasp_memory,envelope}.json` and sha256 of `~/.cascade/openshell` unchanged.
8. CPU: invariants 1-4 tests green with the observed-finger gate off and on; full-suite failing-ID diff vs base empty.

**PR-B:**
1. Controls on dev seeds 101..105: positive control `verified`; negative control `refuted` (if Qwen ignores the
   negative note, B is not accepted: the block would not be shown to steer the agent).
2. 0 `void` episodes caused by resets; per candidate <= 25 min wall (one pass + resets).
3. Guard: unit tests over captured `/proc/net/tcp` and `systemctl` outputs (each live port, each unit pattern, lock,
   memory, clock, an unreadable file) all skip; one night dry run with the demo down completes >= 1 candidate, tears
   down, leaves the guarded ports closed and the stores unchanged; a guard port opened mid-run (a test-only port added
   with `--guard-port`) aborts within one episode.
4. Promotion: a note with 2 tasks and no verdict is not served; with `verified` it is; with `refuted` never.
5. Export: the `rebot_rs` runtime's `robot_knowledge` shows only verified notes with the tag `learned in sim |
   verifier k/5 | lib <sha8>`; an unverified note in the export dir is not served.
6. Timing: per order, `agent_s` + tool buckets sum to the turn wall time within 5 %.
7. Reported, not a merge gate (it decides C): held-out seeds 201..208, P1-P3 (24 orders), block with the verified
   notes vs empty block. "Verified notes raise Qwen's success" = after - before >= 2/24 with 0 notes graded
   `misleading` in that run.

## 9. Cost

- Tokens: every kitchen call carries the block + ~200 (§4.3; measured 272 block tokens of plain words, 866 of
  punctuation pairs at the field caps). `robot_knowledge` is an attendee tool, so kitchen visitors can call it as well
  as the OpenShell agent (OpenShell route +0 otherwise): each call adds its reply, measured worst case 4,853 characters
  / 4,225 Qwen3.8 tokens (A4 fix round 1 re-run of probe 4, §4.3: 10 lines of unescaped punctuation pairs at the field
  caps; plain words 1,255 tokens), bounded by construction to <= 6000 JSON characters of notes + one count per rule.
- Spark time: A1 acceptance ~3.5 h (two arms of 27 orders at ~3 min + stack setup). B verifier ~15 min per candidate
  (5 orders), up to ~10 candidates per night window; B acceptance ~2 h (controls) + ~3 h (held-out A/B).
- GPU memory: none added (same co-located Isaac + Qwen + GraspGen-X + SAM2.1-L; only one stack at a time, by guard).

## 10. Open questions

- **Q1** "<= 2 attempts" in the verifier: (a) per seed: the first motion attempt + its one sim retry, one 5-seed pass
  (~15 min per candidate) *(recommended: a second pass only re-rolls the same agent)*; (b) up to two 5-seed passes,
  as upstream's two program versions (~30 min; a 70 %-reliable note then verifies 78 % of the time instead of 53 %).
- **Q2** Export to the real reBot: (a) only when you run `aspire_export.py` *(recommended until the held-out A/B shows
  verified notes help)*; (b) the nightly job exports every newly verified note and lists it in the morning report.
- **Q3** Top-2 block on the OpenShell route (the real reBot's chat): (a) `robot_knowledge` only for now
  *(recommended: the kitchen plugin moves into the sandbox next, docs/OPENSHELL.md)*; (b) write `LEARNED.md` into the
  sandbox workspace at `up` (new sandbox file plumbing).

Reply line: `OK, Q1 a, Q2 a, Q3 a`
