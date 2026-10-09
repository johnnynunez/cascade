# B44-live: local Qwen as the outcome judge over recorded live Isaac picks (9 Oct 2026)

**Question.** Does the judge-vs-physics confusion matrix that B44 (#274) puts in the launcher's run summary mean
anything with the local brain as the judge? This measures it on recorded live Isaac picks.

**Data.** 72 live Isaac runs, each with exactly one `pick_and_place` that the physics channel graded. 65 come from
the B36 pick-reliability A/B series on the bare reBot scene (the pink cube to the drop zone, PhysX and Newton,
several gripper variants). 7 come from the B35 NemoClaw runtime runs. The physics grades are 40 `confirmed`,
27 `refuted` and 5 `unverified`. `manifest.json` gives, for every run, its source directory and the sha256 of its
`trace.jsonl` and of the BEFORE/AFTER keyframes (1280×720 JPEG; not committed, 21 MB). The judge read copies,
never the sources.

**Judge.** `scripts/judge_run.py --skills pick_and_place` with the GRM prompt verbatim and a single front view
(wrist slots = front repeat). The model is `Qwen/Qwen3.8-27B`, served by llama.cpp on `127.0.0.1:8080` (local,
OpenAI-compatible). `extra_body.chat_template_kwargs.enable_thinking: false`, temperature 0.1, top_p 0.9.
- Pass 1 used `max_tokens: 1536`, the budget `deploy/runtime/runtime.py` ships for the Spark judge.
- Pass 2 re-judged, once, only the steps pass 1 left unscored, with `max_tokens: 8192`.

The script is `judge_all.py` and the per-step records are in `results.json`: hop, tokens, finish reason, the tail
of the raw answer, and the physics evidence string. Wall time for pass 1 was 1107 s for 72 calls.

**Decision rule.** This is the code's rule (`StepVerdict.agrees_with_physics`): the judge says "progress" iff
`hop > 0`, and physics says success iff the postcondition is `confirmed` on the physics channel. Unverified steps
are never counted.

## Results

| | scored | physics-graded | tp | tn | fp | fn | agreement |
| --- | --- | --- | --- | --- | --- | --- | --- |
| pass 1 (1536 tokens) | 60 / 72 | 56 | 40 | 1 | 15 | 0 | 73.2 % |
| pass 1 + pass 2 (8192 for the 12 unscored) | 71 / 72 | 66 | 40 | 7 | 19 | 0 | 71.2 % |

- **Confirmed picks.** All 40 scored exactly +100 %, so fn = 0: the judge never missed a physics-confirmed pick.
  This is the `fn` regression signal B44 exists for.
- **Refuted picks.** They scored from −100 % to +90 % (26 scored, 1 never scored). Most moved the cube part of the
  way, and the GRM prompt rates partial progress as a positive hop. Under `hop > 0`, that counts as `fp`.
- **Unscored.** Pass 1 left 12 steps unscored, 11 of them refuted and 0 confirmed. Despite
  `enable_thinking: false` and the prompt's "ONLY one line", Qwen writes its reasoning into the answer. On hard
  cases it runs past 1536 tokens without emitting `<score>`. At 8192 tokens, 11 of the 12 got a score. Sampling is
  stochastic: one pass-1 abstention scored after 308 tokens on the re-run.
- **Post-hoc threshold.** Reading success as `hop ≥ 1.0` would separate all 66 physics-graded scored steps
  (tp=40, tn=26, fp=0, fn=0). That threshold was picked **in-sample** on two scenes. It is not shipped.

## Not claimed

- These are recorded runs re-judged offline, not a launcher proof turn judged with `--judge vlm`. The launcher
  path is CPU-tested (#274); a live full launch with the judge pass is still owed.
- Nothing here generalises beyond these two Isaac scenes, one camera view, and the pink cube with a few other props.
- No GRM number: ROADMAP #14 is still open.
- The `hop ≥ 1.0` reading is not validated out of sample.

## Follow-ups

- **(B65)** An opt-in success threshold for the judge metric (e.g. `eval.judge.success_hop`), validated on runs
  held out from this set.
- **(B66)** Make the score fit the budget: a reasoning-then-score two-step, or a constrained final line, so a
  local judge cannot abstain by running long.
