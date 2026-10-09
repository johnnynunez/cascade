# B40: a size-consistency check in the belief store's fusion gate (9 Oct 2026, CPU only)

**Question.** In the B32b live runs the side camera `isaac_side` named the open bin AND a yellow 5 × 5 × 8 cm prop
inside it "yellow". On frames where it saw the bin but not the prop, its view of the bin was fused into the PROP's
belief, because the names matched and the prop's centre was the nearer one. Can a size check in the fusion gate
refuse that, without splitting one object's views (partial, occluded, from two cameras) into two beliefs?

**Answer (CPU, not yet live).** Yes on the measured data. A view's size is its robust horizontal diameter; a belief
remembers the largest one fused into it. A fusion is refused when the view is more than **2.0 ×** that size **and**
more than **5 cm** larger. On the live clouds of B32b, views of one object differ by at most × 1.29 and a container
view is at least × 2.48 (and 10.9 cm) larger than the prop's largest view. Both numbers are estimates between those
margins; the live A/B on Isaac is owed (below).

## The defect, live (B32b assignment logs, Isaac 6.2 PhysX + YOLOE, source 90a9f0c)

Two logged runs of the in-bin-centre scene (`results_b32c_adv/assign_in_bin_centre_{on,off}.json`, one per colour
arm): **6** side-camera views of the bin (OBB 0.218–0.222 m) went into the yellow cube's belief (born from a 0.09 m
view), 3 per run, all from `isaac_side`, identical with the one-name rule. For those ticks the prop's belief carried
the bin's cloud (what grasp-from-memory plans on, `SkillRuntime._fix_from_belief`), its extent and its label as an
alias (`building block`); the next top-camera frame overwrote them again.

## The size measure

`memory.beliefs._view_diameter`: on the cloud the store remembers (≤ 384 points) without its lowest centimetre (the
B32c floor band, so a table skirt in the mask does not count), the largest 2nd–98th percentile span of the x-y
projection over 8 directions 22.5° apart. Horizontal on purpose: how much of a prop's height a camera sees depends
on the rim in front of it.

Live clouds (`measure_sizes.py --lab`, every logged B32b decision cloud plus the committed fixture; labelled by
geometry as in B32b):

| population | n | diameter min / median / max (m) | robust a-a box, max x-y span | OBB max extent |
| --- | --- | --- | --- | --- |
| the bin, both cameras | 205 | 0.183 / 0.195 / 0.235 | 0.142 / 0.147 / 0.215 | 0.184 / 0.227 / 0.299 |
| the 5 × 5 × 8 cm prop, side camera, in / next to the bin | 15 | 0.066 / 0.067 / 0.074 | 0.048 / 0.049 / 0.065 | 0.090 / 0.092 / 0.328 |

| ratio (live) | diameter | a-a box x-y | OBB |
| --- | --- | --- | --- |
| same object, bin: largest / smallest view | × 1.29 | × 1.52 | × 1.63 |
| same object, prop: largest / smallest view | × 1.12 | × 1.34 | × 3.63 |
| smallest container view / largest prop view | **× 2.48, +10.9 cm** | × 2.19, +7.7 cm | × 0.56 |

- **Why not `extent`.** The OBB is a min/max box: one live side view of the prop in the bin corner has an OBB of 33 cm
  (bleed far behind it), larger than the bin's smallest OBB. It cannot separate the two.
- **Why not the axis-aligned box.** Smaller live margin, and it changes when an object turns: a 15 cm square turned
  45° grows 20 % in its robust a-a box, 3 % in the diameter (`test_the_diameter_is_the_spec_and_ignores_yaw`).

Ray-cast (the bare Isaac scene of `tests/test_colour_identity_beliefs.py`: calibrated poses, bridge optics, exact
masks at 320 × 180; both cameras; `sizes_summary.json` → `raycast_ratios`):

| case (worst over both cameras) | full view / part |
| --- | --- |
| the bin's full views: 3 scenes, mask bleed 0–2 px | 0.183–0.199 m (× 1.08) |
| half of the bin hidden (left/right/top/bottom of the mask), bleed ≤ 1 px | ≤ × 1.38 |
| half of the prop hidden, no bleed (bare scene: ≤ 1 px) | ≤ × 1.68 |
| only a quarter of the prop left, no bleed | ≤ × 1.86 |
| only a quarter of the 5 cm cube left, no bleed | × 2.63, but only +4 cm: passes the 5 cm excess |
| only a quarter of the bin left | × 2.00 (prop in it), × 2.09 (cube in its corner), **× 2.64** (prop next to it) |
| the prop INSIDE the bin, 1 px of bleed (≈ 4 px at the live 1280 × 720) | **× 2.53** its clean view |

## The rule and where it can be wrong

Refuse fusing a view into a belief when `view > 2.0 × belief AND view − belief > 5 cm` (`SIZE_GATE_RATIO`,
`SIZE_GATE_EXCESS_M`), belief = the largest diameter of any real-mask view fused into it (`ObjectBelief.diameter_m`,
a running max: a partial view only looks smaller, and a full view after it must still pass). It only refuses: a
view or belief without a real-mask cloud is never vetoed. Margins on the live data: × 1.55 above the largest
same-object ratio, × 1.24 below the smallest container ratio; the 5 cm excess keeps small objects (live props
+0.8 cm between views) from ever being refused on noise.

Where the ray-cast says it can be wrong:
- **A belief born from a quarter of a container's view** (something hides 3/4 of the bin from its very first frame)
  refuses the container's full view (up to × 2.64, +11 cm): a second belief for the bin. Half views pass (≤ × 1.38).
- **Mask bleed onto the container.** At ≥ 4 px of bleed (live resolution) the view of a prop inside the bin takes in
  the bin's far wall and looks × 2.5 its size, so it is refused from the prop's own belief (it becomes a belief of
  its own, or joins the bin). Live YOLOE views of that prop measured 0.066–0.074 m: no such bleed on this rig.
- **Not addressed:** a prop's view fusing into the CONTAINER's belief. A smaller view always passes, because it looks
  exactly like an occluded view of the container.

**Default: on** (`memory.size_gate: true`), with `false` = the store before B40 byte for byte, as B32b shipped. The
live data separates with a margin on both sides; the two ray-cast failure modes need conditions not seen live
(a quarter-only first view, ≥ 4 px bleed onto a container), and each costs a duplicate belief, the same class of
error as the defect it replaces (a prop belief carrying the bin's geometry).

## Tests (`tests/test_fusion_size_gate.py`, 36)

- RED on an export of main 4e896c3 (`git archive origin/main`, the new test files copied in):
  `26 failed, 10 passed in 1.19s` (EXIT=1), and again after the rebase on an export of main 4d0947b:
  `26 failed, 10 passed in 1.24s` (EXIT=1; `memory/beliefs.py` is identical in the two). The 8 behaviour tests of
  the defect fail on the assertion itself: the
  bin view lands in the prop's belief (its OBB becomes 0.218 m, alias `box` / `building block`, the prop relabelled
  `building block` in the watcher and `_reobserve` paths; the golden's switch-on replay still shows `[1]`). The
  other 18 fail on the missing API (`size_gate`, `diameter_m`, `_view_diameter`, `memory.size_gate`). The 10 that
  pass on main pass by design:
  3 premises (measured on a test-local copy of the diameter spec), 4 must-still-fuse guards on the bin's live view
  pairs, `test_a_larger_accepted_view_raises_the_belief_size`, `test_the_gate_only_refuses`, and the golden's
  receipt check.
- GREEN in the worktree: `36 passed`.
- Through the real paths: `BeliefStore.update_frame` (live fixture clouds), `update()` and the per-detection path,
  the two-camera `WorldWatcher._tick`, `SkillRuntime._update_beliefs_from_frame` + `_reobserve`.
- Golden: `size_gate=False` replays a fixed scenario (the defect, the bin's live views, twins, `update()` with and
  without a cloud; default store, one-name rule, per-detection path) byte for byte as main 4e896c3 wrote it
  (`tests/fusion_size_gate_fixture.py`, `tests/fixtures/b40_size_gate_golden/`).
- Mutation check on a scratch copy: 26/26 mutants killed (re-run on the rebased branch: unmutated `109 passed`,
  26/26 killed).

## Live validation owed (parent, Isaac 6.2 PhysX, GPU 0) — NOT YET RUN

1. Private bridge in the B40 port block (bare scene, pixel mask on):
   ```bash
   cd <item>/cascade && env -u PYTHONPATH CASCADE_ISAAC_PIXEL_MASK=1 CUDA_VISIBLE_DEVICES=<GPU-0 UUID> \
     /home/johnny/Projects/demo/omni_isaac_sim/_build/linux-x86_64/release/python.sh \
     scripts/isaac_bridge.py --port 45250 --usd <item>/cascade/assets/usd/RS-rebot-dev-arm/RS-rebot-dev-arm.usda \
     --engine physx > <item>/live/bridge.log 2>&1
   ```
2. `mkdir -p <item>/live && cp docs/evidence/b40-fusion-size-gate-20261009/{b40_live_ab.py,run_ab.sh} <item>/live/`
3. Smoke one arm: `cd <models dir with mobileclip_blt.ts> && python <item>/live/b40_live_ab.py --size-gate on --frames 4 --json /tmp/smoke.json`
4. Interleaved A/B, ≥ 3 runs per arm: `GPU=<GPU-0 UUID> bash <item>/live/run_ab.sh` (REPS=5 if the arms are not uniform).
5. Read every run's `beliefs_final` and `container_views_in_prop_beliefs_detail`. Target: `on` 0 container views in
   prop beliefs (B32b: 3 per run), the bin belief `{isaac: orange, isaac_side: yellow}`, the cube belief without a bin
   alias, final count 3, precision/recall not lower than `off`; `prop_views_in_container_beliefs` reported for both.

## Not claimed

- No live measurement of the gate; 2.0 and 5 cm are CPU estimates from B32b's live clouds and the ray-cast.
- One rig, one scene (bare reBot, PhysX), one detector (YOLOE prompt-free); no kitchen, no Newton, no real camera.
- A prop's view fusing into its container's belief is not handled; neither is a detector that returns the prop and
  its container as ONE detection, nor `_frame_instances` grouping a part with its whole (unchanged on purpose).
