# MicroDuck continuation evidence — 2 October 2026

The mobile implementation is reviewable, but locomotion is **not physically
admitted**. This campaign confirms an actual MCP → CASCADE → official ONNX →
native BAM → Newton/MJWarp → independent-verifier route, including solved foot
support and camera frames. Forward/reverse tracking fails; both turns expire.
Successful standing and stopping do not change those motion outcomes.

## Source and recipe

The collaborator's 53-file working snapshot was copied without modifying its
checkout and preserved as `0fb4c0f`, based on merged PRs #65/#66 (`69037b7`).
Continuation fixes bind the effective native model and exact support registry,
extract solved contact forces, preserve unsafe observations across stop timing
boundaries, and author a reproducible camera product. Native runs below use
`dcdb386`; the following `047a06a` only corrects the unused candidate YAML's
engine from PhysX to Newton. Runtime Python is unchanged between those pins.

| Input | Frozen value |
|---|---|
| Native engine | Isaac Sim 6.1 rc26, Newton 1.6.0, Warp 1.17, MuJoCo/MJWarp 3.12 |
| BAM | IsaacLab #8161, `28aa1fca5843208ff9a67935695a4d5376e44d50` |
| Profile | `nominal_no_current_limit_no_delay`, 7.4 V, explicit drive/joint limits both 0.96 N·m |
| Policy | Pollen `velstand.onnx`, SHA-256 `1c659be55da94bc5753b707de5c6a3e7c49931e05ca3b6991615cef1a8ba9a45` |
| Converted bundle receipt | SHA-256 `910c4ddb7e25122f808888f01e97227af0344bdabbb26572e3f021384584490d` |
| Effective model | SHA-256 `43a814f055e9fc4ff410cdaea09b4eb399c8956f5b7b18835d88eb6bd3cf5165` |
| Physics/policy | Native dt `0.004999999888241291` s; one inference per four solves |

The foundation's complete canonical identity was inspected and pinned before
the MCP launch. Both independently opened models produced identical identity
documents. No gains, policy weights, production speed caps, current limit or
verifier tolerances changed to obtain these results. This diagnostic recipe is
not an admitted production profile. The template still requires explicit
identity, support registry and verifier configuration.

## Native measurements

`native-foundation3` completed 800 solves, 200 ONNX evaluations and 41 direct
contact-force comparisons. All comparisons had zero component discrepancy.
All 800 physical records contain known contact-force observations; the first eight have
empty contact sets and are retained as such. No forbidden loaded body-ground
contact was found. The last 100 samples averaged 7.232532796 N upward sole
force against 7.232355852 N model weight (measured mass 0.737243175 kg).
Native Python and its launcher both exited 0; the owned scope is inactive.

`native-mcp2` completed 2,400 solves, 600 ONNX commits and 121 direct force
comparisons, again with zero discrepancy, no unavailable support and no
forbidden loaded body contact. A 1.5 s unscored settling interval precedes
commands. The same physical episode supplies MCP observations, policy inputs,
independent verification, trace/memory and rendered frames.

| Request | Measured result | Verdict |
|---|---|---|
| Zero velocity for 0.6 s | −0.076 mm longitudinal displacement, supported rest | Confirmed |
| +0.1 m/s for 1 s | +3.432 mm versus +100 mm requested | Refuted: insufficient progress |
| −0.1 m/s for 1 s | −1.071 mm versus −100 mm requested | Refuted: insufficient progress |
| Turn +0.2 rad | Controller measured +0.0434 rad before wall deadline | Unverified; execution failed |
| Turn −0.2 rad | Controller measured −0.0502 rad before wall deadline | Unverified; execution failed |
| Emergency stop | ACK initially unverified, later independent supported-rest confirmation with matching receipt | Confirmed stop only |
| Emergency stop during active command | Independent reader saw active state, then a newer latched generation | Command permission invalidated |
| MCP EOF during active command | Same independent transition; MCP exited 0 | Command permission invalidated |

The last two cases establish command cancellation, not a separately scored
full rest window. The interrupted emergency-stop command was cancelled before
its first nonzero policy commit; the disconnected command had one nonzero
commit, followed by zero-command commits. Neither demonstrates braking from
an established gait. The formal stop began after already slow failed turns.
Turn execution did not return a bound completion receipt,
so its partial angle must not be upgraded to verified physical success.
`reset_stop` restores permission only; world reset remains unimplemented.
The final diagnostic shutdown deliberately signalled only the owned native
process: child exit 143, Isaac shell exit 1, receipt signal 15, no teardown
errors, scope inactive. That expected shutdown is distinct from a crash.

## Software checks and retained failures

Full regression at `0fb8b18`: **4,969 passed, 253 skipped, 4 deselected**
in 389.09 s, including the visitor Python suites. Skips retain unavailable
optional dependency/reference coverage; native evidence is reported separately.
Tracked-source and shared-memory hashes were unchanged throughout.

The separate CI packaging checks passed all 62 cases; the booth/extension
Node suite passed all 88 cases. Both retained source and shared-memory
identities unchanged.

An additional 268 unique optional CPU cases passed with actual USD conversion,
native BAM/MJWarp and official policy/model references at `1883d4d`; these
overlap the normal suite and must not be added to its count. Four initial
environment/fixture failures remain in the evidence; only those cases were
rerun after correcting private import paths and supplying the full admitted
source. No shared SDK or environment was modified. Native CPU BAM comparisons
cover 12 numerical cases (largest absolute error `4.017e-7`) and live MJWarp
friction/load extraction, without claiming MicroDuck gait acceptance.

The first full-suite attempt retained 134 failures: 132 needed the kitchen
asset bundle fetched by the normal CI setup, and two synthetic mobile fixtures
implicitly inherited the old PhysX template default. The verified ignored
bundle was fetched and those fixtures now pin their own declared engine.
Runtime Python did not change. Ruff F/E9 has no new diagnostics against
`69037b7`; its 57 existing findings are not claimed clean.

The first remote CI exposed another test-admission error: its environment had
Warp but no Newton and no `CUDA_VISIBLE_DEVICES`. The BAM fixture checked the
CPU-only variable before skipping the absent optional Newton dependency,
causing 96 setup errors. Moving the dependency check first preserves the
CPU-only requirement when native tests actually run. A local reproduction with
that variable absent then passed 2 cases and skipped the 96 unavailable ones;
no runtime code or native recipe changed.

Remote run `36981791068` then retained one Linux x86 failure (a mocked
five-second deadline rounded to `5.000000000000057`), one ARM fixture timeout,
and 25 macOS fixture timing failures. The follow-up test changes give real
socket reads and deliberately slow synthetic producers appropriate wall-time
budgets while preserving physical windows, rest/velocity/drift thresholds,
explicit deadline failures and reader quarantine. Fast and slow renewal/stop
cases are both exercised. The release test permits only `1e-12` absolute
rounding while still requiring positive, decreasing budgets.

The combined corrected mobile, MicroDuck, IsaacBase, BAM and release suite
passed **1,187 cases with 182 skips** in 112.21 s, with tracked source and shared
memory unchanged. The [portability record](../benchmark/results/microduck_ci_portability_20261002.json)
binds all six edited test files and retained logs by hash. This targeted run
is separate from the earlier full regression; remote checks must be assessed
at the PR's current head. No production runtime, policy or native recipe
changed in these test corrections.

At `3c3543b`, remote Linux x86 and ARM each passed 4,958 cases (269
skipped, 4 deselected) plus 62 packaging cases; booth and minimal-install
checks also passed. macOS retained six failures, with 4,918 passed and 303
skipped. Its receipts show fast TCP reads but too few synthetic steps: relative
post-publication waits accumulated publication/wakeup overhead. The log does
not identify a particular operating-system mechanism.

The follow-up uses absolute fixture deadlines, completes every due step and
checks cancellation between steps; excessive backlog fails explicitly. It
preserves all physical windows, freshness, lease and deadline limits. With the
same injected delays, the new scheduler passed all six affected cases in two
separate controls; replacing only the scheduler with the old relative pattern
reproduced all six failures. The preflight case now also supplies valid support
and requires zero post-admission displacement with the specific motion veto.
Three deterministic helper regressions cover late wakeups, publication cost,
cancellation and excess backlog. The combined suite passed **1,190 cases,
182 skipped**, in 111.98 s with source and shared memory unchanged. The
[scheduling record](../benchmark/results/microduck_ci_schedule_20261002.json)
retains exact file hashes, controls and the previous CI result. Production
runtime and native recipe are unchanged; the new remote checks remain separate.

Earlier failed attempts are retained:

- `native-foundation`: completed the physical episode, then the external
  cleanup harness failed because that Python lacked its pidfd wrapper. An
  independent check found the scope closed; the missing original exit-code
  observation is not retroactively invented.
- `native-mcp`: rejected the second launch before MCP commands because the
  SDK's autogenerated render-product name changed model identity.
- `native-foundation2`: the first deterministic-product fix lacked the
  RenderVar/Hydra synchronization required by the actual SDK and failed before
  physics. The complete product and stopped-timeline synchronization were then
  validated by the two current native launches.

The shared envelope hash remains unchanged and shared grasp memory remains
absent. Child tests use private memory paths. Other agents' processes and the
original working checkout were not stopped or edited.

## Evidence and next decisions

The [machine-readable summary](../benchmark/results/microduck_continuation_20261002.json)
records artifact hashes and source pins. Raw logs, JSONL, JPEGs, source
inventories and same-episode video remain in the local `MICRODUCK_CODEX/`
artifact directory beside the worktree; they are not included in this PR.
`native-mcp2/mcp-simtime.mp4` contains 120 original frames with physical
timestamps (11.901 s container duration), SHA-256
`a1c6a33e0a1d61d9b8aa9793435bd4e1e7adda8c7a26a2c65d92b86eb6d40509`.
The last 0.1 s of physics has no frame; the video does not invent that interval.
An independent audit matched 1,527 verification/interruption samples to the
native trajectory, without pose, support or identity mismatches.
The model assets and policy also remain external with their original notices.

The [policy startup research](research/microduck-policy-startup-2026-10-02.md)
identifies an upstream report of the same low-speed startup symptom, but no
validated cure or explanation for every signed command. The
[BAM follow-up](research/microduck-bam-2026-10-02.md) separates the pinned
actuator implementation from newer fits and battery-sag semantics. A future
full-environment comparison must distinguish startup from sustained walking,
record actual sampled commands and pushes, and preserve failed episodes.

The subsequent [full-mjlab comparison](research/microduck-fullenv-startup-2026-10-02.md)
completed with the same official ONNX and a separately frozen upstream stack.
Across 32 worlds, a cold +0.1 m/s command produced mean body speed
0.000077 m/s; +0.3 produced 0.143978 m/s. Reducing that command to +0.1
produced mean 0.001737 m/s, median 0.000028 m/s. No world exceeded the
descriptive 0.05 m/s phase-mean threshold at +0.1, while one averaged
0.048689 m/s. This supports the low-speed failure in that full environment;
it does not validate our native actuator/model equivalence or establish a
remedy. Production limits, source and physical-admission status are unchanged.

PhysX needs its own faithful actuator/load contract and native campaign.
World reset, full gait admission, hardware and the
[hosted conversation implementation](MICRODUCK_CONVERSATION_DESIGN.md) remain
separate unfinished work; none are certified by this report.
