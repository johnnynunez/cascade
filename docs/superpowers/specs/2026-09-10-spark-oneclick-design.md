# DGX Spark one-command demo distribution

## Required outcome

On a DGX Spark running Linux / DGX OS with an operational NVIDIA driver, one curl-to-bash command installs and starts CASCADE, Isaac Sim **6.1.0** (Python package **6.1.0.0**, CPython 3.12), OpenClaw, and the local Cosmos3-Edge reasoner. The attendee uses OpenClaw prompts to manipulate the scene. A Mac can explicitly select MuJoCo plus an already authenticated hosted brain for development; it is not the production default or evidence of Isaac/CUDA validation.

The user has no Spark available for this development session. Hardware rehearsal remains an explicit delivery gate, never inferred from mocked installer tests or the passing Mac suite. No publishing, pushing, driver replacement, OS reinstallation, or silent license acceptance is authorized.

## Existing failures motivating the change

- `scripts/bootstrap.sh` assumes an existing source checkout through BASH_SOURCE and never installs Isaac.
- `scripts/install.sh` defaults to laptop and does not install OpenClaw by default.
- `scripts/launch.sh --check --brain cosmos` returns OK without checking the requested model.
- The launcher warns on failed brain / robot proof but still prints READY.
- Its proof uses isolated agent exec calls, so reset is not guaranteed to address the manipulated world.
- Local provider registration writes the global ~/.openclaw file, preventing profile isolation.
- The Cosmos XML parser is in CASCADE's internal client, which an MCP host bypasses. Serving must expose native OpenAI tool_calls for OpenClaw.
- The Isaac bridge resolves its Newton experience only through a source-build path, not a pip installation.

## Approach and boundaries

Extend the existing installer and launcher rather than introduce a second runtime or a fleet controller. Every Spark is independent and uses the same pinned stack. Use native, isolated Python environments for CASCADE, Isaac and Cosmos; use the OpenClaw rootless installer with a pinned release. Preserve downloaded caches and reuse successful environments; fail on conflicting ports, wrong models, failed downloads, or incomplete proofs. Existing source changes must never be reset or silently overwritten.

Default entry: `scripts/bootstrap.sh` delegates to `scripts/install.sh`, which can be executed from stdin without a checkout. `--profile spark` is the default; `--profile laptop` and `--profile ci` remain explicit. Support `--dry-run`, `--check`, `--accept-eula`, `--no-open`, `--prepare-only`, and a source `--ref`. Ref identity is recorded; release certification requires an immutable published ref, which this session does not publish.

Isaac installation produces `$REPO/.isaacsim/bin/python`, passed to the launcher through `ISAACSIM_PYTHON_EXE`. Its installed `isaacsim/apps/isaacsim.exp.full.newton.kit` is resolved from package metadata, preserving the current Newton default. Do not change the controller, safety harness, world semantics or physics tuning.

Spark defaults use OpenClaw profile `cascade-demo`, no unrelated tools or personal workspace content. Python/model servers bind loopback. The launcher respects the selected profile and reads the config path through OpenClaw rather than hardcoding it. The existing no-profile local development path remains available.

Cosmos is pinned to the actual model snapshot, served with a parser that converts its XML tool dialect to native OpenAI tool_calls. Probe the requested model and native tool-call capability before any READY claim. Hosted OpenAI/Anthropic are explicit development options, never hidden Cosmos substitutes.

## Proof and failure semantics

One persistent OpenClaw gateway session brackets the proof. Require a real brain answer, correctly listed CASCADE tools, a successful pick trace with `postcondition.status=confirmed` AND `channel=physics`, and a successful reset in the SAME trace/world. Check per-row timestamps and expected target, not merely the newest file. Missing/failing/ambiguous evidence is nonzero. Explicit proof bypasses print unverified STARTED, not READY. Log exact model/provider, sim, session and trace directory. The optional outcome judge cannot veto valid physics or fabricate it.

## Verification

- Subprocess tests execute the actual shell entry points with installer dependencies replaced only at external boundaries; label these as orchestration tests, not Isaac runs.
- Real dependency metadata and the official v6.1.0 release / wheel are inspected; Isaac 6.1.0.0 has a CPython 3.12 aarch64 wheel and the Newton experience.
- Read-only dry-run/check tests, rerun/idempotency tests, missing-model failure, wrong-model refusal, proof/reset mismatch tests, profile isolation and source preservation.
- Full Python suite and focused lint / shell syntax.
- Live Mac OpenClaw + OpenAI + MuJoCo rehearsal, with physical before/after evidence and reset. Baseline observed: 753 passed, 2 hardware deselected; two cubes moved via a Spanish prompt, both physics-confirmed, zero tool failures.
- Pending until a Spark is available: clean installation of all GPU packages, cold first launch, Cosmos native tool reasoning with images, Isaac RGB-D and real manipulation on the shipped asset. No release-ready claim until this passes.
