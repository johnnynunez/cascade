# DGX Spark One-click Implementation Plan

> For agentic workers: execute scoped tasks with test-driven development and review the integrated diff. No commits or publication in this session.

Goal: make the existing curl installer install and launch the Spark demo, with truthful proof gates.
Architecture: retain install.sh -> launch.sh -> OpenClaw MCP -> SkillRuntime. Add a dedicated Isaac installer and isolate the model environment; reuse robot and physics code.
Tech Stack: bash 3.2+, Python 3.12, uv, Isaac Sim 6.1.0.0, OpenClaw, vLLM, pytest.
Spec: docs/superpowers/specs/2026-09-10-spark-oneclick-design.md

## Current environment policy (user correction)

Keep the current stable OpenClaw/backend stack separate from Isaac's official
runtime. Do not downgrade vLLM just to reuse an older test result.

- OpenClaw: private Node prefix, latest stable 2026.9.3.
- Control/MCP/perception `.venv`: torch2.14.0+cu130 / torchvision0.29.0+cu130.
- Model `.cosmos`: latest vLLM0.29.0, REQUIRED torch2.13.0, transformers5.17.0.
- Simulator `.isaacsim`: latest Isaac6.1.0.0, its REQUIRED torch2.11.0+cu130,
  Python3.12. Do not force the app's torch into Isaac.
- Model snapshot remains current HF a9d944e2c6a1bf9f48b92ad16348e70c5f1836ba.
- All three Linux/aarch64/Python3.12 dependency graphs resolve with the above
  constraints. Forcing torch2.14 into Isaac6.1 was tested and is incompatible.
- Parent installed `/tmp/cascade-cosmos-latest-cpu` with torch2.13/vision0.28/
  transformers5.17: six CPU RoPE cases pass (guarded 5.17 patch).
- Parser probe now executes actual vLLM0.29 commit
  98dff2a81d747d1dba01a47f939f48c3526d4206, including moved serve/generate
  protocols and include_reasoning. Current official-tokenizer fixtures pass
  non-streaming, thinking-disabled and four streaming cases. This is NOT
  model inference. Offline cache now checks source revisions and SHA256.
- Version regression: 81 serving/installer tests passed before adding two
  cache-provenance tests; serving-only suite now 38 passed. Final full
  integration suite still required. Earlier version ledger entries below are
  historical and superseded by this matrix.
- Newton parent independently reran both 1.5.1 and 1.5.0: 6 passes each,
  zero failures/skips/warnings, same source hashes, read-back/count asserted.
  Metrics are runs/newton-validation/parent-newton-{version}-cpu/metrics.json.

## Global constraints

- Target Linux DGX Spark; Isaac release v6.1.0, package 6.1.0.0, Python 3.12.
- Local development may explicitly use MuJoCo and existing OpenAI auth.
- No Spark access: GPU rehearsal must remain pending and visible.
- Preserve all existing uncommitted changes. No commit/push/publish.
- No driver replacement or silent EULA acceptance; require --accept-eula for unattended Spark install.
- Separate .venv, .isaacsim, and .cosmos environments under the chosen checkout.
- Installer passes ISAACSIM_PYTHON_EXE=<checkout>/.isaacsim/bin/python and CASCADE_OPENCLAW_PROFILE=cascade-demo to launch.sh. Local development can omit the profile.
- Port/model health is not physical proof. No READY on failed or skipped verification.

## Task 1: Installation boundary

Files: scripts/install.sh, scripts/bootstrap.sh, scripts/install_isaac.sh, tests/test_spark_install.py.
Owner: scoped installer worker. No edits to launch.sh or Cosmos serve script.

1. Add a failing subprocess test: running installer from stdin with --dry-run defaults to spark and reports Isaac 6.1.0.0, Cosmos and OpenClaw without creating directories or downloading anything.
2. Run `.venv/bin/python -m pytest tests/test_spark_install.py -q` and observe the failure.
3. Implement the front door, exact argument validation, clean clone/reuse with explicit --ref, preserved dirty sources, isolated environments, consent check before mutations, exact Isaac installation, cached downloads, --prepare-only and launch delegation.
4. Add vertical tests for no EULA / unsupported host rejection, failed external command propagation, rerun reuse and --check with no side effects. Simulated host tests must not claim GPU verification.
5. Re-run focused tests and shell syntax. Report unresolved external compatibility, not fabricated outputs.

## Task 2: Cosmos serving contract

Files: scripts/serve_cosmos_vllm.sh and focused Cosmos-serving tests (no installer or launcher edits).
Owner: scoped model worker.

1. Verify upstream parser support for the existing XML dialect and exact model snapshot.
2. Add failing command-construction / configuration tests for pinned model revision, isolated stack and native tool parsing; execute them before modifications.
3. Remove floating transformer git-main dependency where a released version supports the architecture. Do not silently patch third-party code after upstream has changed; probe the actual path. Export atomically and validate completion before serving.
4. Expose native OpenAI tool_calls through the serving engine, retaining in-process XML support. Test non-GPU parts with actual upstream source/tokenizer where feasible.
5. Report separately the CPU-tested serving contract and unavailable live CUDA model validation.

## Task 3: Launcher proof and Isaac package discovery

Files: scripts/launch.sh, scripts/demo_proof.py, scripts/isaac_runtime.py, scripts/isaac_bridge.py (experience discovery only), tests/test_demo_proof.py, tests/test_launch_delivery.py, tests/test_isaac_runtime.py.
Owner: controller.

1. Write and execute a failing test for an absent explicitly selected brain passing preflight.
2. Make --check/--dry-run non-mutating; respect ISAACSIM_PYTHON_EXE and OpenClaw profile; stop hardcoding global configuration paths.
3. Extract proof-envelope and trace validation into an importable helper tested on success, missing toolSummary, wrong channel, false ok, stale rows, ambiguous traces and reset of another world.
4. Run proof turns in one session; use a scene-appropriate object (pink for Isaac, red for MuJoCo), gate READY on full proof and label explicit bypasses unverified.
5. Resolve installed Newton .kit through the Isaac package metadata; add tests for source-build and wheel layouts and meaningful failure when absent.
6. Run focused tests after each vertical slice, then real Mac rehearsal using the existing authenticated host.

## Task 4: Integration and delivery documentation

Files: README.md, docs/QUICKSTART.md, docs/ROADMAP.md, docs/ARCHITECTURE.md, CLAUDE.md, docs/SPARK_DELIVERY.md.

1. Review worker diffs, reconcile interfaces from the global constraints and fix test-proven gaps.
2. Run `.venv/bin/python -m pytest tests/ -q`; run `bash -n` on changed shell scripts and focused ruff checks.
3. Exercise a clean exported checkout with no inherited project environment. Save actual outputs; dependency boundary fixtures are orchestration evidence only.
4. Repeat the real OpenClaw/MuJoCo prompt -> physical verdict -> reset proof through the final launcher.
5. Document the single entry command, consent flag, exact version, cache/preparation behavior, per-host logs, check/down commands, and pending Spark certification. Do not describe an unpushed script as available at a new public URL.

## Progress

Baseline audited and measured before edits. The user confirmed the target release is Isaac Sim v6.1.0 and that no Spark is currently accessible. Implementation uses the existing checkout on a new branch; pre-existing changes remain preserved and uncommitted.

### Controller ledger

- Branch: `delivery/spark-oneclick`; no commits/pushes.
- Task 1 installer worker: `sa-0-c4b2bdba` (batch `deleg_80050e75`), delivered install.sh/bootstrap.sh/install_isaac.sh/install_support.py/test_spark_install.py. The original bare Cosmos PID interface was superseded: installer and launcher now share process_owner, profile-scoped state and JSON identity receipts.
- Task 2 Cosmos worker: `sa-1-97c50fd7` (same batch), owns serve_cosmos_vllm.sh/cosmos_serving.py/cosmos_upstream_probe.py/tests. Parent native probe is `demo_proof.py --probe-native BASE_URL --model cosmos3-edge`, requests `cascade_readiness(ready=true)` with native OpenAI tool_calls.
- Isaac reset worker: `sa-0-55faa591` (batch `deleg_2d6c8236`), owns IsaacArm reset, runtime's Isaac reset block, bridge reset/exec queue and new isaac_reset helper/tests. It must return checked props and refuse unsuccessful physical reset.
- Parent Task 3 implemented: strict brain preflight, no writes in check/dry-run, native tool probe, profile-aware CLI, exact model reporting, persistent-session proof, current physics trace + same-world reset, old receipt invalidation, READY versus unverified STARTED, pip Newton experience discovery for 6.1.0.0, health retry across config-triggered gateway restart.
- Parent focused tests: 18 passed (`test_demo_proof.py`, `test_launch_delivery.py`, `test_isaac_runtime.py`).
- Historical launcher rehearsal, before the ownership refactor: process `proc_c8bbf0bc6806` exited 0 with MuJoCo/OpenAI, occupancy Warp CPU and an explicit GraspGen-X protocol stub, physical pick/reset and viewer open. Its receipt `runs/.launch-delivery/proof.json` and trace `runs/mcp_67340/trace.jsonl` do not certify the current launcher or Spark.
- Parent reviewer: `sa-0-4f5b8c98` (batch `deleg_7b0744ae`) delivered the five findings addressed by the subsequent ownership/proof/discovery change below.
- Additional semantic-reference fix: `sa-0-ba3ec14b` (batch `deleg_5b3ca8b0`), owns reference.py/grounding.py/reference tests. Existing negation silently discarded impossible constraints and ordinal clamping picked another object. Worker must refuse these through the real entrypoint and report any runtime fallback bypass needing parent coordination.
- Initial read-only audits delivered: standalone Newton remains optional; Isaac's bridge already defaults Newton. Same-color identity tracking and arbitrary assets remain unproven, not part of the installer change. Do not silently claim a two-cube scene covers them.
- Remaining acceptance: current live prompt-to-physics rehearsal is blocked by provider authentication. Both configured `openai/gpt-5.6-sol` and the user-requested per-run override `openai/gpt-5.6-terra` returned HTTP401 before motion. Basic `openclaw proxy validate` passes but does not certify OpenAI authentication/routing. Default model and VPN were not changed. The failed receipt remains unverified; task-owned processes were stopped and the registry read back with zero live owners. Spark GPU rehearsal remains unavailable; publication is not authorized. vLLM-Omni scope remains unresolved and no migration is claimed.
- User requested real Newton tests instead of relying only on MuJoCo. Installed isolated `/tmp/cascade-newton-nhASTv`: Newton 1.5.1, MuJoCo/MJW 3.11.0, Warp 1.17.0, CPU only; project .venv remains MuJoCo/MJW 3.12.0. Worker `sa-0-bb661719` / `deleg_6686fa64` owns real contact + real SO101 MJCF tests under benchmark/diagnostics/test_newton_delivery.py, metrics runs/newton-validation.
- Review found 5 concrete gaps: cross-session/process proof, cross-profile shutdown, source Isaac discovery, partial reset, inconsistent auto brain resolution. Fix worker `sa-0-b28ae7bf` / `deleg_2d0bdb24` delivered. Parent re-read the implementation and independently repeated its acceptance subset: 91 passed. The installer interface is now reconciled and exercised with real local child processes/HTTP boundaries, including final exec, alternate state roots, borrowed services, replacement receipts and failure cleanup; no GPU inference is implied.
- Real non-GPU platform resolution: `uv pip compile` for Isaac6.1.0.0 / aarch64-manylinux_2_35 / Python3.12 failed first-index (mujoco-usd-converter0.5.0 shadowed on NVIDIA index), succeeded with `--index-strategy unsafe-best-match`. Output `/tmp/cascade-spark-isaac61-resolved.txt`; installer worker informed. This proves dependency resolution, not installation or GPU execution.
- Isaac reset worker delivered. Parent independently reviewed the client/validator/runtime, bridge helper and main-loop integration, then reran `tests/test_isaac_reset_delivery.py`: 23 passed. Client returns validated physics read-back, checks 2 cm tolerance, non-finite/partial/contradictory data fail. Full suite from worker was concurrent and NOT final evidence. Scope preserved. No GPU claim.
- Superseded Cosmos version note: the earlier claim pairing vLLM0.28 with torch2.14/vision0.26 was incorrect; that matrix must not be used. The configured serving matrix is vLLM0.29 / torch2.13 / transformers5.17, separate from app torch2.14/vision0.29 and Isaac torch2.11. The updated upstream parser/tokenizer probe is CPU contract evidence, not GPU model inference. See the current environment policy above.
- Reference worker left runtime fallback integration outside its ownership. Parent reproduced 6 failures (fresh beliefs, secondary camera and retry overwrite), propagated ReferenceResolutionError through both catches, then added 2 RED memory-only constrained-query cases and blocked unqualified fallback for those queries. Reference+Isaac reset suite: 68 passed; focused ruff clean. Runtime's reset block was preserved.
- README, QUICKSTART, ROADMAP, ARCHITECTURE and CLAUDE now document Spark6.1/Cosmos default distribution, consent and pending publication/GPU certification. SPARK_DELIVERY updated with exact serving/reset contracts. Doc paths/links and `git diff --check HEAD` passed; final count/ownership details await stabilized interfaces.
- Installer/model batch delivered. Parent reran installer suite: 43 passed, but actual target resolution then caught torchvision0.28.0 requiring torch2.13.0 against pinned torch2.10.0. Two RED tests now guard compatible0.25.0 and metadata preflight; scripts/install.sh/install_support.py corrected. Actual `uv pip compile --torch-backend=cu130` Linux/aarch64/Python3.12 passes with torch2.10.0+cu130/vision0.25.0+cu130 (`/tmp/cascade-spark-detector-resolved.txt`).
- Fresh working-tree copy (no source commit created) at `/tmp/cascade-clean-install-hkrpkxkn/checkout`: 566 source files, initially no venv/CLI/MJCF. `install.sh --profile laptop --brain keep --prepare-only --no-open` completed exit0 with real dependency installs, 24 downloaded robot assets, Node24.19.0 + OpenClaw2026.9.3 rootless. Its read-only laptop `--check` passed. No services or physical proof yet; resync final edited files before launcher rehearsal.
- Parent extended actual process cleanup test to launcher descendants. It reproduced a surviving worker after an exited failing launcher; `install_support.launch` now cleans its private group on any failure, not only while the leader remains alive. Profile-state ownership reconciliation is complete: 51 installer tests pass. Failed processes retain inert JSON audit receipts, rather than unlinking a possible replacement from another invocation. Parent full suite: 949 passed / 2 deselected, no failures or skips; JUnit at `runs/verification/parent-integrated.xml`.
- Final-code cold laptop preparation: `/var/folders/4z/21mcbq45453d2891y5pdlzp00000gn/T/cascade-ownership-clean-p702lyvs/checkout` was populated from 566 current source files with no prior venv or private CLI. The real laptop installer completed exit0; its read-only installation check also exited0 and the installed package imported the new process_owner helper from that checkout. No GPU package execution or physical proof is implied by prepare-only.
