# PAAI on NVIDIA DGX Spark

PAAI means Physical Agentic AI. This installs the Build a Claw kitchen demo
with Isaac Sim, local Qwen Q4 and OpenClaw.

## 1. Check the machine

Use NVIDIA DGX Spark with Linux `aarch64`, glibc 2.35 or newer, a working
NVIDIA driver compatible with CUDA 13, and the CUDA toolkit with `nvcc`.
Use a normal user account with a writable home directory. The installer does
not install system packages or change the driver or operating system.

The host also needs Bash, Git, Git LFS, curl, Python 3, CA certificates,
`libgomp1` and a C++ compiler. If these are missing on DGX OS, an administrator
can install them before continuing:

```bash
sudo apt-get update && sudo apt-get install -y git git-lfs curl python3 ca-certificates build-essential libgomp1
```

```bash
uname -m
nvidia-smi
/usr/local/cuda/bin/nvcc --version
command -v bash git curl python3 c++
git lfs version
getconf GNU_LIBC_VERSION
python3 -c 'import ctypes; ctypes.CDLL("libgomp.so.1")'
free -h
df -h "$HOME" "${TMPDIR:-/tmp}"
```

Every command above must succeed. Plan for at least 150 GB free for the
checkout, runtime environments, download caches and build products, plus
space for evidence and shader caches. This is a planning allowance, not an
installer-enforced minimum. Check the cache volume too if `XDG_CACHE_HOME` or
`UV_CACHE_DIR` points elsewhere. The model and vision projector alone use
18.85 GB; the compressed kitchen archive adds 0.71 GB before extraction.

Expect tens of GB of downloads and a local CUDA build; first installation
depends on connection speed and can take much longer than a rerun. Outbound
HTTPS must reach GitHub (including raw files, releases and Git LFS), Hugging
Face and its file hosts, PyPI, `pypi.nvidia.com`, `download.pytorch.org`,
`astral.sh`, `openclaw.ai`, Node.js and the npm registry, including redirects.
No model API key is needed for this local setup.

## 2. Install

Start with a new `$HOME/paai-spark` destination and the default runtime/model
paths. Existing `ISAACSIM_PATH`, `ISAACSIM_PYTHON_EXE`, `CASCADE_QWEN_MODEL`,
`CASCADE_QWEN_MMPROJ`, `LLAMA_DIR` or `LLAMA_SERVER` overrides select other
files; remove those overrides from this shell for a clean install.

Review the [NVIDIA Isaac Sim / Omniverse EULA](https://docs.omniverse.nvidia.com/eula)
before running the command. **`--accept-eula` is explicit acceptance**, not
a prompt: the command below proceeds without asking again. An agent needs
its operator's explicit consent before using that flag. Without it, Spark
installation exits 2 before dependency installation. Setting
`OMNI_KIT_ACCEPT_EULA=YES` alone does not grant consent. `--dry-run` and
`--check` need no consent and do not accept the license.

The source URL and `--ref` use the same pinned commit. This single command
preserves a nonzero download/install exit status and saves installation output:

```bash
bash -o pipefail -c 'curl -fsSL https://raw.githubusercontent.com/johnnynunez/cascade/485ad2654c8fab5fac16b212ce5443c3b4de939f/scripts/bootstrap.sh | bash -s -- --ref 485ad2654c8fab5fac16b212ce5443c3b4de939f --profile spark --accept-eula --prepare-only --dir "$HOME/paai-spark" 2>&1 | tee "$HOME/paai-spark-install.log"'
```

The installer clones the pinned source and prepares the following files.
Paths are relative to `$HOME/paai-spark` unless stated otherwise.

| Download or build | Location |
| --- | --- |
| App and Python 3.12 packages, including CUDA perception wheels | `.venv/` |
| Isaac Sim `isaacsim[all,extscache]==6.1.0.0` and its separate Python 3.12 environment | `.isaacsim/` |
| OpenClaw 2026.9.3 and its private Node.js runtime | `.openclaw-cli/` |
| Pinned Qwen Q4 model and BF16 vision projector, with size/hash receipts | `models/qwen3.8-27b/` |
| Pinned llama.cpp source, local CUDA build and runtime receipt | `.llama.cpp/`, `.llama.cpp/build/.cascade-runtime.json` |
| Robot Git LFS files and perception weights | `assets/`, `models/` |
| Verified 166-file kitchen release, including asset license notices | `demo/scene/assets/`, `demo/vendor-kitchen/` |
| Source/package identities, explicit consent and reusable environment | `runs/.install/install.json`, `runs/.install/env.sh` |

The installer also uses `~/.local/bin/uv`, the uv cache (normally
`~/.cache/uv`) and `~/.cache/cascade/installers`. uv obtains Python 3.12 if
needed. Desktop shortcuts are registered under the user's application and
desktop directories. No manual release download is needed.

Wait for `PREPARED` and exit code 0. Preparation starts no demo services.
A rerun reuses verified files. Consent is recorded against the absolute
checkout path in `runs/.install/install.json`; subsequent launches use that
receipt. Moving the checkout requires installation with explicit consent
again. Do not manufacture or edit consent receipts. The desktop install
shortcut asks for `I agree` (or the consent dialog); the launch shortcut
does not install dependencies or grant consent.

## 3. Check preparation without starting services

```bash
cd "$HOME/paai-spark"
bash scripts/install.sh --profile spark --dir "$PWD" --check
python3 -m json.tool runs/.install/install.json
```

The read-only `--check` must exit 0. It checks package metadata, pinned model
and build identities, robot files and kitchen checksums without downloads,
license acceptance, Kit startup or robot motion. The install receipt must
show this checkout's `repo`, the pinned `source_commit`, `source_dirty: false`,
`profile: "spark"`, `brain: "qwen"` and `eula_accepted: true`.

| Result | Meaning and next action |
| --- | --- |
| `PREPARED`, installer exit 0 | Dependencies/assets are ready for a launch attempt; no live proof yet. |
| `--check` exit 0 | Preparation checks passed; continue to launch. |
| `--check` exit 3 | A prerequisite, runtime or asset is missing/invalid; stop and inspect `MISSING` diagnostics. |
| Exit 2 | Consent, arguments or host prerequisites were rejected; fix that diagnostic before retrying. |
| Other nonzero exit | Installation or launch failed; keep its output and inspect the relevant log. |
| Desktop exit 75 | Another install/launch holds the checkout lock; inspect `desktop-latest.json`, do not start a duplicate. |
| `STARTED (UNVERIFIED: robot proof skipped)` | Services may be running, but the physical proof was skipped; this is not `READY`. |
| `READY`, launch exit 0 | Require the current health and proof checks below before handing over the demo. |

## 4. Start and verify

Check [ports and coexistence](#7-ports-and-coexistence) before starting. From
the checkout:

```bash
python3 scripts/desktop.py launch --repo "$PWD" --headless --no-open
```

This works from a remote shell. The cameras and physical checks run without
opening windows. From a logged-in desktop session, omit `--headless --no-open`
to open the Isaac editor and chat.

Cold shader and collision preparation can take more than ten minutes.
The launcher allows 30 minutes for model health and 20 minutes for Isaac
startup, followed by the two physical acceptance cases. Let those bounded
checks finish; launching a second copy does not speed them up.
It uses the same PhysX CUDA scene and arm profile as the working Brev demo.
It starts local Qwen and the dedicated OpenClaw `cascade-demo` profile.
It then tests both orders below, with a reset after each.

Wait for `READY` and launch exit 0. Check current health and receipts:

```bash
OPENCLAW_STATE_DIR="$PWD/runs/.launch/profile-cascade-demo/openclaw" \
  .openclaw-cli/bin/openclaw --profile cascade-demo health --json
python3 -m json.tool runs/.install/desktop-latest.json
python3 -m json.tool runs/.launch/profile-cascade-demo/proof.json
```

Require all of these facts, not just a listening port or an English success
message:

- Health contains `"ok": true`; the desktop receipt has `action: "launch"`
  and `exit_code: 0`.
- `proof.json` has `verified: true`, `sim: "isaac"` and the selected
  provider's `Qwen/Qwen3.8-27B` model. Its `started_at` falls between the
  desktop receipt's `started_at` and `finished_at`; an earlier proof is stale.
- `cases` contains green cube → green square and orange → open box. Each
  case has `physics.pass: true`, `physics.event_cameras.pass: true` and
  `props_reset` containing the manipulated prop (`green_cube` or `orange`).
- Each case's `evidence_dir` contains the native turns and
  `physics/gpu-physical-audit.json`, with `physics/placed-<camera>.jpg` and
  `physics/reset-<camera>.jpg` screenshots. Inspect those audits and the
  current cameras: `cam0`, `side` and `proof` must all advance. A missing,
  failed or stale observation means stop, even if chat sounds successful.

## 5. Use natural English in OpenClaw

To open the chat from the Spark's desktop, run:

```bash
OPENCLAW_STATE_DIR="$PWD/runs/.launch/profile-cascade-demo/openclaw" \
  .openclaw-cli/bin/openclaw --profile cascade-demo dashboard
```

Write normal English in the message box. Start with:

> What can you see on the table?

Compare the answer with the live cameras. Send one order at a time:

> Could you put the green cube in the green square?

Wait for the result. Check that the cube was released inside the square.
Then send:

> Let's start over.

After reset completes, send:

> Please put the orange in the open box.

Check that the orange was released inside the box. Reset and inspect again
before the next visitor.

OpenClaw may say "Placement unverified" because its tool reports only the
object's center position. The startup proof checks full placement separately.

## 6. Stop cleanly

After the last request finishes, stop this installation from its checkout:

```bash
./run.sh down
./run.sh down --dry-run
```

The first command must exit 0. The read-only second command must list no
`would stop` processes. Stop uses recorded checkout/profile/process identities;
it leaves unrelated services and cached installation files intact. Retained
receipts are diagnostic history, not evidence that a stopped demo is ready.

## 7. Ports and coexistence

| Default loopback port | Service |
| --- | --- |
| 18790 | This checkout's foreground OpenClaw gateway, profile `cascade-demo` |
| 8080 | This checkout's local Qwen API |
| 8611 | Isaac bridge |
| 18789 | Separate personal OpenClaw gateway; leave it running |

The optional live-view dashboard uses `0.0.0.0:8090` when explicitly opened
or eager streaming is selected. It remains unbound in the default lazy mode;
the Spark attendee tools do not open it. Leave unrelated listeners on 8090 alone.

The Spark defaults do not start occupancy or GraspGen-X sidecars on 5557/5556.
OpenClaw configuration, workspace and memory stay under
`runs/.launch/profile-cascade-demo/openclaw/`; no gateway system service is
installed. Run only one Spark demo stack on these default ports at a time.

Before a clean launch, `ss -ltnp` can identify occupied ports, but cannot prove
readiness. Unrelated listeners on 8080 or 18790 cause startup to fail and are
preserved. The launcher can reuse an existing Isaac bridge on 8611 only after
health and scene-identity checks; it does not acquire ownership of that
borrowed process. For a clean install demonstration, stop if 8611 already
belongs to another stack and let that stack's owner resolve it. Do not kill
processes by port or process name. Keep unrelated services running.

The chat is bound to loopback. Use the Spark desktop's dashboard, or an
authenticated SSH tunnel to port 18790 for a remote browser; do not expose the
gateway publicly to make the demo reachable.

## 8. Recover with a bounded retry

For a missed grasp, ask OpenClaw to reset and inspect before another order.
Do the same after `did not settle at home`. This occurred after the tested
orange placement; the following reset passed.
For an installer error, keep `$HOME/paai-spark-install.log`, correct only the
reported prerequisite/network/disk issue and rerun the same pinned command
once. The script performs its own bounded download retries. An existing dirty
checkout is preserved; it will not switch refs over local changes. An
incomplete environment or changed llama.cpp source/build is refused rather
than silently deleted. Keep those paths for diagnosis instead of resetting
them or reinstalling in an endless loop.

For a stalled camera or failed startup, inspect the logs below, then stop this
installation and retry launch once:

```bash
./run.sh down && python3 scripts/desktop.py launch --repo "$PWD" --headless --no-open
```

If the same failure returns, stop this installation and report the failing
command, exit status and evidence paths. Do not extend timeouts repeatedly,
skip the robot proof, change the model/physics/controller settings or delete
receipts to obtain `READY`.

Find the latest desktop log and physical proof here:

```bash
python3 -m json.tool runs/.install/desktop-latest.json
python3 -m json.tool runs/.launch/profile-cascade-demo/proof.json
```

Full logs are in `runs/.install/desktop-*/progress.log` and
`runs/.launch/profile-cascade-demo/`, particularly `qwen.log`,
`isaac_bridge.log` and `openclaw-gateway.log`. Case screenshots, native turns
and audits are in the `evidence_dir` named by the proof. Early bootstrap
failures may have only terminal output and `$HOME/paai-spark-install.log`;
absence of an install receipt is a failure to investigate, not consent.

## 9. Agent checklist

1. Confirm the prerequisite commands, available disk/network and operator EULA
   consent. Stop before installation if any requirement is missing.
2. Run the pinned preparation command once. Require exit 0, `PREPARED` and
   the matching `install.json` source/path/consent fields.
3. Run the read-only installation check. Require exit 0; exit 3 is a stop.
4. Check port ownership, then launch once. Require exit 0 and the current
   health, desktop receipt, both physical cases, both resets and advancing
   cameras described above. An open port, old receipt or agent prose never
   establishes readiness.
5. Send natural English requests one at a time; preserve failed or unverified
   results. Reset and inspect before another attempt. Follow the single-retry
   recovery limit; if it fails again, stop and report the evidence.
6. When finished, run the clean stop and require no remaining `would stop`
   entries in the read-only ownership check. Leave unrelated services alone.

See [Spark delivery](SPARK_DELIVERY.md) for file identities and acceptance details.
