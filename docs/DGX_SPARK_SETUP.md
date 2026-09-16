# PAAI on DGX Spark

PAAI means Physical Agentic AI. This installs the Build a Claw kitchen demo
with Isaac Sim, local Qwen Q4 and OpenClaw.

## 1. Check the machine

Use Linux on DGX Spark with a working NVIDIA driver and CUDA toolkit.
The host needs Bash, Git, Git LFS, curl, Python 3, CA certificates,
`libgomp1` and a C++ compiler (`build-essential` on DGX OS).

```bash
uname -m
nvidia-smi
/usr/local/cuda/bin/nvcc --version
git lfs version
getconf GNU_LIBC_VERSION
free -h
df -h "$HOME" /tmp
```

Expect `aarch64` and glibc 2.35 or newer. Allow space for the runtime,
models and caches. The model and vision projector alone use about 19 GB.
HTTPS access is needed for GitHub, Hugging Face, PyPI, NVIDIA packages,
PyTorch, Astral, Node.js and npm.

## 2. Install

The source URL and `--ref` below use the same tested commit.

```bash
curl -fsSL https://raw.githubusercontent.com/johnnynunez/cascade/ed2f73a4915bf5e6140e7b806023485ddbcf173e/scripts/bootstrap.sh | bash -s -- --ref ed2f73a4915bf5e6140e7b806023485ddbcf173e --profile spark --accept-eula --prepare-only --dir "$HOME/paai-spark"
```

`--accept-eula` accepts NVIDIA's Isaac Sim / Omniverse license.
The command installs the app packages, Isaac runtime, OpenClaw,
Qwen model, vision projector, llama.cpp and kitchen assets.
It verifies downloaded files. No manual release download is needed.

The first install downloads large files and builds llama.cpp for Spark.
Wait for `PREPARED` and exit code 0. Preparation starts no demo services.
A repeated install reuses verified files.

```bash
cd "$HOME/paai-spark"
bash scripts/install.sh --profile spark --dir "$PWD" --check
```

This read-only check must exit 0. Before installation, exit 3 means a
required environment or asset is missing.

## 3. Start

```bash
python3 scripts/desktop.py launch --repo "$PWD" --headless --no-open
```

This works from a remote shell. The cameras and physical checks run without
opening windows. From a logged-in desktop session, omit `--headless --no-open`
to open the Isaac editor and chat.

Cold shader and collision preparation can take more than ten minutes.
The launcher shows progress and allows 20 minutes for Isaac startup.
It uses the same PhysX CUDA scene and arm profile as the working Brev demo.
It starts local Qwen and the dedicated OpenClaw `cascade-demo` profile.
It then tests both orders below, with a reset after each.

Wait for `READY`. Check the current result:

```bash
OPENCLAW_STATE_DIR="$PWD/runs/.launch/profile-cascade-demo/openclaw" \
  .openclaw-cli/bin/openclaw --profile cascade-demo health --json
python3 -m json.tool runs/.launch/profile-cascade-demo/proof.json
```

The health result must contain `"ok": true`. The proof must contain
`"verified": true`, the selected Qwen model, two cases and passing physical
audits. All three cameras must advance. Each audit includes a successful
reset. `PREPARED` or `STARTED / UNVERIFIED` is not a READY result.

## 4. Use OpenClaw

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

## 5. Recover

For a missed grasp, ask OpenClaw to reset and inspect before another order.
Do the same after `did not settle at home`. This occurred after the tested
orange placement; the following reset passed.
For an installer error, keep the diagnostic and rerun the same install command.
For a stalled camera or failed startup, restart this installation:

```bash
./run.sh down
python3 scripts/desktop.py launch --repo "$PWD" --headless --no-open
```

Find the latest desktop log and physical proof here:

```bash
python3 -m json.tool runs/.install/desktop-latest.json
python3 -m json.tool runs/.launch/profile-cascade-demo/proof.json
```

Full logs are in `runs/.install/desktop-*/progress.log` and
`runs/.launch/profile-cascade-demo/`. Case screenshots and audits are in
the `evidence_dir` named by the proof.

The demo uses gateway port 18790, model port 8080 and Isaac bridge port 8611.
A port conflict stops startup with a diagnostic. Keep the existing personal
OpenClaw gateway on 18789 and other services running.

See [Spark delivery](SPARK_DELIVERY.md) for file identities and acceptance details.
