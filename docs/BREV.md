# Brev kitchen deployment

**Next action:** after [preparing once](#prepare-once-then-start), start the kitchen:

```bash
./deploy/brev/deploy.sh start --profile /srv/cascade/site.json
```

x86_64 only, with its own acceptance boundary: do not apply Spark defaults or a later source's
acceptance to this profile without its own deployment check (capability status:
[ROADMAP](ROADMAP.md)). For the arm64/GB10 Spark use [DGX Spark setup](DGX_SPARK_SETUP.md); the
x86_64 container is not a Spark image, and only the scene, tool API, guide, extension and
source-level health checks are shared. Upstream's Cosmos3-Edge profile is separate and
uncertified here.

Tested machine: AWS `g7e.2xlarge`, one RTX PRO 6000 Blackwell Server Edition (97,887 MiB VRAM),
8 vCPUs, 62.27 GiB RAM, a separate 1.7 TB NVMe volume. Isaac Sim 6.1 runs CUDA PhysX; the brain
is Qwen3.8-27B Q8_0 with its BF16 vision projector, fully on the GPU (one Qwen/Isaac run used
36,590 MiB with video off).

## Choose and connect a machine

1. Check availability with `brev search` or the console (filters `--gpu-name`,
   `--min-vram 90`, `--min-disk 1700`, `--flex-ports`; the disk filter does not resize root:
   confirm the mounted storage after boot), then create the tested type:

   ```bash
   brev create cascade-kitchen --type g7e.2xlarge --flex-ports
   ```

2. Controller needs Python 3.10+, `rsync`, the Brev CLI and Tailscale. `brev login` gives
   `~/.brev/brev.pem` (set mode 0600; the key stays on the controller).
3. Join controller and VM to one tailnet, allow the VM's port 22, keep `ubuntu`'s passwordless
   sudo. Brev's managed SSH relay can close connections while the VM runs (not a reason to reset
   the GPU machine) and public port 22 was unreachable: Tailscale to the VM's own SSH worked.
   Check the node-key expiry of **both** devices.
4. Supply an executable OpenSSH-compatible transport wrapper that accepts
   `-F <private-config> <alias> <remote-command>` and forwards arguments and stdio unchanged
   (rsync too). Default `~/.local/bin/totally-not-ssh.sh` (not in the repo); override:

   ```bash
   export PAAI_TRANSPORT_WRAPPER="/usr/local/bin/cascade-transport"
   ```

   The access gate checks the executable before writing a private config, creates a private
   alias for the Tailscale peer and keeps existing host entries.

## Prepare once, then start

1. Copy [the RTX profile](../deploy/brev/profiles/brev-rtx6000.json) outside the checkout and
   set `network.tailscale_hostname`. Keep models, Docker layers, build inputs, Isaac caches and
   recordings under `/opt/dlami/nvme/paai-demo` (root is ~117 GB). A separate Docker daemon on
   `unix:///run/paai-demo-docker.sock` leaves the system daemon's data alone.
2. Supply the external files at the relative paths in
   [`bundle_assets.json`](../deploy/brev/bundle_assets.json) (robot assets, perception weights,
   pinned runtime archives) with their license notices, plus the pinned OpenClaw and CLIP
   archives.
3. Assemble the distribution outside the checkout and import it (valid receipts let later
   operations reuse the bytes):

```bash
python3 scripts/kitchen_assets.py
python3 deploy/brev/build_bundle.py --assets /srv/cascade/assets \
  --output /srv/cascade/distribution --profile /srv/cascade/site.json
./deploy/brev/deploy.sh prepare --profile /srv/cascade/site.json \
  --source /srv/cascade/distribution/source \
  --manifest /srv/cascade/distribution/PORTABLE_BUNDLE.json \
  --openclaw /srv/cascade/openclaw.tar.zst \
  --clip /srv/cascade/clip-source.tar.gz
./deploy/brev/deploy.sh install --profile /srv/cascade/site.json
```

`scripts/kitchen_assets.py` fetches the audited `Cocina_Asier_01` release and verifies the
archive and every unpacked file (`--check` verifies offline; an offline source machine passes
the ZIP with `--archive /srv/cascade/cocina-asier-v1.zip`). The orange and bowl are generated
from the checkout. Assembly and admission are offline.

**Operations:** `deploy.sh {prepare,install,access,preflight,start,status,restart,stop}
--profile <site.json>`. They address the owned containers, not the Brev instance.

1. A changed installed source bundle is rejected, never silently replaced; validate a new
   bundle on a separate target. A repository `STOP` file blocks new controller operations.
2. Restart checks preparation, image, source compatibility and preflight before stopping the
   owned services.
3. Preflight checks the exact GPU, compute capability, free VRAM (70,000 MiB before start,
   12,288 MiB headroom while running), the separate data volume and existing GPU process
   ownership. One simulator per GPU; never alter the NVIDIA driver or kernel. Compose accepts
   the Isaac EULA explicitly.
4. First boot without `docker0` (bridge containers fail with `Device does not exist`): run
   `sudo systemctl restart docker` (the **system** daemon) only for that fault. The demo daemon
   and container have restart policies; routine disconnects never need a VM restart.

Licensing: CASCADE is MIT; the runtime installs [Ultralytics](https://www.ultralytics.com/license)
and its pinned [CLIP fork](https://github.com/ultralytics/CLIP/blob/a13192f8cb767260d7dfd98c843b0716593169e7/LICENSE)
under AGPL-3.0 (vendor licensing may apply). The [scene notice](../demo/scene/NOTICE.md) records
the room geometry, CC0 textures and original props. Preserve every notice when distributing.

## Visitor access

The private frontend serves `/guide/`, `/openclaw/` and `/cameras/` on port 8092; OpenClaw
bootstrap and admin stay on Tailscale/loopback. The optional public visitor serves its page,
status, three camera videos and JPEG fallbacks through an allowlist on loopback 8093. Enable it in
the external profile before preparing; the ngrok token lives in a 0600 file owned by the runtime
user, the profile holds only its path:

```json
"public_visitor": {
  "enabled": true,
  "token_file": "/secure/location/ngrok.token",
  "local_auth": false,
  "video": true
}
```

1. The same `deploy.sh install|start|status|restart|stop` manage the visitor and ngrok services;
   `status` prints the HTTPS visitor URL after checking auth, cameras and blocked admin routes.
   `deploy/brev/public.py` exposes the visitor-only operations for maintenance.
2. Basic-auth credentials: `data/private/public-visitor/auth.json` under the deployment root;
   share only those. If the account rejects ngrok traffic policies, set `local_auth: true`: the
   visitor server enforces basic auth in both modes. ngrok request inspection stays off
   ([ngrok Linux](https://ngrok.com/download/linux)).
3. `video: true` shares each camera through one host FFmpeg/NVENC encoder (`h264_nvenc`; the
   installer adds Ubuntu's FFmpeg, never a software encoder) as H.264 fragmented MP4 over the
   same HTTPS origin: 960 × 540, nominal 3 fps, 2 Mbps CBR, 1 s keyframes, no B-frames. Sources
   update ~2.5–3 times per second; the Isaac camera is not turned into a 30 fps renderer, and
   the output stays 960 × 540 when the offline card replaces a capture.
4. The player drops old buffers, reconnects after a stall and falls back to JPEG snapshots.
   All three cameras played through the public URL in Chrome (14 September 2026). The native
   Isaac RTSP/WebRTC profile is off and unaccepted. A camera image is not evidence of a grasp.

## Operating the kitchen

1. **Sessions.** One operator chat at a time: all sessions share the physical scene. Start a new
   session per attendee after the reset. Recovery: [staff guide](BOOTH_RUNBOOK.md#10-paai-on-brev-staff-guide).
2. **Reset.** If the browser disconnects during Reset, reconnect and read the original result
   before retrying (the button never resends an uncertain order; a reload never replays an
   accepted reset). The readiness check can briefly report `BridgeError` (2 s bridge timeouts
   while ping and cameras still answer); wait for Ready before the next order.
3. **Model limits.** 16,384-token context; replies capped at 1,024 tokens via
   `extra_body.max_completion_tokens` (the image estimate can cut `maxTokens` to one); compaction
   keeps 2,048 recent tokens; requests time out after 180 s. Tool sessions request cameras at
   most 4 times per second; idle tool runtimes retire after one minute (not a kitchen reset).
4. **Trust the measured output.** `annotated_view` badges are tracked estimates and can be stale
   or mislabeled, and the chat can miscopy coordinates; neither establishes reachability. Kitchen
   profiles cap each pick at two grasp attempts; a tool's distance from the green square or open
   box centre is reported as unverified containment and release.
5. **Measured matrix.** Orange and tomato can passed; lemon lifted once and reached the green
   zone but failed strict settling (unverified); orange-on-can failed stability and was reset.
   `localize_object` resolved `the lemon` from yellow pixels and depth (articles "the"/"a"/"an"
   keep the configured visual description). The unpacked Chrome extension passed as camera
   companion. The relocated publication layout still needs its own complete deployment and
   sustained-runtime check.

Booth support: Johnny Núñez and Asier Arranz on Slack.
