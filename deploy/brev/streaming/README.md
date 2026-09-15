# Optional Isaac camera streaming

This optional x86_64 Brev profile configures native H.264/NVENC outputs for
Kitchen, Worktop and Side, with MediaMTX forwarding compressed video to Chrome
through WebRTC. It is disabled by default and still requires live encoding,
playback, reconnect and combined Qwen/Isaac VRAM validation. The public ngrok
view uses a separate H.264/NVENC path with JPEG fallback; see [the Brev guide](../../../docs/BREV.md).

## Configuration and lifecycle

Set `streaming.enabled` to `true` in a private copy of
[the RTX profile](../profiles/brev-rtx6000.json) to select
[the Compose overlay](../compose.streaming.yaml). Deployment generates the
camera and relay configuration for the selected Tailscale IPv4 and visitor
origin `http://<tailscale-ip>:8092`. The overlay supplies read-only configuration
and firewall admission mounts, plus a supervised MediaMTX relay pinned by digest.

[`camera_stream_lifecycle.py`](camera_stream_lifecycle.py) reads the absolute
JSON path in `PAAI_CAMERA_VIDEO_CONFIG`. The file requires `schema: 1` and an
explicit boolean `enabled`; enabled video also requires `tailnet_ipv4` and
`visitor_origin`. Optional fields are `rtsp_ports`, `http_port` and `media_port`.
An unset path or `enabled: false` creates no video owner or native imports.

The existing bridge owns the lifecycle on the Isaac main thread. Its lower-level
API in [`camera_streams.py`](camera_streams.py) is:

```python
from camera_stream_profile import StreamProfile
from camera_streams import RtspCameraStreams
from network_guard import verify_isolation

video = RtspCameraStreams(StreamProfile(tailnet_ipv4, visitor_origin))
video.start(sensors, verify_rtsp_isolation=verify_isolation)
```

`sensors` maps existing camera names to initialized sensors: Kitchen uses
`proof`, Worktop uses `cam0`, and Side uses `side`. Each camera must already have
`omni:sensor:tickRate` set to 30 Hz. Startup validates it without changing physics
or camera timing. Each output owns a separate 1280 × 720 render product because
H.264 compression changes its `LdrColor` RenderVar. RGB-D perception products
remain separate; video requests no extra depth or segmentation outputs.

Keep one owner in the bridge. Call `video.stop()` before closing Isaac, and use
`video.restart_camera("kitchen")` on the main thread only after media checks
identify a writer failure. The integrated lifecycle rechecks firewall admission
every five seconds and disables video on failure. A failed lifecycle requires
operator reconciliation. Browser reconnects use the relay without requiring a
simulator restart. This module never advances physics.

## Relay and browser path

To generate relay configuration separately, run from the repository root:

```bash
python3 deploy/brev/streaming/camera_stream_profile.py \
  --tailnet-ipv4 "$PAAI_TAILNET_IPV4" \
  --visitor-origin "$PAAI_VISITOR_ORIGIN" \
  --output "$PAAI_RUNTIME_DIR/mediamtx.yml"
```

The output is JSON accepted by MediaMTX's YAML parser. `visitor_origin` accepts
HTTPS, or HTTP on the exact configured Tailscale IPv4. Credentials, paths,
queries, fragments, whitespace, control characters and invalid ports are
rejected. Selected ports must be distinct and avoid reserved application ports.
The current host firewall admits only the default RTSP ports below; custom
ports require corresponding deployment support.

| Camera | Local RTSP input | Authenticated visitor endpoint |
| --- | --- | --- |
| Kitchen | `rtsp://127.0.0.1:8554/kitchen` | `/camera-video/kitchen/whep` |
| Worktop | `rtsp://127.0.0.1:8555/worktop` | `/camera-video/worktop/whep` |
| Side | `rtsp://127.0.0.1:8556/side` | `/camera-video/side/whep` |

The relay runs with host networking and no GPU access or transcoding. Its HTTP
listener binds to `127.0.0.1:8889`. The [WHEP proxy](../../runtime/whep_proxy.py)
authenticates every method, including `OPTIONS`, `POST`, `PATCH` and `DELETE`,
bounds requests, and rewrites session `Location` headers within the selected
camera path. The frontend enables this route and the video viewer only when
`camera_video.transport` is `whep`. OpenClaw administration remains private.

Media uses the selected Tailscale IPv4 on UDP 8189, with public interface
candidates and external STUN/TURN disabled. The browser needs tailnet access;
an HTTP tunnel alone does not carry this media path. Isaac's RTSP writer has
no documented bind-address option. A root host service enforces nftables rules
blocking non-loopback RTSP traffic. Its protected, read-only admission receipt
must match the boot, network namespace, ports and rules, and be at most 15 seconds
old. An unreachable listener alone is not evidence of firewall protection.

The [camera viewer](../../runtime/web/camera-player/README.md) uses the MediaMTX
v1.21.0 reader with video-only setup, bounded requests and WHEP session cleanup.
Its upstream [MIT notice](../../runtime/web/camera-player/MEDIAMTX-LICENSE)
retains copyright 2019 aler9. Camera changes and `pagehide` close the previous
reader. Browser delivery of cleanup requests and relay expiry for sessions whose
creation response was lost still need live verification.

## Validation limits

The configured profile is H.264 at 1280 × 720 with a 30 Hz sensor rate. Delivered
FPS, source freshness and capture-to-display latency cannot be inferred from
attachment or configuration. Viewer FPS, frame age, retries and drops describe
browser presentation only. `status()` always reports `live_verified: false`.

The writer exposes no bitrate, profile or GOP constructor controls. The desired
6 Mbps bitrate and one-second keyframes are unverified; encoder limits remain
`null`. Validate the actual H.264 bitstream, including absence of B-frames, before
claiming Chrome compatibility. Measure presented frames, stalls, three-camera
identity, combined VRAM and switch/refresh cleanup on the target. The exact
Tailscale HTTP origin also needs browser API and autoplay checks; use an
authenticated HTTPS origin if required by browser policy. Preserve the JPEG and
RGB-D paths for observations and independent manipulation verification.
