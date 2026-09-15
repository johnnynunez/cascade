# Optional camera video viewer

The deployment serves this light English page at `/cameras/` only when
`camera_video.transport` is `whep`. Its four browser assets have a fixed
allowlist and the same authenticated visitor check as the WHEP route.
`/cameras/?transport=jpeg`, `/cameras/stream/*`, and `/cameras/state` retain the
existing image/perception path. The OpenClaw button opens the real local
OpenClaw UI and its existing authentication flow.

One viewer creates one receiving video peer. Camera selection closes the old
peer before opening another; `pagehide` closes the reader, video tracks and
timers. A page restored from the browser back/forward cache reopens its selected
camera. A visible tab reconnects after 12 seconds without another presented
frame, or after 25 seconds without the first frame. Hidden-tab throttling does
not trigger those reconnects. There are no encoding quality controls.

FPS counts the change in `requestVideoFrameCallback`'s `presentedFrames` over
each measured reporting interval. Frame age uses its `presentationTime`, and
drops use `getVideoPlaybackQuality().droppedVideoFrames` since the current
stream attachment. Retries count starts reported by the reader plus explicit
player reconnects. These measure browser presentation; they do not establish
capture-to-display latency, source camera freshness, encoder bitrate or GPU
health. Missing browser measurements remain unavailable.

## MediaMTX attribution and changes

`mediamtx-reader.js` derives from MediaMTX **v1.21.0**:
<https://github.com/bluenviron/mediamtx/blob/v1.21.0/internal/servers/webrtc/reader.js>.
Its original SHA-256 is
`a802f229b803c33713d4c69c4cc0d480108a5bf384947aeee4aaf04268bf85c1`.
The unmodified upstream MIT notice is in `MEDIAMTX-LICENSE` (copyright 2019
aler9).

Local changes preserve the upstream SDP and ICE implementation:

- `videoOnly` skips audio codec probes, audio transceivers and data channels.
- Requests stay on the visitor origin, use same-origin credentials, refuse
  redirects and time out after 15 seconds; peer setup has a 20-second deadline.
- Close and error paths send WHEP DELETE with keepalive and a 3-second timeout.
  Late POST responses are checked against the camera/session path and deleted
  if their reader attempt has ended. Generation checks discard old answers,
  PATCH failures and peer callbacks after switching or retrying.
- `onRetry` reports an actual retry start. The viewer passes no model, gateway
  or relay credential to the reader; the frontend checks every WHEP method.

A POST may create a relay session whose response is lost during navigation or
network failure. The browser cannot delete an unknown session; relay timeout
cleanup still requires live observation. Successful local fake tests do not
certify unload request delivery or MediaMTX session expiry.

## Pending live checks

The page feature-detects `RTCPeerConnection`; its absence offers the image view.
It does not request camera/microphone capture permissions. No live Chrome
support is claimed for the configured exact Tailscale HTTP origin: validate the
actual browser's API availability, autoplay, ICE path and frame callbacks. If
that browser policy requires a secure context, use the authenticated TLS
visitor profile and verify its WHEP methods and tailnet media routing.

RTX PRO 6000 NVENC playback, H.264 compatibility, three-camera identity, combined VRAM,
refresh/switch session cleanup and a physical-pick recording remain deployment
acceptance work. All current browser tests use fake media/HTTP objects.
