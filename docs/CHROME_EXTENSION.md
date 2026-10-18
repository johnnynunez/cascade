# Chrome camera companion

**Next action:** load the extension (2 min):

1. Open `chrome://extensions` in Chrome 116 or later and enable **Developer mode**.
2. Select **Load unpacked** and choose this checkout's `extensions/chrome` directory.
3. Open the private demo's `/guide/` page, click **Open OpenClaw** and wait for **Ready**. In the
   connected chat tab click the extension icon or press **Ctrl+Shift+Y** (**Command+Shift+Y** on
   macOS).
4. Click **Connect cameras** and allow access to that demo's host.
5. Select **Side panel** or **Show in chat**, then a camera. Use **Reconnect** after a server
   restart; reload the extension after source changes.

The optional extension puts live Kitchen, Worktop and Side cameras in a Chrome side panel or a
movable panel inside OpenClaw. Orders still go through OpenClaw; the public camera page works
without it. A connected chat or advancing camera is not evidence of a successful manipulation,
campaign or restart (capability status: [ROADMAP](ROADMAP.md)).

Click the image (or Tab to it and press **Enter** / **Space**) to enlarge; click again or press
**Escape** to return. In the side panel it opens a larger window (closing it returns the
camera); in **Show in chat** it expands in the tab. The stream keeps running across sizes.

The badge says **Live** while the camera advances. The Spark camera server
(`scripts/spark_cameras.py`) keeps serving its last image for up to 5 s when the simulator skips
frames and marks each image with its `X-Frame-Age`; once the image's age (that header plus the
time it has been on screen) passes 1 s the badge reads **Last frame · N s old** instead, counting
up every second even when no further image arrives, and **Live** returns with the next fresh
image; after 5 s without one it reads **Not updating**. Screen readers hear only the state
changes (Live, Last frame, Not updating), not every second of the count. Other servers send no
age, so their badge is unchanged; the visitor proxy (`deploy/brev/visitor.py`) does not forward
the header, so visitors see no change.

## Discovery

1. The page may advertise its status endpoint with
   `<meta name="openclaw-demo" content="/api/status">`; otherwise discovery uses the active
   page's origin and port. Every camera URL must share the status host and scheme.
2. **Real rig (no Spark camera server):** on a loopback OpenClaw page on any port (e.g.
   NemoClaw's Control UI on `127.0.0.1:18789`) that advertises no cameras, the extension tries
   the Spark camera server (`127.0.0.1:8091`), then cascade's live view (`127.0.0.1:8090/state`,
   `apps/stream_server.py`), which lists the real cameras (e.g. `d455f_scene`, `d455f_wrist`) as
   MJPEG. All three share the `http://127.0.0.1/*` grant. The fallback never applies to remote,
   `localhost` or https pages, or to the camera servers themselves.
3. Start the live view with the runtime via `CASCADE_STREAM=eager`; the default `lazy` opens it
   only when the agent calls `open_live_view`.
4. **The nvblox map as a view:** with an occupancy map, the live view also announces
   `nvblox_map` (after the real cameras, labelled **3D map (nvblox)**): occupied voxels in a slow
   orbit with the safety workspace, a top-down height map with objects, and the map's age; MJPEG,
   rendered only while shown, and plain about having nothing to show. Its `/state` entry is
   `"virtual": true, "kind": "map"` and the agent's camera tools never list it.
   `CASCADE_NVBLOX_MAP=1` shows it with occupancy off, `0` hides it
   ([details](NVBLOX.md#the-map-as-a-live-view-tile-nvblox_map)).
5. **Privacy:** it stores only camera configuration and the selected view, never reads chat or
   stores gateway credentials, and camera requests omit credentials. Its history permission
   removes only exact OpenClaw token-fragment visits on recognized origins. The authenticated
   ngrok visitor is separate (its own H.264 video and JPEG fallback, no MJPEG, no gateway admin):
   use the private demo for the companion.

## Verification status

1. **DGX Spark, 30 September 2026:** real Chromium on the fresh install (Xvfb) loaded the shipped
   extension; Kitchen, Worktop and Side decoded **1280 × 720** and advanced; chat connected with
   Send and Start over enabled (both disabled while a native order ran). After a full stop and a
   second READY launch the open page recovered chat and cameras without a reload
   ([receipt](../benchmark/results/spark_clean_delivery_20260930.json),
   [screenshot](../benchmark/results/images/spark_clean_three_cameras_20260930.png),
   [after restart](../benchmark/results/images/spark_clean_three_cameras_after_restart_20260930.png)).
   GNOME app grid and a visible Isaac editor were not exercised.
2. **Brev, two separate runs.** On 14 September 2026, Chrome for Testing 153 on the Brev RTX PRO
   6000 loaded the unpacked extension and opened its native side panel (host permission accepted
   via the browser UI); Kitchen, Worktop and Side decoded 960 × 540 and advanced. On 15 September,
   Chrome for Testing 151 on the controller, connected to the running Brev demo, passed
   enlarge/return with mouse, Enter, Space, Escape and window close, fit a 1366 × 768 window, and
   reused the same image element without a new MJPEG request. The native side panel was shown
   only in the 153 run.
3. **OpenClaw pages:** opening the side panel from the guide's **Open OpenClaw** kept the
   authenticated chat connected (no message or robot command sent); **Show in chat** passed move,
   collapse, restore and refresh inside the real OpenClaw login page.
4. Chrome for Testing suits automated unpacked checks; regular Chrome uses **Load unpacked**.
   Opening the side panel needs a user gesture
   ([Chrome docs](https://developer.chrome.com/docs/extensions/reference/api/sidePanel)).

## Firefox

The same companion runs in Firefox 140 or later; only the manifest differs.

1. Run `python3 extensions/firefox/package.py` (writes `extensions/firefox/build/` and the `.xpi`).
2. Open `about:debugging`, choose **This Firefox**, click **Load Temporary Add-on** and pick the `.xpi`.
3. Firefox drops temporary add-ons on restart; a permanent install needs an AMO-signed package
   (`web-ext sign --channel=unlisted`).
