# PAAI — Physical Agentic AI camera companion

A camera companion for OpenClaw and NVIDIA Isaac Sim. Load this directory
with Chrome → Extensions → Developer mode → Load unpacked. Open the private
booth guide, click Open OpenClaw and wait for Ready. In the connected chat tab,
invoke the extension, click Connect cameras and allow the demo host. Choose
Kitchen, Worktop or Side in the side panel or chat overlay.

Click the camera image to enlarge it. Click again or press Escape to return.
You can also reach the image with Tab and open it with Enter or Space. The
side panel opens a larger window; the chat overlay expands within the tab.
The existing camera connection keeps running while you switch sizes.

Discovery uses page metadata or the active page's origin and preserves its
port. Camera endpoints must share that host and scheme. No chat messages,
model data or gateway credentials are stored. The history permission only
removes exact token-fragment visits on recognized OpenClaw origins.

The companion expects MJPEG; the public ngrok H.264/JPEG visitor works separately.
See [installation and verification](../../docs/CHROME_EXTENSION.md). Run
`node --test tests/*.test.mjs` for local behavior checks and `python3 package.py`
for an allowlisted ZIP. Browser profiles and evidence are excluded.

System fonts only; no NVIDIA font or logo is redistributed.
