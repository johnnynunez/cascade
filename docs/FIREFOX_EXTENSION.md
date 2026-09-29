# Firefox camera companion

The camera companion also runs in Firefox 140 or later. It uses the same
JavaScript, HTML and CSS as the [Chrome companion](CHROME_EXTENSION.md); only
the manifest differs. Robot orders still go through OpenClaw.

## Build and load

```bash
python3 extensions/firefox/package.py
```

This writes `extensions/firefox/build/` and
`extensions/firefox/Physical-Agentic-AI-OpenClaw-Demo.xpi`, plus its SHA256.

1. Open `about:debugging` and choose **This Firefox**.
2. Click **Load Temporary Add-on** and select the `.xpi` or
   `extensions/firefox/build/manifest.json`.
3. The toolbar button **Physical Agentic AI · OpenClaw Demo** opens the popup.
   **Ctrl+Shift+Y** (**Command+Shift+Y** on macOS) also opens it.

Firefox removes temporary add-ons when it restarts. A permanent install on
release Firefox needs a package signed by addons.mozilla.org, for example with
`web-ext sign --channel=unlisted` and the maintainer's AMO API key. Firefox
ESR, Developer Edition and Nightly can instead set
`xpinstall.signatures.required` to `false` in `about:config` and install the
unsigned `.xpi`.

## DGX Spark

The PAAI demo page, `http://127.0.0.1:8092`, is the demo's own web UI. Its chat
box relays to the OpenClaw `cascade-demo` agent. With this add-on loaded, the
page shows Kitchen, Worktop and Side automatically, as it does in the PAAI
Chromium profile.

The native OpenClaw Control UI runs on port 18790. On the Spark desktop,
`./run.sh dashboard` opens it in the PAAI Chromium profile. To use Firefox
there instead, run `./run.sh dashboard --no-open` and open the printed address
in Firefox. Treat that address like a password: it signs in to the gateway.

The Control UI advertises no cameras and its own `/api/status` answers 404.
Discovery then falls back to the Spark camera server on port 8091 under the
same loopback host grant, so **Show in chat** and the sidebar find the cameras
without extra setup.

## Differences from Chrome

- **Side panel** opens the Firefox sidebar. Firefox allows this only from a
  click in the extension's own popup or pages, so use the popup button.
- In **Show in chat**, Firefox runs the panel inside the page with the
  restricted content script API. It has no side panel or permission controls,
  so the panel hides its side panel button there. Use the popup for those
  actions.
- The toolbar button is placed directly in the toolbar
  (`"default_area": "navbar"`) instead of the Extensions menu.

## Verification status

On 29 September 2026, Firefox 155.0.1 (snap, headless, driven by geckodriver
0.37.1) loaded the built `.xpi` as a temporary add-on on the DGX Spark while
the demo was running. The final run used the shared code including the port
8091 discovery fallback:

- The add-on installed with no manifest warnings or errors, and its button was
  placed in the toolbar.
- The PAAI demo page on port 8092 showed all three cameras as **Live**, with
  advancing frame counters and decoded 1280 × 720 images. This needed no
  setup.
- A test page repeated the popup's calls to inject **Show in chat** into the
  OpenClaw login page on port 18790, with the discovery address the popup
  saves for that page and no saved camera configuration. The overlay found
  the cameras on port 8091 by itself and showed a live Worktop camera. The
  saved address stayed the page's own, so reopening repeats the same
  discovery.
- With `popup.html` opened as a tab, a WebDriver click on **Side panel**
  opened the sidebar, and `sidebarAction.isOpen` returned `true`. The sidebar
  used the discovery address saved for the 18790 page, found the cameras
  through the same fallback and showed a live camera.
- `panel.html?surface=side`, opened directly in a tab, showed the live MJPEG
  stream. That is a test path, not a user path: the test saved
  `http://127.0.0.1:8091/api/status` as the discovery address and allowed the
  tab in `storage.session`, which `worker.js` requires before a panel page in
  a tab may connect. Without that entry the page stays at
  **Permission needed**, as designed.

An earlier run on the code before the fallback showed **Offline** in the
18790 overlay until the test saved the port 8091 address.

Not tested: the popup opened from the toolbar button, the keyboard shortcut,
a visible window (all runs were headless), a browser on another computer
through the SSH tunnel, the host permission prompt for a remote demo, the
Brev deployment, the larger camera window, and a signed installation.
