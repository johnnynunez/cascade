# PAAI camera companion for Firefox

The same camera companion as [`extensions/chrome`](../chrome), packaged for
Firefox 140 or later. All JavaScript, HTML and CSS come from `extensions/chrome`;
this directory holds only the Firefox `manifest.json`, this README, the
packaging script and tests.

Build it with `python3 package.py`. That writes a loadable directory, `build/`,
and a deterministic `Physical-Agentic-AI-OpenClaw-Demo.xpi` with its SHA256.
Both outputs are ignored by Git.

Load it for testing: open `about:debugging`, choose **This Firefox**, then
**Load Temporary Add-on**, and select the `.xpi` or `build/manifest.json`.
Firefox removes temporary add-ons when it restarts. A permanent install in
release Firefox needs a signed package from addons.mozilla.org, for example
`web-ext sign --channel=unlisted` with the maintainer's AMO API key. Firefox
ESR, Developer Edition and Nightly can instead set
`xpinstall.signatures.required` to `false` in `about:config`.

The toolbar button opens the same popup as in Chrome, and **Side panel** opens
the Firefox sidebar. **Show in chat** works as in Chrome, but Firefox gives the
panel inside the page no side panel or permission controls, so the panel hides
its side panel button there. Use the toolbar popup for those actions.

On the DGX Spark, the PAAI demo page on port 8092 shows the three cameras
automatically, as in Chromium. `./run.sh dashboard` opens the native OpenClaw
Control UI in the PAAI Chromium profile on the Spark desktop; open the address
from `./run.sh dashboard --no-open` in Firefox to use this build there. The
gateway advertises no cameras, so discovery falls back to port 8091, as in
Chrome.

Run `node --test tests/*.test.mjs` for the Firefox manifest and package checks.
See [installation and verification](../../docs/FIREFOX_EXTENSION.md).
