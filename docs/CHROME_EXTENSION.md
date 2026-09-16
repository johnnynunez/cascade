# Chrome camera companion

The optional extension places live Kitchen, Worktop and Side cameras in a
Chrome side panel or a movable panel inside OpenClaw. Robot orders still go
through OpenClaw. The public camera page works without this extension.

1. Open `chrome://extensions` in Chrome 116 or later and enable **Developer mode**.
2. Select **Load unpacked** and choose this checkout's `extensions/chrome` directory.
3. Open the private demo's `/guide/` page, click **Open OpenClaw**, and wait for
   the connection monitor to show **Ready**. In the connected chat tab, click
   the extension icon, or press **Ctrl+Shift+Y** (**Command+Shift+Y** on macOS).
4. Click **Connect cameras** and allow access to that demo's host.
5. Select **Side panel** or **Show in chat**, then choose a camera. Use
   **Reconnect** after a server restart. Reload the extension after source changes.

Click the camera image to enlarge it. Click again or press **Escape** to return.
With the keyboard, Tab to the image and press **Enter** or **Space**. In the side
panel, it opens in a larger window; closing that window also returns the camera.
In **Show in chat**, the camera expands within the current tab. The stream keeps
running while you switch sizes.

The page may advertise its status endpoint with
`<meta name="openclaw-demo" content="/api/status">`. Otherwise discovery
uses the active page's origin, including its port. All discovered camera
URLs must share the status host and scheme. The extension stores only
camera configuration and its selected view; it does not read chat messages
or store gateway credentials. Its history permission removes only exact
OpenClaw token-fragment visits on recognized origins.

The extension uses MJPEG camera endpoints. The ngrok visitor is a separate,
authenticated video and snapshot view; it does not provide the extension's MJPEG
transport or gateway administration. Use the private demo for the camera
companion.

## Verification status

On 15 September 2026, the extension was loaded in Chrome for Testing 151 on
the controller, connected to the running Brev demo. All three cameras enlarged
and returned with live 960 × 540 images. Mouse, Enter, Space, Escape and closing
the larger window were checked. The larger view fit a 1366 × 768 browser window.
Repeated toggles kept the same image element and did not start another MJPEG
request or abort the existing one. The embedded view also passed repeated
enlarge/return checks. The rendered views were inspected.

On 14 September 2026, Chrome for Testing 153 on the Brev RTX PRO 6000
loaded the unpacked extension and opened its genuine native side panel.
The Chrome host-permission dialog was accepted through the browser UI.
Kitchen, Worktop and Side each decoded 960 × 540 images with advancing
source frame IDs. Current desktop screenshots and a short recording verified
the extension itself. The test installed missing Chrome display libraries;
it did not require a second simulator or a model restart.

A subsequent Brev check used the guide's **Open OpenClaw** button, confirmed
the native authenticated connection and enabled message box, then opened the
camera side panel while the chat stayed connected. The revised PAAI text and
light theme rendered in the native popup and panel. This check reused the host
grant and sent no chat message or robot command.

**Show in chat** was also exercised inside the real OpenClaw login page on
the controller, including movement, collapse, restore and refresh. This is a
camera companion. The guide establishes OpenClaw's authenticated connection;
the extension does not supply or store its credentials.
Camera requests omit credentials, so use the private demo's supported MJPEG
endpoints. The authenticated public visitor has its own H.264 video and JPEG
fallback and does not use this extension.

Chrome for Testing is suitable for automated unpacked-extension checks;
regular Chrome installations should use **Load unpacked** in the extensions
page. Opening the side panel requires a user gesture, as documented by
[Chrome](https://developer.chrome.com/docs/extensions/reference/api/sidePanel).
