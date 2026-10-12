#!/usr/bin/env python3
"""Booth screen orientation: portrait (camera above the chat) or landscape (camera beside it).

The camera companion follows the window shape, so rotating the screen re-docks the panel
without reloading OpenClaw. Rotation goes through Mutter's DisplayConfig (what GNOME Settings
> Displays uses) as a temporary config: a persistent one makes GNOME ask "Keep these display
settings?" and revert after 20 s when nobody answers. The choice is saved in
~/.config/paai-orientation and reapplied at login by the autostart entry from `install`.

  python3 scripts/screen_orientation.py toggle      # portrait <-> landscape
  python3 scripts/screen_orientation.py portrait|landscape|status
  python3 scripts/screen_orientation.py install     # Ctrl+Alt+R, desktop icon, login autostart

PAAI_PORTRAIT_ROTATION=right picks clockwise portrait (default left, counterclockwise).
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

TRANSFORMS = {"normal": 0, "left": 1, "inverted": 2, "right": 3}
STATE = Path.home() / ".config/paai-orientation"
BUS = "org.gnome.Mutter.DisplayConfig"
KEYS = "org.gnome.settings-daemon.plugins.media-keys"
BINDING_PATH = f"/{KEYS.replace('.', '/')}/custom-keybindings/paai-orientation/"
SHORTCUT = "<Control><Alt>r"


def _proxy():
    from gi.repository import Gio
    return Gio.DBusProxy.new_for_bus_sync(Gio.BusType.SESSION, Gio.DBusProxyFlags.NONE, None,
                                          BUS, "/org/gnome/Mutter/DisplayConfig", BUS, None)


def _call(proxy, method, args=None):
    from gi.repository import Gio
    return proxy.call_sync(method, args, Gio.DBusCallFlags.NONE, -1, None).unpack()


def current() -> str:
    """'portrait' or 'landscape' from the primary logical monitor's transform."""
    _serial, _monitors, logical, _props = _call(_proxy(), "GetCurrentState")
    primary = next((m for m in logical if m[4]), logical[0])
    return "portrait" if primary[3] % 2 else "landscape"


def rotate(transform: str) -> None:
    from gi.repository import GLib
    proxy = _proxy()
    serial, monitors, logical, _props = _call(proxy, "GetCurrentState")
    modes = {}
    for (connector, *_ids), monitor_modes, _mprops in monitors:
        for mode_id, *_rest, mode_props in monitor_modes:
            if mode_props.get("is-current"):
                modes[connector] = mode_id
    configs = [(x, y, scale, TRANSFORMS[transform], primary,
                [(spec[0], modes[spec[0]], {}) for spec in specs if spec[0] in modes])
               for x, y, scale, _transform, primary, specs, _lprops in logical]
    _call(proxy, "ApplyMonitorsConfig",
          GLib.Variant("(uua(iiduba(ssa{sv}))a{sv})", (serial, 1, configs, {})))
    STATE.parent.mkdir(parents=True, exist_ok=True)
    STATE.write_text(transform + "\n")


def remaximize_browser() -> None:
    """Maximized windows keep their old size after a rotation on some setups."""
    xdotool = shutil.which("xdotool")
    if not xdotool or not os.environ.get("DISPLAY"):
        return
    time.sleep(2)
    found = subprocess.run([xdotool, "search", "--onlyvisible", "--class", "chromium"],
                           capture_output=True, text=True, check=False)
    for window in found.stdout.split():
        subprocess.run([xdotool, "windowactivate", "--sync", window, "key", "--clearmodifiers", "super+Up"],
                       capture_output=True, check=False)


def set_mode(mode: str) -> None:
    portrait = os.environ.get("PAAI_PORTRAIT_ROTATION", "left")
    if portrait not in ("left", "right"):
        raise SystemExit("PAAI_PORTRAIT_ROTATION must be left or right")
    rotate(portrait if mode == "portrait" else "normal")
    remaximize_browser()
    print(f"screen: {mode}")


def install(script: Path) -> None:
    command = f"/usr/bin/python3 {script} toggle"
    keys = subprocess.run(["gsettings", "get", KEYS, "custom-keybindings"],
                          capture_output=True, text=True, check=True).stdout.strip()
    paths = [] if keys in ("@as []", "[]") else [p.strip(" '") for p in keys.strip("[]").split(",")]
    if BINDING_PATH not in paths:
        paths.append(BINDING_PATH)
        subprocess.run(["gsettings", "set", KEYS, "custom-keybindings", str(paths)], check=True)
    schema = f"{KEYS}.custom-keybinding:{BINDING_PATH}"
    for key, value in (("name", "PAAI rotate screen"), ("command", command), ("binding", SHORTCUT)):
        subprocess.run(["gsettings", "set", schema, key, value], check=True)

    entry = ("[Desktop Entry]\nVersion=1.0\nType=Application\n"
             "Name=PAAI - ROTATE SCREEN (vertical / horizontal)\n"
             "Comment=Toggle the booth screen between portrait and landscape (Ctrl+Alt+R)\n"
             f"Exec={command}\nIcon=object-rotate-left\nTerminal=false\n")
    desktop = Path(subprocess.run(["xdg-user-dir", "DESKTOP"], capture_output=True, text=True,
                                  check=False).stdout.strip() or Path.home() / "Desktop")
    icon = desktop / "paai-rotate-screen.desktop"
    icon.write_text(entry)
    icon.chmod(0o755)
    subprocess.run(["gio", "set", str(icon), "metadata::trusted", "true"], capture_output=True, check=False)

    autostart = Path.home() / ".config/autostart/paai-orientation.desktop"
    autostart.parent.mkdir(parents=True, exist_ok=True)
    autostart.write_text("[Desktop Entry]\nType=Application\nName=PAAI screen orientation\n"
                         f"Exec=/usr/bin/python3 {script} restore\nNoDisplay=true\n"
                         "X-GNOME-Autostart-enabled=true\n")
    print(f"installed: {SHORTCUT} shortcut, {icon}, {autostart}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("action", choices=["toggle", "portrait", "landscape", "status", "restore", "install"])
    action = parser.parse_args().action
    if action == "status":
        print(current())
    elif action == "install":
        install(Path(__file__).resolve())
    elif action == "restore":
        if STATE.is_file() and STATE.read_text().strip() in TRANSFORMS:
            for _ in range(20):  # Mutter may still be starting at login.
                try:
                    rotate(STATE.read_text().strip())
                    break
                except Exception:
                    time.sleep(1)
    else:
        set_mode(action if action != "toggle" else
                 ("landscape" if current() == "portrait" else "portrait"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
