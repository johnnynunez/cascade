#!/usr/bin/env bash
# Install wrc_demo into the shared .demo venv on this rig (uv-managed; the
# venv has no pip module, so everything goes through `uv pip`).
set -euo pipefail

PY="${PY:-/home/spark/Projects/demo/.demo/bin/python}"
REPO="$(cd "$(dirname "$0")/.." && pwd)"

echo "[+] installing wrc_demo (editable) + extras into $PY"
uv pip install --python "$PY" -e "$REPO" ultralytics openai anthropic motorbridge pytest

# YOLOE / YOLO-World open-vocabulary text prompts need ultralytics' CLIP fork
# (closed-set yolo11n.pt works without it):
uv pip install --python "$PY" "git+https://github.com/ultralytics/CLIP.git" || \
    echo "[!] CLIP install skipped - open-vocab prompts unavailable until installed"

cat <<'EOF'
[+] done. Reminders for the live rig:
  - pyrealsense2 comes from the librealsense L515 fork build:
      ~/Projects/librealsense/build/Release/pyrealsense2*.so
    (already copied into .demo site-packages on this machine)
  - CAN bus: sudo ip link set can0 up type can bitrate 1000000
  - do not run motorbridge-gateway / MotorBridge Studio while the demo runs
EOF
