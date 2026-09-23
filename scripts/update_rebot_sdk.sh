#!/usr/bin/env bash
# Keep the reBotArm_control_py SDK fresh, in the place the `rebot_rs` arm
# profile expects it: one directory up from this repo
# (<repo>/../reBotArm_control_py), matching the profile's portable
# `sdk_path: ${repo}/../reBotArm_control_py`.
#
#   bash scripts/update_rebot_sdk.sh
#
# - Clones from the canonical Seeed GitHub repo if missing.
# - Otherwise points `origin` at GitHub and pulls the latest `main`.
# - `--autostash` shelves any local tuning (e.g. hand-tuned MIT gains in
#   config/rebotarm_rs.yaml) across the rebase and re-applies it. If upstream
#   touched the same lines, git stops and asks you to resolve -- it never
#   silently drops your changes.
set -euo pipefail
cd "$(dirname "$0")/.."

SDK_NAME="reBotArm_control_py"
SDK_DIR="$(cd .. && pwd)/$SDK_NAME"
UPSTREAM="https://github.com/Seeed-Projects/reBotArm_control_py.git"
BRANCH="main"

# GitHub is behind the GFW on this network; harmless when a proxy is already set.
export HTTPS_PROXY="${HTTPS_PROXY:-http://127.0.0.1:7897}"
export HTTP_PROXY="${HTTP_PROXY:-http://127.0.0.1:7897}"

if [ ! -d "$SDK_DIR/.git" ]; then
    echo "==> cloning $UPSTREAM"
    git clone "$UPSTREAM" "$SDK_DIR"
else
    # Point origin at the canonical repo (a clone made elsewhere may carry a
    # mirror origin); then fast-forward to the latest main.
    git -C "$SDK_DIR" remote set-url origin "$UPSTREAM"
    git -C "$SDK_DIR" pull --rebase --autostash origin "$BRANCH"
fi

echo "==> SDK: $(git -C "$SDK_DIR" rev-parse --short HEAD)  ($(git -C "$SDK_DIR" remote get-url origin))"
echo "==> cascade sdk_path resolves to: $SDK_DIR"
