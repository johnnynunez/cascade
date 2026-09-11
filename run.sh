#!/usr/bin/env bash
# cascade -- one click. Clone, then:
#
#     ./run.sh                    # Isaac Sim if installed, else MuJoCo; + OpenClaw chat
#     ./run.sh isaac              # force Isaac Sim (needs ISAACSIM_PATH or a standard install)
#     ./run.sh mujoco             # force MuJoCo (CPU, any laptop)
#     ./run.sh check [isaac]      # preflight only: what is missing, nothing started
#     ./run.sh down               # stop everything this script started
#
# First run on a fresh machine does the setup (uv venv, python extras, OpenClaw
# CLI) and is safe to re-run; every later run is just the launch. Everything
# else (flags like --headless, --brain, --no-open) is passed through to
# scripts/launch.sh -- see its header.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO"
# Desktop shells do not inherit the terminal used for installation. Restore
# its exact source/wheel selection and isolated chat profile on every entry.
[[ ! -f "$REPO/runs/.install/env.sh" ]] || source "$REPO/runs/.install/env.sh"
default_sim=auto
if [[ "${CASCADE_INSTALL_PROFILE:-}" == spark ]]; then default_sim=isaac; fi

mode="${1:-$default_sim}"
case "$mode" in
    isaac|mujoco|none|auto) shift || true ;;
    check)
        shift
        check_mode="$default_sim"
        case "${1:-}" in isaac|mujoco|none|auto) check_mode="$1"; shift ;; esac
        if [[ "${CASCADE_INSTALL_PROFILE:-}" == spark ]]; then set -- --brain cosmos "$@"; fi
        exec "$REPO/scripts/launch.sh" --check --sim "$check_mode" "$@" ;;
    down)  shift; exec "$REPO/scripts/launch.sh" --down "$@" ;;
    -h|--help|help) sed -n '2,12p' "$0"; exit 0 ;;
    --*) mode="$default_sim" ;;  # flags only: ./run.sh --headless
    *) echo "unknown mode '$mode' (isaac|mujoco|none|auto|check|down)" >&2; exit 2 ;;
esac

# Setup is needed until the venv imports cascade AND the OpenClaw CLI exists.
# (a plain string, not an array: macOS ships bash 3.2, where an empty array
# is "unbound" under set -u)
setup=""
if [[ "${CASCADE_INSTALL_PROFILE:-}" == spark ]]; then
    # Do not repair a deleted app environment with arbitrary platform wheels.
    exec "$REPO/scripts/launch.sh" --sim "$mode" --brain cosmos "$@"
fi
if ! [[ -x "$REPO/.venv/bin/python" ]] || ! "$REPO/.venv/bin/python" -c "import cascade" >/dev/null 2>&1 \
   || { [[ ! -x "$REPO/.openclaw-cli/bin/openclaw" ]] && ! command -v openclaw >/dev/null 2>&1; }; then
    setup="--setup"
fi
# shellcheck disable=SC2086  # $setup is intentionally word-split (empty or one flag)
exec "$REPO/scripts/launch.sh" $setup --sim "$mode" "$@"
