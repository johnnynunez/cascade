#!/usr/bin/env bash
# Compatibility front door. Real installation is always delegated to install.sh.
# A cold stdin invocation keeps only the small read-only plan/consent gate here,
# so --dry-run/--check do not even download the real installer.
set -euo pipefail

if [[ -n "${BASH_SOURCE[0]:-}" && -f "${BASH_SOURCE[0]}" ]]; then
    LOCAL="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/install.sh"
    [[ ! -f "$LOCAL" ]] || exec bash "$LOCAL" "$@"
fi
bootstrap() {
    local profile=spark dir="${CASCADE_HOME:-$HOME/cascade}" ref=main brain="" dry=0 check=0 accept=0
    local arg
    # Validate before downloading/executing anything. Keep this small cold-loader
    # contract in sync with install.sh; subprocess tests cover both stdin paths.
    while [[ $# -gt 0 ]]; do
        arg="$1"
        case "$arg" in
            --profile|--dir|--ref|--brain)
                [[ $# -ge 2 && -n "$2" && "$2" != -* ]] || { printf 'missing value for %s\n' "$arg" >&2; return 2; }
                case "$arg" in --profile) profile="$2" ;; --dir) dir="$2" ;; --ref) ref="$2" ;; --brain) brain="$2" ;; esac
                shift 2 ;;
            --dry-run) dry=1; shift ;;
            --check) check=1; shift ;;
            --accept-eula) accept=1; shift ;;
            --no-open|--prepare-only) shift ;;
            -h|--help)
                printf '%s\n' 'bootstrap.sh -> install.sh [--profile spark|laptop|ci] [--dir PATH] [--ref REF]' \
                    '  --dry-run --check --accept-eula --prepare-only --no-open --brain qwen|keep'
                return 0 ;;
            *) printf 'unknown flag %s\n' "$arg" >&2; return 2 ;;
        esac
    done
    case "$profile" in spark) brain="${brain:-qwen}" ;; laptop|ci) brain="${brain:-keep}" ;; *) printf 'unknown --profile %s\n' "$profile" >&2; return 2 ;; esac
    case "$brain" in qwen|keep) ;; *) printf 'unknown --brain %s\n' "$brain" >&2; return 2 ;; esac
    [[ "$profile" == spark || "$brain" == keep ]] || { printf '%s\n' '--brain qwen requires --profile spark' >&2; return 2; }
    [[ "$profile" != spark || "$brain" == qwen ]] || { printf '%s\n' 'Spark delivery requires --brain qwen' >&2; return 2; }
    [[ "$ref" =~ ^[A-Za-z0-9_][A-Za-z0-9_./-]*$ && "$ref" != *..* && "$ref" != */ && "$ref" != *. && "$ref" != *.lock && "$ref" != */.* && "$ref" != *//* ]] || { printf 'invalid --ref %s\n' "$ref" >&2; return 2; }
    if [[ -f "$dir/scripts/install.sh" ]]; then
        return 10  # use local source; no network needed
    fi
    printf '[cascade-install] profile=%s brain=%s source=%s dir=%s\n' "$profile" "$brain" "$ref" "$dir"
    printf '[cascade-install] CASCADE Python 3.12 in %s/.venv\n' "$dir"
    if [[ "$profile" == spark ]]; then
        printf '[cascade-install] Isaac Sim 6.1.0, exact wheel 6.1.0.0 / Python 3.12 in %s/.isaacsim\n' "$dir"
        printf '[cascade-install] Qwen Q4 + vision projector: pinned downloads in %s/models/qwen3.8-27b; project-owned CUDA llama.cpp, loopback :8080\n' "$dir"
        printf '%s\n' '[cascade-install] Kitchen: download and verify the 166-file kitchen-v1 release automatically.'
    fi
    [[ "$profile" == ci ]] || printf '[cascade-install] OpenClaw 2026.9.3 rootless in %s/.openclaw-cli (Spark profile cascade-demo)\n' "$dir"
    [[ "$dry" != 1 ]] || return 0
    if [[ "$check" == 1 ]]; then
        printf '%s\n' "MISSING: source checkout, $dir/.venv, $dir/.isaacsim, local model / llama-server, OpenClaw private CLI"
        return 3
    fi
    [[ "$profile" != spark || "$accept" == 1 ]] || { printf '%s\n' 'Spark installation requires --accept-eula' >&2; return 2; }
    [[ "$profile" != spark || ( "$(uname -s)" == Linux && "$(uname -m)" == aarch64 ) ]] || { printf '%s\n' 'Spark requires Linux aarch64; use --profile laptop or ci explicitly' >&2; return 2; }
    command -v git >/dev/null 2>&1 || { printf '%s\n' 'git required; no system packages are installed automatically' >&2; return 2; }
    git check-ref-format --allow-onelevel "$ref" >/dev/null 2>&1 || { printf 'invalid --ref %s\n' "$ref" >&2; return 2; }
    return 11  # consent given; download the selected entry point in memory
}
# Retain original arguments without an empty array (bash 3.2 + nounset).
STATUS=0
bootstrap "$@" || STATUS=$?
case "$STATUS" in
    0|2|3) exit "$STATUS" ;;
    10|11)
        DIR="${CASCADE_HOME:-$HOME/cascade}"
        REF=main
        # bootstrap has validated the arity of these options.
        EXPECT=""
        for ARG in "$@"; do
            if [[ "$EXPECT" == --dir ]]; then DIR="$ARG"; EXPECT=""
            elif [[ "$EXPECT" == --ref ]]; then REF="$ARG"; EXPECT=""
            elif [[ "$ARG" == --dir || "$ARG" == --ref ]]; then EXPECT="$ARG"
            fi
        done
        if [[ "$STATUS" == 10 ]]; then exec bash "$DIR/scripts/install.sh" "$@"; fi
        PAYLOAD="$(curl -fLsS --retry 3 --retry-delay 1 "https://raw.githubusercontent.com/johnnynunez/cascade/$REF/scripts/install.sh")"
        [[ -n "$PAYLOAD" ]] || { printf '%s\n' 'empty installer download' >&2; exit 1; }
        exec bash -c "$PAYLOAD" -- "$@" ;;
    *) exit "$STATUS" ;;
esac
