#!/usr/bin/env bash
set -euo pipefail
campaign_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec python3 "$campaign_dir/deploy.py" "$@"
