#!/usr/bin/env bash
# One-shot: install cascade core deps (kinematics + arm + llm + dev) into .venv.
# Run:  bash /home/seeed/cascade/install_deps.sh
set -euo pipefail
cd "$(dirname "$0")"

export HTTPS_PROXY=http://127.0.0.1:7897
export HTTP_PROXY=http://127.0.0.1:7897

uv pip install --python .venv/bin/python -e '.[dev,kinematics,arm,llm]'

echo
echo "=== installed ==="
.venv/bin/python -c "import importlib.util as u; [print(f'{m:14s}', 'OK' if u.find_spec(m) else 'MISSING') for m in ['pin','motorbridge','openai','anthropic','cv2','yaml','numpy']]"
