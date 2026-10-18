#!/usr/bin/env bash
# Isaac Sim 6.2 (source build) headless, NEWTON engine only, Spark kitchen scene, for the "beside Isaac" timings of
# PR perf/bolt-default-fixes. COPY of the brain comparison's run_isaac_cmp.sh with ONE change: the bridge listens on
# :8641 (8611/18498 are other runs' ports). Code = the demo-video worktree (origin/main 425fb67), read-only.
set -euo pipefail
WT=/home/spark/Projects/cascade-wt/demo-video
source /home/spark/Projects/isaac/env.sh
cd "$WT"
unset PYTHONPATH DISPLAY WAYLAND_DISPLAY
export CASCADE_PROOF_CAMERA=1 CASCADE_ISAAC_PIXEL_MASK=1 CASCADE_ISAAC_DT=0.008333333333333333 \
       CASCADE_PHYSICS_DEVICE=cuda:0 CASCADE_REQUIRE_CUDA=1 \
       CASCADE_ISAAC_CAM_EVERY=6 CASCADE_OBSERVED_FINGER_GATE=1
echo "env: ENGINE=newton SUBSTEPS=5 DT=$CASCADE_ISAAC_DT CAM_EVERY=$CASCADE_ISAAC_CAM_EVERY FINGER_GATE=$CASCADE_OBSERVED_FINGER_GATE LD_PRELOAD=${LD_PRELOAD:-} git=$(git -C "$WT" rev-parse --short HEAD)"
SCENE="$(python3 -c 'import pathlib,sys; print(pathlib.Path(sys.argv[1]).resolve(strict=True))' "$WT/demo/scene/kitchen_config.json")"
exec "$WT/.venv/bin/python" "$WT/scripts/isaac_launch.py" --python "$ISAACSIM_PATH/python.sh" -- \
    "$WT/scripts/isaac_bridge.py" --port 8641 --usd "$WT/assets/usd/RS-rebot-dev-arm/RS-rebot-dev-arm.usda" \
    --engine newton --newton-substeps 5 --scene-config "$SCENE"
