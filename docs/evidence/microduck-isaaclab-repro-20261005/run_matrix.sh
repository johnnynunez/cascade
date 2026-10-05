#!/usr/bin/env bash
# Run the Lab reproduction matrix, two processes at a time, each in its own systemd scope (Hermes shells are memcg-capped).
set -u
cd "$(dirname "$0")"
export CUDA_VISIBLE_DEVICES=GPU-4c811d02-a79b-2837-425e-5f62ac3c578a
PY=fork/.venv/bin/python
mkdir -p results logs
run_one() {  # task policy variant seed
  local task=$1 policy=$2 variant=$3 seed=$4
  local tag="${task}_${policy}_${variant}_seed${seed}"
  systemd-run --user --scope --collect --unit="lab-repro-${tag}" -p MemoryMax=64G -- \
    bash -c "cd fork && timeout 1800 ../$PY ../lab_microduck_walk_test.py --task $task --policy $policy --seed $seed --variant $variant --out ../results/${tag}.json > ../logs/run_${tag}.log 2>&1; echo EXIT=\$? >> ../logs/run_${tag}.log"
}
# matrix (task policy variant seed)
JOBS=(
  "flat velocity_flat lab_play 0"
  "flat velocity_flat lab_play 1"
  "rough velocity_rough lab_play 0"
  "rough velocity_rough lab_play 1"
  "flat velocity_flat norand 0"
  "rough velocity_rough norand 0"
  "flat velocity_flat cascade_nominal 0"
  "rough velocity_rough cascade_nominal 0"
  "flat velocity_rough lab_play 0"
)
i=0
for job in "${JOBS[@]}"; do
  run_one $job &
  i=$((i+1))
  if (( i % 2 == 0 )); then wait; fi
done
wait
echo "ALL DONE"
