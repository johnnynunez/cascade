#!/usr/bin/env bash
# live-w6 pre-Isaac guard: the user's benchmarks always win.
# exit 0 = clear; exit 3 = a user benchmark runs or GPU 0 has a compute process that is not ours.
GPU0=GPU-87c0fe81-bcfb-b636-9eec-e56fd88c33c3
bench="$(ps -eo pid,args | grep -E 'blogs-gpu-repair|blog3-verification|supervisor.py' | grep -v -E 'grep|guard.sh')"
if [[ -n "$bench" ]]; then
    echo "GUARD: user benchmark running: $bench"
    exit 3
fi
foreign=""
while IFS=, read -r pid uuid mem; do
    pid="${pid// /}"; uuid="${uuid// /}"
    [[ -z "$pid" || "$uuid" != "$GPU0" ]] && continue
    if ! grep -q "hermes-live-w6" "/proc/$pid/cgroup" 2>/dev/null; then
        foreign+=" $pid($mem)"
    fi
done < <(nvidia-smi --query-compute-apps=pid,gpu_uuid,used_memory --format=csv,noheader 2>/dev/null)
if [[ -n "$foreign" ]]; then
    echo "GUARD: GPU 0 has compute processes that are not ours:$foreign"
    exit 3
fi
echo "GUARD: clear $(date -Is) GPU0=$(nvidia-smi -i 0 --query-gpu=memory.used,utilization.gpu --format=csv,noheader)"
exit 0
