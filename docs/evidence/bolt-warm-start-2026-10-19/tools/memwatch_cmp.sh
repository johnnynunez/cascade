#!/usr/bin/env bash
# Memory guard for a comparison brain unit (throwaway).  memwatch_cmp.sh <unit> <log> <kill_below_GB> [container]
# Every 2 s: MemAvailable; logs every 10 s plus the running minimum. Below the threshold: docker kill the container
# (a container brain is outside the unit's cgroup) and SIGKILL the unit, so the host never freezes.
U=$1; LOG=$2; THR_KB=$(( ${3:-3} * 1024 * 1024 )); C=${4:-}
: > "$LOG"; min=999999999; i=0; sleep 3
while systemctl --user is-active -q "$U"; do
  a=$(awk '/MemAvailable/ {print $2}' /proc/meminfo)
  [ "$a" -lt "$min" ] && min=$a
  [ $((i % 5)) -eq 0 ] && echo "$(date +%T) avail_GB=$((a / 1048576)) min_GB=$((min / 1048576)) swap_used_MB=$(awk '/SwapTotal/{t=$2}/SwapFree/{f=$2}END{print int((t-f)/1024)}' /proc/meminfo) load=$(cut -d' ' -f1 /proc/loadavg)" >> "$LOG"
  if [ "$a" -lt "$THR_KB" ]; then
    echo "$(date +%T) KILL: MemAvailable $((a / 1024)) MB" >> "$LOG"
    [ -n "$C" ] && docker kill "$C" >> "$LOG" 2>&1
    systemctl --user kill -s KILL "$U"; break
  fi
  i=$((i + 1)); sleep 2
done
echo "$(date +%T) END min_avail_GB=$(awk "BEGIN{printf \"%.1f\", $min/1048576}")" >> "$LOG"
