#!/usr/bin/env bash
# Kernel-log Xid watcher for one GPU window (throwaway).  xidwatch_bf.sh <status file> <since epoch> <driver unit> <units...>
# Every 5 s: any `NVRM: Xid` since <since> -> stop every listed unit (Isaac first, then the brain), write
# <status dir>/xid.txt and exit 1. Exits 0 when the driver unit is gone.
ST=$1; SINCE=$2; DRIVER=$3; shift 3
while systemctl --user is-active -q "$DRIVER"; do
  x=$(journalctl -k --since "@$SINCE" --no-pager 2>/dev/null | grep -E "NVRM: Xid" || true)
  if [ -n "$x" ]; then
    echo "$x" > "$(dirname "$ST")/xid.txt"
    echo "$(date +%T) XID WATCHER: $(echo "$x" | head -n 1 | cut -c1-200) -> stopping $*" >> "$ST"
    for u in "$@"; do systemctl --user stop "$u" 2>/dev/null; done
    systemctl --user stop "$DRIVER" 2>/dev/null
    exit 1
  fi
  sleep 5
done
exit 0
