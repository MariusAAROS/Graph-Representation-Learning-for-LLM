#!/usr/bin/env bash
# Start both queues in the background, detached from the shell, with a shared deadline.
#
#   bash scripts/pretrained_recursion/launch.sh            # deadline = now + 11 h
#   HOURS=10 bash scripts/pretrained_recursion/launch.sh
#
# Re-running it after a crash keeps finished jobs (done markers) and resets the deadline,
# so only re-run with a HOURS that matches the time actually left on the machine.
set -euo pipefail
source "$(dirname "$0")/common.sh"

HOURS=${HOURS:-11}
echo $(( $(date +%s) + HOURS * 3600 )) > "$ROOT/deadline"
log "deadline: $(date -d @"$(cat "$ROOT/deadline")" '+%F %T') (${HOURS} h)"

for gpu in 0 1; do
  nohup setsid bash "$GRL/scripts/pretrained_recursion/queue_gpu${gpu}.sh" \
    >> "$ROOT/logs/queue_gpu${gpu}.log" 2>&1 < /dev/null &
  log "queue gpu${gpu} started (pid $!), log: $ROOT/logs/queue_gpu${gpu}.log"
done
