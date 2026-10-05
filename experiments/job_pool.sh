#!/bin/bash
# Generic GPU job pool. Job lines: "script|model|arm|split_or_-|tag|ENV=V ENV2=V|extra args"
# split "-" means the script takes no --split (FEVER). No automatic retries (deterministic failures are logged, not repeated).
# Usage: ./job_pool.sh <jobs_file> <log_file> <gpu> [<gpu> ...]
PY=python
JOBS=$1; LOG=$2; shift 2; LOCK=$JOBS.lock
pop () { ( flock 9; l=$(head -1 "$JOBS"); [ -n "$l" ] && sed -i '1d' "$JOBS"; echo "$l" ) 9>"$LOCK"; }
worker () { G=$1
  while true; do
    J=$(pop); [ -z "$J" ] && break
    IFS='|' read -r S M ARM SPLIT TAG ENVS EXTRA <<< "$J"
    LG=results_v2/job_${S%.py}_${M}_${ARM}_${TAG}.log
    SPL=""; [ "$SPLIT" != "-" ] && SPL="--split $SPLIT"
    env X=1 $ENVS $PY $S --arm $ARM --model $M --gpu $G $SPL --tag $TAG $EXTRA > $LG 2>&1; EC=$?
    echo "JOB|$S|$M|$ARM|$SPLIT|$TAG|$EC|$(grep RUN_DIRECTORY $LG | tail -1 | sed 's/RUN_DIRECTORY: //')" >> "$LOG"
  done
  echo "WORKER_DONE|gpu$G" >> "$LOG"
}
for G in "$@"; do worker $G & done
wait; echo "POOL_DONE" >> "$LOG"
