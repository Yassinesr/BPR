#!/usr/bin/env bash
set -eu

########################################
# Config - edit these for your setup
########################################
CONFIG="configs/bpr/hrnet48_256.py"    # config path (relative to repo root)
WORK_DIR="/home/yassine/projects/BPR/work_dirs/hrnet48_256"
CHECKPOINT="${WORK_DIR}/latest.pth"    # checkpoint to resume from (optional)
NGPUS=1                                # number of GPUs to use with dist_train.sh
RESTART_DELAY=10                       # seconds to wait before restart after crash
LOGFILE="${WORK_DIR}/auto_train.log"   # main log file for this wrapper
########################################

# Ensure work dir exists
mkdir -p "$WORK_DIR"

echo "=== Auto-train wrapper started ===" | tee -a "$LOGFILE"
echo "CONFIG: $CONFIG" | tee -a "$LOGFILE"
echo "WORK_DIR: $WORK_DIR" | tee -a "$LOGFILE"
echo "NGPUS: $NGPUS" | tee -a "$LOGFILE"
echo "CHECKPOINT: $CHECKPOINT" | tee -a "$LOGFILE"
echo "Restart delay: ${RESTART_DELAY}s" | tee -a "$LOGFILE"

# Loop: run training, if it exits with code 0 -> finish; otherwise restart
while true; do
    echo "$(date '+%Y-%m-%d %H:%M:%S') - Launching training..." | tee -a "$LOGFILE"

    # Launch. Use dist_train.sh wrapper which sets RANK/WORLD_SIZE properly.
    # Modify the next line if you prefer python train.py directly for single-gpu.
    bash tools/dist_train.sh "$CONFIG" "$NGPUS" --resume-from "$CHECKPOINT" 2>&1 | tee -a "$LOGFILE"
    RC=$?

    if [ "$RC" -eq 0 ]; then
        echo "$(date '+%Y-%m-%d %H:%M:%S') - Training exited normally (RC=0). Stopping auto-restart." | tee -a "$LOGFILE"
        break
    else
        echo "$(date '+%Y-%m-%d %H:%M:%S') - Training crashed or was terminated (RC=$RC)." | tee -a "$LOGFILE"
        echo "$(date '+%Y-%m-%d %H:%M:%S') - Waiting ${RESTART_DELAY}s before restarting..." | tee -a "$LOGFILE"
        sleep "$RESTART_DELAY"
        # Optionally update CHECKPOINT to latest.pth before restarting
        if [ -f "${WORK_DIR}/latest.pth" ]; then
            CHECKPOINT="${WORK_DIR}/latest.pth"
            echo "$(date '+%Y-%m-%d %H:%M:%S') - Found updated checkpoint: ${CHECKPOINT}. Will resume from it." | tee -a "$LOGFILE"
        else
            echo "$(date '+%Y-%m-%d %H:%M:%S') - No latest.pth found; will attempt restart with current CHECKPOINT=$CHECKPOINT" | tee -a "$LOGFILE"
        fi
    fi
done

echo "=== Auto-train wrapper finished ===" | tee -a "$LOGFILE"

