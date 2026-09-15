#!/bin/bash
# Full 10-case validation run
# Run from: D:\大创\v3_20260525

cd "D:\大创\v3_20260525"

LOG="output/full_validation_run_$(date +%Y%m%d_%H%M%S).log"

echo "============================================================" | tee -a "$LOG"
echo "  Full v3 validation started at $(date)" | tee -a "$LOG"
echo "============================================================" | tee -a "$LOG"

PYTHONUNBUFFERED=1 /d/python/python.exe -u validate_v3.py 2>&1 | tee -a "$LOG"

EXIT_CODE=${PIPESTATUS[0]}
echo "============================================================" | tee -a "$LOG"
echo "  Finished at $(date) with exit code: $EXIT_CODE" | tee -a "$LOG"
echo "============================================================" | tee -a "$LOG"
