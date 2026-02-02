#!/bin/bash
#
# Batch Submission Script for NMT Training
#
# Submits Slurm jobs for all supported languages using the 'xlarge' config.
# Logs submission details and Job IDs to logs/job_history.csv.
#
# Usage:
#   ./scripts/submit_all_jobs.sh [CONFIG]
#   ./scripts/submit_all_jobs.sh small  # Override config (default: xlarge)

CONFIG=${1:-xlarge}
LOG_DIR="logs"
HISTORY_FILE="${LOG_DIR}/job_history.csv"

# Supported languages
LANGS=("as" "bn" "gu" "hi" "kn" "ml" "mr" "or" "pa" "ta" "te")

# Create logs directory
mkdir -p "$LOG_DIR"

# Initialize history file if not exists
if [[ ! -f "$HISTORY_FILE" ]]; then
    echo "Date,Time,Language,Config,Job_ID,Status" > "$HISTORY_FILE"
fi

echo "========================================================"
echo "Batch Submitting NMT Training Jobs"
echo "Config: $CONFIG"
echo "Target Languages: ${LANGS[*]}"
echo "========================================================"
echo ""

for lang in "${LANGS[@]}"; do
    echo -n "Submitting $lang... "
    
    # Submit job and capture ID
    # sbatch output format: "Submitted batch job 123456"
    OUTPUT=$(sbatch scripts/train.slurm "$lang" "$CONFIG")
    RET=$?
    
    if [[ $RET -eq 0 ]]; then
        # Extract Job ID
        JOB_ID=$(echo "$OUTPUT" | awk '{print $4}')
        
        echo "✓ Job ID: $JOB_ID"
        
        # Log to history
        TIMESTAMP=$(date "+%Y-%m-%d,%H:%M:%S")
        echo "$TIMESTAMP,$lang,$CONFIG,$JOB_ID,Submitted" >> "$HISTORY_FILE"
    else
        echo "✗ Failed"
        echo "  Error: $OUTPUT"
    fi
    
    # Small delay to be nice to the scheduler
    sleep 1
done

echo ""
echo "--------------------------------------------------------"
echo "Submission Complete!"
echo "History saved to: $HISTORY_FILE"
echo "Monitor jobs with: squeue --me"
echo "========================================================"
