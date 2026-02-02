#!/bin/bash
#SBATCH --job-name=nmt-train
#SBATCH --output=logs/slurm-%x-%j.out
#SBATCH --error=logs/slurm-%x-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=3-00:00:00
#SBATCH --mem=32G
#SBATCH --partition=workq

# ============================================================================
# NMT Training Slurm Submission Script
# ============================================================================
# Usage:
#   sbatch scripts/train.slurm [TARGET_LANG] [CONFIG_NAME]
#
# Example:
#   sbatch scripts/train.slurm hi base
#   sbatch scripts/train.slurm ta small
# ============================================================================

# Default arguments
LANG=${1:-hi}
CONFIG=${2:-base}

echo "Date: $(date)"
echo "Host: $(hostname)"
echo "Job Directory: $(pwd)"
echo "Job ID: $SLURM_JOB_ID"
echo "Target Language: $LANG"
echo "Configuration: $CONFIG"
echo ""

# Load environment
# Ensure we source the correct virtual environment
if [ -d ".venv" ]; then
    echo "Activating virtual environment (.venv)..."
    source .venv/bin/activate
elif [ -d "venv" ]; then
    echo "Activating virtual environment (venv)..."
    source venv/bin/activate
else
    echo "Warning: No virtual environment found. Assuming modules/conda already loaded."
fi

# Run the training pipeline
# We use --yes to skip interactive prompts and --lang to specify the language
echo "Starting training pipeline..."
./scripts/train_pipeline.sh --lang "$LANG" --config "$CONFIG" --yes

echo ""
echo "Job finished at: $(date)"
