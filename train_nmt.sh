#!/bin/bash
#SBATCH --partition=workq
#SBATCH --job-name=nmt-${TARGET_LANG:-hi}
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/dist_home/nooglers/nooglers/Hari/Subtitle-Generator/temp/logs/%x_%j.out
#SBATCH --error=/dist_home/nooglers/nooglers/Hari/Subtitle-Generator/temp/logs/%x_%j.err

set -e

BASE_DIR="/dist_home/nooglers/nooglers/Hari/Subtitle-Generator"
LANG="${TARGET_LANG:-hi}"
LOG_DIR="${BASE_DIR}/temp/logs/${LANG}"

# directory already exists (safe anyway)
mkdir -p "${LOG_DIR}"

# load cuda safely (DO NOT EXIT if missing)
module purge || true
module load cuda || echo "CUDA module not found, relying on system CUDA"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1

cd "${BASE_DIR}"

# activate venv safely
if [ -f ".venv/bin/activate" ]; then
  source .venv/bin/activate
else
  echo "ERROR: .venv not found at ${BASE_DIR}/.venv" >&2
  exit 1
fi

# run training
./scripts/train_pipeline.sh \
  --lang "${LANG}" \
  --yes \
  2>&1 | tee "${LOG_DIR}/train_${SLURM_JOB_ID}.log"



