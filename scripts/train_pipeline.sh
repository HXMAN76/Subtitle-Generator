#!/bin/bash
#
# NMT Training Pipeline for Samanantar Dataset
# 
# This script handles the complete training pipeline:
# 1. Download dataset for specified language
# 2. Create tokenizer corpus
# 3. Create validation split
# 4. Train tokenizer
# 5. Train translation model
#
# Usage:
#   ./scripts/train_pipeline.sh hi          # Train Hindi model
#   ./scripts/train_pipeline.sh ta          # Train Tamil model
#   ./scripts/train_pipeline.sh hi --force  # Force re-download
#
# Supported Languages:
#   as - Assamese    bn - Bengali     gu - Gujarati
#   hi - Hindi       kn - Kannada     ml - Malayalam
#   mr - Marathi     or - Odia        pa - Punjabi
#   ta - Tamil       te - Telugu
#

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default values
LANG="${1:-hi}"
FORCE="${2:-}"
DATA_DIR="data/raw"
MODEL_DIR="models/translation"
CONFIG="base"

# Validate language
VALID_LANGS="as bn gu hi kn ml mr or pa ta te"
if [[ ! " $VALID_LANGS " =~ " $LANG " ]]; then
    echo -e "${RED}Error: Invalid language '$LANG'${NC}"
    echo "Supported languages: $VALID_LANGS"
    exit 1
fi

# Language names
declare -A LANG_NAMES=(
    ["as"]="Assamese"
    ["bn"]="Bengali"
    ["gu"]="Gujarati"
    ["hi"]="Hindi"
    ["kn"]="Kannada"
    ["ml"]="Malayalam"
    ["mr"]="Marathi"
    ["or"]="Odia"
    ["pa"]="Punjabi"
    ["ta"]="Tamil"
    ["te"]="Telugu"
)

echo ""
echo -e "${BLUE}╔══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║           NMT Training Pipeline - Samanantar Dataset         ║${NC}"
echo -e "${BLUE}╚══════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "  Target Language: ${GREEN}${LANG_NAMES[$LANG]} ($LANG)${NC}"
echo -e "  Config: ${CONFIG}"
echo ""

# Helper function
print_step() {
    echo ""
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${YELLOW}Step $1: $2${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
}

print_skip() {
    echo -e "${GREEN}✓ $1 - Skipping${NC}"
}

print_done() {
    echo -e "${GREEN}✓ $1${NC}"
}

# ============================================================================
# Step 1: Download Dataset
# ============================================================================
print_step "1/5" "Download Dataset"

TRAIN_FILE="${DATA_DIR}/train-en-${LANG}.jsonl"

if [[ -f "$TRAIN_FILE" && "$FORCE" != "--force" ]]; then
    print_skip "Dataset already exists: $TRAIN_FILE"
else
    echo "Downloading ${LANG_NAMES[$LANG]} dataset from HuggingFace..."
    python scripts/download_dataset.py --lang "$LANG" ${FORCE:+--force}
    print_done "Dataset downloaded"
fi

# ============================================================================
# Step 2: Create Tokenizer Corpus
# ============================================================================
print_step "2/5" "Create Tokenizer Corpus"

CORPUS_FILE="${DATA_DIR}/spm_corpus_multilang.txt"

if [[ -f "$CORPUS_FILE" && "$FORCE" != "--force" ]]; then
    print_skip "Corpus already exists: $CORPUS_FILE"
else
    echo "Creating tokenizer corpus..."
    python scripts/download_dataset.py --lang "$LANG" --create-corpus
    print_done "Corpus created"
fi

# ============================================================================
# Step 3: Create Validation Split
# ============================================================================
print_step "3/5" "Create Validation Split"

VAL_FILE="${DATA_DIR}/validation-en-${LANG}.jsonl"

if [[ -f "$VAL_FILE" && "$FORCE" != "--force" ]]; then
    print_skip "Validation file already exists: $VAL_FILE"
else
    echo "Creating validation split..."
    python scripts/download_dataset.py --lang "$LANG" --create-val-split
    print_done "Validation split created"
fi

# ============================================================================
# Step 4: Train Tokenizer
# ============================================================================
print_step "4/5" "Train Tokenizer"

TOKENIZER_FILE="${MODEL_DIR}/nmt_spm.model"

if [[ -f "$TOKENIZER_FILE" && "$FORCE" != "--force" ]]; then
    print_skip "Tokenizer already exists: $TOKENIZER_FILE"
    TRAIN_TOKENIZER=""
else
    echo "Tokenizer will be trained during model training..."
    TRAIN_TOKENIZER="--train-tokenizer"
fi

# ============================================================================
# Step 5: Train Model
# ============================================================================
print_step "5/5" "Train Translation Model"

echo ""
echo "Configuration:"
echo "  - Language: ${LANG_NAMES[$LANG]} ($LANG)"
echo "  - Model Config: $CONFIG"
echo "  - Streaming: Yes (memory efficient)"
echo "  - Train Data: $TRAIN_FILE"
echo "  - Val Data: $VAL_FILE"
echo ""

read -p "Start training? [Y/n] " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Nn]$ ]]; then
    echo ""
    echo -e "${GREEN}Starting training...${NC}"
    echo ""
    
    python scripts/train_nmt.py \
        --target-lang "$LANG" \
        --config "$CONFIG" \
        --streaming \
        $TRAIN_TOKENIZER
    
    print_done "Training complete"
else
    echo -e "${YELLOW}Training skipped${NC}"
fi

# ============================================================================
# Summary
# ============================================================================
echo ""
echo -e "${BLUE}╔══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║                         Summary                              ║${NC}"
echo -e "${BLUE}╚══════════════════════════════════════════════════════════════╝${NC}"
echo ""

echo "Data Files:"
echo -e "  Training:   $(ls -lh "$TRAIN_FILE" 2>/dev/null | awk '{print $5, $9}' || echo 'Not found')"
echo -e "  Validation: $(ls -lh "$VAL_FILE" 2>/dev/null | awk '{print $5, $9}' || echo 'Not found')"
echo -e "  Corpus:     $(ls -lh "$CORPUS_FILE" 2>/dev/null | awk '{print $5, $9}' || echo 'Not found')"
echo ""

echo "Model Files:"
echo -e "  Tokenizer:  $(ls -lh "$TOKENIZER_FILE" 2>/dev/null | awk '{print $5, $9}' || echo 'Not found')"
echo ""

if ls "${MODEL_DIR}"/*.pt 1> /dev/null 2>&1; then
    echo "Checkpoints:"
    ls -lh "${MODEL_DIR}"/*.pt | while read line; do
        echo -e "  ${GREEN}$(echo $line | awk '{print $9, "("$5")"}')"
    done
else
    echo -e "  ${YELLOW}No checkpoints found yet${NC}"
fi

echo ""
echo -e "${GREEN}Pipeline complete!${NC}"
echo ""
