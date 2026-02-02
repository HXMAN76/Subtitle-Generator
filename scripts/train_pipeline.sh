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
#   ./scripts/train_pipeline.sh                    # Interactive mode
#   ./scripts/train_pipeline.sh --lang hi          # Non-interactive (Hindi)
#   ./scripts/train_pipeline.sh --lang ta --yes    # Auto-confirm training
#   ./scripts/train_pipeline.sh --lang hi --force  # Force re-download
#
# Supported Languages:
#   as - Assamese    bn - Bengali     gu - Gujarati
#   hi - Hindi       kn - Kannada     ml - Malayalam
#   mr - Marathi     or - Odia        pa - Punjabi
#   ta - Tamil       te - Telugu
#

set -e  # Exit on error

# Ensure we are in the project root
if [[ ! -f "scripts/train_pipeline.sh" ]]; then
    echo "Error: Please run this script from the project root."
    echo "Usage: ./scripts/train_pipeline.sh"
    exit 1
fi

# Check for python
if ! command -v python &> /dev/null; then
    echo "Error: python could not be found."
    exit 1
fi


# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Default values
DATA_DIR="data/raw"
MODEL_DIR_BASE="models/translation"
CONFIG="base"
LANG=""
FORCE=""
AUTO_YES=""
SKIP_TRAIN=""

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

# Language codes in order
LANG_CODES=("hi" "ta" "te" "bn" "mr" "gu" "kn" "ml" "pa" "or" "as")
VALID_LANGS="as bn gu hi kn ml mr or pa ta te"w

# ============================================================================
# Parse Command Line Arguments
# ============================================================================
while [[ $# -gt 0 ]]; do
    case $1 in
        --lang|-l)
            LANG="$2"
            shift 2
            ;;
        --force|-f)
            FORCE="--force"
            shift
            ;;
        --yes|-y)
            AUTO_YES="true"
            shift
            ;;
        --no|-n)
            SKIP_TRAIN="true"
            shift
            ;;
        --config|-c)
            CONFIG="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --lang, -l LANG    Target language code (hi, ta, te, etc.)"
            echo "  --yes, -y          Auto-confirm training (non-interactive)"
            echo "  --no, -n           Skip training (only download and prepare data)"
            echo "  --force, -f        Force re-download even if files exist"
            echo "  --config, -c CFG   Model config: base, small, debug (default: base)"
            echo "  --help, -h         Show this help message"
            echo ""
            echo "Supported languages:"
            echo "  as - Assamese    bn - Bengali     gu - Gujarati"
            echo "  hi - Hindi       kn - Kannada     ml - Malayalam"
            echo "  mr - Marathi     or - Odia        pa - Punjabi"
            echo "  ta - Tamil       te - Telugu"
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# ============================================================================
# Header
# ============================================================================
echo ""
echo -e "${BLUE}╔══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║           NMT Training Pipeline - Samanantar Dataset         ║${NC}"
echo -e "${BLUE}╚══════════════════════════════════════════════════════════════╝${NC}"
echo ""

# ============================================================================
# Language Selection (Interactive or from CLI)
# ============================================================================
if [[ -z "$LANG" ]]; then
    # Interactive mode
    echo -e "${YELLOW}Select target language for translation (English → ?):${NC}"
    echo ""

    for i in "${!LANG_CODES[@]}"; do
        code="${LANG_CODES[$i]}"
        name="${LANG_NAMES[$code]}"
        num=$((i + 1))
        printf "  ${CYAN}%2d${NC}) %-3s - %s\n" "$num" "$code" "$name"
    done

    echo ""
    read -p "Enter choice [1-11] (default: 1 for Hindi): " choice
    echo ""

    if [[ -z "$choice" ]]; then
        choice=1
    fi

    if ! [[ "$choice" =~ ^[0-9]+$ ]] || [ "$choice" -lt 1 ] || [ "$choice" -gt 11 ]; then
        echo -e "${RED}Error: Invalid choice. Please enter a number between 1 and 11.${NC}"
        exit 1
    fi

    LANG="${LANG_CODES[$((choice - 1))]}"
else
    # Validate CLI language
    if [[ ! " $VALID_LANGS " =~ " $LANG " ]]; then
        echo -e "${RED}Error: Invalid language '$LANG'${NC}"
        echo "Supported: $VALID_LANGS"
        exit 1
    fi
fi

echo -e "${GREEN}Target: ${LANG_NAMES[$LANG]} ($LANG)${NC}"
echo -e "Config: ${CONFIG}"

# Set language-specific model directory
MODEL_DIR="${MODEL_DIR_BASE}/${LANG}"
echo -e "Output: ${MODEL_DIR}"

if [[ -n "$FORCE" ]]; then
    echo -e "Mode: ${YELLOW}Force re-download${NC}"
fi
if [[ -n "$AUTO_YES" ]]; then
    echo -e "Mode: ${YELLOW}Auto-confirm (non-interactive)${NC}"
fi
echo ""

# ============================================================================
# Helper Functions
# ============================================================================
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

if [[ -f "$TRAIN_FILE" && -z "$FORCE" ]]; then
    print_skip "Dataset already exists: $TRAIN_FILE"
else
    echo "Downloading ${LANG_NAMES[$LANG]} dataset from HuggingFace..."
    python scripts/download_dataset.py --lang "$LANG" ${FORCE}
    print_done "Dataset downloaded"
fi

# ============================================================================
# Step 2: Create Tokenizer Corpus
# ============================================================================
print_step "2/5" "Create Tokenizer Corpus"

CORPUS_FILE="${DATA_DIR}/spm_corpus_multilang.txt"

if [[ -f "$CORPUS_FILE" && -z "$FORCE" ]]; then
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

if [[ -f "$VAL_FILE" && -z "$FORCE" ]]; then
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

TOKENIZER_FILE="${MODEL_DIR}/tokenizer.model"

if [[ -f "$TOKENIZER_FILE" && -z "$FORCE" ]]; then
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

# Skip training if --no flag
if [[ -n "$SKIP_TRAIN" ]]; then
    echo -e "${YELLOW}Training skipped (--no flag)${NC}"
else
    # Confirm training (or auto-confirm)
    if [[ -z "$AUTO_YES" ]]; then
        read -p "Start training? [Y/n] " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Nn]$ ]]; then
            echo -e "${YELLOW}Training skipped${NC}"
            SKIP_TRAIN="true"
        fi
    fi

    if [[ -z "$SKIP_TRAIN" ]]; then
        echo ""
        echo -e "${GREEN}Starting training...${NC}"
        echo ""

        python scripts/train_nmt.py \
            --target-lang "$LANG" \
            --config "$CONFIG" \
            --streaming \
            $TRAIN_TOKENIZER

        print_done "Training complete"
    fi
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
