# Multi-Language Evaluation Guide

This guide explains how to evaluate NMT models across all 10 supported Indic languages.

---

## Prerequisites

On your **server** where Samanantar data is stored:

1. ✅ Training data in JSONL format: `data/raw/train-en-{lang}.jsonl`
2. ✅ Trained models: `models/translation/{lang}/best.pt`
3. ✅ Tokenizer: `models/translation/nmt_spm.model`

---

## Step 1: Create Test Splits (Server)

First, create test and validation splits from your Samanantar training data:

```bash
# On server: Create test splits for all available languages
python scripts/create_test_splits.py

# Or for specific languages only
python scripts/create_test_splits.py --lang hi ta bn

# Custom sizes (default: 1000 test, 2000 validation)
python scripts/create_test_splits.py --test-size 2000 --val-size 3000
```

This will create:
- `data/raw/test-en-{lang}.json` (1,000 samples per language)
- `data/raw/validation-en-{lang}.json` (2,000 samples per language)
- `data/raw/train-en-{lang}.json` (converted from JSONL)

---

## Step 2: Run Multi-Language Evaluation (Server)

### Evaluate All Languages

```bash
# Basic evaluation (BLEU + METEOR)
python scripts/evaluate_nmt.py --all-languages

# With COMET scoring (slower, requires GPU)
python scripts/evaluate_nmt.py --all-languages --comet

# Save detailed report
python scripts/evaluate_nmt.py --all-languages \
    --output results/nmt_evaluation_report.json
```

### Evaluate Single Language

```bash
# Evaluate Hindi with samples
python scripts/evaluate_nmt.py \
    --checkpoint models/translation/hi/best.pt \
    --target-lang hi \
    --samples 5

# Save translations
python scripts/evaluate_nmt.py \
    --checkpoint models/translation/hi/best.pt \
    --target-lang hi \
    --output results/translations-hi.json
```

---

## Understanding the Output

### Console Output

```
======================================================================
MULTI-LANGUAGE EVALUATION SUMMARY
======================================================================
Language        Code   BLEU       METEOR     COMET     
----------------------------------------------------------------------
Assamese        as     28.45      0.5234     0.7456    
Bengali         bn     32.12      0.5678     0.7823    
Gujarati        gu     30.89      0.5456     0.7645    
Hindi           hi     34.23      0.5789     0.7912    
...
----------------------------------------------------------------------
AVERAGE                31.24      0.5538     0.7721    
======================================================================
```

### Metrics Explanation

| Metric | Range | Description |
|--------|-------|-------------|
| **BLEU** | 0-100 | N-gram precision (higher = better) |
| **METEOR** | 0-1 | Synonym/paraphrase matching (higher = better) |
| **COMET** | 0-1 | Neural metric, best human correlation (higher = better) |

### JSON Report

```json
{
  "timestamp": "2026-01-20T18:00:00",
  "languages": 10,
  "results": {
    "hi": {
      "language_name": "Hindi",
      "bleu": 34.23,
      "meteor": 0.5789,
      "comet": 0.7912
    }
  },
  "averages": {
    "bleu": 31.24,
    "meteor": 0.5538,
    "comet": 0.7721
  }
}
```

---

## Quick Reference

### File Locations

```
data/raw/
├── train-en-{lang}.jsonl    # Original Samanantar data (server)
├── train-en-{lang}.json     # Converted for training
├── validation-en-{lang}.json # Validation split
└── test-en-{lang}.json      # Test split for evaluation

models/translation/
├── {lang}/
│   └── best.pt              # Trained model checkpoint
└── nmt_spm.model            # Shared tokenizer
```

### Supported Languages

```
as - Assamese    bn - Bengali     gu - Gujarati
hi - Hindi       kn - Kannada     ml - Malayalam
mr - Marathi     or - Odia        pa - Punjabi
ta - Tamil       te - Telugu
```

---

## Troubleshooting

### Issue: "No test data found"

**Solution:** Run `create_test_splits.py` first to create test splits from training data.

### Issue: "No language models found"

**Solution:** Check that model checkpoints exist in `models/translation/{lang}/best.pt`

### Issue: Missing JSONL files

**Solution:** Ensure you're on the server with Samanantar data, or download using:
```bash
python scripts/download_dataset.py --all-langs
```

---

## Expected Performance

Based on Samanantar dataset quality and model size:

- **High-resource languages** (hi, bn, ta): BLEU 30-35
- **Medium-resource languages** (gu, kn, ml, mr, te): BLEU 25-30
- **Low-resource languages** (as, or, pa): BLEU 20-25

> [!NOTE]
> COMET scores correlate better with human judgment than BLEU scores.
