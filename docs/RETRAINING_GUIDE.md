# NMT Re-training Recommendations

Actionable plan for improving underperforming models based on evaluation results.

---

## 🚨 Critical: Data Integrity Check

### Issue: Bengali & Marathi (BLEU = 100%)

**Steps:**
1. Verify test set doesn't overlap with training data
2. Check if model is identity-mapping (copying source to target)
3. Manually inspect sample translations

**Diagnostic Script:**
```bash
# Check for duplicate sentences between train and test
python scripts/check_data_leakage.py \
    --train data/raw/train-en-bn.json \
    --test data/raw/test-en-bn.json

# Inspect sample translations
python -c "
import json
with open('results/nmt_evaluation_report.json/translations-en-bn.json') as f:
    data = json.load(f)
    for i, item in enumerate(data['translations'][:5]):
        print(f'Example {i+1}:')
        print(f'  Source: {item[\"source\"]}')
        print(f'  Reference: {item[\"reference\"]}')
        print(f'  Hypothesis: {item[\"hypothesis\"]}')
        print()
"
```

**Resolution:**
- If data leakage: Regenerate test splits with different seed/samples
- If identity mapping: Check model architecture and training logs

---

## 🔧 Fix METEOR Computation

### Issue: METEOR = 0.0 for all languages

**Root Cause:** Missing NLTK data or tokenization issues

**Fix:**
```bash
# Install required NLTK data
pip install nltk
python -c "
import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')
"

# Re-run evaluation for Hindi (test)
python scripts/evaluate_nmt.py \
    --checkpoint models/translation/hi/best.pt \
    --target-lang hi \
    --samples 3
```

---

## 🎯 Dravidian Languages Retraining

### Languages: Kannada (12.14), Malayalam (15.97), Tamil (15.44)

**Problem:** All Dravidian languages severely underperform despite large datasets

**Root Causes:**
1. **Tokenization mismatch**: BPE trained on all languages may not capture Dravidian morphology
2. **Script complexity**: Different phonetic structure from Indo-Aryan
3. **Training convergence**: May need different hyperparameters

### Retraining Strategy

#### Option 1: Separate Tokenizer (Recommended)

**Create Dravidian-specific tokenizer:**
```bash
# Extract Dravidian text corpus
python scripts/create_dravidian_corpus.py \
    --langs kn ml ta \
    --output data/raw/spm_corpus_dravidian.txt

# Train specialized tokenizer
python src/nmt/tokenizer.py \
    --corpus data/raw/spm_corpus_dravidian.txt \
    --output models/translation/nmt_spm_dravidian \
    --vocab-size 32000 \
    --langs "<en>" "<kn>" "<ml>" "<ta>"

# Retrain models with Dravidian tokenizer
for lang in kn ml ta; do
    python scripts/train_nmt.py \
        --lang $lang \
        --tokenizer models/translation/nmt_spm_dravidian.model \
        --epochs 30 \
        --batch-size 64 \
        --lr 0.0001
done
```

#### Option 2: Extended Training

**Continue training existing models:**
```bash
# Resume from checkpoint with lower learning rate
python scripts/train_nmt.py \
    --lang kn \
    --resume models/translation/kn/best.pt \
    --epochs 10 \
    --lr 0.00005 \
    --warmup-steps 0
```

#### Option 3: Hyperparameter Tuning

**Try different configurations:**
```python
# configs/dravidian_config.py
{
    "model": {
        "d_model": 512,
        "n_heads": 8,
        "n_encoder_layers": 8,  # Increase from 6
        "n_decoder_layers": 8,  # Increase from 6
        "d_ff": 2048,
        "dropout": 0.1
    },
    "training": {
        "batch_size": 32,  # Reduce for stability
        "lr": 0.0001,
        "warmup_steps": 4000,
        "max_epochs": 40,  # Train longer
        "gradient_clip": 1.0
    }
}
```

---

## 📊 Other Language Improvements

### Assamese (14.37 BLEU) - Low Resource

**Strategy: Data Augmentation**

```bash
# Back-translation from Bengali (related language)
python scripts/back_translate.py \
    --source-lang as \
    --pivot-lang bn \
    --target-lang en \
    --samples 50000

# Add synthetic data to training
python scripts/augment_dataset.py \
    --original data/raw/train-en-as.json \
    --synthetic data/augmented/backtrans-en-as.json \
    --output data/raw/train-en-as-augmented.json
```

### Odia (63.89 BLEU) - Verify Legitimacy

**Diagnostic:**
```bash
# Manual inspection
python scripts/inspect_translations.py \
    --file results/nmt_evaluation_report.json/translations-en-or.json \
    --samples 20 \
    --random-seed 42

# Compare with human evaluation
# If legitimate, document as high performer
```

---

## 📋 Retraining Checklist

### Before Retraining

- [ ] Back up current models to `models/translation/{lang}/backup/`
- [ ] Verify data quality (no duplicates, proper format)
- [ ] Check GPU memory availability
- [ ] Set up experiment tracking (TensorBoard logs)

### During Retraining

- [ ] Monitor training loss convergence
- [ ] Track validation BLEU every epoch
- [ ] Watch for gradient issues (NaN/Inf)
- [ ] Save checkpoints every 5 epochs

### After Retraining

- [ ] Evaluate on test set
- [ ] Compare with baseline (old model)
- [ ] Manual quality check (at least 20 samples)
- [ ] Update model registry with new scores

---

## 🎯 Priority Order

1. **Immediate (Do First)**
   - Fix METEOR computation
   - Investigate Bengali/Marathi data leakage
   - Verify test set integrity

2. **High Priority (This Week)**
   - Retrain Dravidian models (Kannada, Malayalam, Tamil)
   - Create specialized Dravidian tokenizer
   - Re-evaluate all models with fixed METEOR

3. **Medium Priority (Next Week)**
   - Data augmentation for Assamese
   - Verify Odia results
   - Fine-tune Punjabi for better performance

4. **Low Priority (Future)**
   - Implement COMET metric
   - Add chrF and TER metrics
   - Experiment with deeper architectures

---

## 📁 Required Scripts to Create

1. **`scripts/check_data_leakage.py`** - Detect test/train overlap
2. **`scripts/create_dravidian_corpus.py`** - Extract Dravidian text
3. **`scripts/inspect_translations.py`** - Manual quality inspection tool
4. **`scripts/back_translate.py`** - Data augmentation via back-translation
5. **`configs/dravidian_config.py`** - Dravidian-specific hyperparameters

---

## Expected Improvements

After implementing these recommendations:

| Language | Current BLEU | Target BLEU | Strategy |
|----------|--------------|-------------|----------|
| Kannada | 12.14 | **25-30** | Specialized tokenizer + extended training |
| Malayalam | 15.97 | **25-30** | Specialized tokenizer + extended training |
| Tamil | 15.44 | **25-30** | Specialized tokenizer + extended training |
| Assamese | 14.37 | **20-25** | Data augmentation |
| Punjabi | 34.93 | **38-42** | Fine-tuning |

**Timeline:** 2-3 weeks for complete retraining cycle
