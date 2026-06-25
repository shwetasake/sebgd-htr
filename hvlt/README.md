# HVLT — Hierarchical Vision-Language Transformer
## Handwritten Text Recognition on ICDAR Word-Level English Dataset

Implementation of:
> "Hierarchical Vision-Language Transformers for Handwritten Text: Bridging Sequence Generalization and Token Memorization"
> Siddiqui, Thakre, Ajankar — VJTI Mumbai

---

## Project Structure

```
hvlt/
├── data/
│   └── dataset.py          # ICDAR dataset loader, vocab, tokenizer
├── models/
│   ├── tps_stn.py          # Stage II: TPS-STN geometric rectification (K=16)
│   ├── encoder.py          # Stage III: Gated CNN + Swin Transformer
│   │                       # Stage IV: Artifact Classification Gate (ACG)
│   ├── decoder.py          # Stage V: Positional Bridge + RoBERTa decoder
│   └── hvlt.py             # Full pipeline + HVLTLoss
├── utils/
│   └── metrics.py          # CAR / WAR (Equations 1 & 2 from paper)
├── train.py                # Main training script
├── predict.py              # Inference on labeled / unlabeled images
└── requirements.txt
```

---

## Setup

```bash
pip install -r requirements.txt
```

Pretrained weights are automatically downloaded:
- `swin_small_patch4_window7_224` via `timm` (~48MB)
- `roberta-base` via HuggingFace `transformers` (~480MB)

---

## Dataset Format

`train.txt` format (one entry per line):
```
image/74_1.jpg hello
image/74_2.jpg world
```

---

## About the Unlabeled Test Set

> **The test set without ground truth should be SKIPPED for evaluation.**

This is the correct approach. You cannot compute CAR/WAR without labels.
Instead:
1. We split `train.txt` into **85% train / 15% val** for training and evaluation.
2. Use `predict.py` to generate predictions on the test folder (for submission or inspection only).

---

## Training

### Single run (quick test):
```bash
cd hvlt
python train.py \
  --train_txt "/home/mca/Shweta/handwritten text detection/dataset/Word_Level_English_Training_Set/Word_Level_Training_Set/train.txt" \
  --root_dir  "/home/mca/Shweta/handwritten text detection/dataset/Word_Level_English_Training_Set/Word_Level_Training_Set" \
  --batch_size 32 \
  --max_epochs 10 \
  --seeds 42 \
  --output_dir outputs/
```

### Full 3-seed run (as in paper):
```bash
python train.py \
  --train_txt "..." \
  --root_dir  "..." \
  --batch_size 32 \
  --max_epochs 10 \
  --seeds 42 123 7 \
  --use_amp \
  --output_dir outputs/
```

### Key training flags:
| Flag | Default | Notes |
|------|---------|-------|
| `--lr` | 5e-5 | Constant Adam LR — no scheduler (by design) |
| `--patience` | 3 | WAR-based early stopping |
| `--acg_lambda` | 0.1 | ACG auxiliary loss weight |
| `--num_fiducial` | 16 | TPS control points (K) |
| `--use_amp` | False | Mixed precision (recommended if GPU) |

---

## Key Training Insight from Paper

The paper documents **Sequence Memorisation Collapse**:

| Epoch | Train CAR | Val CAR | Val WAR |
|-------|-----------|---------|---------|
| 1     | 48.20%    | 48.20%  | 35.40%  |
| 2     | 96.16%    | 96.16%  | 81.71%  |
| **3** | **99.61%**| **99.61%**| **95.20%** ← BEST |
| 4     | 99.95%    | 99.95%  | 32.28%  |
| 6–30  | ~100%     | ~100%   | 0.00%   |

**Early stopping on VAL WAR (not loss) is critical.**
The code handles this automatically.

---

## Inference on Unlabeled Test Images

```bash
# Predict on entire test folder
python predict.py \
  --checkpoint outputs/seed_42/best_model.pt \
  --image_dir  "/home/mca/Shweta/handwritten text detection/dataset/test_images/" \
  --output_csv predictions.csv

# Predict on a single image
python predict.py \
  --checkpoint outputs/seed_42/best_model.pt \
  --image      "/path/to/image.jpg"
```

---

## Architecture Summary (Table IV from paper)

| Hyperparameter | Value |
|---|---|
| TPS fiducial points K | 16 |
| Swin layer distribution | [2, 2, 18, 2] |
| Swin attention heads | [4, 8, 16, 32] |
| Language decoder | RoBERTa-base (L=12, H=768) |
| Cross-attention heads | 12 |
| Sequence dimension d_model | 768 |
| Total parameters | ~142M |
| Output classes | 99 |
| Max sequence length | 25 tokens |
| Optimizer / LR | Adam / 5×10⁻⁵ |
| ACG dropout | 0.3 |
| ACG auxiliary loss weight λ | 0.1 |

---

## Results on GNHK (from paper)

| Metric | Score |
|--------|-------|
| CAR | 99.14% ± 0.12% |
| WAR | 94.74% ± 0.15% |

---

## Paper-Specific Notes

1. **No LR schedule**: The constant LR is intentional — LR scheduling masks cross-modal attention dynamics that reveal the collapse.
2. **ACG labels**: For ICDAR English word dataset, all ACG labels are 0 (no math/symbol artifacts). The ACG still contributes via feature routing.
3. **3 seeds**: Paper reports mean ± std over seeds [42, 123, 7] — this is replicated here.
4. **`swin_small`** matches the paper's C=96, depths=[2,2,18,2] configuration exactly.
