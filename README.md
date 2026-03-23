# DeepLens Submission


## Task 1: Multi-Class Strong Lensing Classification

**Objective:** Classify strong lensing images into three substructure classes.

**Classes:**
- `no` — No substructure
- `sphere` — Spherical halo substructure  
- `vort` — Vortex substructure

### Method: Vision Transformer (ViT-B/16)

```
    Image (224×224)
      ↓
    Patch Embedding (14×14 = 196 patches)
      ↓
    Positional Encoding + Class Token
      ↓
    12 Transformer Blocks (Multi-head Self-Attention)
      ↓
    Classification Head (3 classes)
      ↓
    Output: Class Logits
```

**Key Strategy:**

| Phase | Approach | Epochs |
|-------|----------|--------|
| Phase 1 | Freeze early layers, train task-specific heads | 1–7 |
| Phase 2 | Unfreeze all with layer-wise learning rate decay | 8+ |

**Evaluation:** ROC-AUC (one-vs-rest) on validation set (90:10 split)

**Weights:** `task-1-weights.pth`

---

## Task 5: Binary Lens Classification

**Objective:** Identify strong gravitational lenses from astronomical images.

**Classes:**
- `lens` — Strong lens detected
- `non-lens` — No lens

### Method: ResNet-18

```
    Image (3, 64, 64)
      ↓
    Conv Block 1 (32 filters) → ReLU → MaxPool
      ↓
    Conv Block 2 (64 filters) → ReLU → MaxPool
      ↓
    Conv Block 3 (128 filters) → ReLU → MaxPool
      ↓
    Global Avg Pool
      ↓
    FC Layers (256 → 128 → 2)
      ↓
    Output: Binary Logits
```

**Class Imbalance Mitigation:**

| Technique | Purpose |
|-----------|---------|
| **Weighted Loss** | Higher penalty for minority class (lenses) |
| **Balanced Sampling** | Upsample lenses during batch creation |
| **Threshold Tuning** | Optimize decision boundary using ROC curve |
| **ROC-AUC Metric** | Evaluation robust to class imbalance |

**Hyperparameters:**
- Batch size: 32
- Learning rate: 0.001 (Adam optimizer)
- Early stopping: patience = 10 epochs
- L2 regularization: 1e-4

**Weights:** `task-5-weights.pth`

---

## Dataset Structure

```
dataset/                          lens-dataset/
├── train/                        ├── train_lenses/
│   ├── no/                       ├── train_nonlenses/
│   ├── sphere/                   ├── test_lenses/
│   └── vort/                     └── test_nonlenses/
└── val/
    ├── no/
    ├── sphere/
    └── vort/
```

---

## Evaluation Metrics

Both tasks use **ROC-AUC** for robust multi-class and imbalanced classification evaluation.

**Task 1:** One-vs-Rest ROC curves (3 binary classifiers)  
**Task 5:** Single ROC curve + threshold optimization

---

## Files

- `task-1.ipynb` — Multi-class ViT implementation
- `task-1-weights.pth` — Pre-trained ViT model
- `task-5.ipynb` — Binary lens classifier
- `task-5-weights.pth` — Pre-trained CNN model
