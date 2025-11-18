# GPT-X - Complete Pipeline Flowchart

## High-Level Architecture Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                         START PIPELINE                          │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│  STEP 1: DATA LOADING & PREPROCESSING                           │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│  STEP 2: TOKENIZATION                                           │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│  STEP 3: ENCODE TEXT TO TOKEN IDS                               │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│  STEP 4: MODEL INITIALIZATION (model.py)                       │
│                                                                 │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ GPT Architecture Components:                             │  │
│  │                                                           │  │
│  │ Input Layer:                                             │  │
│  │   • Token Embedding (vocab_size → 432D)                 │  │
│  │   • Position Embedding (0-255 → 432D)                   │  │
│  │                                                           │  │
│  │ Transformer Blocks (x6):                                 │  │
│  │   ┌────────────────────────────────────────┐            │  │
│  │   │ Block 1:                                │            │  │
│  │   │  → Layer Norm                           │            │  │
│  │   │  → Multi-Head Attention (6 heads)       │            │  │
│  │   │  → Residual Connection                  │            │  │
│  │   │  → Layer Norm                           │            │  │
│  │   │  → Feed Forward Network                 │            │  │
│  │   │  → Residual Connection                  │            │  │
│  │   └────────────────────────────────────────┘            │  │
│  │   ┌────────────────────────────────────────┐            │  │
│  │   │ Block 2: (same structure)               │            │  │
│  │   └────────────────────────────────────────┘            │  │
│  │                                                           │  │
│  │ Output Layer:                                            │  │
│  │   • Final Layer Norm                                     │  │
│  │   • Linear Projection (432D → vocab_size)               │  │
│  │                                                           │  │
│  │ Total Parameters: ~20M                                  │  │
│  └──────────────────────────────────────────────────────────┘  │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│  STEP 5: TRAINING LOOP                                          │
│                                                                 │
│  For 10000 iterations:                                          │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │                                                           │  │
│  │  1. Get random batch of data (64 sequences)              │  │
│  │     Input X:  [token_ids] shape: (64, 432D)              │  │
│  │     Target Y: [token_ids shifted by 1] shape: (64, 432D) │  │
│  │                                                           │  │
│  │  2. Forward Pass                                         │  │
│  │     X → Model → Predictions (logits)                     │  │
│  │                                                           │  │
│  │  3. Calculate Loss                                       │  │
│  │     Compare predictions with targets                     │  │
│  │     CrossEntropy Loss = measure of error                 │  │
│  │                                                           │  │
│  │  4. Backward Pass                                        │  │
│  │     Compute gradients (how to adjust weights)            │  │
│  │                                                           │  │
│  │  5. Update Parameters                                    │  │
│  │     Adjust model weights using AdamW optimizer           │  │
│  │                                                           │  │
│  │  Every 500 steps:                                        │  │
│  │     → Evaluate on validation set                         │  │
│  │     → Print train/val loss                               │  │
│  │                                                           │  │
│  └──────────────────────────────────────────────────────────┘  │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│  STEP 6: TEXT GENERATION                                       │
│                                                                 │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                         END PIPELINE                            │
└─────────────────────────────────────────────────────────────────┘
```

---

This GPT implementation:
1. ✅ Uses BPE tokenization 
4. ✅ Clear data flow
5. ✅ Follows transformer architecture
6. ✅ Trains on Hindi text
7. ✅ Generates new text autoregressively
