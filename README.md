# GPT-X - Complete Pipeline Flowchart

## High-Level Architecture Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                         START PIPELINE                          │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│  STEP 1: DATA LOADING & PREPROCESSING (data_loader.py)         │
│                                                                 │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ 1. Load dataset.txt file                                 │  │
│  │ 2. Clean text (remove non-Hindi characters)             │  │
│  │ 3. Normalize whitespace                                  │  │
│  │ 4. Remove short lines                                    │  │
│  └──────────────────────────────────────────────────────────┘  │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│  STEP 2: TOKENIZATION (tokenizer.py)                           │
│                                                                 │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ BPE Tokenizer Training Process:                          │  │
│  │                                                           │  │
│  │ a) Split text into characters                            │  │
│  │    "hello" → ['h', 'e', 'l', 'l', 'o', '</w>']          │  │
│  │                                                           │  │
│  │ b) Count character pair frequencies                      │  │
│  │    ('l', 'l') appears 50 times                          │  │
│  │    ('e', 'l') appears 30 times                          │  │
│  │                                                           │  │
│  │ c) Merge most frequent pair                              │  │
│  │    ['h', 'e', 'll', 'o', '</w>']                        │  │
│  │                                                           │  │
│  │ d) Repeat until vocabulary size reached                  │  │
│  │    Build ~500 tokens (subword units)                     │  │
│  │                                                           │  │
│  │ e) Create token ↔ ID mappings                           │  │
│  │    token_to_id: {'h': 0, 'e': 1, 'll': 2, ...}         │  │
│  │    id_to_token: {0: 'h', 1: 'e', 2: 'll', ...}         │  │
│  └──────────────────────────────────────────────────────────┘  │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│  STEP 3: ENCODE TEXT TO TOKEN IDS                              │
│                                                                 │
│  Text: "नमस्ते दुनिया"                                         │
│    ↓                                                            │
│  Tokens: ['नम', 'स्', 'ते', ' ', 'दु', 'नि', 'या', '</w>']   │
│    ↓                                                            │
│  Token IDs: [45, 12, 89, 3, 67, 34, 91, 1]                    │
│                                                                 │
│  Create train/val split: 90% train, 10% validation            │
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
│  │   • Token Embedding (vocab_size → 128D)                 │  │
│  │   • Position Embedding (0-127 → 128D)                   │  │
│  │                                                           │  │
│  │ Transformer Blocks (x2):                                 │  │
│  │   ┌────────────────────────────────────────┐            │  │
│  │   │ Block 1:                                │            │  │
│  │   │  → Layer Norm                           │            │  │
│  │   │  → Multi-Head Attention (4 heads)       │            │  │
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
│  │   • Linear Projection (128D → vocab_size)               │  │
│  │                                                           │  │
│  │ Total Parameters: ~0.8M                                  │  │
│  └──────────────────────────────────────────────────────────┘  │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│  STEP 5: TRAINING LOOP (trainer.py)                            │
│                                                                 │
│  For 4000 iterations:                                          │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │                                                           │  │
│  │  1. Get random batch of data (32 sequences)              │  │
│  │     Input X:  [token_ids] shape: (32, 128)              │  │
│  │     Target Y: [token_ids shifted by 1] shape: (32, 128) │  │
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
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ Autoregressive Generation Process:                       │  │
│  │                                                           │  │
│  │ Start: [<start_token>]                                   │  │
│  │   ↓                                                       │  │
│  │ Model predicts next token → "नम" (ID: 45)              │  │
│  │   ↓                                                       │  │
│  │ Context: [<start>, 45]                                   │  │
│  │   ↓                                                       │  │
│  │ Model predicts next token → "स्" (ID: 12)              │  │
│  │   ↓                                                       │  │
│  │ Context: [<start>, 45, 12]                               │  │
│  │   ↓                                                       │  │
│  │ Repeat 10,000 times...                                   │  │
│  │   ↓                                                       │  │
│  │ Final: [<start>, 45, 12, 89, 3, 67, ...]                │  │
│  │   ↓                                                       │  │
│  │ Decode IDs back to text: "नमस्ते दुनिया..."           │  │
│  │   ↓                                                       │  │
│  │ Save to file: generated_output.txt                       │  │
│  └──────────────────────────────────────────────────────────┘  │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                         END PIPELINE                            │
└─────────────────────────────────────────────────────────────────┘
```

---

## Detailed Component Explanations

### 1. **Attention Mechanism (Head class)**

```
Input: "The cat sat"
Token Embeddings: [vec_the, vec_cat, vec_sat]

For each token, compute:
├─ Query (Q): "What am I looking for?"
├─ Key (K): "What do I contain?"
└─ Value (V): "What information do I pass forward?"

Attention Score Matrix:
        the    cat    sat
the    [0.7   0.2   0.1]  ← "the" attends mostly to itself
cat    [0.3   0.5   0.2]  ← "cat" attends to "the" and itself
sat    [0.2   0.4   0.4]  ← "sat" attends to "cat" and itself

Causal Mask (prevents looking at future tokens):
        the    cat    sat
the    [✓     ✗     ✗  ]  ← Can only see "the"
cat    [✓     ✓     ✗  ]  ← Can see "the" and "cat"
sat    [✓     ✓     ✓  ]  ← Can see all previous tokens

Weighted sum of values → Output for each token
```

### 2. **Multi-Head Attention**

```
Instead of 1 attention mechanism, use 4 parallel heads:

Head 1: Learns syntax patterns
Head 2: Learns semantic relationships
Head 3: Learns positional dependencies
Head 4: Learns long-range dependencies

Concatenate all outputs → Project back to embedding dimension
```

### 3. **Feed-Forward Network**

```
Purpose: Process information gathered by attention

Input (128D) → Expand to (512D) → ReLU → Project to (128D)
               ↓
          Allows non-linear transformations
          and feature learning
```

### 4. **Residual Connections**

```
Without residual:  x → Block → output
With residual:     x → Block → output + x

Benefits:
- Helps gradients flow during training
- Allows model to learn identity function
- Stabilizes training of deep networks
```

---

## Data Flow Example

### Complete Forward Pass Example:

```
Input Text: "नमस्ते"

1. Tokenization:
   "नमस्ते" → ['नम', 'स्', 'ते', '</w>'] → [45, 12, 89, 1]

2. Embedding:
   Token IDs [45, 12, 89, 1]
   ↓
   Token Embeddings: (4, 128)  [each ID → 128D vector]
   +
   Position Embeddings: (4, 128)  [positions 0,1,2,3 → 128D vectors]
   =
   Combined: (4, 128)

3. Transformer Block 1:
   (4, 128) → LayerNorm → MultiHeadAttention → Add → (4, 128)
           → LayerNorm → FeedForward → Add → (4, 128)

4. Transformer Block 2:
   (4, 128) → [same process] → (4, 128)

5. Output:
   (4, 128) → LayerNorm → Linear Projection → (4, vocab_size)
   
   For each position, we get probabilities for next token:
   Position 0: P(token | "नम")
   Position 1: P(token | "नम", "स्")
   Position 2: P(token | "नम", "स्", "ते")
   Position 3: P(token | "नम", "स्", "ते", "</w>")

6. Loss Calculation:
   Compare predictions with actual next tokens
   CrossEntropy Loss = -log(P(correct_token))
```

---

## Training vs Generation Mode

### Training Mode:
```
Input:  [45, 12, 89, 1]
Target: [12, 89, 1, 67]  (shifted by 1)

Predict all positions simultaneously
Calculate loss for all predictions
Update weights to minimize loss
```

### Generation Mode:
```
Start: [<start>]
↓
Predict: [45]
↓
Context: [<start>, 45]
↓
Predict: [12]
↓
Context: [<start>, 45, 12]
↓
Continue until max_tokens generated...
```

---

## Key Concepts

### Why BPE instead of Character Tokenization?

**Character Level:**
- Vocabulary: ~50 characters
- Sequence: Very long
- "नमस्ते" = 7 characters

**BPE:**
- Vocabulary: ~500 subword units
- Sequence: Shorter
- "नमस्ते" = 3-4 tokens
- Captures common patterns (like "ते", "स्")
- More efficient for model

### Why Dropout?

Randomly "turn off" neurons during training to:
- Prevent overfitting
- Force model to learn robust features
- Improve generalization

### Why Layer Normalization?

- Stabilizes training
- Speeds up convergence
- Normalizes activations within each layer

---

## File Structure Summary

```
project/
│
├── config.py           # All hyperparameters and settings
├── tokenizer.py        # BPE tokenizer implementation
├── data_loader.py      # Data loading and preprocessing
├── model.py            # GPT architecture (attention, blocks, etc.)
├── trainer.py          # Training loop and evaluation
└── main.py             # Orchestrates entire pipeline
```

---

## How to Run

1. **Update config.py** with your data path
2. **Run main.py**:
   ```python
   python main.py
   ```
3. **Output**: Generated text in `generated_output.txt`

---

## Summary

This GPT implementation:
1. ✅ Uses BPE tokenization (more efficient than character-level)
2. ✅ Modular design (separate files for each component)
3. ✅ Well-commented code
4. ✅ Clear data flow
5. ✅ Follows transformer architecture
6. ✅ Trains on Hindi text
7. ✅ Generates new text autoregressively
