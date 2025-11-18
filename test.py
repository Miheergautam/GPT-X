"""
GPT-X Training + Generation Pipeline (FAST BPE Version)
"""

import torch
from config import *
from fast_tokenizer import FastBPETokenizer
from fast_data_loader import DataProcessor
from model2 import GPTLanguageModel
from trainer import Trainer
import os


def main():
    torch.manual_seed(SEED)

    print("="*60)
    print("GPT-X Training Pipeline (FAST Version)")
    print("="*60)

    # ---------------------------------------
    # 1. TOKENIZER
    # ---------------------------------------
    print("\n[Step 1/5] Initializing Fast BPE Tokenizer...")

    tokenizer_path = "tokenizer.json"
    tokenizer = FastBPETokenizer(
        vocab_size=VOCAB_SIZE,
        min_frequency=MIN_FREQUENCY
    )

    if os.path.exists(tokenizer_path):
        print(f"Found existing tokenizer at {tokenizer_path}. Loading...")
        tokenizer.load(tokenizer_path)
    else:
        print("Training tokenizer (super fast)...")
        tokenizer.train(DATA_PATH)
        tokenizer.save(tokenizer_path)

    print(f"Tokenizer ready. Vocab size = {tokenizer.get_vocab_size()}")

    # ---------------------------------------
    # 2. DATA
    # ---------------------------------------
    print("\n[Step 2/5] Loading and Processing Data...")
    data_processor = DataProcessor(DATA_PATH, tokenizer)
    train_data, val_data = data_processor.prepare_data()

    # ---------------------------------------
    # 3. MODEL INIT
    # ---------------------------------------
    print("\n[Step 3/5] Initializing GPT Model...")
    actual_vocab_size = tokenizer.get_vocab_size()
    model = GPTLanguageModel(vocab_size=actual_vocab_size).to(DEVICE)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params/1e6:.2f}M")

    # ---------------------------------------
    # 4. TRAINING
    # ---------------------------------------
    print("\n[Step 4/5] Training Model...")
    trainer = Trainer(model, data_processor)
    trainer.train()

    # trainer.save_model("gpt_model.pth")

    # ---------------------------------------
    # 5. TEXT GENERATION
    # ---------------------------------------
    print("\n[Step 5/5] Generating Text...")
    print("-" * 60)

    # Start with an empty context token
    context = torch.zeros((1, 1), dtype=torch.long, device=DEVICE)

    # Generate tokens
    generated_ids = model.generate(
        idx=context,
        max_new_tokens=2000,        # you can increase this anytime
        temperature=1.0,            # adjustable
        top_k=50,                   # cleaner output
    )[0].tolist()

    # Decode
    generated_text = tokenizer.decode(generated_ids)

    print("\nGenerated Text (preview):")
    print("-" * 60)
    print(generated_text[:500])
    print("...\n")

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        f.write(generated_text)

    print(f"Full generated text saved to: {OUTPUT_PATH}")
    print("\n" + "=" * 60)
    print("Pipeline Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
