"""
Main execution script for training and generating text with GPT.

This script orchestrates the entire pipeline:
1. Load and preprocess data
2. Train BPE tokenizer
3. Initialize GPT model
4. Train the model
5. Generate text
"""

import torch
from config import *
from tokenizer import BPETokenizer
from data_loader import DataProcessor
from model import GPTLanguageModel
from trainer import Trainer


def main():
    """
    Main execution function.
    """
    # Set random seed for reproducibility
    torch.manual_seed(SEED)
    
    print("="*60)
    print("GPT-X Training Pipeline")
    print("="*60)
    
    # Step 1: Initialize tokenizer
    print("\n[Step 1/5] Initializing BPE Tokenizer...")
    tokenizer = BPETokenizer(vocab_size=VOCAB_SIZE, min_frequency=MIN_FREQUENCY)
    
    # Step 2: Prepare data
    print("\n[Step 2/5] Loading and Processing Data...")
    data_processor = DataProcessor(DATA_PATH, tokenizer)
    train_data, val_data = data_processor.prepare_data()
    
    # Get actual vocabulary size after tokenization
    actual_vocab_size = tokenizer.get_vocab_size()
    print(f"\nActual vocabulary size: {actual_vocab_size}")
    
    # Step 3: Initialize model
    print("\n[Step 3/5] Initializing GPT Model...")
    model = GPTLanguageModel(vocab_size=actual_vocab_size)
    model = model.to(DEVICE)
    
    # Print model size
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params/1e6:.2f}M")
    
    # # Step 4: Train model
    # print("\n[Step 4/5] Training Model...")
    # trainer = Trainer(model, data_processor)
    # trainer.train()
    
    # Optionally save the model
    # trainer.save_model('gpt_model.pth')
    
    # # Step 5: Generate text
    # print("\n[Step 5/5] Generating Text...")
    # print("-"*60)
    
    # # Start with empty context
    # context = torch.zeros((1, 1), dtype=torch.long, device=DEVICE)
    
    # # Generate tokens
    # generated_ids = model.generate(context, max_new_tokens=10000)[0].tolist()
    
    # # Decode to text
    # generated_text = tokenizer.decode(generated_ids)
    
    # # Print to console (first 500 characters)
    # print("\nGenerated Text (preview):")
    # print("-"*60)
    # print(generated_text[:500])
    # print("...\n")
    
    # # Save full generated text to file
    # with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
    #     f.write(generated_text)
    
    # print(f"Full generated text saved to: {OUTPUT_PATH}")
    # print("\n" + "="*60)
    # print("Pipeline Complete!")
    # print("="*60)


if __name__ == "__main__":
    main()