# trainer.py
"""
Training utilities for the GPT model.
Handles the training loop, loss estimation, and model evaluation.
"""

import torch
from config import MAX_ITERS, EVAL_INTERVAL, EVAL_ITERS, LEARNING_RATE, DEVICE


class Trainer:
    """
    Trainer class to handle model training and evaluation.
    """
    
    def __init__(self, model, data_processor):
        """
        Initialize trainer.
        
        Args:
            model: GPT model to train
            data_processor: DataProcessor instance for batch generation
        """
        self.model = model
        self.data_processor = data_processor
        
        # Create optimizer (AdamW is Adam with weight decay)
        self.optimizer = torch.optim.AdamW(
            model.parameters(), 
            lr=LEARNING_RATE,
            weight_decay=0.1,      # ← EDIT #1
            betas=(0.9, 0.95)      # (recommended for transformers)
        )
        # LR Schedule parameters
        self.warmup_iters = 1000           # EDIT #2
        self.min_lr = LEARNING_RATE / 10   # EDIT #2
        self.max_lr = LEARNING_RATE        # EDIT #2
        self.total_iters = MAX_ITERS       # EDIT #2
    
    @torch.no_grad()
    def estimate_loss(self):
        """
        Estimate average loss on train and validation sets.
        
        Why estimate? Training loss is noisy, so we average over
        multiple batches to get a more stable measure of performance.
        
        Returns:
            Dictionary with 'train' and 'val' loss values
        """
        out = {}
        
        # Set model to evaluation mode (disables dropout, etc.)
        self.model.eval()
        
        for split in ['train', 'val']:
            losses = torch.zeros(EVAL_ITERS)
            
            # Average loss over EVAL_ITERS batches
            for k in range(EVAL_ITERS):
                X, Y = self.data_processor.get_batch(split)
                logits, loss = self.model(X, Y)
                losses[k] = loss.item()
            
            # Calculate mean loss
            out[split] = losses.mean()
        
        # Set model back to training mode
        self.model.train()
        
        return out
        
    def get_lr(self, it):
        # EDIT #3
        # Linear warmup
        if it < self.warmup_iters:
            return self.max_lr * (it / self.warmup_iters)
    
        # Cosine decay
        progress = (it - self.warmup_iters) / max(1, self.total_iters - self.warmup_iters)
        cosine_decay = 0.5 * (1 + torch.cos(torch.pi * progress))
        return self.min_lr + (self.max_lr - self.min_lr) * cosine_decay
    
    def train(self):
        """
        Main training loop.
        
        Training process:
        1. Sample a batch of data
        2. Forward pass: compute predictions and loss
        3. Backward pass: compute gradients
        4. Update parameters using optimizer
        5. Repeat for MAX_ITERS iterations
        
        Periodically evaluates on both train and validation sets.
        """
        print("\n" + "="*60)
        print("Starting Training")
        print("="*60)
        print(f"Total iterations: {MAX_ITERS}")
        print(f"Evaluation interval: {EVAL_INTERVAL}")
        print(f"Learning rate: {LEARNING_RATE}")
        print(f"Device: {DEVICE}")
        print("="*60 + "\n")
        
        for iter in range(MAX_ITERS):
            # Evaluate loss periodically
            if iter % EVAL_INTERVAL == 0 or iter == MAX_ITERS - 1:
                losses = self.estimate_loss()
                print(
                    f"Step {iter:4d}: "
                    f"train loss {losses['train']:.4f}, "
                    f"val loss {losses['val']:.4f}"
                )
            
            # Sample a batch of training data
            xb, yb = self.data_processor.get_batch('train')
            
            # Forward pass: compute predictions and loss
            logits, loss = self.model(xb, yb)
            
            # Zero out gradients from previous iteration
            self.optimizer.zero_grad(set_to_none=True)
            
            # Backward pass: compute gradients
            loss.backward()
            
            # Update learning rate
            lr = self.get_lr(iter)
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = lr
            
            # Update model parameters
            self.optimizer.step()
        
        print("\n" + "="*60)
        print("Training Complete!")
        print("="*60 + "\n")
    
    def save_model(self, path):
        """
        Save model weights to file.
        
        Args:
            path: Path to save model checkpoint
        """
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)
        print(f"Model saved to {path}")
    
    def load_model(self, path):
        """
        Load model weights from file.
        
        Args:
            path: Path to model checkpoint
        """
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"Model loaded from {path}")