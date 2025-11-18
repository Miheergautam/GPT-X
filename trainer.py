# trainer.py
"""
Improved Trainer with:
- Early Stopping
- Best Model Checkpoint
- Cosine LR Decay
- Gradient Clipping
- Fine-tuning support
"""

import torch
from config import (
    MAX_ITERS, EVAL_INTERVAL, EVAL_ITERS,
    LEARNING_RATE, DEVICE, FINETUNE
)
import os


class Trainer:
    def __init__(self, model, data_processor):
        self.model = model
        self.data_processor = data_processor

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=LEARNING_RATE
        )

        # Cosine LR Scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=MAX_ITERS,
            eta_min=1e-5
        )

        # Early stopping
        self.best_val_loss = float("inf")
        self.ckpt_path = "best_model.pth"

        # Load old checkpoint if fine-tuning
        if FINETUNE:
            self.load_if_exists()

    # -----------------------------------------------------
    #  LOSS ESTIMATION
    # -----------------------------------------------------
    @torch.no_grad()
    def estimate_loss(self):
        out = {}
        self.model.eval()

        for split in ['train', 'val']:
            losses = torch.zeros(EVAL_ITERS)

            for k in range(EVAL_ITERS):
                X, Y = self.data_processor.get_batch(split)
                _, loss = self.model(X, Y)
                losses[k] = loss.item()

            out[split] = losses.mean()

        self.model.train()
        return out

    # -----------------------------------------------------
    #  MAIN TRAIN LOOP
    # -----------------------------------------------------
    def train(self):
        print("\n" + "="*60)
        print("Starting Training")
        print("="*60)
        print(f"Total iterations: {MAX_ITERS}")
        print(f"Evaluation interval: {EVAL_INTERVAL}")
        print(f"Learning rate: {LEARNING_RATE}")
        print(f"Device: {DEVICE}")
        print("="*60 + "\n")

        for it in range(MAX_ITERS):

            # Periodic evaluation
            if it % EVAL_INTERVAL == 0 or it == MAX_ITERS - 1:
                losses = self.estimate_loss()
                print(
                    f"Step {it:4d}: "
                    f"train loss {losses['train']:.4f}, "
                    f"val loss {losses['val']:.4f}"
                )

                # Early stopping + checkpointing
                if losses["val"] < self.best_val_loss:
                    self.best_val_loss = losses["val"]
                    self.save_checkpoint()
                else:
                    print("⚠ Validation loss did not improve — possible overfitting.")

            # Get training batch
            xb, yb = self.data_processor.get_batch("train")

            # Forward pass
            logits, loss = self.model(xb, yb)

            # Backward pass
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()

            # Gradient clipping (stabilizes training)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

            # Step optimizer + LR scheduler
            self.optimizer.step()
            self.scheduler.step()

        print("\n" + "="*60)
        print("Training Complete!")
        print("="*60 + "\n")

    # -----------------------------------------------------
    # CHECKPOINTING
    # -----------------------------------------------------
    def save_checkpoint(self):
        torch.save({
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "best_val_loss": self.best_val_loss,
        }, self.ckpt_path)
        print(f"💾 Saved BEST checkpoint (val={self.best_val_loss:.4f})")

    def load_if_exists(self):
        if os.path.exists(self.ckpt_path):
            print("🔄 Fine-tuning: Loading existing checkpoint...")
            ckpt = torch.load(self.ckpt_path, map_location=DEVICE)
            self.model.load_state_dict(ckpt["model"])
            self.optimizer.load_state_dict(ckpt["optimizer"])
            self.best_val_loss = ckpt["best_val_loss"]
            print(f"Loaded checkpoint (best val loss = {self.best_val_loss:.4f})")
        else:
            print("⚠ No previous checkpoint found — training from scratch.")

    # -----------------------------------------------------
    # MANUAL SAVE/LOAD (optional usage)
    # -----------------------------------------------------
    def save_model(self, path):
        torch.save(self.model.state_dict(), path)
        print(f"Model saved to {path}")

    def load_model(self, path):
        self.model.load_state_dict(torch.load(path))
        print(f"Model loaded from {path}")
