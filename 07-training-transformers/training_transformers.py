"""
Training Transformers
Complete implementation of training strategies, optimization techniques, and distributed training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from torch.optim.lr_scheduler import LambdaLR
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import autocast, GradScaler
import numpy as np
import math
import time
import os
import json
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from collections import defaultdict
import matplotlib.pyplot as plt
from tqdm import tqdm
import wandb


@dataclass
class TrainingConfig:
    """Configuration for transformer training."""
    # Model
    model_dim: int = 768
    num_heads: int = 12
    num_layers: int = 12
    vocab_size: int = 50000
    max_seq_length: int = 512
    dropout: float = 0.1
    
    # Training
    batch_size: int = 32
    gradient_accumulation_steps: int = 1
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    num_epochs: int = 10
    warmup_steps: int = 1000
    
    # Optimization
    use_mixed_precision: bool = True
    gradient_checkpointing: bool = False
    
    # Distributed
    distributed: bool = False
    world_size: int = 1
    rank: int = 0
    
    # Logging
    log_interval: int = 10
    eval_interval: int = 100
    save_interval: int = 1000
    
    # Paths
    output_dir: str = "./output"
    checkpoint_dir: str = "./checkpoints"


class TransformerDataset(Dataset):
    """Dataset for transformer training."""
    
    def __init__(self, data: List[Dict], tokenizer, max_length: int = 512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Tokenize
        encoding = self.tokenizer(
            item['text'],
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': torch.tensor(item.get('label', 0))
        }


class DynamicBatchSampler:
    """Sampler that groups sequences by length to minimize padding."""
    
    def __init__(self, dataset, batch_size: int, shuffle: bool = True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        
        # Group by length
        self.length_groups = defaultdict(list)
        for idx in range(len(dataset)):
            length = len(dataset[idx]['input_ids'])
            self.length_groups[length].append(idx)
            
    def __iter__(self):
        # Create batches
        batches = []
        
        for length, indices in self.length_groups.items():
            if self.shuffle:
                np.random.shuffle(indices)
                
            # Create batches from this length group
            for i in range(0, len(indices), self.batch_size):
                batch = indices[i:i + self.batch_size]
                if len(batch) == self.batch_size:  # Skip incomplete batches
                    batches.append(batch)
        
        if self.shuffle:
            np.random.shuffle(batches)
            
        return iter(batches)
    
    def __len__(self):
        return sum(len(indices) // self.batch_size 
                  for indices in self.length_groups.values())


# Learning Rate Schedulers
def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps):
    """Linear warmup and linear decay."""
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(0.0, float(num_training_steps - current_step) / 
                   float(max(1, num_training_steps - num_warmup_steps)))
    
    return LambdaLR(optimizer, lr_lambda)


def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, 
                                   num_cycles=0.5):
    """Cosine learning rate schedule with warmup."""
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))
    
    return LambdaLR(optimizer, lr_lambda)


def get_polynomial_decay_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps,
                                             lr_end=1e-7, power=1.0):
    """Polynomial decay schedule with warmup."""
    lr_init = optimizer.defaults["lr"]
    
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        elif current_step > num_training_steps:
            return lr_end / lr_init
        else:
            lr_range = lr_init - lr_end
            decay_steps = num_training_steps - num_warmup_steps
            pct_remaining = 1 - (current_step - num_warmup_steps) / decay_steps
            decay = lr_range * pct_remaining ** power + lr_end
            return decay / lr_init
    
    return LambdaLR(optimizer, lr_lambda)


class TransformerTrainer:
    """Complete trainer for transformer models."""
    
    def __init__(self, model: nn.Module, config: TrainingConfig,
                 train_dataset: Dataset, eval_dataset: Optional[Dataset] = None):
        self.model = model
        self.config = config
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        
        # Setup device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # Setup distributed training
        if config.distributed:
            self.setup_distributed()
            
        # Create data loaders
        self.create_dataloaders()
        
        # Setup optimizer and scheduler
        self.setup_optimization()
        
        # Setup mixed precision
        if config.use_mixed_precision:
            self.scaler = GradScaler()
            
        # Logging
        self.setup_logging()
        
        # Training state
        self.global_step = 0
        self.epoch = 0
        
    def setup_distributed(self):
        """Initialize distributed training."""
        dist.init_process_group(backend='nccl')
        self.model = DDP(self.model, device_ids=[self.config.rank])
        
    def create_dataloaders(self):
        """Create data loaders with appropriate samplers."""
        if self.config.distributed:
            train_sampler = DistributedSampler(
                self.train_dataset,
                num_replicas=self.config.world_size,
                rank=self.config.rank
            )
        else:
            train_sampler = None
            
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            sampler=train_sampler,
            shuffle=(train_sampler is None),
            num_workers=0,
            pin_memory=True
        )
        
        if self.eval_dataset:
            self.eval_loader = DataLoader(
                self.eval_dataset,
                batch_size=self.config.batch_size * 2,  # Larger batch for eval
                shuffle=False,
                num_workers=0,
                pin_memory=True
            )
            
    def setup_optimization(self):
        """Setup optimizer and learning rate scheduler."""
        # Separate parameters for weight decay
        no_decay = ['bias', 'LayerNorm.weight', 'layer_norm.weight']
        optimizer_grouped_parameters = [
            {
                'params': [p for n, p in self.model.named_parameters() 
                          if not any(nd in n for nd in no_decay)],
                'weight_decay': self.config.weight_decay,
            },
            {
                'params': [p for n, p in self.model.named_parameters() 
                          if any(nd in n for nd in no_decay)],
                'weight_decay': 0.0,
            }
        ]
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters,
            lr=self.config.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        # Scheduler
        num_training_steps = len(self.train_loader) * self.config.num_epochs
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=self.config.warmup_steps,
            num_training_steps=num_training_steps
        )
        
    def setup_logging(self):
        """Setup logging and monitoring."""
        logging.basicConfig(
            format='%(asctime)s - %(levelname)s - %(message)s',
            level=logging.INFO
        )
        self.logger = logging.getLogger(__name__)
        
        # Weights & Biases
        if self.config.rank == 0:  # Only log from main process
            try:
                wandb.init(project="transformer-training", config=self.config.__dict__, mode="disabled")
                wandb.watch(self.model)
            except:
                print("Warning: wandb not configured, continuing without it")
            
    def train_epoch(self):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        total_samples = 0
        
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {self.epoch}")
        
        for step, batch in enumerate(progress_bar):
            # Move to device
            batch = {k: v.to(self.device) for k, v in batch.items()}
            
            # Forward pass
            if self.config.use_mixed_precision:
                with autocast():
                    outputs = self.model(**batch)
                    loss = outputs.loss / self.config.gradient_accumulation_steps
            else:
                outputs = self.model(**batch)
                loss = outputs.loss / self.config.gradient_accumulation_steps
            
            # Backward pass
            if self.config.use_mixed_precision:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # Gradient accumulation
            if (step + 1) % self.config.gradient_accumulation_steps == 0:
                # Gradient clipping
                if self.config.use_mixed_precision:
                    self.scaler.unscale_(self.optimizer)
                    
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    self.config.max_grad_norm
                )
                
                # Optimizer step
                if self.config.use_mixed_precision:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                    
                self.scheduler.step()
                self.optimizer.zero_grad()
                
                # Update metrics
                total_loss += loss.item() * self.config.gradient_accumulation_steps
                total_samples += batch['input_ids'].size(0)
                
                # Log
                if self.global_step % self.config.log_interval == 0:
                    self.log_metrics({
                        'train/loss': loss.item() * self.config.gradient_accumulation_steps,
                        'train/learning_rate': self.scheduler.get_last_lr()[0],
                        'train/gradient_norm': self.get_gradient_norm(),
                    })
                    
                # Evaluate
                if self.global_step % self.config.eval_interval == 0 and self.eval_dataset:
                    eval_metrics = self.evaluate()
                    self.log_metrics(eval_metrics, prefix='eval/')
                    
                # Save checkpoint
                if self.global_step % self.config.save_interval == 0:
                    self.save_checkpoint()
                    
                self.global_step += 1
                
                # Update progress bar
                progress_bar.set_postfix({
                    'loss': f"{loss.item() * self.config.gradient_accumulation_steps:.4f}",
                    'lr': f"{self.scheduler.get_last_lr()[0]:.2e}"
                })
        
        return total_loss / len(self.train_loader)
    
    def evaluate(self):
        """Evaluate the model."""
        self.model.eval()
        total_loss = 0
        total_correct = 0
        total_samples = 0
        
        with torch.no_grad():
            for batch in tqdm(self.eval_loader, desc="Evaluating"):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                if self.config.use_mixed_precision:
                    with autocast():
                        outputs = self.model(**batch)
                else:
                    outputs = self.model(**batch)
                
                total_loss += outputs.loss.item() * batch['input_ids'].size(0)
                
                # Accuracy (if applicable)
                if hasattr(outputs, 'logits'):
                    predictions = outputs.logits.argmax(dim=-1)
                    total_correct += (predictions == batch['labels']).sum().item()
                    
                total_samples += batch['input_ids'].size(0)
        
        metrics = {
            'loss': total_loss / total_samples,
        }
        
        if total_correct > 0:
            metrics['accuracy'] = total_correct / total_samples
            
        self.model.train()
        return metrics
    
    def get_gradient_norm(self):
        """Calculate gradient norm."""
        total_norm = 0
        for p in self.model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        return total_norm ** 0.5
    
    def log_metrics(self, metrics: Dict[str, float], prefix: str = ''):
        """Log metrics to wandb and console."""
        # Add prefix
        metrics = {f"{prefix}{k}": v for k, v in metrics.items()}
        metrics['global_step'] = self.global_step
        
        # Log to wandb
        if self.config.rank == 0:
            wandb.log(metrics)
            
        # Log to console
        log_str = f"Step {self.global_step}: "
        log_str += ", ".join([f"{k}={v:.4f}" for k, v in metrics.items() if k != 'global_step'])
        self.logger.info(log_str)
        
    def save_checkpoint(self):
        """Save model checkpoint."""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'global_step': self.global_step,
            'epoch': self.epoch,
            'config': self.config.__dict__,
        }
        
        if self.config.use_mixed_precision:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
            
        checkpoint_path = os.path.join(
            self.config.checkpoint_dir, 
            f"checkpoint-{self.global_step}.pt"
        )
        torch.save(checkpoint, checkpoint_path)
        self.logger.info(f"Saved checkpoint to {checkpoint_path}")
        
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.global_step = checkpoint['global_step']
        self.epoch = checkpoint['epoch']
        
        if self.config.use_mixed_precision and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
            
        self.logger.info(f"Loaded checkpoint from {checkpoint_path}")
        
    def train(self):
        """Main training loop."""
        self.logger.info("Starting training...")
        
        for epoch in range(self.config.num_epochs):
            self.epoch = epoch
            
            # Train epoch
            avg_loss = self.train_epoch()
            self.logger.info(f"Epoch {epoch} - Average Loss: {avg_loss:.4f}")
            
            # Final evaluation
            if self.eval_dataset:
                eval_metrics = self.evaluate()
                self.logger.info(f"Epoch {epoch} - Eval metrics: {eval_metrics}")
                
            # Save epoch checkpoint
            self.save_checkpoint()
            
        self.logger.info("Training completed!")


# Advanced Training Utilities
class GradientAccumulator:
    """Handles gradient accumulation for large batch training."""
    
    def __init__(self, model, accumulation_steps: int = 1):
        self.model = model
        self.accumulation_steps = accumulation_steps
        self.step_count = 0
        
    def step(self, loss, optimizer, scheduler=None, scaler=None):
        """Perform gradient accumulation step."""
        # Scale loss
        loss = loss / self.accumulation_steps
        
        # Backward
        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()
            
        self.step_count += 1
        
        # Update weights
        if self.step_count % self.accumulation_steps == 0:
            if scaler is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                
            if scheduler is not None:
                scheduler.step()
                
            optimizer.zero_grad()
            self.step_count = 0
            return True
        
        return False


class TrainingMonitor:
    """Monitor training progress and detect issues."""
    
    def __init__(self, patience: int = 100):
        self.patience = patience
        self.best_loss = float('inf')
        self.best_step = 0
        self.loss_history = []
        self.gradient_history = []
        
    def update(self, loss: float, gradient_norm: float, step: int):
        """Update monitoring statistics."""
        self.loss_history.append(loss)
        self.gradient_history.append(gradient_norm)
        
        # Check for improvement
        if loss < self.best_loss:
            self.best_loss = loss
            self.best_step = step
            
        # Check for issues
        issues = []
        
        # Loss explosion
        if loss > 10 * self.best_loss or math.isnan(loss):
            issues.append("Loss explosion detected!")
            
        # Gradient explosion
        if gradient_norm > 1000:
            issues.append("Gradient explosion detected!")
            
        # No improvement
        if step - self.best_step > self.patience:
            issues.append("No improvement for {} steps".format(self.patience))
            
        return issues
    
    def plot_training_curves(self):
        """Plot training curves."""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        
        # Loss curve
        ax1.plot(self.loss_history)
        ax1.set_title('Training Loss')
        ax1.set_xlabel('Step')
        ax1.set_ylabel('Loss')
        ax1.set_yscale('log')
        
        # Gradient norm curve
        ax2.plot(self.gradient_history)
        ax2.set_title('Gradient Norm')
        ax2.set_xlabel('Step')
        ax2.set_ylabel('Gradient Norm')
        ax2.set_yscale('log')
        
        plt.tight_layout()
        plt.show()


def analyze_model_gradients(model):
    """Analyze gradient statistics across model layers."""
    gradient_stats = {}
    
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad = param.grad.data
            gradient_stats[name] = {
                'mean': grad.mean().item(),
                'std': grad.std().item(),
                'max': grad.max().item(),
                'min': grad.min().item(),
                'norm': grad.norm().item(),
            }
            
    return gradient_stats


def visualize_attention_entropy(model, dataloader, num_samples=10):
    """Visualize attention entropy to detect collapsed attention."""
    model.eval()
    entropies = defaultdict(list)
    
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= num_samples:
                break
                
            outputs = model(**batch, output_attentions=True)
            
            # Calculate entropy for each head
            for layer_idx, attn in enumerate(outputs.attentions):
                # attn shape: [batch, heads, seq_len, seq_len]
                attn_probs = attn.mean(dim=0)  # Average over batch
                
                # Calculate entropy for each head
                entropy = -(attn_probs * torch.log(attn_probs + 1e-9)).sum(dim=-1).mean(dim=-1)
                entropies[f"layer_{layer_idx}"].extend(entropy.cpu().numpy())
    
    # Plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    layers = sorted(entropies.keys())
    data = [entropies[layer] for layer in layers]
    
    ax.boxplot(data, labels=layers)
    ax.set_xlabel('Layer')
    ax.set_ylabel('Attention Entropy')
    ax.set_title('Attention Entropy Distribution Across Layers')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    
    model.train()


# Example usage
if __name__ == "__main__":
    # Create dummy data
    train_data = [
        {"text": f"This is training example {i}", "label": i % 2}
        for i in range(1000)
    ]
    
    eval_data = [
        {"text": f"This is evaluation example {i}", "label": i % 2}
        for i in range(100)
    ]
    
    # Dummy tokenizer
    class DummyTokenizer:
        def __call__(self, text, **kwargs):
            # Simple character-level tokenization
            tokens = [ord(c) for c in text]
            max_length = kwargs.get('max_length', 512)
            
            # Padding
            if len(tokens) < max_length:
                tokens = tokens + [0] * (max_length - len(tokens))
            else:
                tokens = tokens[:max_length]
                
            return {
                'input_ids': torch.tensor([tokens]),
                'attention_mask': torch.tensor([[1 if t != 0 else 0 for t in tokens]])
            }
    
    tokenizer = DummyTokenizer()
    
    # Create datasets
    train_dataset = TransformerDataset(train_data, tokenizer)
    eval_dataset = TransformerDataset(eval_data, tokenizer)
    
    # Create model (dummy for demonstration)
    from torch.nn import TransformerEncoder, TransformerEncoderLayer
    
    class DummyTransformerModel(nn.Module):
        def __init__(self, vocab_size=256, d_model=128, nhead=8, num_layers=6):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, d_model)
            encoder_layer = TransformerEncoderLayer(d_model, nhead)
            self.transformer = TransformerEncoder(encoder_layer, num_layers)
            self.classifier = nn.Linear(d_model, 2)
            
        def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
            # Embed
            x = self.embedding(input_ids)
            
            # Transform
            x = self.transformer(x)
            
            # Pool and classify
            x = x.mean(dim=1)
            logits = self.classifier(x)
            
            # Calculate loss
            loss = None
            if labels is not None:
                loss = F.cross_entropy(logits, labels)
                
            class Output:
                pass
            
            output = Output()
            output.loss = loss
            output.logits = logits
            
            return output
    
    model = DummyTransformerModel()
    
    # Training config
    config = TrainingConfig(
        batch_size=8,
        num_epochs=2,
        learning_rate=1e-3,
        warmup_steps=100,
        log_interval=10,
        eval_interval=50,
        save_interval=100,
        use_mixed_precision=False  # Set to True if GPU available
    )
    
    # Create trainer
    trainer = TransformerTrainer(model, config, train_dataset, eval_dataset)
    
    print("=== Training Transformers Demo ===")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Evaluation samples: {len(eval_dataset)}")
    print(f"Device: {trainer.device}")
    
    # Demonstrate learning rate schedules
    print("\n=== Learning Rate Schedules ===")
    
    # Create dummy optimizer
    dummy_optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    # Test different schedules
    schedules = {
        'Linear': get_linear_schedule_with_warmup(dummy_optimizer, 100, 1000),
        'Cosine': get_cosine_schedule_with_warmup(dummy_optimizer, 100, 1000),
        'Polynomial': get_polynomial_decay_schedule_with_warmup(dummy_optimizer, 100, 1000)
    }
    
    # Plot schedules
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for name, scheduler in schedules.items():
        lrs = []
        for step in range(1000):
            scheduler.step()
            lrs.append(scheduler.get_last_lr()[0])
        ax.plot(lrs, label=name)
    
    ax.set_xlabel('Step')
    ax.set_ylabel('Learning Rate')
    ax.set_title('Learning Rate Schedules')
    ax.legend()
    plt.tight_layout()
    plt.show()
    
    # Demonstrate gradient analysis
    print("\n=== Gradient Analysis ===")
    
    # Forward and backward pass
    batch = next(iter(trainer.train_loader))
    outputs = model(**batch)
    outputs.loss.backward()
    
    # Analyze gradients
    grad_stats = analyze_model_gradients(model)
    
    print("\nGradient statistics for first 5 parameters:")
    for i, (name, stats) in enumerate(grad_stats.items()):
        if i >= 5:
            break
        print(f"\n{name}:")
        for stat_name, value in stats.items():
            print(f"  {stat_name}: {value:.6f}")
    
    # Training would start here
    # trainer.train()
    
    print("\n✅ Training setup complete! Ready to train.")