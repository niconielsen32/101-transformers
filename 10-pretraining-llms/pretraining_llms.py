"""
Pretraining Large Language Models
Complete implementation of LLM pretraining pipeline including dataset processing,
distributed training, and monitoring.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from torch.optim.lr_scheduler import LambdaLR
import numpy as np
import math
import time
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from collections import defaultdict
import matplotlib.pyplot as plt
from tqdm import tqdm
import logging
from datetime import datetime


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Configuration
@dataclass
class PretrainingConfig:
    """Configuration for LLM pretraining."""
    # Model
    model_name: str = "llama-7b"
    vocab_size: int = 32000
    hidden_size: int = 4096
    num_layers: int = 32
    num_heads: int = 32
    intermediate_size: int = 11008
    max_position_embeddings: int = 2048
    
    # Training
    batch_size: int = 4
    gradient_accumulation_steps: int = 8
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    adam_epsilon: float = 1e-8
    max_grad_norm: float = 1.0
    
    # Schedule
    num_train_steps: int = 1000000
    warmup_steps: int = 10000
    logging_steps: int = 100
    save_steps: int = 5000
    eval_steps: int = 1000
    
    # Data
    train_data_path: str = "./data/train"
    val_data_path: str = "./data/val"
    max_length: int = 2048
    mlm_probability: float = 0.15
    
    # Distributed
    world_size: int = 1
    rank: int = 0
    local_rank: int = 0
    backend: str = "nccl"
    
    # Optimization
    mixed_precision: bool = True
    gradient_checkpointing: bool = True
    cpu_offload: bool = False
    
    # Checkpointing
    checkpoint_dir: str = "./checkpoints"
    resume_from_checkpoint: Optional[str] = None


# Dataset Classes
class PretrainingDataset(Dataset):
    """Dataset for language model pretraining."""
    
    def __init__(self, data_path: str, tokenizer, max_length: int = 2048):
        self.data_path = Path(data_path)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.documents = self._load_documents()
        
    def _load_documents(self) -> List[str]:
        """Load all documents from data path."""
        documents = []
        for file_path in self.data_path.glob("*.txt"):
            with open(file_path, 'r', encoding='utf-8') as f:
                documents.extend(f.read().split('\n\n'))
        return documents
    
    def __len__(self):
        return len(self.documents)
    
    def __getitem__(self, idx):
        document = self.documents[idx]
        
        # Tokenize
        encoded = self.tokenizer(
            document,
            max_length=self.max_length,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoded['input_ids'].squeeze(0),
            'attention_mask': encoded['attention_mask'].squeeze(0),
            'labels': encoded['input_ids'].squeeze(0)
        }


class PackedDataset(Dataset):
    """Dataset that packs multiple documents into single examples."""
    
    def __init__(self, documents: List[List[int]], max_length: int, 
                 pad_token_id: int):
        self.examples = self._pack_documents(documents, max_length, pad_token_id)
        
    def _pack_documents(self, documents: List[List[int]], max_length: int, 
                       pad_token_id: int) -> List[List[int]]:
        """Pack documents into fixed-length examples."""
        packed_examples = []
        current_example = []
        current_length = 0
        
        for doc in documents:
            doc_length = len(doc)
            
            if current_length + doc_length <= max_length:
                current_example.extend(doc)
                current_length += doc_length
            else:
                # Pad and save current example
                padding = [pad_token_id] * (max_length - current_length)
                packed_examples.append(current_example + padding)
                
                # Start new example
                if doc_length <= max_length:
                    current_example = doc
                    current_length = doc_length
                else:
                    # Split long document
                    current_example = doc[:max_length]
                    packed_examples.append(current_example)
                    current_example = []
                    current_length = 0
        
        # Handle last example
        if current_example:
            padding = [pad_token_id] * (max_length - current_length)
            packed_examples.append(current_example + padding)
            
        return packed_examples
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        input_ids = torch.tensor(self.examples[idx], dtype=torch.long)
        attention_mask = (input_ids != 0).long()
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': input_ids
        }


# Data Processing
class DataProcessor:
    """Process and prepare data for pretraining."""
    
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        
    def process_text(self, text: str) -> str:
        """Clean and normalize text."""
        # Remove multiple spaces
        text = ' '.join(text.split())
        
        # Fix common encoding issues
        text = text.replace('"', '"').replace('"', '"')
        text = text.replace(''', "'").replace(''', "'")
        
        # Remove control characters
        text = ''.join(char for char in text if ord(char) >= 32 or char == '\n')
        
        return text
    
    def filter_document(self, doc: str, min_length: int = 100, 
                       max_length: int = 50000) -> bool:
        """Filter documents based on quality criteria."""
        # Length filter
        word_count = len(doc.split())
        if word_count < min_length or word_count > max_length:
            return False
        
        # Repetition filter
        lines = doc.split('\n')
        if len(lines) > 10:
            unique_lines = len(set(lines))
            if unique_lines / len(lines) < 0.5:
                return False
        
        # Language detection (simplified)
        common_english_words = {'the', 'be', 'to', 'of', 'and', 'a', 'in'}
        doc_words = set(doc.lower().split()[:100])
        if len(doc_words & common_english_words) < 2:
            return False
        
        return True
    
    def create_mlm_batch(self, batch: Dict[str, torch.Tensor], 
                        mlm_probability: float = 0.15) -> Dict[str, torch.Tensor]:
        """Create masked language modeling batch."""
        input_ids = batch['input_ids'].clone()
        labels = batch['input_ids'].clone()
        
        # Create mask
        probability_matrix = torch.full(labels.shape, mlm_probability)
        special_tokens_mask = self._get_special_tokens_mask(input_ids)
        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # Only compute loss on masked tokens
        
        # 80% of time, replace with [MASK]
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        input_ids[indices_replaced] = self.tokenizer.mask_token_id
        
        # 10% of time, replace with random word
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
        input_ids[indices_random] = random_words[indices_random]
        
        # 10% of time, keep original
        
        return {
            'input_ids': input_ids,
            'attention_mask': batch['attention_mask'],
            'labels': labels
        }
    
    def _get_special_tokens_mask(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Get mask for special tokens."""
        special_tokens_mask = torch.zeros_like(input_ids, dtype=torch.bool)
        special_ids = {
            self.tokenizer.pad_token_id,
            self.tokenizer.cls_token_id,
            self.tokenizer.sep_token_id,
            self.tokenizer.mask_token_id
        }
        for token_id in special_ids:
            if token_id is not None:
                special_tokens_mask |= (input_ids == token_id)
        return special_tokens_mask


# Training Components
class LearningRateScheduler:
    """Custom learning rate scheduler for pretraining."""
    
    def __init__(self, optimizer, num_warmup_steps: int, 
                 num_training_steps: int, min_lr_ratio: float = 0.1):
        self.optimizer = optimizer
        self.num_warmup_steps = num_warmup_steps
        self.num_training_steps = num_training_steps
        self.min_lr_ratio = min_lr_ratio
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]
        
    def get_lr(self, step: int) -> List[float]:
        """Calculate learning rate for current step."""
        if step < self.num_warmup_steps:
            # Linear warmup
            factor = step / max(1, self.num_warmup_steps)
        else:
            # Cosine decay
            progress = (step - self.num_warmup_steps) / (self.num_training_steps - self.num_warmup_steps)
            factor = self.min_lr_ratio + (1 - self.min_lr_ratio) * (1 + math.cos(math.pi * progress)) / 2
            
        return [base_lr * factor for base_lr in self.base_lrs]
    
    def step(self, step: int):
        """Update learning rate."""
        lrs = self.get_lr(step)
        for param_group, lr in zip(self.optimizer.param_groups, lrs):
            param_group['lr'] = lr


class GradientAccumulator:
    """Handle gradient accumulation for large batch training."""
    
    def __init__(self, model, accumulation_steps: int):
        self.model = model
        self.accumulation_steps = accumulation_steps
        self.step_count = 0
        
    def backward(self, loss: torch.Tensor):
        """Backward pass with gradient accumulation."""
        loss = loss / self.accumulation_steps
        loss.backward()
        self.step_count += 1
        
    def should_update(self) -> bool:
        """Check if gradients should be updated."""
        return self.step_count % self.accumulation_steps == 0
    
    def reset(self):
        """Reset accumulation counter."""
        self.step_count = 0


class TrainingMonitor:
    """Monitor training metrics and performance."""
    
    def __init__(self, config: PretrainingConfig):
        self.config = config
        self.metrics = defaultdict(list)
        self.start_time = time.time()
        self.tokens_processed = 0
        
    def log_step(self, step: int, loss: float, learning_rate: float, 
                 grad_norm: float, batch_size: int):
        """Log metrics for current step."""
        self.metrics['step'].append(step)
        self.metrics['loss'].append(loss)
        self.metrics['learning_rate'].append(learning_rate)
        self.metrics['grad_norm'].append(grad_norm)
        self.metrics['perplexity'].append(math.exp(min(loss, 10)))
        
        # Calculate tokens per second
        elapsed = time.time() - self.start_time
        self.tokens_processed += batch_size * self.config.max_length
        tokens_per_second = self.tokens_processed / elapsed
        self.metrics['tokens_per_second'].append(tokens_per_second)
        
        # Log to console
        if step % self.config.logging_steps == 0:
            logger.info(
                f"Step {step}/{self.config.num_train_steps} | "
                f"Loss: {loss:.4f} | "
                f"PPL: {math.exp(min(loss, 10)):.2f} | "
                f"LR: {learning_rate:.2e} | "
                f"Grad: {grad_norm:.2f} | "
                f"Tokens/s: {tokens_per_second:.0f}"
            )
            
        # Check for anomalies
        if math.isnan(loss):
            logger.error(f"NaN loss detected at step {step}")
            raise ValueError("NaN loss detected")
            
        if grad_norm > 100:
            logger.warning(f"Large gradient norm: {grad_norm:.2f} at step {step}")
    
    def plot_training_curves(self, save_path: str = "training_curves.png"):
        """Plot training metrics."""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Loss
        axes[0, 0].plot(self.metrics['step'], self.metrics['loss'])
        axes[0, 0].set_title('Training Loss')
        axes[0, 0].set_xlabel('Step')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_yscale('log')
        
        # Perplexity
        axes[0, 1].plot(self.metrics['step'], self.metrics['perplexity'])
        axes[0, 1].set_title('Perplexity')
        axes[0, 1].set_xlabel('Step')
        axes[0, 1].set_ylabel('Perplexity')
        
        # Learning rate
        axes[0, 2].plot(self.metrics['step'], self.metrics['learning_rate'])
        axes[0, 2].set_title('Learning Rate')
        axes[0, 2].set_xlabel('Step')
        axes[0, 2].set_ylabel('LR')
        
        # Gradient norm
        axes[1, 0].plot(self.metrics['step'], self.metrics['grad_norm'])
        axes[1, 0].set_title('Gradient Norm')
        axes[1, 0].set_xlabel('Step')
        axes[1, 0].set_ylabel('Norm')
        axes[1, 0].set_yscale('log')
        
        # Tokens per second
        axes[1, 1].plot(self.metrics['step'], self.metrics['tokens_per_second'])
        axes[1, 1].set_title('Training Throughput')
        axes[1, 1].set_xlabel('Step')
        axes[1, 1].set_ylabel('Tokens/second')
        
        # Loss over time
        elapsed_hours = [(s * self.config.logging_steps * 0.1) / 3600 for s in range(len(self.metrics['loss']))]
        axes[1, 2].plot(elapsed_hours, self.metrics['loss'])
        axes[1, 2].set_title('Loss over Time')
        axes[1, 2].set_xlabel('Hours')
        axes[1, 2].set_ylabel('Loss')
        
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()


# Distributed Training
class DistributedTrainer:
    """Handle distributed training setup and execution."""
    
    def __init__(self, config: PretrainingConfig):
        self.config = config
        self.setup_distributed()
        
    def setup_distributed(self):
        """Initialize distributed training."""
        if self.config.world_size > 1:
            dist.init_process_group(
                backend=self.config.backend,
                init_method='env://',
                world_size=self.config.world_size,
                rank=self.config.rank
            )
            torch.cuda.set_device(self.config.local_rank)
            
    def wrap_model(self, model: nn.Module) -> nn.Module:
        """Wrap model for distributed training."""
        if self.config.world_size == 1:
            return model
            
        if self.config.cpu_offload or model.num_parameters() > 1e9:
            # Use FSDP for very large models
            from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
            auto_wrap_policy = functools.partial(
                transformer_auto_wrap_policy,
                transformer_layer_cls={TransformerBlock},
            )
            
            model = FSDP(
                model,
                auto_wrap_policy=auto_wrap_policy,
                mixed_precision=mixed_precision_policy,
                sharding_strategy=ShardingStrategy.FULL_SHARD,
                cpu_offload=CPUOffload(offload_params=True),
            )
        else:
            # Use DDP for smaller models
            model = DDP(
                model.cuda(self.config.local_rank),
                device_ids=[self.config.local_rank],
                output_device=self.config.local_rank,
                find_unused_parameters=False
            )
            
        return model
    
    def get_dataloader(self, dataset: Dataset, shuffle: bool = True) -> DataLoader:
        """Create distributed dataloader."""
        sampler = None
        if self.config.world_size > 1:
            sampler = DistributedSampler(
                dataset,
                num_replicas=self.config.world_size,
                rank=self.config.rank,
                shuffle=shuffle
            )
            shuffle = False
            
        return DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=shuffle,
            sampler=sampler,
            num_workers=4,
            pin_memory=True,
            drop_last=True
        )


# Checkpointing
class CheckpointManager:
    """Manage model checkpoints and recovery."""
    
    def __init__(self, config: PretrainingConfig):
        self.config = config
        self.checkpoint_dir = Path(config.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
    def save_checkpoint(self, model: nn.Module, optimizer, step: int, 
                       metrics: Dict, keep_last_n: int = 5):
        """Save training checkpoint."""
        checkpoint_path = self.checkpoint_dir / f"checkpoint-{step}.pt"
        
        # Prepare checkpoint data
        checkpoint = {
            'step': step,
            'model_state_dict': self._get_model_state_dict(model),
            'optimizer_state_dict': optimizer.state_dict(),
            'metrics': metrics,
            'config': self.config,
            'rng_state': torch.get_rng_state(),
        }
        
        if torch.cuda.is_available():
            checkpoint['cuda_rng_state'] = torch.cuda.get_rng_state()
            
        # Save checkpoint
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Saved checkpoint to {checkpoint_path}")
        
        # Clean up old checkpoints
        self._cleanup_checkpoints(keep_last_n)
        
    def load_checkpoint(self, checkpoint_path: str, model: nn.Module, 
                       optimizer=None) -> Dict:
        """Load training checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Load model state
        model_state_dict = checkpoint['model_state_dict']
        model.load_state_dict(model_state_dict)
        
        # Load optimizer state
        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
        # Restore RNG states
        torch.set_rng_state(checkpoint['rng_state'])
        if torch.cuda.is_available() and 'cuda_rng_state' in checkpoint:
            torch.cuda.set_rng_state(checkpoint['cuda_rng_state'])
            
        logger.info(f"Loaded checkpoint from {checkpoint_path}")
        
        return checkpoint
    
    def _get_model_state_dict(self, model: nn.Module) -> Dict:
        """Extract model state dict handling DDP/FSDP."""
        if hasattr(model, 'module'):
            return model.module.state_dict()
        else:
            return model.state_dict()
            
    def _cleanup_checkpoints(self, keep_last_n: int):
        """Remove old checkpoints keeping only the last n."""
        checkpoints = sorted(self.checkpoint_dir.glob("checkpoint-*.pt"))
        if len(checkpoints) > keep_last_n:
            for checkpoint in checkpoints[:-keep_last_n]:
                checkpoint.unlink()
                logger.info(f"Removed old checkpoint: {checkpoint}")


# Main Training Loop
class PretrainingPipeline:
    """Complete pretraining pipeline."""
    
    def __init__(self, config: PretrainingConfig):
        self.config = config
        self.setup()
        
    def setup(self):
        """Setup training components."""
        # Initialize distributed training
        self.distributed_trainer = DistributedTrainer(self.config)
        
        # Create model
        self.model = self.create_model()
        self.model = self.distributed_trainer.wrap_model(self.model)
        
        # Create optimizer
        self.optimizer = self.create_optimizer()
        
        # Create scheduler
        self.scheduler = LearningRateScheduler(
            self.optimizer,
            num_warmup_steps=self.config.warmup_steps,
            num_training_steps=self.config.num_train_steps
        )
        
        # Setup monitoring
        self.monitor = TrainingMonitor(self.config)
        self.checkpoint_manager = CheckpointManager(self.config)
        
        # Gradient accumulation
        self.grad_accumulator = GradientAccumulator(
            self.model, 
            self.config.gradient_accumulation_steps
        )
        
        # Mixed precision
        self.scaler = torch.cuda.amp.GradScaler() if self.config.mixed_precision else None
        
    def create_model(self) -> nn.Module:
        """Create the model to be trained."""
        # Placeholder - would import actual model
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            vocab_size=self.config.vocab_size,
            hidden_size=self.config.hidden_size,
            num_hidden_layers=self.config.num_layers,
            num_attention_heads=self.config.num_heads,
            intermediate_size=self.config.intermediate_size,
            max_position_embeddings=self.config.max_position_embeddings,
        )
        
        if self.config.gradient_checkpointing:
            model.gradient_checkpointing_enable()
            
        return model
    
    def create_optimizer(self) -> torch.optim.Optimizer:
        """Create optimizer with weight decay."""
        # Separate parameters that should and shouldn't have weight decay
        decay_params = []
        no_decay_params = []
        
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
                
            if 'bias' in name or 'layer_norm' in name or 'layernorm' in name:
                no_decay_params.append(param)
            else:
                decay_params.append(param)
                
        optimizer_groups = [
            {'params': decay_params, 'weight_decay': self.config.weight_decay},
            {'params': no_decay_params, 'weight_decay': 0.0}
        ]
        
        optimizer = torch.optim.AdamW(
            optimizer_groups,
            lr=self.config.learning_rate,
            betas=(self.config.adam_beta1, self.config.adam_beta2),
            eps=self.config.adam_epsilon
        )
        
        return optimizer
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Tuple[float, float]:
        """Single training step."""
        # Move batch to device
        batch = {k: v.cuda() for k, v in batch.items()}
        
        # Mixed precision context
        with torch.cuda.amp.autocast(enabled=self.config.mixed_precision):
            outputs = self.model(**batch)
            loss = outputs.loss
            
        # Backward pass
        if self.scaler is not None:
            self.scaler.scale(loss).backward()
        else:
            self.grad_accumulator.backward(loss)
            
        # Gradient clipping and optimizer step
        if self.grad_accumulator.should_update():
            if self.scaler is not None:
                self.scaler.unscale_(self.optimizer)
                
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), 
                self.config.max_grad_norm
            )
            
            if self.scaler is not None:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()
                
            self.optimizer.zero_grad()
            self.grad_accumulator.reset()
        else:
            grad_norm = 0.0
            
        return loss.item(), grad_norm
    
    def train(self, train_dataloader: DataLoader, val_dataloader: Optional[DataLoader] = None):
        """Main training loop."""
        global_step = 0
        
        # Resume from checkpoint if specified
        if self.config.resume_from_checkpoint:
            checkpoint = self.checkpoint_manager.load_checkpoint(
                self.config.resume_from_checkpoint,
                self.model,
                self.optimizer
            )
            global_step = checkpoint['step']
            
        # Training loop
        self.model.train()
        
        for epoch in range(100):  # Large number, will break on num_train_steps
            if hasattr(train_dataloader.sampler, 'set_epoch'):
                train_dataloader.sampler.set_epoch(epoch)
                
            for batch in train_dataloader:
                # Training step
                loss, grad_norm = self.train_step(batch)
                
                # Update learning rate
                self.scheduler.step(global_step)
                
                # Log metrics
                self.monitor.log_step(
                    global_step,
                    loss,
                    self.optimizer.param_groups[0]['lr'],
                    grad_norm,
                    self.config.batch_size * self.config.world_size
                )
                
                # Validation
                if val_dataloader and global_step % self.config.eval_steps == 0:
                    val_loss = self.validate(val_dataloader)
                    logger.info(f"Validation loss: {val_loss:.4f}")
                    self.model.train()
                    
                # Checkpointing
                if global_step % self.config.save_steps == 0:
                    self.checkpoint_manager.save_checkpoint(
                        self.model,
                        self.optimizer,
                        global_step,
                        self.monitor.metrics
                    )
                    
                global_step += 1
                
                if global_step >= self.config.num_train_steps:
                    logger.info("Reached maximum training steps")
                    return
                    
    def validate(self, dataloader: DataLoader) -> float:
        """Run validation."""
        self.model.eval()
        total_loss = 0
        total_tokens = 0
        
        with torch.no_grad():
            for batch in dataloader:
                batch = {k: v.cuda() for k, v in batch.items()}
                
                with torch.cuda.amp.autocast(enabled=self.config.mixed_precision):
                    outputs = self.model(**batch)
                    
                total_loss += outputs.loss.item() * batch['input_ids'].numel()
                total_tokens += batch['input_ids'].numel()
                
        return total_loss / total_tokens


# Utility Functions
def estimate_training_time(config: PretrainingConfig, tokens_per_second: float) -> Dict[str, float]:
    """Estimate training time and cost."""
    total_tokens = config.num_train_steps * config.batch_size * config.max_length * config.world_size
    
    training_time_seconds = total_tokens / tokens_per_second
    training_time_hours = training_time_seconds / 3600
    training_time_days = training_time_hours / 24
    
    # Cost estimation (rough)
    gpu_hours = training_time_hours * config.world_size
    cost_per_gpu_hour = 2.0  # Example: $2/hour for A100
    total_cost = gpu_hours * cost_per_gpu_hour
    
    return {
        'total_tokens': total_tokens,
        'training_hours': training_time_hours,
        'training_days': training_time_days,
        'gpu_hours': gpu_hours,
        'estimated_cost': total_cost
    }


def analyze_dataset_statistics(dataset: Dataset) -> Dict[str, float]:
    """Analyze dataset statistics."""
    lengths = []
    vocab = set()
    
    for i in tqdm(range(min(10000, len(dataset))), desc="Analyzing dataset"):
        example = dataset[i]
        input_ids = example['input_ids']
        
        lengths.append(len(input_ids))
        vocab.update(input_ids.tolist())
        
    return {
        'num_examples': len(dataset),
        'avg_length': np.mean(lengths),
        'std_length': np.std(lengths),
        'min_length': np.min(lengths),
        'max_length': np.max(lengths),
        'vocab_size': len(vocab)
    }


# Example Usage
if __name__ == "__main__":
    print("=== LLM Pretraining Pipeline Demo ===\n")
    
    # Configuration
    config = PretrainingConfig(
        model_name="gpt2",  # Using smaller model for demo
        vocab_size=50257,
        hidden_size=768,
        num_layers=12,
        num_heads=12,
        intermediate_size=3072,
        batch_size=2,
        gradient_accumulation_steps=4,
        num_train_steps=1000,
        warmup_steps=100,
        world_size=1
    )
    
    print("Configuration:")
    print(f"  Model: {config.model_name}")
    print(f"  Batch size: {config.batch_size} x {config.gradient_accumulation_steps} accumulation")
    print(f"  Training steps: {config.num_train_steps}")
    print(f"  Learning rate: {config.learning_rate}")
    
    # Data processing demo
    print("\n--- Data Processing Demo ---")
    processor = DataProcessor(None)  # Would pass tokenizer
    
    sample_text = """
    The transformer architecture has revolutionized natural language processing.
    It uses self-attention mechanisms to process sequences in parallel.
    This is much more efficient than recurrent neural networks.
    """
    
    cleaned_text = processor.process_text(sample_text)
    print(f"Original length: {len(sample_text)}")
    print(f"Cleaned length: {len(cleaned_text)}")
    print(f"Passes filter: {processor.filter_document(cleaned_text)}")
    
    # Learning rate schedule visualization
    print("\n--- Learning Rate Schedule ---")
    optimizer = torch.optim.Adam([torch.zeros(1)], lr=config.learning_rate)
    scheduler = LearningRateScheduler(optimizer, config.warmup_steps, config.num_train_steps)
    
    steps = list(range(0, config.num_train_steps, 10))
    lrs = [scheduler.get_lr(step)[0] for step in steps]
    
    plt.figure(figsize=(10, 6))
    plt.plot(steps, lrs)
    plt.axvline(x=config.warmup_steps, color='r', linestyle='--', label='End of warmup')
    plt.xlabel('Training Step')
    plt.ylabel('Learning Rate')
    plt.title('Learning Rate Schedule')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    # Training time estimation
    print("\n--- Training Time Estimation ---")
    tokens_per_second = 50000  # Example throughput
    estimates = estimate_training_time(config, tokens_per_second)
    
    print(f"Total tokens: {estimates['total_tokens']:,.0f}")
    print(f"Training time: {estimates['training_days']:.1f} days")
    print(f"GPU hours: {estimates['gpu_hours']:,.0f}")
    print(f"Estimated cost: ${estimates['estimated_cost']:,.2f}")
    
    # Distributed training setup
    print("\n--- Distributed Training Setup ---")
    print("For multi-GPU training, use:")
    print("  torchrun --nproc_per_node=8 pretraining_llms.py")
    print("\nFor multi-node training, use:")
    print("  torchrun --nproc_per_node=8 --nnodes=4 --node_rank=0 \\")
    print("    --master_addr=master_ip --master_port=29500 pretraining_llms.py")
    
    print("\n✅ Pretraining pipeline components demonstrated successfully!")