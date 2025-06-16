# Training Transformers

Master the art and science of training transformer models effectively. From optimization strategies to distributed training, this module covers everything you need to train transformers at scale.

## 🎯 Learning Objectives

By the end of this module, you will understand:
- Data preparation and batching strategies
- Learning rate schedules and warmup
- Gradient accumulation and mixed precision
- Distributed training techniques
- Debugging training issues
- Monitoring and evaluation

## 📚 Table of Contents

1. [Data Preparation](#1-data-preparation)
2. [Optimization Strategies](#2-optimization-strategies)
3. [Learning Rate Scheduling](#3-learning-rate-scheduling)
4. [Advanced Training Techniques](#4-advanced-training-techniques)
5. [Distributed Training](#5-distributed-training)
6. [Training Stability](#6-training-stability)
7. [Monitoring and Debugging](#7-monitoring-and-debugging)
8. [Best Practices](#8-best-practices)

## 1. Data Preparation

### 1.1 Efficient Data Loading

```python
class TransformerDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Tokenize
        tokens = self.tokenizer.encode(
            item['text'],
            max_length=self.max_length,
            truncation=True,
            padding='max_length'
        )
        
        return {
            'input_ids': torch.tensor(tokens),
            'attention_mask': torch.tensor([1 if t != 0 else 0 for t in tokens]),
            'labels': torch.tensor(item['label'])
        }
```

### 1.2 Dynamic Batching

Group sequences of similar length to minimize padding:

```python
def dynamic_batching(dataset, batch_size):
    # Sort by length
    sorted_data = sorted(dataset, key=lambda x: len(x['input_ids']))
    
    # Create batches
    batches = []
    current_batch = []
    
    for item in sorted_data:
        current_batch.append(item)
        if len(current_batch) == batch_size:
            batches.append(current_batch)
            current_batch = []
    
    return batches
```

### 1.3 Data Augmentation

- **Token masking**: Randomly mask tokens (BERT-style)
- **Sequence shuffling**: Permute sentences
- **Back-translation**: Translate and back-translate
- **Paraphrasing**: Generate variations

## 2. Optimization Strategies

### 2.1 Optimizer Choice

**AdamW** (most common):
```python
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=5e-5,
    betas=(0.9, 0.999),
    eps=1e-8,
    weight_decay=0.01
)
```

**Key differences**:
- **Adam**: Adaptive learning rates, momentum
- **AdamW**: Decoupled weight decay
- **LAMB**: Layer-wise adaptive rates
- **Adafactor**: Memory-efficient for large models

### 2.2 Gradient Clipping

Essential for training stability:

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

### 2.3 Weight Initialization

```python
def init_weights(module):
    if isinstance(module, nn.Linear):
        module.weight.data.normal_(mean=0.0, std=0.02)
        if module.bias is not None:
            module.bias.data.zero_()
    elif isinstance(module, nn.Embedding):
        module.weight.data.normal_(mean=0.0, std=0.02)
    elif isinstance(module, nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)
```

## 3. Learning Rate Scheduling

### 3.1 Linear Warmup with Decay

The standard transformer schedule:

```python
def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(0.0, float(num_training_steps - current_step) / 
                   float(max(1, num_training_steps - num_warmup_steps)))
    
    return LambdaLR(optimizer, lr_lambda)
```

### 3.2 Cosine Schedule

```python
def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    
    return LambdaLR(optimizer, lr_lambda)
```

### 3.3 Inverse Square Root (Original Transformer)

```python
class InverseSquareRootSchedule:
    def __init__(self, optimizer, warmup_steps=4000):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.step = 0
        
    def get_lr(self):
        step = max(1, self.step)
        return min(step ** (-0.5), step * self.warmup_steps ** (-1.5))
```

## 4. Advanced Training Techniques

### 4.1 Gradient Accumulation

Train with larger effective batch sizes:

```python
accumulation_steps = 4
optimizer.zero_grad()

for i, batch in enumerate(dataloader):
    outputs = model(**batch)
    loss = outputs.loss / accumulation_steps
    loss.backward()
    
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

### 4.2 Mixed Precision Training

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for batch in dataloader:
    optimizer.zero_grad()
    
    with autocast():
        outputs = model(**batch)
        loss = outputs.loss
    
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

### 4.3 Gradient Checkpointing

Save memory by recomputing activations:

```python
from torch.utils.checkpoint import checkpoint

class CheckpointedTransformerLayer(nn.Module):
    def forward(self, hidden_states):
        def create_custom_forward(module):
            def custom_forward(*inputs):
                return module(*inputs)
            return custom_forward
        
        hidden_states = checkpoint(
            create_custom_forward(self.self_attention),
            hidden_states
        )
        return hidden_states
```

## 5. Distributed Training

### 5.1 Data Parallel (DP)

Simple but less efficient:

```python
model = nn.DataParallel(model)
```

### 5.2 Distributed Data Parallel (DDP)

```python
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

# Initialize process group
dist.init_process_group("nccl")

# Create model
model = model.to(local_rank)
model = DistributedDataParallel(model, device_ids=[local_rank])

# Create distributed sampler
sampler = DistributedSampler(dataset)
dataloader = DataLoader(dataset, sampler=sampler)
```

### 5.3 Model Parallel

For models too large for single GPU:

```python
class ModelParallelTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size).to('cuda:0')
        self.layer1 = TransformerLayer().to('cuda:0')
        self.layer2 = TransformerLayer().to('cuda:1')
        self.output = nn.Linear(hidden_size, vocab_size).to('cuda:1')
        
    def forward(self, x):
        x = self.embedding(x)
        x = self.layer1(x)
        x = x.to('cuda:1')
        x = self.layer2(x)
        x = self.output(x)
        return x
```

### 5.4 ZeRO Optimization

DeepSpeed's memory optimization:

```python
import deepspeed

model_engine, optimizer, _, _ = deepspeed.initialize(
    model=model,
    model_parameters=model.parameters(),
    config={
        "train_batch_size": 64,
        "gradient_accumulation_steps": 4,
        "fp16": {"enabled": True},
        "zero_optimization": {
            "stage": 2,  # ZeRO-2
            "offload_optimizer": {"device": "cpu"}
        }
    }
)
```

## 6. Training Stability

### 6.1 Gradient Monitoring

```python
def log_gradients(model, step):
    total_norm = 0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    
    total_norm = total_norm ** 0.5
    wandb.log({"gradient_norm": total_norm}, step=step)
```

### 6.2 Loss Spike Recovery

```python
class TrainingStabilizer:
    def __init__(self, model, threshold=10.0):
        self.threshold = threshold
        self.best_loss = float('inf')
        self.best_state = None
        
    def check_and_recover(self, loss):
        if loss > self.threshold * self.best_loss:
            # Recover from checkpoint
            model.load_state_dict(self.best_state)
            return True
        
        if loss < self.best_loss:
            self.best_loss = loss
            self.best_state = copy.deepcopy(model.state_dict())
        
        return False
```

### 6.3 Learning Rate Finder

```python
def find_learning_rate(model, dataloader, init_lr=1e-8, end_lr=1):
    optimizer = torch.optim.AdamW(model.parameters(), lr=init_lr)
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=1.1)
    
    losses = []
    lrs = []
    
    for batch in dataloader:
        optimizer.zero_grad()
        
        outputs = model(**batch)
        loss = outputs.loss
        
        losses.append(loss.item())
        lrs.append(optimizer.param_groups[0]['lr'])
        
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        
        if optimizer.param_groups[0]['lr'] > end_lr:
            break
    
    return lrs, losses
```

## 7. Monitoring and Debugging

### 7.1 Key Metrics to Track

```python
metrics = {
    'loss': loss.item(),
    'learning_rate': optimizer.param_groups[0]['lr'],
    'gradient_norm': compute_gradient_norm(model),
    'weight_norm': compute_weight_norm(model),
    'gpu_memory': torch.cuda.max_memory_allocated() / 1024**3,
    'tokens_per_second': batch_size * seq_length / step_time
}
```

### 7.2 Attention Pattern Analysis

```python
def analyze_attention_patterns(model, dataloader):
    attention_stats = defaultdict(list)
    
    for batch in dataloader:
        with torch.no_grad():
            outputs = model(**batch, output_attentions=True)
            
        for layer_idx, attn in enumerate(outputs.attentions):
            # Average attention entropy
            entropy = -torch.sum(attn * torch.log(attn + 1e-9), dim=-1).mean()
            attention_stats[f'layer_{layer_idx}_entropy'].append(entropy.item())
            
            # Attention to special tokens
            cls_attention = attn[:, :, 0, :].mean()
            attention_stats[f'layer_{layer_idx}_cls_attention'].append(cls_attention.item())
```

### 7.3 Dead Neurons Detection

```python
def detect_dead_neurons(model):
    dead_neurons = {}
    
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            # Check for neurons that never activate
            weight_norm = module.weight.norm(dim=1)
            dead_mask = weight_norm < 1e-6
            
            if dead_mask.any():
                dead_neurons[name] = dead_mask.sum().item()
    
    return dead_neurons
```

## 8. Best Practices

### 8.1 Training Recipe

1. **Start small**: Debug with tiny model/data
2. **Overfit single batch**: Ensure model can learn
3. **Scale gradually**: Increase model/data size
4. **Monitor everything**: Loss, gradients, activations
5. **Save checkpoints**: Regular and best models
6. **Use validation**: Early stopping, hyperparameter selection

### 8.2 Hyperparameter Guidelines

| Parameter | Small Model | Base Model | Large Model |
|-----------|------------|------------|-------------|
| Learning Rate | 5e-4 | 3e-4 | 1e-4 |
| Batch Size | 32 | 256 | 2048+ |
| Warmup Steps | 500 | 2000 | 10000 |
| Weight Decay | 0.01 | 0.01 | 0.01 |
| Gradient Clip | 1.0 | 1.0 | 1.0 |

### 8.3 Common Issues and Solutions

| Issue | Symptoms | Solutions |
|-------|----------|-----------|
| Loss explosion | NaN loss, huge gradients | Lower LR, gradient clipping, check data |
| Slow convergence | Loss plateaus early | Increase LR, check initialization |
| Overfitting | Train good, val bad | Dropout, weight decay, more data |
| OOM errors | CUDA out of memory | Gradient checkpointing, smaller batch |
| Unstable training | Loss spikes | Learning rate warmup, better init |

## 📊 Training Efficiency Tips

1. **Use mixed precision**: 2x speedup, half memory
2. **Optimize data loading**: Multiple workers, prefetching
3. **Gradient accumulation**: Simulate large batches
4. **Efficient attention**: Flash Attention, sparse patterns
5. **Model parallelism**: For very large models

## 🔍 Advanced Techniques

### Curriculum Learning
```python
def curriculum_scheduler(epoch, max_length=512):
    # Gradually increase sequence length
    return min(128 + epoch * 32, max_length)
```

### Stochastic Depth
```python
class StochasticDepth(nn.Module):
    def __init__(self, module, drop_rate=0.1):
        super().__init__()
        self.module = module
        self.drop_rate = drop_rate
        
    def forward(self, x):
        if not self.training or random.random() > self.drop_rate:
            return self.module(x)
        return x  # Skip this layer
```

## 📝 Summary

Training transformers effectively requires:
- **Careful optimization**: Right optimizer, schedule, hyperparameters
- **Stability techniques**: Gradient clipping, warmup, monitoring
- **Efficiency methods**: Mixed precision, distributed training
- **Debugging tools**: Metrics, visualization, analysis
- **Best practices**: Start small, scale gradually, monitor everything

## ➡️ Next Steps

Ready to explore different transformer architectures? Head to [Topic 8: Transformer Variants](../08-transformer-variants/) to learn about BERT, GPT, T5, and more!