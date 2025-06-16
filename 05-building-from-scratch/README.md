# Building a Transformer from Scratch

Welcome to the hands-on implementation! In this module, we'll build a complete transformer model from the ground up, train it on a real task, and understand every single component.

## 🎯 Learning Objectives

By the end of this module, you will:
- Implement a complete transformer from scratch
- Understand every component in detail
- Train on a translation task
- Debug common issues
- Optimize performance
- Deploy for inference

## 📚 Table of Contents

1. [Project Setup](#1-project-setup)
2. [Data Preparation](#2-data-preparation)
3. [Building the Components](#3-building-the-components)
4. [Assembling the Transformer](#4-assembling-the-transformer)
5. [Training Loop](#5-training-loop)
6. [Inference and Generation](#6-inference-and-generation)
7. [Debugging and Visualization](#7-debugging-and-visualization)
8. [Performance Optimization](#8-performance-optimization)
9. [Common Pitfalls](#9-common-pitfalls)

## 1. Project Setup

### 1.1 Dependencies

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
```

### 1.2 Configuration

```python
@dataclass
class TransformerConfig:
    vocab_size: int = 10000
    d_model: int = 512
    n_heads: int = 8
    n_layers: int = 6
    d_ff: int = 2048
    max_seq_length: int = 512
    dropout: float = 0.1
    attention_dropout: float = 0.1
    
    # Computed properties
    @property
    def d_k(self):
        return self.d_model // self.n_heads
```

## 2. Data Preparation

### 2.1 Dataset Class

```python
class TranslationDataset(Dataset):
    def __init__(self, src_texts, tgt_texts, src_vocab, tgt_vocab, max_length=512):
        self.src_texts = src_texts
        self.tgt_texts = tgt_texts
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.max_length = max_length
        
    def __len__(self):
        return len(self.src_texts)
    
    def __getitem__(self, idx):
        src = self.encode(self.src_texts[idx], self.src_vocab)
        tgt = self.encode(self.tgt_texts[idx], self.tgt_vocab)
        
        return {
            'src': src,
            'tgt_input': tgt[:-1],  # All but last token
            'tgt_output': tgt[1:]   # All but first token
        }
```

### 2.2 Tokenization and Vocabulary

```python
class SimpleTokenizer:
    def __init__(self, vocab_size=10000):
        self.vocab_size = vocab_size
        self.word_to_id = {'<pad>': 0, '<sos>': 1, '<eos>': 2, '<unk>': 3}
        self.id_to_word = {v: k for k, v in self.word_to_id.items()}
        self.next_id = 4
        
    def fit(self, texts):
        # Build vocabulary from texts
        word_counts = {}
        for text in texts:
            for word in text.split():
                word_counts[word] = word_counts.get(word, 0) + 1
        
        # Keep most frequent words
        sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
        for word, _ in sorted_words[:self.vocab_size - 4]:
            if word not in self.word_to_id:
                self.word_to_id[word] = self.next_id
                self.id_to_word[self.next_id] = word
                self.next_id += 1
```

## 3. Building the Components

### 3.1 Scaled Dot-Product Attention

```python
class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_k, dropout=0.1):
        super().__init__()
        self.d_k = d_k
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, Q, K, V, mask=None):
        # Q, K, V: [batch_size, n_heads, seq_len, d_k]
        batch_size = Q.size(0)
        
        # Calculate scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # Apply mask
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Apply softmax
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        context = torch.matmul(attention_weights, V)
        
        return context, attention_weights
```

### 3.2 Multi-Head Attention

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.d_model % config.n_heads == 0
        
        self.d_model = config.d_model
        self.n_heads = config.n_heads
        self.d_k = config.d_model // config.n_heads
        
        # Linear projections
        self.W_q = nn.Linear(config.d_model, config.d_model, bias=False)
        self.W_k = nn.Linear(config.d_model, config.d_model, bias=False)
        self.W_v = nn.Linear(config.d_model, config.d_model, bias=False)
        self.W_o = nn.Linear(config.d_model, config.d_model, bias=False)
        
        self.attention = ScaledDotProductAttention(self.d_k, config.attention_dropout)
        self.dropout = nn.Dropout(config.dropout)
        
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        seq_len = query.size(1)
        
        # 1. Linear projections in batch from d_model => h x d_k
        Q = self.W_q(query).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        
        # 2. Apply attention
        context, attention_weights = self.attention(Q, K, V, mask)
        
        # 3. Concatenate heads
        context = context.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.d_model
        )
        
        # 4. Final linear projection
        output = self.W_o(context)
        output = self.dropout(output)
        
        return output, attention_weights
```

### 3.3 Position-wise Feed-Forward Network

```python
class PositionwiseFeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.linear1 = nn.Linear(config.d_model, config.d_ff)
        self.linear2 = nn.Linear(config.d_ff, config.d_model)
        self.dropout = nn.Dropout(config.dropout)
        self.activation = nn.GELU()
        
    def forward(self, x):
        # FFN(x) = max(0, xW1 + b1)W2 + b2
        return self.linear2(self.dropout(self.activation(self.linear1(x))))
```

### 3.4 Positional Encoding

```python
class PositionalEncoding(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.d_model = config.d_model
        self.dropout = nn.Dropout(config.dropout)
        
        # Create positional encoding matrix
        pe = torch.zeros(config.max_seq_length, config.d_model)
        position = torch.arange(0, config.max_seq_length).unsqueeze(1).float()
        
        # Create div_term
        div_term = torch.exp(torch.arange(0, config.d_model, 2).float() * 
                           -(math.log(10000.0) / config.d_model))
        
        # Apply sine to even indices
        pe[:, 0::2] = torch.sin(position * div_term)
        # Apply cosine to odd indices
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Add batch dimension and register as buffer
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        # x: [batch_size, seq_len, d_model]
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len, :]
        return self.dropout(x)
```

### 3.5 Layer Normalization

```python
class LayerNorm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        
    def forward(self, x):
        # x: [batch_size, seq_len, d_model]
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta
```

## 4. Assembling the Transformer

### 4.1 Encoder Layer

```python
class EncoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self_attention = MultiHeadAttention(config)
        self.feed_forward = PositionwiseFeedForward(config)
        self.norm1 = LayerNorm(config.d_model)
        self.norm2 = LayerNorm(config.d_model)
        self.dropout = nn.Dropout(config.dropout)
        
    def forward(self, x, mask=None):
        # Self-attention with residual connection
        attn_output, _ = self.self_attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feed-forward with residual connection
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x
```

### 4.2 Decoder Layer

```python
class DecoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.masked_self_attention = MultiHeadAttention(config)
        self.cross_attention = MultiHeadAttention(config)
        self.feed_forward = PositionwiseFeedForward(config)
        self.norm1 = LayerNorm(config.d_model)
        self.norm2 = LayerNorm(config.d_model)
        self.norm3 = LayerNorm(config.d_model)
        self.dropout = nn.Dropout(config.dropout)
        
    def forward(self, x, encoder_output, src_mask=None, tgt_mask=None):
        # Masked self-attention
        attn_output, _ = self.masked_self_attention(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Cross-attention
        attn_output, attention_weights = self.cross_attention(
            x, encoder_output, encoder_output, src_mask
        )
        x = self.norm2(x + self.dropout(attn_output))
        
        # Feed-forward
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))
        
        return x, attention_weights
```

### 4.3 Complete Transformer

```python
class Transformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Embeddings
        self.src_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.tgt_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.positional_encoding = PositionalEncoding(config)
        
        # Encoder
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(config) for _ in range(config.n_layers)
        ])
        
        # Decoder
        self.decoder_layers = nn.ModuleList([
            DecoderLayer(config) for _ in range(config.n_layers)
        ])
        
        # Output projection
        self.output_projection = nn.Linear(config.d_model, config.vocab_size)
        
        # Initialize parameters
        self._init_parameters()
        
    def _init_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
                
    def encode(self, src, src_mask=None):
        # Embed and add positional encoding
        x = self.src_embedding(src) * math.sqrt(self.config.d_model)
        x = self.positional_encoding(x)
        
        # Pass through encoder layers
        for layer in self.encoder_layers:
            x = layer(x, src_mask)
            
        return x
    
    def decode(self, tgt, encoder_output, src_mask=None, tgt_mask=None):
        # Embed and add positional encoding
        x = self.tgt_embedding(tgt) * math.sqrt(self.config.d_model)
        x = self.positional_encoding(x)
        
        # Pass through decoder layers
        attention_weights = []
        for layer in self.decoder_layers:
            x, attn_weights = layer(x, encoder_output, src_mask, tgt_mask)
            attention_weights.append(attn_weights)
            
        return x, attention_weights
    
    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        encoder_output = self.encode(src, src_mask)
        decoder_output, attention_weights = self.decode(
            tgt, encoder_output, src_mask, tgt_mask
        )
        output = self.output_projection(decoder_output)
        return output, attention_weights
```

## 5. Training Loop

### 5.1 Loss and Optimizer

```python
def create_masks(src, tgt, pad_idx=0):
    # Source mask: hide padding tokens
    src_mask = (src != pad_idx).unsqueeze(1).unsqueeze(2)
    
    # Target mask: hide padding + future tokens
    tgt_seq_len = tgt.size(1)
    tgt_mask = (tgt != pad_idx).unsqueeze(1).unsqueeze(3)
    nopeak_mask = torch.triu(torch.ones(1, 1, tgt_seq_len, tgt_seq_len), diagonal=1).bool()
    tgt_mask = tgt_mask & ~nopeak_mask
    
    return src_mask, tgt_mask

class LabelSmoothingLoss(nn.Module):
    def __init__(self, vocab_size, padding_idx=0, smoothing=0.1):
        super().__init__()
        self.criterion = nn.KLDivLoss(reduction='sum')
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.vocab_size = vocab_size
        
    def forward(self, pred, target):
        pred = pred.log_softmax(dim=-1)
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.vocab_size - 2))
            true_dist.scatter_(1, target.unsqueeze(1), self.confidence)
            true_dist[:, self.padding_idx] = 0
            mask = torch.nonzero(target == self.padding_idx)
            if mask.dim() > 0:
                true_dist.index_fill_(0, mask.squeeze(), 0.0)
        return self.criterion(pred, true_dist)
```

### 5.2 Training Function

```python
def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    total_tokens = 0
    
    progress_bar = tqdm(dataloader, desc='Training')
    for batch in progress_bar:
        src = batch['src'].to(device)
        tgt_input = batch['tgt_input'].to(device)
        tgt_output = batch['tgt_output'].to(device)
        
        # Create masks
        src_mask, tgt_mask = create_masks(src, tgt_input)
        
        # Forward pass
        output, _ = model(src, tgt_input, src_mask, tgt_mask)
        
        # Reshape for loss calculation
        output = output.reshape(-1, output.size(-1))
        tgt_output = tgt_output.reshape(-1)
        
        # Calculate loss
        loss = criterion(output, tgt_output)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # Update statistics
        total_loss += loss.item()
        total_tokens += (tgt_output != 0).sum().item()
        
        # Update progress bar
        progress_bar.set_postfix({'loss': loss.item()})
    
    return total_loss / total_tokens
```

### 5.3 Learning Rate Schedule

```python
class TransformerScheduler:
    def __init__(self, optimizer, d_model, warmup_steps=4000):
        self.optimizer = optimizer
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.step_num = 0
        
    def step(self):
        self.step_num += 1
        lr = self.get_lr()
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
            
    def get_lr(self):
        return self.d_model ** (-0.5) * min(
            self.step_num ** (-0.5),
            self.step_num * self.warmup_steps ** (-1.5)
        )
```

## 6. Inference and Generation

### 6.1 Greedy Decoding

```python
def greedy_decode(model, src, src_mask, max_length, sos_idx, eos_idx, device):
    model.eval()
    
    # Encode source
    encoder_output = model.encode(src, src_mask)
    
    # Initialize decoder input with SOS
    tgt = torch.LongTensor([[sos_idx]]).to(device)
    
    for _ in range(max_length):
        # Create target mask
        tgt_mask = torch.triu(torch.ones(1, 1, tgt.size(1), tgt.size(1)), diagonal=1).bool()
        tgt_mask = ~tgt_mask.to(device)
        
        # Decode
        decoder_output, _ = model.decode(tgt, encoder_output, src_mask, tgt_mask)
        
        # Get next token
        next_token_logits = model.output_projection(decoder_output[:, -1, :])
        next_token = next_token_logits.argmax(dim=-1).unsqueeze(0)
        
        # Append to target sequence
        tgt = torch.cat([tgt, next_token], dim=1)
        
        # Stop if EOS is generated
        if next_token.item() == eos_idx:
            break
            
    return tgt
```

### 6.2 Beam Search

```python
class BeamSearchDecoder:
    def __init__(self, model, beam_size=4, max_length=100):
        self.model = model
        self.beam_size = beam_size
        self.max_length = max_length
        
    def decode(self, src, src_mask, sos_idx, eos_idx, device):
        self.model.eval()
        
        # Encode source
        encoder_output = self.model.encode(src, src_mask)
        
        # Initialize beams
        beams = [(torch.LongTensor([[sos_idx]]).to(device), 0.0)]
        completed = []
        
        for _ in range(self.max_length):
            new_beams = []
            
            for seq, score in beams:
                if seq[0, -1].item() == eos_idx:
                    completed.append((seq, score))
                    continue
                
                # Create mask
                tgt_mask = self._create_tgt_mask(seq.size(1), device)
                
                # Decode
                decoder_output, _ = self.model.decode(
                    seq, encoder_output, src_mask, tgt_mask
                )
                
                # Get probabilities for next token
                next_token_logits = self.model.output_projection(decoder_output[:, -1, :])
                next_token_probs = F.log_softmax(next_token_logits, dim=-1)
                
                # Get top k tokens
                top_probs, top_indices = next_token_probs.topk(self.beam_size)
                
                # Create new beams
                for i in range(self.beam_size):
                    new_seq = torch.cat([seq, top_indices[:, i:i+1]], dim=1)
                    new_score = score + top_probs[0, i].item()
                    new_beams.append((new_seq, new_score))
            
            # Select top beams
            beams = sorted(new_beams, key=lambda x: x[1], reverse=True)[:self.beam_size]
            
            # Early stopping if all beams completed
            if len(beams) == 0:
                break
                
        # Return best completed sequence
        if completed:
            completed.sort(key=lambda x: x[1] / x[0].size(1), reverse=True)  # Length normalization
            return completed[0][0]
        else:
            return beams[0][0]
```

## 7. Debugging and Visualization

### 7.1 Attention Visualization

```python
def visualize_attention(model, src, tgt, src_tokens, tgt_tokens, layer_idx=0, head_idx=0):
    model.eval()
    
    # Get attention weights
    with torch.no_grad():
        src_mask, tgt_mask = create_masks(src, tgt)
        output, attention_weights = model(src, tgt, src_mask, tgt_mask)
    
    # Extract specific layer and head
    attn = attention_weights[layer_idx][0, head_idx].cpu().numpy()
    
    # Plot
    plt.figure(figsize=(10, 8))
    plt.imshow(attn, cmap='Blues')
    plt.colorbar()
    
    # Set labels
    plt.xticks(range(len(src_tokens)), src_tokens, rotation=45)
    plt.yticks(range(len(tgt_tokens)), tgt_tokens)
    
    plt.xlabel('Source')
    plt.ylabel('Target')
    plt.title(f'Attention Weights - Layer {layer_idx}, Head {head_idx}')
    plt.tight_layout()
    plt.show()
```

### 7.2 Gradient Flow Analysis

```python
def analyze_gradient_flow(model):
    ave_grads = []
    max_grads = []
    layers = []
    
    for n, p in model.named_parameters():
        if p.requires_grad and "bias" not in n:
            layers.append(n)
            ave_grads.append(p.grad.abs().mean().cpu().item())
            max_grads.append(p.grad.abs().max().cpu().item())
    
    plt.figure(figsize=(12, 6))
    plt.bar(np.arange(len(ave_grads)), ave_grads, alpha=0.5, label='mean')
    plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.5, label='max')
    plt.hlines(0, 0, len(ave_grads)+1, lw=2, color="k")
    plt.xticks(range(len(ave_grads)), layers, rotation=90)
    plt.xlim(left=0, right=len(ave_grads))
    plt.ylim(bottom=-0.001, top=0.02)
    plt.xlabel("Layers")
    plt.ylabel("Average gradient")
    plt.title("Gradient flow")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
```

## 8. Performance Optimization

### 8.1 Mixed Precision Training

```python
from torch.cuda.amp import autocast, GradScaler

def train_with_amp(model, dataloader, optimizer, criterion, device):
    scaler = GradScaler()
    model.train()
    
    for batch in dataloader:
        src = batch['src'].to(device)
        tgt_input = batch['tgt_input'].to(device)
        tgt_output = batch['tgt_output'].to(device)
        
        src_mask, tgt_mask = create_masks(src, tgt_input)
        
        optimizer.zero_grad()
        
        # Mixed precision forward pass
        with autocast():
            output, _ = model(src, tgt_input, src_mask, tgt_mask)
            output = output.reshape(-1, output.size(-1))
            tgt_output = tgt_output.reshape(-1)
            loss = criterion(output, tgt_output)
        
        # Scaled backward pass
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        scaler.step(optimizer)
        scaler.update()
```

### 8.2 Gradient Accumulation

```python
def train_with_gradient_accumulation(model, dataloader, optimizer, criterion, 
                                    device, accumulation_steps=4):
    model.train()
    optimizer.zero_grad()
    
    for i, batch in enumerate(dataloader):
        src = batch['src'].to(device)
        tgt_input = batch['tgt_input'].to(device)
        tgt_output = batch['tgt_output'].to(device)
        
        src_mask, tgt_mask = create_masks(src, tgt_input)
        
        output, _ = model(src, tgt_input, src_mask, tgt_mask)
        output = output.reshape(-1, output.size(-1))
        tgt_output = tgt_output.reshape(-1)
        
        loss = criterion(output, tgt_output)
        loss = loss / accumulation_steps
        loss.backward()
        
        if (i + 1) % accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()
```

## 9. Common Pitfalls and Solutions

### 9.1 Attention Mask Mistakes

```python
# Wrong: Mask not properly broadcasted
# mask = torch.zeros(seq_len, seq_len)  # Missing batch and head dimensions

# Correct: Proper mask shape
mask = torch.zeros(batch_size, 1, 1, seq_len)  # Broadcasts to [batch, heads, seq, seq]
```

### 9.2 Position Encoding Issues

```python
# Wrong: Forgetting to scale embeddings
# x = self.embedding(x) + self.pos_encoding(x)

# Correct: Scale embeddings by sqrt(d_model)
x = self.embedding(x) * math.sqrt(self.d_model)
x = self.pos_encoding(x)
```

### 9.3 Learning Rate Problems

```python
# Wrong: Fixed learning rate
# optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Correct: Use warmup schedule
optimizer = torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9)
scheduler = TransformerScheduler(optimizer, d_model=512, warmup_steps=4000)
```

### 9.4 Memory Issues

```python
# Use gradient checkpointing for large models
from torch.utils.checkpoint import checkpoint

class CheckpointedEncoderLayer(nn.Module):
    def forward(self, x, mask=None):
        # Checkpoint self-attention
        attn_output = checkpoint(self.self_attention, x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Checkpoint feed-forward
        ff_output = checkpoint(self.feed_forward, x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x
```

## 🎯 Complete Training Script

```python
def train_transformer(config, train_data, valid_data, epochs=10):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create model
    model = Transformer(config).to(device)
    
    # Create optimizer and scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9)
    scheduler = TransformerScheduler(optimizer, config.d_model)
    
    # Create loss function
    criterion = LabelSmoothingLoss(config.vocab_size, padding_idx=0, smoothing=0.1)
    
    # Training loop
    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")
        
        # Train
        train_loss = train_epoch(model, train_data, optimizer, criterion, device)
        print(f"Train Loss: {train_loss:.4f}")
        
        # Validate
        valid_loss = evaluate(model, valid_data, criterion, device)
        print(f"Valid Loss: {valid_loss:.4f}")
        
        # Step scheduler
        scheduler.step()
        
        # Save checkpoint
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': train_loss,
        }, f'checkpoint_epoch_{epoch}.pt')
```

## 📊 Results and Analysis

Training a transformer from scratch teaches you:
1. **Component interactions**: How each piece fits together
2. **Hyperparameter sensitivity**: Small changes have big effects
3. **Debugging skills**: Common failure modes and fixes
4. **Performance considerations**: Memory and computation trade-offs

## 🔍 Key Takeaways

1. **Attention is the core**: Everything else supports attention computation
2. **Residual connections are crucial**: They enable deep networks
3. **Proper initialization matters**: Xavier/He initialization prevents gradient issues
4. **Learning rate scheduling is essential**: Warmup prevents early instability
5. **Debugging tools are invaluable**: Visualize attention and gradients

## 📝 Summary

You've built a complete transformer from scratch! This implementation covers:
- All core components with detailed explanations
- Training pipeline with modern techniques
- Inference methods (greedy and beam search)
- Debugging and visualization tools
- Performance optimizations

## ➡️ Next Steps

Ready to dive into tokenization? Head to [Topic 6: Tokenization and Embeddings](../06-tokenization-embeddings/) to learn about the crucial first step in any NLP pipeline!