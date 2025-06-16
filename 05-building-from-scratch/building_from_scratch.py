"""
Building a Transformer from Scratch
Complete implementation with training, inference, and debugging utilities.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
from dataclasses import dataclass
from typing import Optional, Tuple, List
import json
import os


@dataclass
class TransformerConfig:
    """Configuration for Transformer model."""
    vocab_size: int = 10000
    d_model: int = 512
    n_heads: int = 8
    n_encoder_layers: int = 6
    n_decoder_layers: int = 6
    d_ff: int = 2048
    max_seq_length: int = 512
    dropout: float = 0.1
    attention_dropout: float = 0.1
    activation: str = "gelu"
    layer_norm_eps: float = 1e-6
    
    @property
    def d_k(self):
        assert self.d_model % self.n_heads == 0
        return self.d_model // self.n_heads
    
    def save(self, path):
        with open(path, 'w') as f:
            json.dump(self.__dict__, f, indent=2)
    
    @classmethod
    def load(cls, path):
        with open(path, 'r') as f:
            return cls(**json.load(f))


class ScaledDotProductAttention(nn.Module):
    """Scaled Dot-Product Attention mechanism."""
    
    def __init__(self, d_k: int, dropout: float = 0.1):
        super().__init__()
        self.d_k = d_k
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, Q, K, V, mask=None):
        # Q, K, V: [batch_size, n_heads, seq_len, d_k]
        batch_size = Q.size(0)
        n_heads = Q.size(1)
        
        # Calculate attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Apply softmax
        attention_weights = F.softmax(scores, dim=-1)
        
        # Apply dropout
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        context = torch.matmul(attention_weights, V)
        
        return context, attention_weights


class MultiHeadAttention(nn.Module):
    """Multi-Head Attention mechanism."""
    
    def __init__(self, config: TransformerConfig):
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
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        for module in [self.W_q, self.W_k, self.W_v, self.W_o]:
            nn.init.xavier_uniform_(module.weight)
            
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        seq_len = query.size(1)
        
        # 1. Linear projections in batch from d_model => h x d_k
        Q = self.W_q(query).view(batch_size, seq_len, self.n_heads, self.d_k)
        K = self.W_k(key).view(batch_size, -1, self.n_heads, self.d_k)
        V = self.W_v(value).view(batch_size, -1, self.n_heads, self.d_k)
        
        # 2. Transpose for attention calculation
        Q = Q.transpose(1, 2)  # [batch_size, n_heads, seq_len, d_k]
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)
        
        # 3. Apply attention
        context, attention_weights = self.attention(Q, K, V, mask)
        
        # 4. Concatenate heads
        context = context.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.d_model
        )
        
        # 5. Final linear projection
        output = self.W_o(context)
        output = self.dropout(output)
        
        return output, attention_weights


class PositionwiseFeedForward(nn.Module):
    """Position-wise Feed-Forward Network."""
    
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.linear1 = nn.Linear(config.d_model, config.d_ff)
        self.linear2 = nn.Linear(config.d_ff, config.d_model)
        self.dropout = nn.Dropout(config.dropout)
        
        # Select activation function
        if config.activation == "relu":
            self.activation = nn.ReLU()
        elif config.activation == "gelu":
            self.activation = nn.GELU()
        else:
            raise ValueError(f"Unknown activation: {config.activation}")
            
    def forward(self, x):
        return self.linear2(self.dropout(self.activation(self.linear1(x))))


class PositionalEncoding(nn.Module):
    """Positional Encoding using sine and cosine functions."""
    
    def __init__(self, config: TransformerConfig):
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


class LayerNorm(nn.Module):
    """Layer Normalization."""
    
    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        
    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta


class EncoderLayer(nn.Module):
    """Single Encoder Layer."""
    
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.self_attention = MultiHeadAttention(config)
        self.feed_forward = PositionwiseFeedForward(config)
        self.norm1 = LayerNorm(config.d_model, config.layer_norm_eps)
        self.norm2 = LayerNorm(config.d_model, config.layer_norm_eps)
        self.dropout = nn.Dropout(config.dropout)
        
    def forward(self, x, mask=None):
        # Pre-norm architecture
        # Self-attention with residual connection
        attn_output, _ = self.self_attention(
            self.norm1(x), self.norm1(x), self.norm1(x), mask
        )
        x = x + self.dropout(attn_output)
        
        # Feed-forward with residual connection
        ff_output = self.feed_forward(self.norm2(x))
        x = x + self.dropout(ff_output)
        
        return x


class DecoderLayer(nn.Module):
    """Single Decoder Layer."""
    
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.masked_self_attention = MultiHeadAttention(config)
        self.cross_attention = MultiHeadAttention(config)
        self.feed_forward = PositionwiseFeedForward(config)
        self.norm1 = LayerNorm(config.d_model, config.layer_norm_eps)
        self.norm2 = LayerNorm(config.d_model, config.layer_norm_eps)
        self.norm3 = LayerNorm(config.d_model, config.layer_norm_eps)
        self.dropout = nn.Dropout(config.dropout)
        
    def forward(self, x, encoder_output, src_mask=None, tgt_mask=None):
        # Masked self-attention
        norm_x = self.norm1(x)
        attn_output, _ = self.masked_self_attention(norm_x, norm_x, norm_x, tgt_mask)
        x = x + self.dropout(attn_output)
        
        # Cross-attention
        norm_x = self.norm2(x)
        attn_output, attention_weights = self.cross_attention(
            norm_x, encoder_output, encoder_output, src_mask
        )
        x = x + self.dropout(attn_output)
        
        # Feed-forward
        ff_output = self.feed_forward(self.norm3(x))
        x = x + self.dropout(ff_output)
        
        return x, attention_weights


class Transformer(nn.Module):
    """Complete Transformer Model."""
    
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        
        # Embeddings
        self.src_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.tgt_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.positional_encoding = PositionalEncoding(config)
        
        # Encoder
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(config) for _ in range(config.n_encoder_layers)
        ])
        self.encoder_norm = LayerNorm(config.d_model, config.layer_norm_eps)
        
        # Decoder
        self.decoder_layers = nn.ModuleList([
            DecoderLayer(config) for _ in range(config.n_decoder_layers)
        ])
        self.decoder_norm = LayerNorm(config.d_model, config.layer_norm_eps)
        
        # Output projection
        self.output_projection = nn.Linear(config.d_model, config.vocab_size)
        
        # Initialize parameters
        self._init_parameters()
        
    def _init_parameters(self):
        """Initialize parameters with Xavier uniform."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
                
    def generate_square_subsequent_mask(self, sz: int) -> torch.Tensor:
        """Generate a square mask for the sequence."""
        mask = torch.triu(torch.ones(sz, sz), diagonal=1).bool()
        return ~mask
    
    def encode(self, src: torch.Tensor, src_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Encode source sequence."""
        # Embed and add positional encoding
        x = self.src_embedding(src) * math.sqrt(self.config.d_model)
        x = self.positional_encoding(x)
        
        # Pass through encoder layers
        for layer in self.encoder_layers:
            x = layer(x, src_mask)
        
        # Final layer norm
        x = self.encoder_norm(x)
        
        return x
    
    def decode(self, tgt: torch.Tensor, encoder_output: torch.Tensor,
               src_mask: Optional[torch.Tensor] = None,
               tgt_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Decode target sequence."""
        # Embed and add positional encoding
        x = self.tgt_embedding(tgt) * math.sqrt(self.config.d_model)
        x = self.positional_encoding(x)
        
        # Pass through decoder layers
        attention_weights = []
        for layer in self.decoder_layers:
            x, attn_weights = layer(x, encoder_output, src_mask, tgt_mask)
            attention_weights.append(attn_weights)
        
        # Final layer norm
        x = self.decoder_norm(x)
        
        return x, attention_weights
    
    def forward(self, src: torch.Tensor, tgt: torch.Tensor,
                src_mask: Optional[torch.Tensor] = None,
                tgt_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Forward pass through the transformer."""
        encoder_output = self.encode(src, src_mask)
        decoder_output, attention_weights = self.decode(
            tgt, encoder_output, src_mask, tgt_mask
        )
        output = self.output_projection(decoder_output)
        return output, attention_weights


# Utility functions
def create_padding_mask(seq: torch.Tensor, pad_idx: int = 0) -> torch.Tensor:
    """Create a mask to hide padding tokens."""
    return (seq != pad_idx).unsqueeze(1).unsqueeze(2)


def create_look_ahead_mask(size: int, device: torch.device) -> torch.Tensor:
    """Create a mask to prevent attending to future positions."""
    mask = torch.triu(torch.ones(size, size, device=device), diagonal=1).bool()
    return ~mask


def create_masks(src: torch.Tensor, tgt: torch.Tensor, 
                pad_idx: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
    """Create all masks needed for training."""
    # Source mask: hide padding
    src_mask = create_padding_mask(src, pad_idx)
    
    # Target mask: hide padding + future
    tgt_seq_len = tgt.size(1)
    tgt_padding_mask = create_padding_mask(tgt, pad_idx)
    tgt_look_ahead_mask = create_look_ahead_mask(tgt_seq_len, tgt.device)
    tgt_mask = tgt_padding_mask & tgt_look_ahead_mask.unsqueeze(0).unsqueeze(0)
    
    return src_mask, tgt_mask


# Training utilities
class TransformerScheduler:
    """Learning rate scheduler for Transformer."""
    
    def __init__(self, optimizer, d_model: int, warmup_steps: int = 4000):
        self.optimizer = optimizer
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.step_num = 0
        
    def step(self):
        """Update learning rate."""
        self.step_num += 1
        lr = self.get_lr()
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
            
    def get_lr(self) -> float:
        """Calculate learning rate."""
        return self.d_model ** (-0.5) * min(
            self.step_num ** (-0.5),
            self.step_num * self.warmup_steps ** (-1.5)
        )


class LabelSmoothingLoss(nn.Module):
    """Label smoothing loss."""
    
    def __init__(self, vocab_size: int, padding_idx: int = 0, smoothing: float = 0.1):
        super().__init__()
        self.criterion = nn.KLDivLoss(reduction='sum')
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.vocab_size = vocab_size
        
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
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


# Inference utilities
def greedy_decode(model: Transformer, src: torch.Tensor, src_mask: torch.Tensor,
                  max_length: int, sos_idx: int, eos_idx: int,
                  device: torch.device) -> torch.Tensor:
    """Greedy decoding."""
    model.eval()
    
    # Encode source
    encoder_output = model.encode(src, src_mask)
    
    # Initialize decoder input
    tgt = torch.LongTensor([[sos_idx]]).to(device)
    
    for _ in range(max_length):
        # Create target mask
        tgt_mask = create_look_ahead_mask(tgt.size(1), device)
        
        # Decode
        decoder_output, _ = model.decode(tgt, encoder_output, src_mask, tgt_mask)
        
        # Get next token
        next_token_logits = model.output_projection(decoder_output[:, -1, :])
        next_token = next_token_logits.argmax(dim=-1).unsqueeze(0)
        
        # Append to target sequence
        tgt = torch.cat([tgt, next_token], dim=1)
        
        # Stop if EOS
        if next_token.item() == eos_idx:
            break
            
    return tgt


class BeamSearchDecoder:
    """Beam search decoder."""
    
    def __init__(self, model: Transformer, beam_size: int = 4, max_length: int = 100):
        self.model = model
        self.beam_size = beam_size
        self.max_length = max_length
        
    def decode(self, src: torch.Tensor, src_mask: torch.Tensor,
               sos_idx: int, eos_idx: int, device: torch.device) -> torch.Tensor:
        """Perform beam search decoding."""
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
                tgt_mask = create_look_ahead_mask(seq.size(1), device)
                
                # Decode
                decoder_output, _ = self.model.decode(
                    seq, encoder_output, src_mask, tgt_mask
                )
                
                # Get probabilities
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
            
            if len(beams) == 0:
                break
                
        # Return best sequence
        if completed:
            completed.sort(key=lambda x: x[1] / x[0].size(1), reverse=True)
            return completed[0][0]
        else:
            return beams[0][0]


# Visualization utilities
def visualize_attention(attention_weights: torch.Tensor, src_tokens: List[str],
                       tgt_tokens: List[str], layer_idx: int = 0,
                       head_idx: int = 0):
    """Visualize attention weights."""
    # Extract specific layer and head
    attn = attention_weights[layer_idx][0, head_idx].cpu().numpy()
    
    plt.figure(figsize=(10, 8))
    plt.imshow(attn, cmap='Blues', aspect='auto')
    plt.colorbar()
    
    # Set labels
    plt.xticks(range(len(src_tokens)), src_tokens, rotation=45, ha='right')
    plt.yticks(range(len(tgt_tokens)), tgt_tokens)
    
    plt.xlabel('Source')
    plt.ylabel('Target')
    plt.title(f'Attention Weights - Layer {layer_idx}, Head {head_idx}')
    plt.tight_layout()
    plt.show()


def plot_learning_rate_schedule(d_model: int = 512, warmup_steps: int = 4000,
                               max_steps: int = 100000):
    """Plot the learning rate schedule."""
    steps = np.arange(1, max_steps + 1)
    lrs = []
    
    for step in steps:
        lr = d_model ** (-0.5) * min(
            step ** (-0.5),
            step * warmup_steps ** (-1.5)
        )
        lrs.append(lr)
    
    plt.figure(figsize=(10, 6))
    plt.plot(steps, lrs)
    plt.xlabel('Training Steps')
    plt.ylabel('Learning Rate')
    plt.title('Transformer Learning Rate Schedule')
    plt.grid(True, alpha=0.3)
    plt.show()


# Demo functions
def create_toy_dataset():
    """Create a toy dataset for demonstration."""
    # Simple copy task
    src_sentences = [
        "1 2 3 4 5",
        "6 7 8 9",
        "10 11 12",
        "13 14 15 16 17 18"
    ]
    
    # Target is the same (copy task)
    tgt_sentences = src_sentences
    
    return src_sentences, tgt_sentences


def demonstrate_transformer():
    """Demonstrate the transformer on a toy task."""
    print("=== Transformer Demonstration ===\n")
    
    # Configuration
    config = TransformerConfig(
        vocab_size=20,
        d_model=128,
        n_heads=4,
        n_encoder_layers=2,
        n_decoder_layers=2,
        d_ff=256,
        max_seq_length=20
    )
    
    # Create model
    model = Transformer(config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params:,}")
    
    # Create toy data
    batch_size = 2
    seq_len = 10
    src = torch.randint(1, config.vocab_size, (batch_size, seq_len)).to(device)
    tgt = torch.randint(1, config.vocab_size, (batch_size, seq_len)).to(device)
    
    # Create masks
    src_mask, tgt_mask = create_masks(src, tgt)
    
    # Forward pass
    output, attention_weights = model(src, tgt, src_mask, tgt_mask)
    
    print(f"\nInput shapes:")
    print(f"  Source: {src.shape}")
    print(f"  Target: {tgt.shape}")
    print(f"  Source mask: {src_mask.shape}")
    print(f"  Target mask: {tgt_mask.shape}")
    
    print(f"\nOutput shape: {output.shape}")
    print(f"Number of attention weight matrices: {len(attention_weights)}")
    
    # Demonstrate positional encoding
    print("\n=== Positional Encoding ===")
    pe = PositionalEncoding(config)
    pos_encoding = pe.pe[0, :20, :].numpy()
    
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(pos_encoding.T, aspect='auto', cmap='RdBu')
    plt.colorbar()
    plt.xlabel('Position')
    plt.ylabel('Dimension')
    plt.title('Positional Encoding Heatmap')
    
    plt.subplot(1, 2, 2)
    dims = [0, 1, 2, 3]
    for d in dims:
        plt.plot(pos_encoding[:, d], label=f'dim {d}')
    plt.xlabel('Position')
    plt.ylabel('Encoding Value')
    plt.title('Positional Encoding for Different Dimensions')
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    # Demonstrate learning rate schedule
    print("\n=== Learning Rate Schedule ===")
    plot_learning_rate_schedule(config.d_model, warmup_steps=4000, max_steps=20000)
    
    print("\n✅ Transformer successfully created and tested!")


if __name__ == "__main__":
    # Run demonstration
    demonstrate_transformer()
    
    # Show model architecture
    print("\n=== Model Architecture ===")
    config = TransformerConfig(vocab_size=10000)
    model = Transformer(config)
    print(model)