"""
The Transformer Architecture
Complete implementation of the transformer model from "Attention Is All You Need".
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from typing import Optional, Tuple


class MultiHeadAttention(nn.Module):
    """Multi-Head Attention mechanism."""
    
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        # Linear projections
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        
        # 1. Linear projections in batch from d_model => h x d_k
        Q = self.W_q(query).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        
        # 2. Apply attention on all the projected vectors in batch
        attn_output, attn_weights = self.scaled_dot_product_attention(Q, K, V, mask)
        
        # 3. Concatenate heads and put through final linear layer
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, -1, self.d_model
        )
        output = self.W_o(attn_output)
        
        return output, attn_weights
    
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        output = torch.matmul(attn_weights, V)
        
        return output, attn_weights


class PositionalEncoding(nn.Module):
    """Positional encoding using sine and cosine functions."""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        
        # Create div_term for the sinusoidal pattern
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           -(math.log(10000.0) / d_model))
        
        # Apply sin to even indices
        pe[:, 0::2] = torch.sin(position * div_term)
        # Apply cos to odd indices
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0)  # Add batch dimension
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        # Add positional encoding to input embeddings
        return x + self.pe[:, :x.size(1)]


class FeedForwardNetwork(nn.Module):
    """Position-wise feed-forward network."""
    
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()
        
    def forward(self, x):
        return self.linear2(self.dropout(self.activation(self.linear1(x))))


class EncoderLayer(nn.Module):
    """Single encoder layer."""
    
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.feed_forward = FeedForwardNetwork(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        # Self-attention with residual connection and layer norm
        attn_output, _ = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feed-forward with residual connection and layer norm
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x


class DecoderLayer(nn.Module):
    """Single decoder layer."""
    
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.feed_forward = FeedForwardNetwork(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, encoder_output, src_mask=None, tgt_mask=None):
        # Masked self-attention
        attn_output, _ = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Cross-attention to encoder output
        attn_output, _ = self.cross_attn(x, encoder_output, encoder_output, src_mask)
        x = self.norm2(x + self.dropout(attn_output))
        
        # Feed-forward
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))
        
        return x


class Transformer(nn.Module):
    """Complete Transformer model."""
    
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        n_heads: int = 8,
        n_encoder_layers: int = 6,
        n_decoder_layers: int = 6,
        d_ff: int = 2048,
        max_len: int = 5000,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.d_model = d_model
        
        # Token embeddings and positional encoding
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len)
        
        # Encoder
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_encoder_layers)
        ])
        
        # Decoder
        self.decoder_layers = nn.ModuleList([
            DecoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_decoder_layers)
        ])
        
        # Output projection
        self.output_projection = nn.Linear(d_model, vocab_size)
        
        self.dropout = nn.Dropout(dropout)
        
        # Initialize parameters
        self._init_parameters()
        
    def _init_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def generate_mask(self, sz: int) -> torch.Tensor:
        """Generate a causal mask for the decoder."""
        mask = torch.triu(torch.ones(sz, sz), diagonal=1).bool()
        return ~mask
    
    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        # Encode
        src_embeddings = self.dropout(self.pos_encoding(self.embedding(src) * math.sqrt(self.d_model)))
        encoder_output = src_embeddings
        for encoder_layer in self.encoder_layers:
            encoder_output = encoder_layer(encoder_output, src_mask)
        
        # Decode
        tgt_embeddings = self.dropout(self.pos_encoding(self.embedding(tgt) * math.sqrt(self.d_model)))
        decoder_output = tgt_embeddings
        for decoder_layer in self.decoder_layers:
            decoder_output = decoder_layer(decoder_output, encoder_output, src_mask, tgt_mask)
        
        # Project to vocabulary
        output = self.output_projection(decoder_output)
        
        return output


def create_padding_mask(seq, pad_idx=0):
    """Create a mask to hide padding tokens."""
    return (seq != pad_idx).unsqueeze(1).unsqueeze(2)


def create_look_ahead_mask(size):
    """Create a mask to hide future tokens (for decoder)."""
    mask = torch.triu(torch.ones(size, size), diagonal=1).bool()
    return ~mask


# Demonstration functions
def demonstrate_positional_encoding():
    """Visualize positional encodings."""
    import matplotlib.pyplot as plt
    
    d_model = 128
    max_len = 100
    
    pe = PositionalEncoding(d_model, max_len)
    encodings = pe.pe[0, :max_len, :].numpy()
    
    plt.figure(figsize=(12, 8))
    plt.imshow(encodings.T, aspect='auto', cmap='RdBu')
    plt.colorbar()
    plt.xlabel('Position')
    plt.ylabel('Dimension')
    plt.title('Positional Encodings')
    plt.show()
    
    # Show specific dimensions
    plt.figure(figsize=(12, 6))
    for i in range(0, 8, 2):
        plt.plot(encodings[:, i], label=f'dim {i}')
    plt.legend()
    plt.xlabel('Position')
    plt.ylabel('Encoding Value')
    plt.title('Positional Encoding for Different Dimensions')
    plt.show()


def demonstrate_attention_heads():
    """Show how different heads attend to different patterns."""
    # Create a simple transformer
    d_model = 64
    n_heads = 4
    seq_len = 10
    
    mha = MultiHeadAttention(d_model, n_heads)
    
    # Random input
    x = torch.randn(1, seq_len, d_model)
    
    # Get attention weights
    with torch.no_grad():
        output, attn_weights = mha(x, x, x)
    
    # Visualize each head
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(1, n_heads, figsize=(15, 4))
    for head in range(n_heads):
        ax = axes[head]
        weights = attn_weights[0, head].numpy()
        im = ax.imshow(weights, cmap='Blues')
        ax.set_title(f'Head {head+1}')
        ax.set_xlabel('Keys')
        ax.set_ylabel('Queries')
        plt.colorbar(im, ax=ax)
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    print("Transformer Architecture Implementation")
    print("=" * 50)
    
    # Create a small transformer
    vocab_size = 1000
    model = Transformer(
        vocab_size=vocab_size,
        d_model=128,
        n_heads=4,
        n_encoder_layers=2,
        n_decoder_layers=2,
        d_ff=512
    )
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    # Test forward pass
    src = torch.randint(0, vocab_size, (2, 10))  # [batch_size, seq_len]
    tgt = torch.randint(0, vocab_size, (2, 8))
    
    output = model(src, tgt)
    print(f"\nInput shapes: src={src.shape}, tgt={tgt.shape}")
    print(f"Output shape: {output.shape}")
    
    # Demonstrate components
    print("\nDemonstrating Positional Encoding...")
    demonstrate_positional_encoding()
    
    print("\nDemonstrating Multi-Head Attention...")
    demonstrate_attention_heads()