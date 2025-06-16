"""
Transformer Variants
Implementation of various transformer architectures including BERT, GPT, T5, and Vision Transformers.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from typing import Optional, Tuple, Dict, List, Union
from dataclasses import dataclass
import matplotlib.pyplot as plt
from einops import rearrange, repeat
from einops.layers.torch import Rearrange


# Configuration classes
@dataclass
class BERTConfig:
    """Configuration for BERT model."""
    vocab_size: int = 30522
    hidden_size: int = 768
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    intermediate_size: int = 3072
    hidden_act: str = "gelu"
    hidden_dropout_prob: float = 0.1
    attention_probs_dropout_prob: float = 0.1
    max_position_embeddings: int = 512
    type_vocab_size: int = 2
    initializer_range: float = 0.02
    layer_norm_eps: float = 1e-12


@dataclass
class GPTConfig:
    """Configuration for GPT model."""
    vocab_size: int = 50257
    n_positions: int = 1024
    n_embd: int = 768
    n_layer: int = 12
    n_head: int = 12
    n_inner: int = 3072
    activation_function: str = "gelu_new"
    resid_pdrop: float = 0.1
    embd_pdrop: float = 0.1
    attn_pdrop: float = 0.1
    layer_norm_epsilon: float = 1e-5
    initializer_range: float = 0.02


@dataclass
class T5Config:
    """Configuration for T5 model."""
    vocab_size: int = 32128
    d_model: int = 768
    d_kv: int = 64
    d_ff: int = 3072
    num_layers: int = 12
    num_decoder_layers: int = 12
    num_heads: int = 12
    relative_attention_num_buckets: int = 32
    dropout_rate: float = 0.1
    layer_norm_epsilon: float = 1e-6
    initializer_factor: float = 1.0
    feed_forward_proj: str = "relu"


@dataclass
class ViTConfig:
    """Configuration for Vision Transformer."""
    image_size: int = 224
    patch_size: int = 16
    num_channels: int = 3
    num_classes: int = 1000
    hidden_size: int = 768
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    intermediate_size: int = 3072
    hidden_dropout_prob: float = 0.0
    attention_probs_dropout_prob: float = 0.0
    initializer_range: float = 0.02
    layer_norm_eps: float = 1e-6
    qkv_bias: bool = True


# BERT Implementation
class BERTEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings."""
    
    def __init__(self, config: BERTConfig):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)
        
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
    def forward(self, input_ids, token_type_ids=None, position_ids=None):
        seq_length = input_ids.size(1)
        
        if position_ids is None:
            position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
            
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)
            
        words_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        
        embeddings = words_embeddings + position_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        
        return embeddings


class BERTSelfAttention(nn.Module):
    """BERT self-attention mechanism."""
    
    def __init__(self, config: BERTConfig):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(f"Hidden size {config.hidden_size} not divisible by num_attention_heads {config.num_attention_heads}")
            
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)
        
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        
    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)
    
    def forward(self, hidden_states, attention_mask=None):
        query_layer = self.transpose_for_scores(self.query(hidden_states))
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        
        # Attention scores
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask
            
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)
        
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        
        return context_layer, attention_probs


class BERTLayer(nn.Module):
    """A single BERT layer."""
    
    def __init__(self, config: BERTConfig):
        super().__init__()
        self.attention = BERTSelfAttention(config)
        self.attention_output = nn.Linear(config.hidden_size, config.hidden_size)
        self.attention_dropout = nn.Dropout(config.hidden_dropout_prob)
        self.attention_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        
        self.intermediate = nn.Linear(config.hidden_size, config.intermediate_size)
        self.output = nn.Linear(config.intermediate_size, config.hidden_size)
        self.output_dropout = nn.Dropout(config.hidden_dropout_prob)
        self.output_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        
        self.activation = nn.GELU() if config.hidden_act == "gelu" else nn.ReLU()
        
    def forward(self, hidden_states, attention_mask=None):
        # Self-attention
        attention_output, attention_probs = self.attention(hidden_states, attention_mask)
        attention_output = self.attention_output(attention_output)
        attention_output = self.attention_dropout(attention_output)
        attention_output = self.attention_layer_norm(attention_output + hidden_states)
        
        # Feed-forward
        intermediate_output = self.activation(self.intermediate(attention_output))
        layer_output = self.output(intermediate_output)
        layer_output = self.output_dropout(layer_output)
        layer_output = self.output_layer_norm(layer_output + attention_output)
        
        return layer_output, attention_probs


class BERTModel(nn.Module):
    """BERT model implementation."""
    
    def __init__(self, config: BERTConfig):
        super().__init__()
        self.config = config
        self.embeddings = BERTEmbeddings(config)
        self.encoder = nn.ModuleList([BERTLayer(config) for _ in range(config.num_hidden_layers)])
        self.pooler = nn.Linear(config.hidden_size, config.hidden_size)
        self.pooler_activation = nn.Tanh()
        
    def forward(self, input_ids, token_type_ids=None, attention_mask=None, output_attentions=False):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
            
        # Convert attention mask to extended attention mask
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        
        # Embeddings
        hidden_states = self.embeddings(input_ids, token_type_ids)
        
        # Encoder
        all_attentions = []
        for layer in self.encoder:
            hidden_states, attention_probs = layer(hidden_states, extended_attention_mask)
            if output_attentions:
                all_attentions.append(attention_probs)
                
        # Pooler
        pooled_output = self.pooler_activation(self.pooler(hidden_states[:, 0]))
        
        outputs = (hidden_states, pooled_output)
        if output_attentions:
            outputs = outputs + (all_attentions,)
            
        return outputs


# GPT Implementation
class GPTAttention(nn.Module):
    """GPT-style causal self-attention."""
    
    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.attn_pdrop
        
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        
        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)
        
        # Causal mask
        self.register_buffer("bias", torch.tril(torch.ones(config.n_positions, config.n_positions))
                                     .view(1, 1, config.n_positions, config.n_positions))
        
    def forward(self, x, layer_past=None):
        B, T, C = x.size()
        
        # Calculate query, key, values
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        
        # Causal self-attention
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        
        # Output projection
        y = self.resid_dropout(self.c_proj(y))
        
        return y


class GPTBlock(nn.Module):
    """A GPT transformer block."""
    
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.attn = GPTAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, config.n_inner),
            nn.GELU(),
            nn.Linear(config.n_inner, config.n_embd),
            nn.Dropout(config.resid_pdrop),
        )
        
    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPTModel(nn.Module):
    """GPT model implementation."""
    
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.n_positions, config.n_embd)
        self.drop = nn.Dropout(config.embd_pdrop)
        
        self.h = nn.ModuleList([GPTBlock(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        
    def forward(self, input_ids, past_key_values=None):
        device = input_ids.device
        b, t = input_ids.size()
        
        assert t <= self.config.n_positions, f"Cannot forward sequence of length {t}, max is {self.config.n_positions}"
        
        # Token and position embeddings
        pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0)
        tok_emb = self.wte(input_ids)
        pos_emb = self.wpe(pos)
        x = self.drop(tok_emb + pos_emb)
        
        # Transformer blocks
        for block in self.h:
            x = block(x)
            
        x = self.ln_f(x)
        logits = self.lm_head(x)
        
        return logits


# T5 Implementation
class T5RelativePositionBias(nn.Module):
    """T5-style relative position bias."""
    
    def __init__(self, config: T5Config, is_decoder=False):
        super().__init__()
        self.is_decoder = is_decoder
        self.relative_attention_num_buckets = config.relative_attention_num_buckets
        self.relative_attention_bias = nn.Embedding(
            config.relative_attention_num_buckets, config.num_heads
        )
        
    def _relative_position_bucket(self, relative_position, num_buckets=32, max_distance=128):
        ret = 0
        n = -relative_position
        
        num_buckets //= 2
        ret += (n < 0).long() * num_buckets
        n = torch.abs(n)
        
        max_exact = num_buckets // 2
        is_small = n < max_exact
        
        val_if_large = max_exact + (
            torch.log(n.float() / max_exact) / math.log(max_distance / max_exact) * (num_buckets - max_exact)
        ).long()
        val_if_large = torch.min(val_if_large, torch.full_like(val_if_large, num_buckets - 1))
        
        ret += torch.where(is_small, n, val_if_large)
        return ret
    
    def forward(self, query_length, key_length):
        context_position = torch.arange(query_length, dtype=torch.long)[:, None]
        memory_position = torch.arange(key_length, dtype=torch.long)[None, :]
        
        relative_position = memory_position - context_position
        relative_position_bucket = self._relative_position_bucket(relative_position)
        
        values = self.relative_attention_bias(relative_position_bucket)
        values = values.permute([2, 0, 1]).unsqueeze(0)
        
        return values


class T5Attention(nn.Module):
    """T5 attention layer."""
    
    def __init__(self, config: T5Config, has_relative_attention_bias=False):
        super().__init__()
        self.is_decoder = config.is_decoder if hasattr(config, 'is_decoder') else False
        self.has_relative_attention_bias = has_relative_attention_bias
        
        self.d_model = config.d_model
        self.key_value_proj_dim = config.d_kv
        self.n_heads = config.num_heads
        self.inner_dim = self.n_heads * self.key_value_proj_dim
        
        # Mesh TensorFlow initialization
        self.q = nn.Linear(self.d_model, self.inner_dim, bias=False)
        self.k = nn.Linear(self.d_model, self.inner_dim, bias=False)
        self.v = nn.Linear(self.d_model, self.inner_dim, bias=False)
        self.o = nn.Linear(self.inner_dim, self.d_model, bias=False)
        
        if self.has_relative_attention_bias:
            self.relative_attention_bias = T5RelativePositionBias(config, is_decoder=self.is_decoder)
            
        self.dropout = nn.Dropout(config.dropout_rate)
        
    def forward(self, hidden_states, mask=None, position_bias=None, past_key_value=None):
        batch_size, seq_length = hidden_states.shape[:2]
        
        # Compute Q, K, V
        query_states = self.q(hidden_states).view(batch_size, seq_length, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
        key_states = self.k(hidden_states).view(batch_size, seq_length, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
        value_states = self.v(hidden_states).view(batch_size, seq_length, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
        
        # Compute attention scores
        scores = torch.matmul(query_states, key_states.transpose(3, 2))
        
        # Add relative position bias
        if position_bias is None and self.has_relative_attention_bias:
            position_bias = self.relative_attention_bias(seq_length, seq_length)
            
        if position_bias is not None:
            scores += position_bias
            
        # Apply mask
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
            
        # Attention weights
        attn_weights = F.softmax(scores.float(), dim=-1).type_as(scores)
        attn_weights = self.dropout(attn_weights)
        
        # Compute attention output
        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_length, self.inner_dim)
        attn_output = self.o(attn_output)
        
        return attn_output, position_bias


class T5LayerNorm(nn.Module):
    """T5-style layer normalization (no bias, no subtraction of mean)."""
    
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps
        
    def forward(self, hidden_states):
        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(hidden_states.dtype)


class T5Block(nn.Module):
    """T5 transformer block."""
    
    def __init__(self, config: T5Config, has_relative_attention_bias=False):
        super().__init__()
        self.is_decoder = config.is_decoder if hasattr(config, 'is_decoder') else False
        
        self.layer = nn.ModuleList()
        self.layer.append(T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon))
        self.layer.append(T5Attention(config, has_relative_attention_bias=has_relative_attention_bias))
        
        if self.is_decoder:
            self.layer.append(T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon))
            self.layer.append(T5Attention(config, has_relative_attention_bias=False))
            
        self.layer.append(T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon))
        self.layer.append(T5DenseReluDense(config))
        
    def forward(self, hidden_states, attention_mask=None, position_bias=None, encoder_hidden_states=None):
        # Self-attention
        normed_hidden_states = self.layer[0](hidden_states)
        attention_output, position_bias = self.layer[1](
            normed_hidden_states, mask=attention_mask, position_bias=position_bias
        )
        hidden_states = hidden_states + attention_output
        
        # Cross-attention (if decoder)
        if self.is_decoder and encoder_hidden_states is not None:
            normed_hidden_states = self.layer[2](hidden_states)
            attention_output, _ = self.layer[3](
                normed_hidden_states, 
                mask=attention_mask,
                key_value_states=encoder_hidden_states
            )
            hidden_states = hidden_states + attention_output
            
        # Feed-forward
        normed_hidden_states = self.layer[-2](hidden_states)
        feed_forward_output = self.layer[-1](normed_hidden_states)
        hidden_states = hidden_states + feed_forward_output
        
        return hidden_states, position_bias


class T5DenseReluDense(nn.Module):
    """T5 feed-forward layer."""
    
    def __init__(self, config: T5Config):
        super().__init__()
        self.wi = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.wo = nn.Linear(config.d_ff, config.d_model, bias=False)
        self.dropout = nn.Dropout(config.dropout_rate)
        
    def forward(self, hidden_states):
        hidden_states = self.wi(hidden_states)
        hidden_states = F.relu(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.wo(hidden_states)
        return hidden_states


# Vision Transformer Implementation
class PatchEmbedding(nn.Module):
    """Convert image to patches and embed them."""
    
    def __init__(self, config: ViTConfig):
        super().__init__()
        self.num_patches = (config.image_size // config.patch_size) ** 2
        self.projection = nn.Conv2d(
            config.num_channels, 
            config.hidden_size, 
            kernel_size=config.patch_size, 
            stride=config.patch_size
        )
        
    def forward(self, pixel_values):
        batch_size = pixel_values.shape[0]
        x = self.projection(pixel_values).flatten(2).transpose(1, 2)
        return x


class ViTEmbeddings(nn.Module):
    """Vision Transformer embeddings."""
    
    def __init__(self, config: ViTConfig):
        super().__init__()
        self.patch_embeddings = PatchEmbedding(config)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size))
        self.position_embeddings = nn.Parameter(
            torch.zeros(1, self.patch_embeddings.num_patches + 1, config.hidden_size)
        )
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
    def forward(self, pixel_values):
        batch_size = pixel_values.shape[0]
        embeddings = self.patch_embeddings(pixel_values)
        
        # Add [CLS] token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        embeddings = torch.cat((cls_tokens, embeddings), dim=1)
        
        # Add position embeddings
        embeddings = embeddings + self.position_embeddings
        embeddings = self.dropout(embeddings)
        
        return embeddings


class ViTSelfAttention(nn.Module):
    """Vision Transformer self-attention."""
    
    def __init__(self, config: ViTConfig):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(f"Hidden size {config.hidden_size} not divisible by num_attention_heads {config.num_attention_heads}")
            
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        
        self.query = nn.Linear(config.hidden_size, self.all_head_size, bias=config.qkv_bias)
        self.key = nn.Linear(config.hidden_size, self.all_head_size, bias=config.qkv_bias)
        self.value = nn.Linear(config.hidden_size, self.all_head_size, bias=config.qkv_bias)
        
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        
    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)
        
    def forward(self, hidden_states):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)
        
        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)
        
        # Attention scores
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        
        # Attention probabilities
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)
        
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        
        return context_layer, attention_probs


class ViTLayer(nn.Module):
    """Vision Transformer layer."""
    
    def __init__(self, config: ViTConfig):
        super().__init__()
        self.attention = ViTSelfAttention(config)
        self.attention_output = nn.Linear(config.hidden_size, config.hidden_size)
        self.attention_dropout = nn.Dropout(config.hidden_dropout_prob)
        self.attention_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        
        self.intermediate = nn.Linear(config.hidden_size, config.intermediate_size)
        self.output = nn.Linear(config.intermediate_size, config.hidden_size)
        self.output_dropout = nn.Dropout(config.hidden_dropout_prob)
        self.output_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        
        self.activation = nn.GELU()
        
    def forward(self, hidden_states):
        # Self-attention
        self_attention_outputs = self.attention(hidden_states)
        attention_output = self_attention_outputs[0]
        attention_probs = self_attention_outputs[1]
        
        # Add & Norm
        attention_output = self.attention_output(attention_output)
        attention_output = self.attention_dropout(attention_output)
        hidden_states = self.attention_norm(hidden_states + attention_output)
        
        # Feed-forward
        intermediate_output = self.activation(self.intermediate(hidden_states))
        layer_output = self.output(intermediate_output)
        layer_output = self.output_dropout(layer_output)
        
        # Add & Norm
        layer_output = self.output_norm(hidden_states + layer_output)
        
        return layer_output, attention_probs


class VisionTransformer(nn.Module):
    """Vision Transformer (ViT) model."""
    
    def __init__(self, config: ViTConfig):
        super().__init__()
        self.config = config
        
        self.embeddings = ViTEmbeddings(config)
        self.encoder = nn.ModuleList([ViTLayer(config) for _ in range(config.num_hidden_layers)])
        self.layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        
        # Classifier head
        self.classifier = nn.Linear(config.hidden_size, config.num_classes) if config.num_classes > 0 else nn.Identity()
        
    def forward(self, pixel_values, output_attentions=False):
        hidden_states = self.embeddings(pixel_values)
        
        all_attentions = []
        for layer in self.encoder:
            hidden_states, attention_probs = layer(hidden_states)
            if output_attentions:
                all_attentions.append(attention_probs)
                
        hidden_states = self.layernorm(hidden_states)
        
        # Use [CLS] token for classification
        pooled_output = hidden_states[:, 0]
        logits = self.classifier(pooled_output)
        
        outputs = (logits, hidden_states)
        if output_attentions:
            outputs = outputs + (all_attentions,)
            
        return outputs


# Model comparison utilities
def count_parameters(model):
    """Count the number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def compare_models():
    """Compare different transformer architectures."""
    # Create models with similar sizes
    bert_config = BERTConfig(num_hidden_layers=6, hidden_size=512, num_attention_heads=8)
    gpt_config = GPTConfig(n_layer=6, n_embd=512, n_head=8)
    t5_config = T5Config(num_layers=6, d_model=512, num_heads=8)
    vit_config = ViTConfig(num_hidden_layers=6, hidden_size=512, num_attention_heads=8)
    
    bert = BERTModel(bert_config)
    gpt = GPTModel(gpt_config)
    vit = VisionTransformer(vit_config)
    
    models = {
        'BERT': bert,
        'GPT': gpt,
        'ViT': vit,
    }
    
    print("Model Parameter Comparison:")
    print("-" * 40)
    for name, model in models.items():
        params = count_parameters(model)
        print(f"{name}: {params:,} parameters")
        
    return models


def visualize_attention_patterns(model, inputs, model_type='bert'):
    """Visualize attention patterns from different models."""
    model.eval()
    
    with torch.no_grad():
        if model_type == 'bert':
            outputs = model(inputs, output_attentions=True)
            attentions = outputs[2]  # List of attention matrices
        elif model_type == 'vit':
            outputs = model(inputs, output_attentions=True)
            attentions = outputs[2]
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
    
    # Visualize attention from first layer, first head
    attention = attentions[0][0, 0].cpu().numpy()
    
    plt.figure(figsize=(10, 8))
    plt.imshow(attention, cmap='hot', interpolation='nearest')
    plt.colorbar()
    plt.title(f'{model_type.upper()} Attention Pattern (Layer 1, Head 1)')
    plt.xlabel('Keys')
    plt.ylabel('Queries')
    plt.show()
    
    return attentions


# Example usage
if __name__ == "__main__":
    print("=== Transformer Variants Demo ===\n")
    
    # Compare models
    models = compare_models()
    
    # Test BERT
    print("\n--- BERT Example ---")
    bert_config = BERTConfig(num_hidden_layers=4, hidden_size=256, num_attention_heads=8)
    bert = BERTModel(bert_config)
    
    # Example input
    batch_size = 2
    seq_length = 10
    input_ids = torch.randint(0, bert_config.vocab_size, (batch_size, seq_length))
    
    outputs = bert(input_ids)
    print(f"Input shape: {input_ids.shape}")
    print(f"Last hidden states shape: {outputs[0].shape}")
    print(f"Pooled output shape: {outputs[1].shape}")
    
    # Test GPT
    print("\n--- GPT Example ---")
    gpt_config = GPTConfig(n_layer=4, n_embd=256, n_head=8)
    gpt = GPTModel(gpt_config)
    
    input_ids = torch.randint(0, gpt_config.vocab_size, (batch_size, seq_length))
    logits = gpt(input_ids)
    print(f"Input shape: {input_ids.shape}")
    print(f"Logits shape: {logits.shape}")
    
    # Test ViT
    print("\n--- Vision Transformer Example ---")
    vit_config = ViTConfig(num_hidden_layers=4, hidden_size=256, num_attention_heads=8)
    vit = VisionTransformer(vit_config)
    
    # Random image input
    pixel_values = torch.randn(batch_size, 3, 224, 224)
    outputs = vit(pixel_values)
    print(f"Input shape: {pixel_values.shape}")
    print(f"Logits shape: {outputs[0].shape}")
    print(f"Hidden states shape: {outputs[1].shape}")
    
    # Demonstrate special tokens
    print("\n--- Special Tokens Usage ---")
    special_tokens = {
        'BERT': {
            '[CLS]': "Classification/Start token",
            '[SEP]': "Separator between sentences",
            '[MASK]': "Masked token for MLM",
            '[PAD]': "Padding token"
        },
        'GPT': {
            '<|endoftext|>': "End of text token",
            '<|startoftext|>': "Start of text token (GPT-3+)"
        },
        'T5': {
            '<pad>': "Padding token",
            '</s>': "End of sequence",
            '<unk>': "Unknown token",
            '<extra_id_0>': "Sentinel tokens for denoising"
        }
    }
    
    for model_name, tokens in special_tokens.items():
        print(f"\n{model_name} Special Tokens:")
        for token, description in tokens.items():
            print(f"  {token}: {description}")
    
    print("\n✅ All transformer variants demonstrated successfully!")