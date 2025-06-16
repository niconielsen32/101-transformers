"""
Modern LLM Architectures
Implementation of cutting-edge LLM architectures including LLaMA, Mixture of Experts,
Flash Attention concepts, and other innovations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from typing import Optional, Tuple, Dict, List, Union
from dataclasses import dataclass
from einops import rearrange
import matplotlib.pyplot as plt


# Configuration classes
@dataclass
class LLaMAConfig:
    """Configuration for LLaMA model."""
    vocab_size: int = 32000
    hidden_size: int = 4096
    intermediate_size: int = 11008
    num_hidden_layers: int = 32
    num_attention_heads: int = 32
    num_key_value_heads: Optional[int] = None  # For GQA
    max_position_embeddings: int = 2048
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10000.0
    attention_dropout: float = 0.0
    hidden_dropout: float = 0.0
    
    def __post_init__(self):
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads


@dataclass
class MixtralConfig:
    """Configuration for Mixtral (MoE) model."""
    vocab_size: int = 32000
    hidden_size: int = 4096
    intermediate_size: int = 14336
    num_hidden_layers: int = 32
    num_attention_heads: int = 32
    num_experts: int = 8
    num_experts_per_tok: int = 2
    max_position_embeddings: int = 32768
    rms_norm_eps: float = 1e-5
    rope_theta: float = 1000000.0
    sliding_window: int = 4096


# Core Components
class RMSNorm(nn.Module):
    """RMSNorm normalization layer used in LLaMA."""
    
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps
        
    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        
        return (self.weight * hidden_states).to(input_dtype)


class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding used in LLaMA."""
    
    def __init__(self, dim: int, max_position_embeddings: int = 2048, 
                 base: float = 10000, device=None):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float().to(device) / dim))
        self.register_buffer("inv_freq", inv_freq)
        
        # Build here to make `torch.jit.trace` work
        self.max_seq_len_cached = max_position_embeddings
        t = torch.arange(self.max_seq_len_cached, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :], persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :], persistent=False)
        
    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        if seq_len > self.max_seq_len_cached:
            self.max_seq_len_cached = seq_len
            t = torch.arange(self.max_seq_len_cached, device=x.device, dtype=self.inv_freq.dtype)
            freqs = torch.einsum("i,j->ij", t, self.inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1).to(x.device)
            self.register_buffer("cos_cached", emb.cos()[None, None, :, :], persistent=False)
            self.register_buffer("sin_cached", emb.sin()[None, None, :, :], persistent=False)
            
        return (
            self.cos_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
            self.sin_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
        )


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids):
    """Apply rotary position embeddings to query and key tensors."""
    cos = cos.squeeze(1).squeeze(0)  # [seq_len, dim]
    sin = sin.squeeze(1).squeeze(0)  # [seq_len, dim]
    cos = cos[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    sin = sin[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class SwiGLU(nn.Module):
    """SwiGLU activation function used in LLaMA."""
    
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.w1 = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.w2 = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.w3 = nn.Linear(intermediate_size, hidden_size, bias=False)
        
    def forward(self, x):
        return self.w3(F.silu(self.w1(x)) * self.w2(x))


class GroupedQueryAttention(nn.Module):
    """Grouped Query Attention (GQA) used in LLaMA-2."""
    
    def __init__(self, config: LLaMAConfig):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.head_dim = config.hidden_size // self.num_heads
        
        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, config.hidden_size, bias=False)
        
        self.rotary_emb = RotaryEmbedding(self.head_dim, max_position_embeddings=config.max_position_embeddings)
        
    def forward(self, hidden_states, attention_mask=None, position_ids=None):
        bsz, q_len, _ = hidden_states.size()
        
        # Project to Q, K, V
        query_states = self.q_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        
        kv_seq_len = key_states.shape[-2]
        if position_ids is None:
            position_ids = torch.arange(kv_seq_len, dtype=torch.long, device=hidden_states.device)
            position_ids = position_ids.unsqueeze(0).expand(bsz, -1)
            
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)
        
        # Repeat k/v heads if num_key_value_heads < num_heads
        if self.num_key_value_heads < self.num_heads:
            key_states = repeat_kv(key_states, self.num_heads // self.num_key_value_heads)
            value_states = repeat_kv(value_states, self.num_heads // self.num_key_value_heads)
            
        # Compute attention
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
        
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
            
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_output = torch.matmul(attn_weights, value_states)
        
        # Reshape and project
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.num_heads * self.head_dim)
        attn_output = self.o_proj(attn_output)
        
        return attn_output, attn_weights


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Repeat key/value heads to match query heads."""
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


class LLaMADecoderLayer(nn.Module):
    """LLaMA decoder layer."""
    
    def __init__(self, config: LLaMAConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = GroupedQueryAttention(config)
        self.mlp = SwiGLU(config.hidden_size, config.intermediate_size)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        
    def forward(self, hidden_states, attention_mask=None, position_ids=None):
        residual = hidden_states
        
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )
        hidden_states = residual + hidden_states
        
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        
        return hidden_states, attn_weights


class LLaMAModel(nn.Module):
    """LLaMA model implementation."""
    
    def __init__(self, config: LLaMAConfig):
        super().__init__()
        self.config = config
        
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([LLaMADecoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        
    def forward(self, input_ids, attention_mask=None, position_ids=None):
        hidden_states = self.embed_tokens(input_ids)
        
        # Create causal mask
        if attention_mask is None:
            batch_size, seq_length = input_ids.shape
            attention_mask = torch.triu(
                torch.ones((seq_length, seq_length), dtype=torch.bool, device=input_ids.device),
                diagonal=1
            )
            attention_mask = attention_mask.masked_fill(attention_mask, float('-inf'))
            attention_mask = attention_mask.unsqueeze(0).unsqueeze(0)
            
        all_attentions = []
        for layer in self.layers:
            hidden_states, attn_weights = layer(hidden_states, attention_mask, position_ids)
            all_attentions.append(attn_weights)
            
        hidden_states = self.norm(hidden_states)
        
        return hidden_states, all_attentions


# Mixture of Experts Implementation
class MoEGate(nn.Module):
    """Mixture of Experts gating mechanism."""
    
    def __init__(self, hidden_size: int, num_experts: int, num_experts_per_tok: int):
        super().__init__()
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.gate = nn.Linear(hidden_size, num_experts, bias=False)
        
    def forward(self, hidden_states):
        batch_size, seq_len, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        
        # Compute gating scores
        scores = self.gate(hidden_states)
        
        # Select top-k experts
        topk_scores, topk_indices = torch.topk(scores, self.num_experts_per_tok, dim=-1)
        topk_scores = F.softmax(topk_scores, dim=-1)
        
        # Compute load balancing loss
        gates = F.softmax(scores, dim=-1)
        load_balancing_loss = self._load_balancing_loss(gates, topk_indices)
        
        return topk_scores, topk_indices, load_balancing_loss
    
    def _load_balancing_loss(self, gates, indices):
        """Compute load balancing loss to ensure equal expert usage."""
        # Fraction of tokens assigned to each expert
        num_tokens = gates.shape[0]
        expert_mask = F.one_hot(indices, num_classes=self.num_experts).float()
        expert_mask = expert_mask.sum(dim=1)  # Sum over top-k dimension
        tokens_per_expert = expert_mask.sum(dim=0) / num_tokens
        
        # Fraction of router probability assigned to each expert
        router_prob_per_expert = gates.mean(dim=0)
        
        # Load balancing loss
        return self.num_experts * (tokens_per_expert * router_prob_per_expert).sum()


class MoELayer(nn.Module):
    """Sparse Mixture of Experts layer."""
    
    def __init__(self, config: MixtralConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_experts = config.num_experts
        self.num_experts_per_tok = config.num_experts_per_tok
        
        self.gate = MoEGate(config.hidden_size, config.num_experts, config.num_experts_per_tok)
        self.experts = nn.ModuleList([
            SwiGLU(config.hidden_size, config.intermediate_size) 
            for _ in range(config.num_experts)
        ])
        
    def forward(self, hidden_states):
        batch_size, seq_len, hidden_dim = hidden_states.shape
        hidden_states_flat = hidden_states.view(-1, hidden_dim)
        
        # Get routing decisions
        topk_scores, topk_indices, lb_loss = self.gate(hidden_states)
        
        # Initialize output
        output = torch.zeros_like(hidden_states_flat)
        
        # Process each expert
        for expert_idx in range(self.num_experts):
            # Find tokens assigned to this expert
            expert_mask = (topk_indices == expert_idx).any(dim=-1)
            expert_tokens = hidden_states_flat[expert_mask]
            
            if expert_tokens.shape[0] > 0:
                # Apply expert
                expert_output = self.experts[expert_idx](expert_tokens)
                
                # Get weights for this expert
                expert_weights = topk_scores[expert_mask]
                expert_weights = expert_weights[topk_indices[expert_mask] == expert_idx]
                
                # Weighted output
                output[expert_mask] += expert_weights.unsqueeze(-1) * expert_output
                
        output = output.view(batch_size, seq_len, hidden_dim)
        
        return output, lb_loss


# Flash Attention Concepts (Simplified)
class FlashAttentionCore:
    """
    Conceptual implementation of Flash Attention algorithm.
    Note: This is a simplified version for educational purposes.
    Real Flash Attention requires CUDA kernels for efficiency.
    """
    
    @staticmethod
    def flash_attention_forward(Q, K, V, block_size=64, scale=None):
        """
        Simplified Flash Attention forward pass.
        
        Key ideas:
        1. Process attention in blocks to fit in SRAM
        2. Use online softmax to avoid materializing full attention matrix
        3. Recompute instead of storing intermediate values
        """
        batch_size, num_heads, seq_len, head_dim = Q.shape
        
        if scale is None:
            scale = 1.0 / math.sqrt(head_dim)
            
        # Initialize output and normalization factors
        O = torch.zeros_like(Q)
        L = torch.zeros((batch_size, num_heads, seq_len, 1), device=Q.device)
        M = torch.full((batch_size, num_heads, seq_len, 1), -float('inf'), device=Q.device)
        
        # Process in blocks
        for j in range(0, seq_len, block_size):
            j_end = min(j + block_size, seq_len)
            Kj = K[:, :, j:j_end]
            Vj = V[:, :, j:j_end]
            
            # Compute scores for current block
            S = torch.matmul(Q, Kj.transpose(-2, -1)) * scale
            
            # Update running max
            M_new = torch.maximum(M, S.max(dim=-1, keepdim=True).values)
            
            # Compute exponentials with stability
            P = torch.exp(S - M_new)
            
            # Update normalization
            L_new = torch.exp(M - M_new) * L + P.sum(dim=-1, keepdim=True)
            
            # Update output
            O = torch.exp(M - M_new) * O + torch.matmul(P, Vj)
            
            # Update running statistics
            L = L_new
            M = M_new
            
        # Final normalization
        O = O / L
        
        return O


class FlashAttention(nn.Module):
    """Flash Attention layer with memory-efficient computation."""
    
    def __init__(self, hidden_size: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)
        
        self.qkv_proj = nn.Linear(hidden_size, 3 * hidden_size, bias=False)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, hidden_states, attention_mask=None):
        batch_size, seq_len, hidden_size = hidden_states.shape
        
        # QKV projection
        qkv = self.qkv_proj(hidden_states)
        qkv = qkv.reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Apply Flash Attention
        if hidden_states.is_cuda and seq_len > 1024:
            # Use Flash Attention for long sequences on GPU
            attn_output = FlashAttentionCore.flash_attention_forward(q, k, v, scale=self.scale)
        else:
            # Standard attention for short sequences or CPU
            attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
            
            if attention_mask is not None:
                attn_scores = attn_scores + attention_mask
                
            attn_probs = F.softmax(attn_scores, dim=-1)
            attn_probs = self.dropout(attn_probs)
            attn_output = torch.matmul(attn_probs, v)
            
        # Reshape and project
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(batch_size, seq_len, hidden_size)
        attn_output = self.o_proj(attn_output)
        
        return attn_output


# Advanced Techniques
class SlidingWindowAttention(nn.Module):
    """Sliding window attention used in Mistral."""
    
    def __init__(self, hidden_size: int, num_heads: int, window_size: int = 4096):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.window_size = window_size
        
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        
    def forward(self, hidden_states):
        batch_size, seq_len, hidden_size = hidden_states.shape
        
        # Project Q, K, V
        q = self.q_proj(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Create sliding window mask
        mask = self._create_sliding_window_mask(seq_len, self.window_size, hidden_states.device)
        
        # Compute attention with window
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        scores = scores.masked_fill(mask == 0, float('-inf'))
        attn_weights = F.softmax(scores, dim=-1)
        
        # Apply attention
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, hidden_size)
        attn_output = self.o_proj(attn_output)
        
        return attn_output
    
    def _create_sliding_window_mask(self, seq_len, window_size, device):
        """Create a sliding window attention mask."""
        mask = torch.ones(seq_len, seq_len, device=device)
        for i in range(seq_len):
            start = max(0, i - window_size // 2)
            end = min(seq_len, i + window_size // 2 + 1)
            mask[i, start:end] = 1
        return mask.unsqueeze(0).unsqueeze(0)


class MultiQueryAttention(nn.Module):
    """Multi-Query Attention (MQA) - single key/value head shared across all query heads."""
    
    def __init__(self, hidden_size: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, self.head_dim, bias=False)  # Single K head
        self.v_proj = nn.Linear(hidden_size, self.head_dim, bias=False)  # Single V head
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        
    def forward(self, hidden_states):
        batch_size, seq_len, hidden_size = hidden_states.shape
        
        # Multi-head queries, single-head keys and values
        q = self.q_proj(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(hidden_states).view(batch_size, seq_len, 1, self.head_dim).transpose(1, 2)
        v = self.v_proj(hidden_states).view(batch_size, seq_len, 1, self.head_dim).transpose(1, 2)
        
        # Expand k, v to match number of heads
        k = k.expand(-1, self.num_heads, -1, -1)
        v = v.expand(-1, self.num_heads, -1, -1)
        
        # Standard attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn_weights = F.softmax(scores, dim=-1)
        attn_output = torch.matmul(attn_weights, v)
        
        # Reshape and project
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, hidden_size)
        attn_output = self.o_proj(attn_output)
        
        return attn_output


# Model Analysis and Visualization
def analyze_model_architecture(model, input_shape=(1, 128)):
    """Analyze model architecture and parameter distribution."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Parameter distribution by layer type
    param_dist = {}
    for name, module in model.named_modules():
        module_type = type(module).__name__
        if module_type not in param_dist:
            param_dist[module_type] = 0
        param_dist[module_type] += sum(p.numel() for p in module.parameters(recurse=False))
    
    # Memory estimation
    param_memory = total_params * 4 / (1024 ** 3)  # GB (assuming float32)
    
    # Activation memory (rough estimate)
    batch_size, seq_len = input_shape
    activation_memory = batch_size * seq_len * 4096 * 32 * 4 / (1024 ** 3)  # GB
    
    return {
        'total_params': total_params,
        'trainable_params': trainable_params,
        'param_distribution': param_dist,
        'param_memory_gb': param_memory,
        'activation_memory_gb': activation_memory,
        'total_memory_gb': param_memory + activation_memory
    }


def visualize_attention_patterns(attention_weights, layer_idx=0, head_idx=0):
    """Visualize attention patterns from a model."""
    # Get specific attention pattern
    attn = attention_weights[layer_idx][0, head_idx].detach().cpu().numpy()
    
    plt.figure(figsize=(10, 8))
    plt.imshow(attn, cmap='hot', interpolation='nearest')
    plt.colorbar()
    plt.title(f'Attention Pattern - Layer {layer_idx}, Head {head_idx}')
    plt.xlabel('Key Position')
    plt.ylabel('Query Position')
    plt.tight_layout()
    plt.show()


def compare_architectures():
    """Compare different modern LLM architectures."""
    architectures = {
        'LLaMA-7B': {
            'params': '7B',
            'layers': 32,
            'hidden': 4096,
            'heads': 32,
            'technique': 'RMSNorm, RoPE, SwiGLU'
        },
        'Mistral-7B': {
            'params': '7B',
            'layers': 32,
            'hidden': 4096,
            'heads': 32,
            'technique': 'GQA, Sliding Window, RoPE'
        },
        'Mixtral-8x7B': {
            'params': '47B (8x7B)',
            'layers': 32,
            'hidden': 4096,
            'experts': 8,
            'technique': 'Sparse MoE, Top-2 routing'
        },
        'GPT-4': {
            'params': '~1.7T*',
            'layers': '120*',
            'hidden': '~16K*',
            'heads': '~128*',
            'technique': 'MoE*, Multimodal'
        }
    }
    
    print("Modern LLM Architecture Comparison:")
    print("-" * 80)
    for name, specs in architectures.items():
        print(f"\n{name}:")
        for key, value in specs.items():
            print(f"  {key}: {value}")
    
    print("\n* Estimated based on reports")


# Example usage and demonstrations
if __name__ == "__main__":
    print("=== Modern LLM Architectures Demo ===\n")
    
    # Compare architectures
    compare_architectures()
    
    # Create a small LLaMA model
    print("\n--- LLaMA Architecture Demo ---")
    llama_config = LLaMAConfig(
        vocab_size=32000,
        hidden_size=512,
        intermediate_size=1024,
        num_hidden_layers=4,
        num_attention_heads=8,
        num_key_value_heads=4  # GQA with 4 KV heads
    )
    
    llama_model = LLaMAModel(llama_config)
    
    # Test forward pass
    input_ids = torch.randint(0, llama_config.vocab_size, (2, 32))
    hidden_states, attentions = llama_model(input_ids)
    
    print(f"Input shape: {input_ids.shape}")
    print(f"Output shape: {hidden_states.shape}")
    print(f"Number of attention layers: {len(attentions)}")
    
    # Analyze model
    analysis = analyze_model_architecture(llama_model, input_shape=(2, 32))
    print(f"\nModel Analysis:")
    print(f"Total parameters: {analysis['total_params']:,}")
    print(f"Parameter memory: {analysis['param_memory_gb']:.2f} GB")
    
    # Demonstrate RoPE
    print("\n--- Rotary Position Embeddings Demo ---")
    rope = RotaryEmbedding(dim=64, max_position_embeddings=128)
    positions = torch.arange(8)
    cos, sin = rope(torch.zeros(1, 1, 8, 64), seq_len=8)
    
    print(f"RoPE cos shape: {cos.shape}")
    print(f"RoPE sin shape: {sin.shape}")
    
    # Visualize RoPE
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.imshow(cos[0, 0, :, :32].numpy(), cmap='coolwarm', aspect='auto')
    plt.title('Cosine Component of RoPE')
    plt.xlabel('Dimension')
    plt.ylabel('Position')
    plt.colorbar()
    
    plt.subplot(1, 2, 2)
    plt.imshow(sin[0, 0, :, :32].numpy(), cmap='coolwarm', aspect='auto')
    plt.title('Sine Component of RoPE')
    plt.xlabel('Dimension')
    plt.ylabel('Position')
    plt.colorbar()
    
    plt.tight_layout()
    plt.show()
    
    # Demonstrate MoE
    print("\n--- Mixture of Experts Demo ---")
    moe_config = MixtralConfig(
        hidden_size=512,
        intermediate_size=1024,
        num_experts=8,
        num_experts_per_tok=2
    )
    
    moe_layer = MoELayer(moe_config)
    hidden_states = torch.randn(2, 16, 512)
    output, lb_loss = moe_layer(hidden_states)
    
    print(f"MoE input shape: {hidden_states.shape}")
    print(f"MoE output shape: {output.shape}")
    print(f"Load balancing loss: {lb_loss.item():.4f}")
    
    # Demonstrate Flash Attention concepts
    print("\n--- Flash Attention Concepts ---")
    print("Key innovations:")
    print("1. Tiling: Process attention in blocks that fit in SRAM")
    print("2. Recomputation: Trade compute for memory")
    print("3. IO-aware: Minimize HBM accesses")
    print("4. Online softmax: Compute softmax without materializing full matrix")
    
    # Performance comparison
    print("\n--- Efficiency Improvements ---")
    improvements = {
        'Technique': ['Standard Attention', 'Flash Attention', 'GQA', 'MQA', 'Sliding Window'],
        'Memory': ['O(n²)', 'O(n)', 'O(n²/g)', 'O(n²/h)', 'O(n·w)'],
        'Compute': ['O(n²)', 'O(n²)', 'O(n²)', 'O(n²)', 'O(n·w)'],
        'Quality': ['100%', '100%', '~99%', '~98%', '~98%']
    }
    
    import pandas as pd
    df = pd.DataFrame(improvements)
    print("\n" + df.to_string(index=False))
    
    print("\n✅ Modern LLM architectures demonstrated successfully!")