# Modern LLM Architectures

Explore the cutting-edge architectures powering today's most advanced language models, from GPT-4 to LLaMA, including the latest innovations in efficiency and scale.

## 🎯 Learning Objectives

By the end of this module, you will understand:
- Architectural innovations in modern LLMs
- Scaling strategies and efficiency improvements
- Mixture of Experts (MoE) models
- Flash Attention and optimized implementations
- Open-source innovations (LLaMA, Mistral)
- Architectural trends and future directions

## 📚 Table of Contents

1. [Evolution of LLM Architectures](#1-evolution-of-llm-architectures)
2. [GPT-4 and Advanced Scaling](#2-gpt-4-and-advanced-scaling)
3. [LLaMA Family](#3-llama-family)
4. [Mixture of Experts](#4-mixture-of-experts)
5. [Flash Attention](#5-flash-attention)
6. [Efficient Architectures](#6-efficient-architectures)
7. [Architectural Innovations](#7-architectural-innovations)
8. [Future Directions](#8-future-directions)

## 1. Evolution of LLM Architectures

### 1.1 Timeline of Innovations

```
2018: GPT-1 (117M) - Unsupervised pre-training
2019: GPT-2 (1.5B) - Zero-shot task transfer
2020: GPT-3 (175B) - In-context learning
2022: ChatGPT - RLHF at scale
2023: GPT-4 - Multimodal, enhanced reasoning
2023: LLaMA - Efficient open models
2024: Mixtral - Open MoE models
```

### 1.2 Key Architectural Trends

1. **Scale**: 100M → 1T+ parameters
2. **Efficiency**: Better performance per parameter
3. **Specialization**: Task-specific adaptations
4. **Multimodality**: Text + vision + audio
5. **Sparsity**: Conditional computation

## 2. GPT-4 and Advanced Scaling

### 2.1 Architectural Improvements (Speculated)

```python
class GPT4Architecture:
    """Hypothetical GPT-4 architecture based on public information."""
    
    def __init__(self):
        self.num_experts = 16  # MoE speculation
        self.expert_capacity = 2
        self.num_layers = 120
        self.hidden_size = 12288
        self.num_heads = 96
        self.context_length = 32768
        
    def routing_mechanism(self, x):
        """Sparse MoE routing."""
        router_logits = self.router(x)
        expert_indices = torch.topk(router_logits, k=self.expert_capacity)
        
        # Load balance loss
        load_balance_loss = self.compute_load_balance_loss(expert_indices)
        
        return expert_indices, load_balance_loss
```

### 2.2 Training Innovations

**Curriculum Learning**:
```python
def curriculum_schedule(step, max_steps):
    # Gradually increase complexity
    progress = step / max_steps
    
    # Sequence length curriculum
    seq_length = int(512 + progress * (8192 - 512))
    
    # Task difficulty curriculum
    if progress < 0.3:
        task_distribution = {'simple': 0.7, 'medium': 0.3, 'hard': 0.0}
    elif progress < 0.7:
        task_distribution = {'simple': 0.2, 'medium': 0.6, 'hard': 0.2}
    else:
        task_distribution = {'simple': 0.1, 'medium': 0.3, 'hard': 0.6}
    
    return seq_length, task_distribution
```

### 2.3 Multimodal Integration

```python
class MultimodalGPT4(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.text_encoder = TransformerEncoder(config.text_config)
        self.image_encoder = VisionTransformer(config.vision_config)
        self.cross_modal_layers = nn.ModuleList([
            CrossModalLayer(config) for _ in range(config.num_cross_layers)
        ])
        
    def forward(self, text_tokens=None, images=None):
        # Encode modalities
        text_features = self.text_encoder(text_tokens) if text_tokens is not None else None
        image_features = self.image_encoder(images) if images is not None else None
        
        # Cross-modal fusion
        if text_features is not None and image_features is not None:
            features = torch.cat([text_features, image_features], dim=1)
            for layer in self.cross_modal_layers:
                features = layer(features)
        else:
            features = text_features or image_features
            
        return features
```

## 3. LLaMA Family

### 3.1 LLaMA Architecture

**Key innovations**:
- RMSNorm instead of LayerNorm
- SwiGLU activation function
- Rotary Position Embeddings (RoPE)
- No bias in linear layers

```python
class LLaMAModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([
            LLaMADecoderLayer(config) for _ in range(config.num_hidden_layers)
        ])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        
    def forward(self, input_ids, attention_mask=None):
        hidden_states = self.embed_tokens(input_ids)
        
        # Apply rotary embeddings
        position_ids = torch.arange(input_ids.shape[1], device=input_ids.device)
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask, position_embeddings)
            
        hidden_states = self.norm(hidden_states)
        return hidden_states
```

### 3.2 RMSNorm

```python
class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps
        
    def forward(self, hidden_states):
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states
```

### 3.3 SwiGLU Activation

```python
class SwiGLU(nn.Module):
    def __init__(self, hidden_size, intermediate_size):
        super().__init__()
        self.w1 = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.w2 = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.w3 = nn.Linear(intermediate_size, hidden_size, bias=False)
        
    def forward(self, x):
        return self.w3(F.silu(self.w1(x)) * self.w2(x))
```

### 3.4 Rotary Position Embeddings (RoPE)

```python
class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        
    def forward(self, x, seq_len=None):
        t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        
        cos_emb = emb.cos()[None, None, :, :]
        sin_emb = emb.sin()[None, None, :, :]
        
        return cos_emb, sin_emb

def apply_rotary_pos_emb(q, k, cos, sin):
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed
```

## 4. Mixture of Experts

### 4.1 MoE Architecture

```python
class MoELayer(nn.Module):
    def __init__(self, hidden_size, num_experts, expert_capacity):
        super().__init__()
        self.num_experts = num_experts
        self.expert_capacity = expert_capacity
        
        # Router
        self.router = nn.Linear(hidden_size, num_experts)
        
        # Experts
        self.experts = nn.ModuleList([
            FeedForward(hidden_size) for _ in range(num_experts)
        ])
        
    def forward(self, hidden_states):
        batch_size, seq_len, hidden_size = hidden_states.shape
        
        # Routing decision
        router_logits = self.router(hidden_states)
        routing_weights = F.softmax(router_logits, dim=-1)
        
        # Select top-k experts
        routing_weights, selected_experts = torch.topk(
            routing_weights, self.expert_capacity, dim=-1
        )
        
        # Normalize routing weights
        routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)
        
        # Dispatch to experts
        final_hidden_states = torch.zeros_like(hidden_states)
        
        for expert_idx in range(self.num_experts):
            # Get tokens for this expert
            expert_mask = (selected_experts == expert_idx).any(dim=-1)
            expert_tokens = hidden_states[expert_mask]
            
            if expert_tokens.shape[0] > 0:
                # Apply expert
                expert_out = self.experts[expert_idx](expert_tokens)
                
                # Weighted combination
                expert_weights = routing_weights[expert_mask]
                expert_weights = expert_weights[selected_experts[expert_mask] == expert_idx]
                
                # Accumulate results
                final_hidden_states[expert_mask] += expert_weights.unsqueeze(-1) * expert_out
                
        return final_hidden_states
```

### 4.2 Load Balancing

```python
def load_balance_loss(router_probs, expert_indices):
    """Encourage equal distribution of tokens to experts."""
    num_experts = router_probs.shape[-1]
    
    # Compute expert load
    expert_mask = F.one_hot(expert_indices, num_experts).float()
    expert_load = expert_mask.sum(dim=[0, 1])
    
    # Target uniform distribution
    total_tokens = expert_mask.sum()
    target_load = total_tokens / num_experts
    
    # Imbalance loss
    imbalance = ((expert_load - target_load) ** 2).sum()
    
    # Importance loss (encourage confident routing)
    importance = router_probs.sum(dim=[0, 1])
    importance_loss = (importance ** 2).sum()
    
    return imbalance * 0.01 + importance_loss * 0.001
```

### 4.3 Switch Transformers

```python
class SwitchTransformer(nn.Module):
    """Simplified Switch Transformer with single expert routing."""
    
    def __init__(self, config):
        super().__init__()
        self.router = nn.Linear(config.hidden_size, config.num_experts)
        self.experts = nn.ModuleList([
            FeedForward(config.hidden_size) for _ in range(config.num_experts)
        ])
        
    def forward(self, x):
        # Route each token to single best expert
        router_logits = self.router(x)
        expert_index = router_logits.argmax(dim=-1)
        
        # Process each expert's tokens in parallel
        output = torch.zeros_like(x)
        for i in range(self.num_experts):
            mask = expert_index == i
            if mask.any():
                output[mask] = self.experts[i](x[mask])
                
        return output
```

## 5. Flash Attention

### 5.1 Hardware-Aware Implementation

```python
# Conceptual Flash Attention (actual implementation in CUDA)
def flash_attention(Q, K, V, block_size=64):
    """
    Flash Attention: Fast and Memory-Efficient Exact Attention
    Key ideas:
    1. Tiling to fit in SRAM
    2. Recomputation to save memory
    3. IO-aware algorithm
    """
    batch_size, num_heads, seq_len, head_dim = Q.shape
    
    # Initialize output and normalization
    O = torch.zeros_like(Q)
    l = torch.zeros((batch_size, num_heads, seq_len, 1))
    m = torch.full((batch_size, num_heads, seq_len, 1), -float('inf'))
    
    # Process in blocks
    for i in range(0, seq_len, block_size):
        # Load Q block
        Q_block = Q[:, :, i:i+block_size]
        
        # Initialize block accumulators
        O_block = torch.zeros_like(Q_block)
        l_block = torch.zeros((batch_size, num_heads, Q_block.shape[2], 1))
        m_block = torch.full((batch_size, num_heads, Q_block.shape[2], 1), -float('inf'))
        
        for j in range(0, seq_len, block_size):
            # Load K, V blocks
            K_block = K[:, :, j:j+block_size]
            V_block = V[:, :, j:j+block_size]
            
            # Compute attention scores
            S_block = torch.matmul(Q_block, K_block.transpose(-2, -1)) / math.sqrt(head_dim)
            
            # Update running max
            m_block_new = torch.maximum(m_block, S_block.max(dim=-1, keepdim=True).values)
            
            # Compute exponentials with stability
            P_block = torch.exp(S_block - m_block_new)
            
            # Update running sum
            l_block_new = torch.exp(m_block - m_block_new) * l_block + P_block.sum(dim=-1, keepdim=True)
            
            # Update output accumulator
            O_block = torch.exp(m_block - m_block_new) * O_block + torch.matmul(P_block, V_block)
            
            # Update trackers
            l_block = l_block_new
            m_block = m_block_new
            
        # Write back to output
        O[:, :, i:i+block_size] = O_block / l_block
        
    return O
```

### 5.2 Memory Optimization

```python
class FlashAttentionLayer(nn.Module):
    def __init__(self, hidden_size, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.o_proj = nn.Linear(hidden_size, hidden_size)
        
    def forward(self, hidden_states):
        batch_size, seq_len, _ = hidden_states.shape
        
        # Project and reshape
        Q = self.q_proj(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Flash attention (would use optimized kernel in practice)
        attn_output = flash_attention(Q, K, V)
        
        # Reshape and project
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        attn_output = self.o_proj(attn_output)
        
        return attn_output
```

## 6. Efficient Architectures

### 6.1 Mistral Architecture

```python
class MistralModel(nn.Module):
    """Mistral: Efficient architecture with sliding window attention."""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        
        self.layers = nn.ModuleList([
            MistralDecoderLayer(config) for _ in range(config.num_hidden_layers)
        ])
        
        self.norm = RMSNorm(config.hidden_size)
        
    def forward(self, input_ids, attention_mask=None):
        hidden_states = self.embed_tokens(input_ids)
        
        # Sliding window mask
        if attention_mask is None:
            attention_mask = self.create_sliding_window_mask(
                input_ids.shape[1], 
                window_size=self.config.sliding_window
            )
        
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask)
            
        return self.norm(hidden_states)
    
    def create_sliding_window_mask(self, seq_len, window_size):
        mask = torch.ones(seq_len, seq_len)
        for i in range(seq_len):
            mask[i, max(0, i - window_size):i + window_size + 1] = 0
        return mask.bool()
```

### 6.2 Grouped Query Attention (GQA)

```python
class GroupedQueryAttention(nn.Module):
    """GQA: Share key/value heads across multiple query heads."""
    
    def __init__(self, hidden_size, num_heads, num_kv_heads):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.num_queries_per_kv = num_heads // num_kv_heads
        self.head_dim = hidden_size // num_heads
        
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, self.num_kv_heads * self.head_dim)
        self.v_proj = nn.Linear(hidden_size, self.num_kv_heads * self.head_dim)
        self.o_proj = nn.Linear(hidden_size, hidden_size)
        
    def forward(self, hidden_states):
        batch_size, seq_len, _ = hidden_states.shape
        
        # Project queries
        Q = self.q_proj(hidden_states).view(
            batch_size, seq_len, self.num_heads, self.head_dim
        ).transpose(1, 2)
        
        # Project keys and values (fewer heads)
        K = self.k_proj(hidden_states).view(
            batch_size, seq_len, self.num_kv_heads, self.head_dim
        ).transpose(1, 2)
        V = self.v_proj(hidden_states).view(
            batch_size, seq_len, self.num_kv_heads, self.head_dim
        ).transpose(1, 2)
        
        # Repeat K, V for each query group
        K = K.repeat_interleave(self.num_queries_per_kv, dim=1)
        V = V.repeat_interleave(self.num_queries_per_kv, dim=1)
        
        # Standard attention
        attn_output = F.scaled_dot_product_attention(Q, K, V)
        
        # Reshape and project
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, -1
        )
        return self.o_proj(attn_output)
```

## 7. Architectural Innovations

### 7.1 Conditioned Generation

```python
class ConditionedLLM(nn.Module):
    """LLM with explicit conditioning mechanisms."""
    
    def __init__(self, config):
        super().__init__()
        self.transformer = TransformerModel(config)
        
        # Conditioning encoders
        self.instruction_encoder = nn.LSTM(config.hidden_size, config.hidden_size // 2, bidirectional=True)
        self.style_embedding = nn.Embedding(config.num_styles, config.hidden_size)
        
        # Fusion layer
        self.condition_fusion = nn.Linear(config.hidden_size * 2, config.hidden_size)
        
    def forward(self, input_ids, instruction_ids=None, style_id=None):
        # Encode main input
        hidden_states = self.transformer(input_ids)
        
        # Add conditioning
        if instruction_ids is not None:
            instruction_emb = self.transformer.embed_tokens(instruction_ids)
            instruction_hidden, _ = self.instruction_encoder(instruction_emb)
            instruction_pooled = instruction_hidden.mean(dim=1)
            
            # Fuse with hidden states
            hidden_states = hidden_states + instruction_pooled.unsqueeze(1)
            
        if style_id is not None:
            style_emb = self.style_embedding(style_id)
            hidden_states = hidden_states + style_emb.unsqueeze(1)
            
        return hidden_states
```

### 7.2 Adaptive Computation

```python
class AdaptiveLLM(nn.Module):
    """LLM with early exit and adaptive computation."""
    
    def __init__(self, config):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerLayer(config) for _ in range(config.num_layers)
        ])
        
        # Exit classifiers at different depths
        self.exit_classifiers = nn.ModuleList([
            nn.Linear(config.hidden_size, 1) 
            for _ in range(config.num_layers)
        ])
        
        self.exit_threshold = 0.9
        
    def forward(self, hidden_states, adaptive=True):
        exit_probs = []
        
        for i, layer in enumerate(self.layers):
            hidden_states = layer(hidden_states)
            
            if adaptive and i > 0:  # Allow early exit after first layer
                exit_score = torch.sigmoid(
                    self.exit_classifiers[i](hidden_states.mean(dim=1))
                )
                exit_probs.append(exit_score)
                
                if exit_score.mean() > self.exit_threshold:
                    return hidden_states, i + 1  # Return early
                    
        return hidden_states, len(self.layers)
```

### 7.3 Retrieval-Augmented Generation

```python
class RALLM(nn.Module):
    """Retrieval-Augmented LLM."""
    
    def __init__(self, config, retriever):
        super().__init__()
        self.llm = TransformerModel(config)
        self.retriever = retriever
        
        # Cross-attention for retrieved docs
        self.cross_attention_layers = nn.ModuleList([
            CrossAttentionLayer(config) for _ in range(config.num_cross_layers)
        ])
        
    def forward(self, input_ids, use_retrieval=True):
        # Standard LLM encoding
        hidden_states = self.llm.embed_tokens(input_ids)
        
        if use_retrieval:
            # Retrieve relevant documents
            query = hidden_states.mean(dim=1)  # Simple pooling
            retrieved_docs = self.retriever(query)
            
            # Encode retrieved documents
            doc_embeddings = self.llm.embed_tokens(retrieved_docs)
            
            # Cross-attend to documents
            for cross_attn in self.cross_attention_layers:
                hidden_states = cross_attn(hidden_states, doc_embeddings)
        
        # Continue with standard LLM layers
        output = self.llm(inputs_embeds=hidden_states)
        
        return output
```

## 8. Future Directions

### 8.1 Emerging Trends

1. **Extreme Scale**: Moving towards 10T+ parameters
2. **Modular Models**: Composable expert modules
3. **Continual Learning**: Adapting without forgetting
4. **Efficient Inference**: Sub-linear scaling
5. **Multimodal Native**: Built-in vision/audio understanding

### 8.2 Research Frontiers

```python
# Conceptual future architecture
class NextGenLLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        # Dynamic architecture
        self.architecture_controller = nn.LSTM(config.hidden_size, config.hidden_size)
        
        # Mixture of architectures
        self.attention_types = nn.ModuleList([
            StandardAttention(config),
            LinearAttention(config),
            RandomAttention(config),
            LocalAttention(config)
        ])
        
        # Neural architecture search
        self.nas_controller = nn.Linear(config.hidden_size, len(self.attention_types))
        
    def forward(self, x):
        # Dynamically select architecture based on input
        architecture_weights = F.softmax(self.nas_controller(x.mean(dim=1)), dim=-1)
        
        # Weighted combination of different architectures
        output = 0
        for i, attn in enumerate(self.attention_types):
            output += architecture_weights[:, i].unsqueeze(-1).unsqueeze(-1) * attn(x)
            
        return output
```

### 8.3 Efficiency Innovations

1. **Speculative Decoding**: Small model proposes, large model verifies
2. **Cascade Models**: Progressive refinement
3. **Quantization**: 1-bit to 4-bit models
4. **Structured Pruning**: Remove entire attention heads
5. **Knowledge Distillation**: Compress into smaller models

## 📊 Performance Comparisons

| Model | Parameters | Context | Training FLOPs | Inference Speed |
|-------|------------|---------|----------------|-----------------|
| GPT-3 | 175B | 2K | 3.14e23 | 20 tokens/s |
| LLaMA-2 | 70B | 4K | 1.7e23 | 30 tokens/s |
| Mistral | 7B | 32K | 2.5e22 | 100 tokens/s |
| GPT-4 | ~1.7T* | 32K | ~2e25* | 40 tokens/s |

*Estimated based on reports

## 🔍 Key Takeaways

1. **Efficiency matters**: Better architecture > more parameters
2. **Sparsity is key**: MoE enables massive scale
3. **Hardware awareness**: Flash Attention shows the way
4. **Open models compete**: LLaMA proves open can match closed
5. **Innovation continues**: New architectures emerging rapidly

## 📝 Summary

Modern LLM architectures represent the cutting edge of AI:
- **Scale**: From billions to trillions of parameters
- **Efficiency**: Innovations like Flash Attention and GQA
- **Sparsity**: MoE enables conditional computation
- **Open source**: LLaMA and Mistral democratize access
- **Future**: Dynamic, adaptive, multimodal architectures

The field continues to evolve rapidly with innovations in:
- Training efficiency
- Inference optimization
- Model quality
- Task adaptation

## ➡️ Next Steps

Ready to learn how to pretrain these massive models? Head to [Topic 10: Pretraining Large Language Models](../10-pretraining-llms/) to understand the process of creating LLMs from scratch!