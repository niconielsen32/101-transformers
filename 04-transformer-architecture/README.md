# The Transformer Architecture

Welcome to the complete transformer architecture! This module brings together attention mechanisms with other key components to create the revolutionary model from "Attention Is All You Need".

## 🎯 Learning Objectives

By the end of this module, you will understand:
- The complete transformer architecture
- Multi-head attention and why it's powerful
- Position encodings and their importance
- Layer normalization and residual connections
- The encoder-decoder structure
- How to implement a transformer from scratch

## 📚 Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [Multi-Head Attention](#2-multi-head-attention)
3. [Position Encodings](#3-position-encodings)
4. [Feed-Forward Networks](#4-feed-forward-networks)
5. [Layer Normalization](#5-layer-normalization)
6. [Residual Connections](#6-residual-connections)
7. [Encoder Stack](#7-encoder-stack)
8. [Decoder Stack](#8-decoder-stack)
9. [Putting It All Together](#9-putting-it-all-together)

## 1. Architecture Overview

### 1.1 The Big Picture

The transformer consists of:
- **Encoder**: Maps input sequence to continuous representations
- **Decoder**: Generates output sequence using encoder output and previous predictions

### 1.2 Key Components

```
Input → Embedding → Positional Encoding → 
    ↓
[Encoder Block] × N
    ↓
Encoder Output
    ↓
[Decoder Block] × N → Linear → Softmax → Output
```

### 1.3 Design Principles

1. **No recurrence**: Fully parallelizable
2. **Self-attention**: Global dependencies
3. **Position encodings**: Inject sequence order
4. **Residual connections**: Easy gradient flow
5. **Layer normalization**: Stable training

## 2. Multi-Head Attention

### 2.1 Why Multiple Heads?

Single attention:
```
Attention(Q,K,V) = softmax(QK^T/√d_k)V
```

Multi-head attention:
```
MultiHead(Q,K,V) = Concat(head_1, ..., head_h)W^O
where head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)
```

### 2.2 Benefits

- **Multiple representation subspaces**: Each head learns different relationships
- **Parallel attention**: Different aspects simultaneously
- **More parameters**: Increased model capacity
- **Robustness**: If one head fails, others compensate

### 2.3 Typical Configuration

- 8 heads is common
- Each head dimension: d_model / n_heads
- Example: d_model=512, heads=8 → d_k=64 per head

## 3. Position Encodings

### 3.1 The Problem

Attention has no inherent notion of position:
- "cat sat mat" = "mat cat sat" = "sat mat cat"

We need to inject positional information!

### 3.2 Sinusoidal Encodings

The original transformer uses:

```
PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

Where:
- pos: Position in sequence
- i: Dimension index
- d_model: Model dimension

### 3.3 Properties

- **Unique for each position**: No two positions have same encoding
- **Consistent across sequences**: Same position always gets same encoding
- **Relative positions**: Can learn to attend by relative position
- **Extrapolation**: Can handle longer sequences than training

### 3.4 Learned vs Fixed

- **Fixed (sinusoidal)**: No parameters, generalizes well
- **Learned**: More flexible but needs training data

## 4. Feed-Forward Networks

### 4.1 Structure

Simple 2-layer network applied to each position:

```
FFN(x) = ReLU(xW_1 + b_1)W_2 + b_2
```

Typically:
- Hidden dimension = 4 × d_model
- Example: d_model=512 → d_ff=2048

### 4.2 Position-wise

Applied **independently** to each position:
- No interaction between positions
- Can be parallelized
- Adds non-linearity

### 4.3 Why So Large?

The FFN acts as a key-value memory:
- First layer: Keys (what patterns to match)
- Second layer: Values (what to output)
- Large hidden dimension = more patterns

## 5. Layer Normalization

### 5.1 Why Normalize?

- **Stable gradients**: Prevents explosion/vanishing
- **Faster training**: Normalized inputs converge faster
- **Smoother landscape**: Easier optimization

### 5.2 Layer Norm vs Batch Norm

Layer Norm normalizes across features:
```
LayerNorm(x) = γ * (x - μ) / σ + β
```

Where μ, σ computed across hidden dimensions (not batch).

### 5.3 Pre-norm vs Post-norm

**Original (Post-norm)**:
```
x + Sublayer(LayerNorm(x))
```

**Modern (Pre-norm)**:
```
x + Sublayer(LayerNorm(x))
```

Pre-norm is more stable for deep models.

## 6. Residual Connections

### 6.1 The Concept

Add input to output:
```
Output = x + Sublayer(x)
```

### 6.2 Benefits

- **Gradient highway**: Direct path for gradients
- **Identity default**: Easy to learn identity function
- **Deep networks**: Enables very deep models
- **Feature reuse**: Preserves information

## 7. Encoder Stack

### 7.1 Encoder Layer

Each encoder layer contains:
1. Multi-head self-attention
2. Position-wise feed-forward network
3. Residual connections around each
4. Layer normalization

```python
def encoder_layer(x):
    # Self-attention
    attn_output = self_attention(x, x, x)
    x = layer_norm(x + attn_output)
    
    # Feed-forward
    ff_output = feed_forward(x)
    x = layer_norm(x + ff_output)
    
    return x
```

### 7.2 Stack of Layers

Typically 6 layers:
- Each layer refines representations
- Lower layers: syntax, local patterns
- Higher layers: semantics, global patterns

## 8. Decoder Stack

### 8.1 Decoder Layer

Each decoder layer contains:
1. Masked multi-head self-attention
2. Multi-head cross-attention (to encoder)
3. Position-wise feed-forward network
4. Residual connections and layer norms

### 8.2 Masked Self-Attention

Prevents looking at future positions:
```
Mask[i,j] = 0 if j > i else 1
```

Ensures autoregressive generation.

### 8.3 Cross-Attention

- **Queries**: From decoder
- **Keys, Values**: From encoder
- Allows decoder to focus on relevant input

## 9. Putting It All Together

### 9.1 Complete Forward Pass

1. **Input Processing**:
   - Token embeddings
   - Add position encodings

2. **Encoder**:
   - Pass through N encoder layers
   - Output: contextualized representations

3. **Decoder**:
   - Start with target embeddings
   - Masked self-attention
   - Cross-attention to encoder
   - Generate predictions

4. **Output**:
   - Linear projection
   - Softmax for probabilities

### 9.2 Training

- **Teacher forcing**: Use true previous outputs
- **Loss**: Cross-entropy over vocabulary
- **Optimization**: Adam with learning rate scheduling

### 9.3 Inference

- **Autoregressive**: Generate one token at a time
- **Beam search**: Keep top-k hypotheses
- **Caching**: Reuse computed key-value pairs

## 📊 Model Configurations

| Model | Layers | d_model | Heads | d_ff | Parameters |
|-------|--------|---------|-------|------|------------|
| Base | 6 | 512 | 8 | 2048 | 65M |
| Big | 6 | 1024 | 16 | 4096 | 213M |
| GPT-2 | 12 | 768 | 12 | 3072 | 117M |
| GPT-3 | 96 | 12288 | 96 | 49152 | 175B |

## 🔍 Key Insights

1. **Attention is all you need**: No recurrence or convolution
2. **Parallelization**: All positions processed simultaneously
3. **Long-range dependencies**: Direct connections between all positions
4. **Scalability**: Easy to make bigger/smaller
5. **Transfer learning**: Pre-train on large data, fine-tune on specific tasks

## 📝 Summary

The transformer architecture combines:
- **Multi-head attention**: Learn multiple relationship types
- **Position encodings**: Inject sequence order
- **Feed-forward networks**: Process each position
- **Residual connections**: Enable deep networks
- **Layer normalization**: Stable training

This creates a powerful, parallelizable architecture that has revolutionized NLP and beyond!

## ➡️ Next Steps

Ready to build a transformer from scratch? Head to [Topic 5: Building from Scratch](../05-building-from-scratch/) to implement everything you've learned!