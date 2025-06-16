# Attention Mechanisms

Welcome to the breakthrough that revolutionized NLP! In this module, we'll demystify attention mechanisms - the key innovation that made transformers possible.

## 🎯 Learning Objectives

By the end of this module, you will understand:
- The intuition behind attention mechanisms
- How attention solves RNN limitations
- Mathematical formulation of attention
- Different types of attention (Bahdanau, Luong, Self-attention)
- Query, Key, Value framework
- How to implement attention from scratch

## 📚 Table of Contents

1. [The Intuition Behind Attention](#1-the-intuition-behind-attention)
2. [Bahdanau Attention - The Beginning](#2-bahdanau-attention---the-beginning)
3. [Attention as Information Retrieval](#3-attention-as-information-retrieval)
4. [Self-Attention - The Game Changer](#4-self-attention---the-game-changer)
5. [Query, Key, Value Framework](#5-query-key-value-framework)
6. [Scaled Dot-Product Attention](#6-scaled-dot-product-attention)
7. [Attention Visualizations](#7-attention-visualizations)
8. [Implementation Details](#8-implementation-details)
9. [Exercises](#9-exercises)

## 1. The Intuition Behind Attention

### 1.1 Human Attention

When you read this sentence, you don't process every word equally:

> "The **cat** that chased the mouse **sat** on the mat."

Your brain naturally focuses on important words (cat, sat) while giving less attention to others. This is exactly what attention mechanisms do!

### 1.2 The Core Idea

Instead of compressing everything into a fixed vector:
```
RNN: [All information] → [Fixed vector] → [Output]
```

Attention allows looking at all inputs:
```
Attention: [All information] → [Weighted combination] → [Output]
                                    ↑
                              (Dynamic weights)
```

### 1.3 Key Benefits

1. **No information bottleneck**: Access all hidden states
2. **Dynamic focus**: Different weights for different contexts
3. **Interpretability**: See what the model is "looking at"
4. **Parallelizable**: Compute all attention scores simultaneously

## 2. Bahdanau Attention - The Beginning

### 2.1 The Problem It Solved

In 2014, Bahdanau et al. noticed that neural machine translation quality dropped for long sentences. Their solution: let the decoder "attend" to different parts of the source sentence.

### 2.2 The Mechanism

For translating sentence X to Y:

1. **Encoder**: Process source sentence
   ```
   h₁, h₂, ..., hₙ = Encoder(x₁, x₂, ..., xₙ)
   ```

2. **Attention**: At each decoder step t
   ```
   score(hᵢ, sₜ) = vᵀ tanh(W₁hᵢ + W₂sₜ)  # Alignment score
   αₜᵢ = softmax(score(hᵢ, sₜ))         # Attention weight
   cₜ = Σᵢ αₜᵢ hᵢ                        # Context vector
   ```

3. **Decoder**: Use context vector
   ```
   yₜ = Decoder(sₜ, cₜ)
   ```

### 2.3 Alignment Scores

The alignment score measures how well input position i matches output position t:
- High score → Strong alignment
- Low score → Weak alignment

## 3. Attention as Information Retrieval

### 3.1 Database Analogy

Think of attention as a soft database lookup:

```python
# Traditional database (hard lookup)
database = {"cat": "animal", "sat": "action", ...}
result = database["cat"]  # Exact match

# Attention (soft lookup)
query = "feline"
scores = similarity(query, all_keys)
result = weighted_sum(all_values, weights=softmax(scores))
```

### 3.2 Content-Based Addressing

Unlike RNNs that use position-based addressing (step t can only see step t-1), attention uses content-based addressing:
- Find relevant information regardless of position
- Multiple positions can be relevant
- Relevance determined by content similarity

## 4. Self-Attention - The Game Changer

### 4.1 The Key Innovation

What if we let a sequence attend to itself? Each position can look at all other positions!

```
Traditional: Different sequences attend to each other
Self-attention: Same sequence attends to itself
```

### 4.2 Why It's Powerful

Consider: "The animal didn't cross the street because it was tired"

Self-attention helps resolve:
- "it" refers to "animal" (not "street")
- Captures long-range dependencies
- Bidirectional context without separate forward/backward passes

### 4.3 The Mathematics

For position i attending to all positions:

```
Attention(i) = Σⱼ softmax(score(i,j)) × value(j)
```

Where score(i,j) measures how much position i should attend to position j.

## 5. Query, Key, Value Framework

### 5.1 The Abstraction

Modern attention uses three components:
- **Query (Q)**: What information am I looking for?
- **Key (K)**: What information do I contain?
- **Value (V)**: What information should I return?

### 5.2 Real-World Analogy

Like a library system:
- **Query**: "Books about transformers"
- **Keys**: Book titles/topics in the catalog
- **Values**: The actual books

You compare your query against all keys, then retrieve the corresponding values.

### 5.3 Mathematical Formulation

```
Q = XWᵠ  # Queries
K = XWᵏ  # Keys  
V = XWᵛ  # Values

Attention(Q,K,V) = softmax(QKᵀ)V
```

### 5.4 Why Three Different Projections?

- **Flexibility**: Different representations for matching vs. retrieving
- **Capacity**: More parameters = more expressive power
- **Specialization**: Keys optimize for matching, values for content

## 6. Scaled Dot-Product Attention

### 6.1 The Formula

The attention mechanism used in transformers:

```
Attention(Q,K,V) = softmax(QKᵀ/√dₖ)V
```

### 6.2 Why Scale by √dₖ?

As dimension dₖ increases:
- Dot products grow larger
- Softmax becomes sharper (approaching one-hot)
- Gradients vanish

Scaling keeps values in reasonable range.

### 6.3 Step-by-Step Computation

1. **Compute scores**: `S = QKᵀ`
2. **Scale**: `S = S / √dₖ`
3. **Apply softmax**: `A = softmax(S)`
4. **Weight values**: `Output = AV`

### 6.4 Complexity Analysis

- Time: O(n²d) where n is sequence length, d is dimension
- Space: O(n²) for attention matrix
- Parallelizable: All positions computed simultaneously

## 7. Attention Patterns and Interpretability

### 7.1 Common Attention Patterns

1. **Positional**: Attending to nearby positions
2. **Syntactic**: Attending to syntactically related words
3. **Semantic**: Attending to semantically related content
4. **Global**: Attending to special tokens (CLS, SEP)

### 7.2 Visualizing Attention

Attention weights form an n×n matrix showing how much each position attends to every other position:

```
        The  cat  sat  on  the  mat
The     0.8  0.1  0.0  0.0  0.1  0.0
cat     0.2  0.6  0.1  0.0  0.0  0.1  
sat     0.1  0.4  0.3  0.1  0.0  0.1
on      0.0  0.0  0.2  0.5  0.2  0.1
the     0.1  0.0  0.0  0.3  0.4  0.2
mat     0.0  0.2  0.1  0.1  0.2  0.4
```

### 7.3 Attention Head Specialization

Different attention heads learn different patterns:
- Head 1: Previous word
- Head 2: Subject-verb relationships  
- Head 3: Determiner-noun relationships
- Head 4: Long-range dependencies

## 8. Implementation Details

### 8.1 Efficient Implementation

```python
def scaled_dot_product_attention(Q, K, V, mask=None):
    # Q, K, V: [batch_size, seq_len, d_k]
    
    # 1. Compute scores
    scores = torch.matmul(Q, K.transpose(-2, -1))
    
    # 2. Scale
    d_k = Q.size(-1)
    scores = scores / math.sqrt(d_k)
    
    # 3. Apply mask (optional)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    
    # 4. Softmax
    attention_weights = F.softmax(scores, dim=-1)
    
    # 5. Apply attention to values
    output = torch.matmul(attention_weights, V)
    
    return output, attention_weights
```

### 8.2 Masking

Masks prevent attending to certain positions:
- **Padding mask**: Ignore padded positions
- **Look-ahead mask**: Prevent attending to future (for autoregressive models)

### 8.3 Computational Tricks

1. **Batched computation**: Process multiple sequences together
2. **Chunking**: For very long sequences
3. **Sparse attention**: Only attend to subset of positions
4. **Flash Attention**: Hardware-aware implementation

## 9. Types of Attention

### 9.1 Comparison

| Type | Query Source | Key/Value Source | Use Case |
|------|--------------|------------------|----------|
| Self-attention | Same sequence | Same sequence | Understanding context |
| Cross-attention | Sequence A | Sequence B | Translation, Q&A |
| Causal attention | Same sequence | Same (past only) | Language generation |

### 9.2 Additive vs Multiplicative

**Additive (Bahdanau)**:
```
score(q,k) = vᵀ tanh(W₁q + W₂k)
```

**Multiplicative (Luong/Transformer)**:
```
score(q,k) = qᵀk
```

Multiplicative is faster and works better in practice.

## 🔍 Historical Context

- **2014**: Bahdanau attention for NMT
- **2015**: Luong attention variants
- **2016**: Self-attention in "A Decomposable Attention Model"
- **2017**: "Attention Is All You Need" - Transformers

## 📊 Performance Impact

On WMT'14 English-German translation:
- RNN without attention: BLEU 16.5
- RNN with attention: BLEU 20.8
- Transformer (all attention): BLEU 28.4

## 🎯 Key Takeaways

1. **Attention solves the bottleneck**: Direct access to all positions
2. **Content-based addressing**: Find relevant information by similarity
3. **Parallelizable**: All positions processed simultaneously
4. **Interpretable**: Can visualize what model focuses on
5. **Foundation of transformers**: Self-attention enables the architecture

## 📝 Summary

You've learned how attention mechanisms work:
- **Why**: Solve RNN limitations (bottleneck, sequential processing)
- **What**: Weighted combination of values based on query-key similarity
- **How**: Compute scores, apply softmax, weight values
- **Types**: Self-attention, cross-attention, various scoring functions

Attention is the foundation that enables transformers to process sequences efficiently and effectively!

## ➡️ Next Steps

Ready to see how attention is used to build transformers? Head to [Topic 4: The Transformer Architecture](../04-transformer-architecture/) to learn how multi-head attention and other components create the complete model!