# Introduction to Sequence Modeling

Welcome to Topic 2! Now that you understand the mathematical foundations, let's explore why transformers were invented. We'll see the limitations of RNNs and understand why attention mechanisms were revolutionary.

## 🎯 Learning Objectives

By the end of this module, you will understand:
- Why sequence modeling is challenging
- How RNNs work and their fundamental limitations
- The vanishing/exploding gradient problem
- Why parallelization matters for modern AI
- The motivation for attention mechanisms

## 📚 Table of Contents

1. [What is Sequence Modeling?](#1-what-is-sequence-modeling)
2. [Recurrent Neural Networks (RNNs)](#2-recurrent-neural-networks-rnns)
3. [Long Short-Term Memory (LSTM)](#3-long-short-term-memory-lstm)
4. [The Fundamental Problems](#4-the-fundamental-problems)
5. [Sequence-to-Sequence Models](#5-sequence-to-sequence-models)
6. [Why We Need Something Better](#6-why-we-need-something-better)
7. [Exercises](#7-exercises)

## 1. What is Sequence Modeling?

### 1.1 Sequential Data is Everywhere

Sequential data has temporal or positional dependencies:
- **Natural Language**: Words depend on previous words
- **Time Series**: Stock prices, weather patterns
- **Audio**: Speech, music
- **Video**: Frames in sequence
- **DNA**: Genetic sequences

### 1.2 The Challenge

Unlike fixed-size inputs (images), sequences have:
- **Variable length**: Sentences can be any length
- **Long-range dependencies**: "The cat, which sat on the mat that was in the house, was black"
- **Order matters**: "Dog bites man" ≠ "Man bites dog"

### 1.3 Traditional Approaches

**Bag of Words**: Ignore order completely
```
"The cat sat" → {"the": 1, "cat": 1, "sat": 1}
"Sat the cat" → {"the": 1, "cat": 1, "sat": 1}  # Same representation!
```

**N-grams**: Fixed-size windows
```
"The cat sat on the mat"
Bigrams: ["the cat", "cat sat", "sat on", "on the", "the mat"]
```

Problems:
- Sparsity: Most n-grams never appear
- No generalization: "the cat" and "a cat" are completely different
- Fixed context window

## 2. Recurrent Neural Networks (RNNs)

### 2.1 The RNN Architecture

RNNs process sequences step by step, maintaining a hidden state:

```
h_t = tanh(W_hh @ h_{t-1} + W_xh @ x_t + b_h)
y_t = W_hy @ h_t + b_y
```

Where:
- `x_t`: Input at time t
- `h_t`: Hidden state at time t
- `y_t`: Output at time t
- `W_hh`: Hidden-to-hidden weights (shared across time)
- `W_xh`: Input-to-hidden weights (shared across time)
- `W_hy`: Hidden-to-output weights (shared across time)

### 2.2 Processing a Sequence

For sequence "The cat sat":
1. Process "The": h₁ = f(x₁, h₀)
2. Process "cat": h₂ = f(x₂, h₁)  # h₁ contains info about "The"
3. Process "sat": h₃ = f(x₃, h₂)  # h₂ contains info about "The cat"

### 2.3 RNN Unrolling

Conceptually, we "unroll" the RNN across time:

```
     x₁ → [RNN] → h₁
            ↓
     x₂ → [RNN] → h₂
            ↓
     x₃ → [RNN] → h₃
```

### 2.4 Backpropagation Through Time (BPTT)

To train RNNs, we backpropagate through the unrolled network:

```
∂L/∂W = Σ_t ∂L_t/∂W
```

This requires computing gradients through many timesteps.

## 3. Long Short-Term Memory (LSTM)

### 3.1 The Vanishing Gradient Problem

In vanilla RNNs, gradients can vanish or explode:

```
∂h_t/∂h_0 = Π_{k=1}^t ∂h_k/∂h_{k-1}
```

If these derivatives are < 1, gradients vanish exponentially.
If they're > 1, gradients explode exponentially.

### 3.2 LSTM Architecture

LSTMs use gates to control information flow:

```python
# Forget gate: what to forget from previous state
f_t = σ(W_f @ [h_{t-1}, x_t] + b_f)

# Input gate: what new information to store
i_t = σ(W_i @ [h_{t-1}, x_t] + b_i)
C̃_t = tanh(W_C @ [h_{t-1}, x_t] + b_C)

# Update cell state
C_t = f_t * C_{t-1} + i_t * C̃_t

# Output gate: what to output based on cell state
o_t = σ(W_o @ [h_{t-1}, x_t] + b_o)
h_t = o_t * tanh(C_t)
```

### 3.3 Why LSTMs Help (But Don't Solve Everything)

- **Cell state**: Highway for gradients
- **Gates**: Learn what to remember/forget
- **But**: Still sequential, can't parallelize

## 4. The Fundamental Problems

### 4.1 Sequential Computation

RNNs must process tokens one by one:

```
Time step 1: Process x₁ → h₁
Time step 2: Process x₂ (needs h₁) → h₂
Time step 3: Process x₃ (needs h₂) → h₃
...
```

**Cannot parallelize!** Each step depends on the previous.

### 4.2 Information Bottleneck

All information must pass through fixed-size hidden states:

```
"The cat sat on the mat in the house that Jack built" → h_final (fixed size!)
```

Like compressing a movie into a single frame!

### 4.3 Long-Range Dependencies

Information degrades over many steps:

```
"The cat [50 words...] was black"
         ↑________________________↓
         Must remember "cat" across 50 steps!
```

### 4.4 Computational Complexity

For sequence length n:
- Time complexity: O(n) - must process sequentially
- Memory during training: O(n) - must store all hidden states
- Gradient computation: O(n) - backprop through all steps

## 5. Sequence-to-Sequence Models

### 5.1 The Encoder-Decoder Architecture

For tasks like translation:

```
Encoder RNN: "The cat sat" → h_final (context vector)
Decoder RNN: h_final → "Le chat s'est assis"
```

### 5.2 The Context Vector Problem

Entire source sentence compressed into one vector!

```
Source: "The quick brown fox jumps over the lazy dog near the river bank"
Context: [0.2, -0.5, 0.8, ...]  # Fixed size, regardless of length!
```

### 5.3 Performance Degradation

As sequences get longer:
- Translation quality drops
- Information is lost
- Especially bad for long documents

## 6. Why We Need Something Better

### 6.1 Modern Requirements

Today's models need to:
- Process thousands of tokens (entire documents)
- Train on massive datasets efficiently
- Capture long-range dependencies
- Run on parallel hardware (GPUs/TPUs)

### 6.2 RNN Limitations Summary

| Problem | Impact | Why It Matters |
|---------|---------|----------------|
| Sequential computation | Can't parallelize | Training is slow |
| Information bottleneck | Fixed-size states | Loses information |
| Vanishing gradients | Can't learn long-range | Forgets context |
| Training instability | Exploding gradients | Hard to train |

### 6.3 The Dream Solution

What if we could:
1. Look at all positions simultaneously?
2. Learn which positions to attend to?
3. Process in parallel?
4. Handle any sequence length?

**This is exactly what attention mechanisms provide!**

### 6.4 Preview: Attention

Instead of sequential processing:
```
RNN: word₁ → word₂ → word₃ → ... → wordₙ
```

Attention allows:
```
Each word can directly look at every other word!
word_i ← → word_j for all i, j
```

## 7. Practical Examples

### 7.1 RNN Failure Case

Try to predict the last word:
```
"The student, who had studied hard for the exam and spent countless 
hours in the library reading through multiple textbooks and research 
papers while also attending every lecture and tutorial session, 
finally ______"

RNN: Likely forgot "student" by now!
Attention: Can look back directly at "student"
```

### 7.2 Translation Alignment

English: "The agreement on the European Economic Area was signed in May 1992."
French: "L'accord sur la zone économique européenne a été signé en mai 1992."

RNN must encode everything before translating.
Attention can align "European" ↔ "européenne" directly!

## 📊 Empirical Evidence

Performance on WMT'14 English-German translation:
- RNN (2014): BLEU score ~16
- RNN + Attention (2015): BLEU score ~20
- Transformer (2017): BLEU score ~28

Training time for equivalent performance:
- RNN: Days to weeks
- Transformer: Hours to days

## 🔍 Further Reading

- [Understanding LSTM Networks](https://colah.github.io/posts/2015-08-Understanding-LSTMs/) - Chris Olah's excellent visualization
- [The Unreasonable Effectiveness of RNNs](http://karpathy.github.io/2015/05/21/rnn-effectiveness/) - Andrej Karpathy
- [Visualizing and Understanding RNNs](https://arxiv.org/abs/1506.02078) - What RNNs learn

## 📝 Summary

You've learned why we need better sequence models:
- **RNNs** process sequentially, limiting parallelization
- **Vanishing gradients** prevent learning long-range dependencies
- **Information bottlenecks** compress everything into fixed-size vectors
- **Modern tasks** require processing longer sequences efficiently

These limitations motivated the development of attention mechanisms, which we'll explore in the next topic!

## ➡️ Next Steps

Ready to learn how attention solves these problems? Head to [Topic 3: Attention Mechanisms](../03-attention-mechanisms/) to discover the key innovation that made transformers possible!