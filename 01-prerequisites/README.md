# Prerequisites & Mathematical Foundations

Welcome to the first topic in our transformer journey! This module covers the essential mathematical concepts and neural network fundamentals you'll need to understand transformers deeply.

## 🎯 Learning Objectives

By the end of this module, you will understand:
- Linear algebra operations crucial for transformers
- How neural networks learn through backpropagation
- Matrix operations and their computational efficiency
- The role of activation functions and gradients

## 📚 Table of Contents

1. [Linear Algebra Essentials](#1-linear-algebra-essentials)
2. [Neural Network Fundamentals](#2-neural-network-fundamentals)
3. [Backpropagation Deep Dive](#3-backpropagation-deep-dive)
4. [Computational Considerations](#4-computational-considerations)
5. [Python Implementation](#5-python-implementation)
6. [Exercises](#6-exercises)

## 1. Linear Algebra Essentials

### 1.1 Vectors and Matrices

Transformers are fundamentally about manipulating high-dimensional vectors and matrices. Let's start with the basics:

**Vector**: A 1-dimensional array of numbers
```
v = [1, 2, 3, 4]  # Shape: (4,)
```

**Matrix**: A 2-dimensional array of numbers
```
M = [[1, 2, 3],
     [4, 5, 6]]   # Shape: (2, 3)
```

### 1.2 Dot Product

The dot product is the foundation of attention mechanisms:

```
a · b = Σ(aᵢ × bᵢ)
```

For vectors a = [1, 2, 3] and b = [4, 5, 6]:
```
a · b = (1×4) + (2×5) + (3×6) = 4 + 10 + 18 = 32
```

**Why it matters**: The dot product measures similarity between vectors - a key concept in attention!

### 1.3 Matrix Multiplication

Matrix multiplication is how we transform embeddings:

```
C = A @ B
```

Where `C[i,j] = Σ(A[i,k] × B[k,j])`

**Rules**:
- A (m×n) @ B (n×p) = C (m×p)
- Inner dimensions must match!

### 1.4 Transpose

Flipping a matrix along its diagonal:

```
A = [[1, 2],     Aᵀ = [[1, 3],
     [3, 4]]           [2, 4]]
```

### 1.5 Broadcasting

NumPy/PyTorch automatically expand dimensions for element-wise operations:

```
[1, 2, 3] + 10 = [11, 12, 13]
```

## 2. Neural Network Fundamentals

### 2.1 The Perceptron

The building block of neural networks:

```
output = activation(W·x + b)
```

Where:
- `x`: input vector
- `W`: weight matrix
- `b`: bias vector
- `activation`: non-linear function

### 2.2 Activation Functions

**ReLU (Rectified Linear Unit)**:
```
ReLU(x) = max(0, x)
```

**Sigmoid**:
```
σ(x) = 1 / (1 + e^(-x))
```

**Tanh**:
```
tanh(x) = (e^x - e^(-x)) / (e^x + e^(-x))
```

**GELU (Gaussian Error Linear Unit)** - Used in transformers:
```
GELU(x) ≈ 0.5x(1 + tanh(√(2/π)(x + 0.044715x³)))
```

### 2.3 Loss Functions

**Mean Squared Error (MSE)**:
```
MSE = (1/n) Σ(yᵢ - ŷᵢ)²
```

**Cross-Entropy** (for classification):
```
CE = -Σ(yᵢ log(ŷᵢ))
```

## 3. Backpropagation Deep Dive

### 3.1 The Chain Rule

The heart of deep learning:

```
∂L/∂w = ∂L/∂z × ∂z/∂w
```

### 3.2 Computing Gradients

For a simple network: `z = Wx + b`, `y = σ(z)`, `L = (y - t)²`

1. **Forward pass**: Compute outputs
2. **Backward pass**: Compute gradients
   ```
   ∂L/∂y = 2(y - t)
   ∂L/∂z = ∂L/∂y × σ'(z)
   ∂L/∂W = ∂L/∂z × xᵀ
   ∂L/∂b = ∂L/∂z
   ```

### 3.3 Gradient Descent

Update rule:
```
W_new = W_old - α × ∂L/∂W
```

Where α is the learning rate.

### 3.4 Variants of Gradient Descent

**Stochastic Gradient Descent (SGD)**:
- Update after each sample

**Mini-batch Gradient Descent**:
- Update after small batches (typical in transformers)

**Adam Optimizer** (most common for transformers):
```
m_t = β₁m_{t-1} + (1-β₁)g_t       # Momentum
v_t = β₂v_{t-1} + (1-β₂)g_t²      # RMSprop
W_t = W_{t-1} - α × m_t/√(v_t + ε)
```

## 4. Computational Considerations

### 4.1 Memory Layout

Row-major vs Column-major storage affects performance:
- PyTorch uses row-major (C-style)
- Operations along contiguous memory are faster

### 4.2 Vectorization

Replace loops with matrix operations:

**Slow**:
```python
for i in range(n):
    for j in range(m):
        C[i,j] = A[i,j] + B[i,j]
```

**Fast**:
```python
C = A + B  # Vectorized
```

### 4.3 GPU Acceleration

GPUs excel at parallel matrix operations:
- Thousands of cores for parallel computation
- High memory bandwidth
- Optimized for matrix multiplication

## 5. Python Implementation

See `prerequisites.py` for complete implementations of:
- Vector and matrix operations
- Neural network forward/backward pass
- Gradient descent optimizers
- Performance comparisons

Key libraries we'll use:
```python
import numpy as np      # Numerical operations
import torch           # Deep learning framework
import matplotlib.pyplot as plt  # Visualization
```

## 6. Exercises

### 6.1 Pen and Paper

1. Compute the dot product of [1, 2, 3] and [4, 5, 6]
2. Multiply matrices:
   ```
   [[1, 2],    [[5, 6],
    [3, 4]]  @  [7, 8]]
   ```
3. Derive the gradient of MSE loss

### 6.2 Coding Exercises

1. Implement matrix multiplication from scratch
2. Build a 2-layer neural network using only NumPy
3. Implement backpropagation for your network
4. Compare your implementation with PyTorch

### 6.3 Conceptual Questions

1. Why do we need non-linear activation functions?
2. What happens if we initialize all weights to zero?
3. How does batch size affect gradient descent?
4. Why is the dot product useful for measuring similarity?

## 🔍 Further Reading

- [The Matrix Calculus You Need For Deep Learning](https://explained.ai/matrix-calculus/)
- [Understanding Backpropagation](http://neuralnetworksanddeeplearning.com/chap2.html)
- [PyTorch Autograd Tutorial](https://pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html)

## 📝 Summary

You've learned the mathematical foundations needed for transformers:
- **Linear algebra**: Vectors, matrices, dot products
- **Neural networks**: Forward pass, activations, loss functions
- **Backpropagation**: Chain rule, gradient computation
- **Optimization**: Gradient descent and its variants

These concepts will appear repeatedly as we build up to transformers. Make sure you're comfortable with matrix multiplication and the chain rule - they're the bread and butter of deep learning!

## ➡️ Next Steps

Ready to see why we need something better than RNNs? Head to [Topic 2: Introduction to Sequence Modeling](../02-sequence-modeling/) to understand the motivation behind attention mechanisms.