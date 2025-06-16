"""
Attention Mechanisms
Complete implementation of various attention mechanisms from scratch.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, Tuple, Dict
import math


class BahdanauAttention:
    """Original additive attention mechanism from Bahdanau et al. 2014."""
    
    def __init__(self, hidden_size: int):
        self.hidden_size = hidden_size
        
        # Learnable parameters
        self.W1 = np.random.randn(hidden_size, hidden_size) * 0.01
        self.W2 = np.random.randn(hidden_size, hidden_size) * 0.01
        self.v = np.random.randn(hidden_size, 1) * 0.01
        
    def forward(self, query: np.ndarray, keys: np.ndarray, values: np.ndarray):
        """
        Args:
            query: [hidden_size] - decoder hidden state
            keys: [seq_len, hidden_size] - encoder hidden states
            values: [seq_len, hidden_size] - encoder hidden states (usually same as keys)
            
        Returns:
            context: [hidden_size] - weighted sum of values
            weights: [seq_len] - attention weights
        """
        seq_len = keys.shape[0]
        
        # Expand query to match keys shape
        query_expanded = np.repeat(query.reshape(1, -1), seq_len, axis=0)
        
        # Compute alignment scores
        # score = v^T * tanh(W1*h_i + W2*s_t)
        scores = []
        for i in range(seq_len):
            score = self.v.T @ np.tanh(self.W1 @ keys[i] + self.W2 @ query)
            scores.append(score[0])
        
        scores = np.array(scores)
        
        # Apply softmax to get weights
        weights = self._softmax(scores)
        
        # Compute weighted sum of values
        context = np.sum(values * weights.reshape(-1, 1), axis=0)
        
        return context, weights
    
    def _softmax(self, x):
        """Stable softmax implementation."""
        exp_x = np.exp(x - np.max(x))
        return exp_x / np.sum(exp_x)


class LuongAttention:
    """Multiplicative attention variants from Luong et al. 2015."""
    
    def __init__(self, hidden_size: int, method: str = 'dot'):
        self.hidden_size = hidden_size
        self.method = method
        
        if method == 'general':
            self.W = np.random.randn(hidden_size, hidden_size) * 0.01
        elif method == 'concat':
            self.W = np.random.randn(hidden_size * 2, hidden_size) * 0.01
            self.v = np.random.randn(hidden_size, 1) * 0.01
            
    def forward(self, query: np.ndarray, keys: np.ndarray, values: np.ndarray):
        """
        Compute attention using different scoring methods.
        """
        if self.method == 'dot':
            # score(s_t, h_i) = s_t^T * h_i
            scores = keys @ query
            
        elif self.method == 'general':
            # score(s_t, h_i) = s_t^T * W * h_i
            scores = keys @ (self.W @ query)
            
        elif self.method == 'concat':
            # score(s_t, h_i) = v^T * tanh(W[s_t; h_i])
            seq_len = keys.shape[0]
            scores = []
            for i in range(seq_len):
                concat = np.concatenate([query, keys[i]])
                score = self.v.T @ np.tanh(self.W @ concat)
                scores.append(score[0])
            scores = np.array(scores)
        
        # Apply softmax
        weights = self._softmax(scores)
        
        # Weighted sum
        context = np.sum(values * weights.reshape(-1, 1), axis=0)
        
        return context, weights
    
    def _softmax(self, x):
        exp_x = np.exp(x - np.max(x))
        return exp_x / np.sum(exp_x)


class SelfAttention:
    """Self-attention mechanism - the foundation of transformers."""
    
    def __init__(self, d_model: int, d_k: int = None):
        self.d_model = d_model
        self.d_k = d_k or d_model
        
        # Linear projections for Q, K, V
        self.W_q = np.random.randn(d_model, self.d_k) * np.sqrt(2.0 / d_model)
        self.W_k = np.random.randn(d_model, self.d_k) * np.sqrt(2.0 / d_model)
        self.W_v = np.random.randn(d_model, self.d_k) * np.sqrt(2.0 / d_model)
        
    def forward(self, x: np.ndarray, mask: Optional[np.ndarray] = None):
        """
        Args:
            x: [seq_len, d_model] - input sequence
            mask: [seq_len, seq_len] - attention mask (0 for positions to mask)
            
        Returns:
            output: [seq_len, d_k] - attended values
            weights: [seq_len, seq_len] - attention weights
        """
        seq_len = x.shape[0]
        
        # 1. Project to Q, K, V
        Q = x @ self.W_q  # [seq_len, d_k]
        K = x @ self.W_k  # [seq_len, d_k]
        V = x @ self.W_v  # [seq_len, d_k]
        
        # 2. Compute attention scores
        scores = Q @ K.T  # [seq_len, seq_len]
        
        # 3. Scale scores
        scores = scores / np.sqrt(self.d_k)
        
        # 4. Apply mask if provided
        if mask is not None:
            scores = np.where(mask == 0, -1e9, scores)
        
        # 5. Apply softmax
        weights = self._softmax_2d(scores)
        
        # 6. Apply attention to values
        output = weights @ V  # [seq_len, d_k]
        
        return output, weights
    
    def _softmax_2d(self, x):
        """Softmax over last dimension."""
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


class ScaledDotProductAttention(nn.Module):
    """PyTorch implementation of scaled dot-product attention."""
    
    def __init__(self, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            Q: [batch, heads, seq_len, d_k]
            K: [batch, heads, seq_len, d_k]
            V: [batch, heads, seq_len, d_v]
            mask: [batch, 1, 1, seq_len] or [batch, 1, seq_len, seq_len]
            
        Returns:
            output: [batch, heads, seq_len, d_v]
            attention: [batch, heads, seq_len, seq_len]
        """
        d_k = Q.size(-1)
        
        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
        
        # Apply mask
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Apply softmax
        attention = F.softmax(scores, dim=-1)
        attention = self.dropout(attention)
        
        # Apply attention to values
        output = torch.matmul(attention, V)
        
        return output, attention


class MultiQueryAttention:
    """Multi-Query Attention - a more efficient variant."""
    
    def __init__(self, d_model: int, n_heads: int):
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        # Multiple queries, single key and value
        self.W_q = np.random.randn(d_model, d_model) * 0.01
        self.W_k = np.random.randn(d_model, self.d_k) * 0.01
        self.W_v = np.random.randn(d_model, self.d_k) * 0.01
        
    def forward(self, x: np.ndarray):
        """Compute multi-query attention."""
        seq_len = x.shape[0]
        
        # Project queries (multi-head)
        Q = x @ self.W_q  # [seq_len, d_model]
        Q = Q.reshape(seq_len, self.n_heads, self.d_k)  # [seq_len, n_heads, d_k]
        
        # Project keys and values (single)
        K = x @ self.W_k  # [seq_len, d_k]
        V = x @ self.W_v  # [seq_len, d_k]
        
        outputs = []
        weights_list = []
        
        # Compute attention for each head
        for h in range(self.n_heads):
            Q_h = Q[:, h, :]  # [seq_len, d_k]
            
            # Standard attention computation
            scores = Q_h @ K.T / np.sqrt(self.d_k)
            weights = self._softmax_2d(scores)
            output = weights @ V
            
            outputs.append(output)
            weights_list.append(weights)
        
        # Concatenate heads
        output = np.concatenate(outputs, axis=-1)
        
        return output, weights_list
    
    def _softmax_2d(self, x):
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


class AttentionVisualizer:
    """Utilities for visualizing attention patterns."""
    
    @staticmethod
    def plot_attention_weights(weights: np.ndarray, 
                             input_labels: list = None,
                             output_labels: list = None,
                             title: str = "Attention Weights"):
        """
        Plot attention weight matrix as a heatmap.
        
        Args:
            weights: [output_len, input_len] attention weights
            input_labels: Labels for input sequence
            output_labels: Labels for output sequence
        """
        plt.figure(figsize=(10, 8))
        
        # Create heatmap
        sns.heatmap(weights, 
                   cmap='Blues',
                   cbar=True,
                   square=True,
                   xticklabels=input_labels or range(weights.shape[1]),
                   yticklabels=output_labels or range(weights.shape[0]),
                   vmin=0,
                   vmax=1)
        
        plt.title(title)
        plt.xlabel('Input Positions')
        plt.ylabel('Output Positions')
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def plot_attention_patterns(patterns: Dict[str, np.ndarray]):
        """Plot multiple attention patterns side by side."""
        n_patterns = len(patterns)
        fig, axes = plt.subplots(1, n_patterns, figsize=(5*n_patterns, 5))
        
        if n_patterns == 1:
            axes = [axes]
        
        for ax, (name, weights) in zip(axes, patterns.items()):
            im = ax.imshow(weights, cmap='Blues', aspect='auto', vmin=0, vmax=1)
            ax.set_title(name)
            ax.set_xlabel('Keys')
            ax.set_ylabel('Queries')
            plt.colorbar(im, ax=ax)
        
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def create_attention_masks():
        """Create common attention mask patterns."""
        seq_len = 8
        
        masks = {}
        
        # 1. No mask (full attention)
        masks['Full Attention'] = np.ones((seq_len, seq_len))
        
        # 2. Causal mask (autoregressive)
        masks['Causal Mask'] = np.tril(np.ones((seq_len, seq_len)))
        
        # 3. Dilated attention
        dilated = np.zeros((seq_len, seq_len))
        for i in range(seq_len):
            for j in range(seq_len):
                if abs(i - j) % 2 == 0:
                    dilated[i, j] = 1
        masks['Dilated Attention'] = dilated
        
        # 4. Local attention (window)
        window_size = 3
        local = np.zeros((seq_len, seq_len))
        for i in range(seq_len):
            for j in range(max(0, i-window_size), min(seq_len, i+window_size+1)):
                local[i, j] = 1
        masks['Local Attention'] = local
        
        # 5. Random sparse
        random_sparse = np.random.binomial(1, 0.3, (seq_len, seq_len))
        random_sparse = np.maximum(random_sparse, random_sparse.T)  # Symmetric
        masks['Random Sparse'] = random_sparse
        
        return masks


def demonstrate_attention_computation():
    """Step-by-step attention computation demonstration."""
    print("=== Attention Computation Walkthrough ===\n")
    
    # Simple example
    seq_len = 4
    d_model = 6
    d_k = 3
    
    # Input sequence (e.g., word embeddings)
    x = np.random.randn(seq_len, d_model)
    
    # Initialize self-attention
    attention = SelfAttention(d_model, d_k)
    
    # Forward pass
    output, weights = attention.forward(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Attention weights shape: {weights.shape}")
    
    print("\nAttention weights (how much each position attends to others):")
    print(weights)
    print(f"\nRow sums (should be 1.0): {weights.sum(axis=1)}")
    
    # Visualize
    labels = [f"Pos {i}" for i in range(seq_len)]
    AttentionVisualizer.plot_attention_weights(weights, labels, labels, 
                                              "Self-Attention Pattern")


def compare_attention_mechanisms():
    """Compare different attention mechanisms."""
    print("=== Comparing Attention Mechanisms ===\n")
    
    # Setup
    hidden_size = 8
    seq_len = 6
    
    # Create dummy encoder outputs
    encoder_outputs = np.random.randn(seq_len, hidden_size)
    decoder_hidden = np.random.randn(hidden_size)
    
    # 1. Bahdanau (Additive) Attention
    bahdanau = BahdanauAttention(hidden_size)
    context_b, weights_b = bahdanau.forward(decoder_hidden, encoder_outputs, encoder_outputs)
    
    # 2. Luong (Multiplicative) Attention
    luong = LuongAttention(hidden_size, method='dot')
    context_l, weights_l = luong.forward(decoder_hidden, encoder_outputs, encoder_outputs)
    
    # 3. Self-Attention
    self_attn = SelfAttention(hidden_size, hidden_size)
    # For comparison, we'll look at first position attending to all
    x = encoder_outputs
    output_s, weights_s_full = self_attn.forward(x)
    weights_s = weights_s_full[0, :]  # First position's attention
    
    # Visualize all three
    patterns = {
        'Bahdanau\n(Additive)': weights_b.reshape(1, -1),
        'Luong\n(Multiplicative)': weights_l.reshape(1, -1),
        'Self-Attention\n(Position 0)': weights_s.reshape(1, -1)
    }
    
    AttentionVisualizer.plot_attention_patterns(patterns)
    
    print("Key differences:")
    print("- Bahdanau: Uses a feedforward network to compute alignment")
    print("- Luong: Simple dot product (faster)")
    print("- Self-Attention: Every position attends to every other position")


def demonstrate_masking():
    """Show how different masks affect attention patterns."""
    print("=== Attention Masking Patterns ===\n")
    
    # Create masks
    masks = AttentionVisualizer.create_attention_masks()
    
    # Visualize
    AttentionVisualizer.plot_attention_patterns(masks)
    
    # Demonstrate causal masking effect
    seq_len = 6
    d_model = 8
    
    x = np.random.randn(seq_len, d_model)
    attention = SelfAttention(d_model)
    
    # Without mask
    output_no_mask, weights_no_mask = attention.forward(x)
    
    # With causal mask
    causal_mask = np.tril(np.ones((seq_len, seq_len)))
    output_masked, weights_masked = attention.forward(x, causal_mask)
    
    # Compare
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    im1 = ax1.imshow(weights_no_mask, cmap='Blues', vmin=0, vmax=1)
    ax1.set_title('Without Mask (Bidirectional)')
    ax1.set_xlabel('Keys')
    ax1.set_ylabel('Queries')
    plt.colorbar(im1, ax=ax1)
    
    im2 = ax2.imshow(weights_masked, cmap='Blues', vmin=0, vmax=1)
    ax2.set_title('With Causal Mask (Autoregressive)')
    ax2.set_xlabel('Keys')
    ax2.set_ylabel('Queries')
    plt.colorbar(im2, ax=ax2)
    
    plt.tight_layout()
    plt.show()


def analyze_attention_properties():
    """Analyze mathematical properties of attention."""
    print("=== Attention Properties Analysis ===\n")
    
    # 1. Permutation equivariance
    print("1. Permutation Equivariance Test:")
    seq_len = 5
    d_model = 8
    x = np.random.randn(seq_len, d_model)
    
    attention = SelfAttention(d_model)
    output1, _ = attention.forward(x)
    
    # Permute input
    perm = np.random.permutation(seq_len)
    x_perm = x[perm]
    output2, _ = attention.forward(x_perm)
    
    # Check if output is also permuted
    output1_perm = output1[perm]
    print(f"Outputs match after permutation: {np.allclose(output1_perm, output2)}")
    
    # 2. Attention weight properties
    print("\n2. Attention Weight Properties:")
    _, weights = attention.forward(x)
    
    print(f"Weights sum to 1: {np.allclose(weights.sum(axis=1), 1.0)}")
    print(f"Weights non-negative: {np.all(weights >= 0)}")
    print(f"Weights shape: {weights.shape} (square matrix)")
    
    # 3. Effect of scaling
    print("\n3. Effect of Temperature Scaling:")
    temps = [0.1, 0.5, 1.0, 2.0, 5.0]
    
    fig, axes = plt.subplots(1, len(temps), figsize=(15, 3))
    
    for idx, temp in enumerate(temps):
        # Modify attention computation with temperature
        Q = x @ attention.W_q
        K = x @ attention.W_k
        scores = (Q @ K.T) / (np.sqrt(attention.d_k) * temp)
        
        exp_scores = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
        weights_temp = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)
        
        axes[idx].imshow(weights_temp, cmap='Blues', vmin=0, vmax=1)
        axes[idx].set_title(f'T={temp}')
        
        # Calculate entropy
        entropy = -np.sum(weights_temp * np.log(weights_temp + 1e-9)) / seq_len
        axes[idx].set_xlabel(f'Entropy: {entropy:.2f}')
    
    plt.suptitle('Temperature Effect on Attention Distribution')
    plt.tight_layout()
    plt.show()
    
    print("\nLower temperature → Sharper attention (more focused)")
    print("Higher temperature → Softer attention (more distributed)")


def benchmark_attention_implementations():
    """Compare performance of different implementations."""
    print("=== Performance Benchmark ===\n")
    
    import time
    
    seq_lengths = [10, 50, 100, 200]
    d_model = 64
    
    numpy_times = []
    torch_times = []
    
    for seq_len in seq_lengths:
        # NumPy implementation
        x_np = np.random.randn(seq_len, d_model)
        attention_np = SelfAttention(d_model)
        
        start = time.time()
        for _ in range(10):
            attention_np.forward(x_np)
        numpy_time = (time.time() - start) / 10
        numpy_times.append(numpy_time)
        
        # PyTorch implementation
        x_torch = torch.randn(1, 1, seq_len, d_model)
        attention_torch = ScaledDotProductAttention()
        
        start = time.time()
        for _ in range(10):
            with torch.no_grad():
                attention_torch(x_torch, x_torch, x_torch)
        torch_time = (time.time() - start) / 10
        torch_times.append(torch_time)
    
    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(seq_lengths, numpy_times, 'b-o', label='NumPy')
    plt.plot(seq_lengths, torch_times, 'g-o', label='PyTorch')
    plt.xlabel('Sequence Length')
    plt.ylabel('Time (seconds)')
    plt.title('Attention Computation Time')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    print("Complexity Analysis:")
    print(f"Time complexity: O(n²d) where n=sequence length, d=dimension")
    print(f"Space complexity: O(n²) for attention matrix")


def demonstrate_query_key_value():
    """Demonstrate the Query-Key-Value framework."""
    print("=== Query-Key-Value Framework ===\n")
    
    # Example: Information retrieval analogy
    print("Library Analogy:")
    print("- Query: What you're looking for")
    print("- Keys: Index/catalog entries")
    print("- Values: Actual content\n")
    
    # Simplified example
    vocab = ["cat", "dog", "sat", "mat", "the"]
    d_model = 4
    
    # Create embeddings (values)
    embeddings = np.random.randn(len(vocab), d_model)
    
    # Create keys (what each word represents)
    keys = embeddings @ np.random.randn(d_model, d_model)
    
    # Create a query (looking for animals)
    query = np.array([1.0, 1.0, -1.0, -1.0])  # High for animals, low for others
    
    # Compute attention
    scores = keys @ query
    weights = np.exp(scores) / np.sum(np.exp(scores))
    
    print("Words:", vocab)
    print("Attention weights:", weights)
    print(f"\nHighest attention: '{vocab[np.argmax(weights)]}' ({weights[np.argmax(weights)]:.3f})")
    
    # Visualize
    plt.figure(figsize=(8, 6))
    plt.bar(vocab, weights)
    plt.title('Attention Weights for Query "looking for animals"')
    plt.ylabel('Attention Weight')
    plt.xlabel('Words')
    plt.ylim(0, max(weights) * 1.2)
    for i, w in enumerate(weights):
        plt.text(i, w + 0.01, f'{w:.3f}', ha='center')
    plt.show()


if __name__ == "__main__":
    print("=" * 60)
    print("ATTENTION MECHANISMS DEMONSTRATIONS")
    print("=" * 60)
    
    # 1. Basic attention computation
    demonstrate_attention_computation()
    print("\n" + "=" * 60 + "\n")
    
    # 2. Compare different mechanisms
    compare_attention_mechanisms()
    print("\n" + "=" * 60 + "\n")
    
    # 3. Masking patterns
    demonstrate_masking()
    print("\n" + "=" * 60 + "\n")
    
    # 4. Mathematical properties
    analyze_attention_properties()
    print("\n" + "=" * 60 + "\n")
    
    # 5. Query-Key-Value
    demonstrate_query_key_value()
    print("\n" + "=" * 60 + "\n")
    
    # 6. Performance benchmark
    benchmark_attention_implementations()
    
    print("\n" + "=" * 60)
    print("Key Insights:")
    print("- Attention allows flexible, content-based information retrieval")
    print("- Self-attention enables parallel processing of sequences")
    print("- Query-Key-Value framework provides a general abstraction")
    print("- Masking controls information flow patterns")
    print("- Scaling prevents gradient issues in deep networks")
    print("=" * 60)