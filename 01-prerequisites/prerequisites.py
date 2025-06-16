"""
Prerequisites and Mathematical Foundations for Transformers
A comprehensive implementation of the mathematical concepts needed to understand transformers.
"""

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from typing import Tuple, List
import time


class LinearAlgebraBasics:
    """Demonstrates fundamental linear algebra operations."""
    
    @staticmethod
    def vector_operations():
        """Basic vector operations."""
        print("=== Vector Operations ===")
        
        # Creating vectors
        v1 = np.array([1, 2, 3])
        v2 = np.array([4, 5, 6])
        
        print(f"v1: {v1}")
        print(f"v2: {v2}")
        
        # Addition
        print(f"\nv1 + v2: {v1 + v2}")
        
        # Scalar multiplication
        print(f"2 * v1: {2 * v1}")
        
        # Dot product
        dot_product = np.dot(v1, v2)
        print(f"\nv1 · v2: {dot_product}")
        
        # Alternative dot product calculation
        manual_dot = sum(a * b for a, b in zip(v1, v2))
        print(f"Manual dot product: {manual_dot}")
        
        # Norm (magnitude)
        norm_v1 = np.linalg.norm(v1)
        print(f"\n||v1||: {norm_v1:.4f}")
        
        # Cosine similarity
        cos_sim = dot_product / (np.linalg.norm(v1) * np.linalg.norm(v2))
        print(f"Cosine similarity: {cos_sim:.4f}")
        
    @staticmethod
    def matrix_operations():
        """Basic matrix operations."""
        print("\n=== Matrix Operations ===")
        
        # Creating matrices
        A = np.array([[1, 2, 3],
                      [4, 5, 6]])
        B = np.array([[7, 8],
                      [9, 10],
                      [11, 12]])
        
        print(f"A shape: {A.shape}")
        print(f"A:\n{A}")
        print(f"\nB shape: {B.shape}")
        print(f"B:\n{B}")
        
        # Matrix multiplication
        C = A @ B  # or np.matmul(A, B)
        print(f"\nC = A @ B, shape: {C.shape}")
        print(f"C:\n{C}")
        
        # Element-wise operations
        A_squared = A ** 2
        print(f"\nA²:\n{A_squared}")
        
        # Transpose
        A_T = A.T
        print(f"\nAᵀ shape: {A_T.shape}")
        print(f"Aᵀ:\n{A_T}")
        
        # Broadcasting example
        vector = np.array([1, 2, 3])
        broadcasted = A + vector
        print(f"\nA + [1,2,3] (broadcasting):\n{broadcasted}")
        
    @staticmethod
    def attention_preview():
        """Preview of attention mechanism using dot products."""
        print("\n=== Attention Mechanism Preview ===")
        
        # Simulating word embeddings
        embeddings = {
            "cat": np.array([0.2, 0.8, 0.1]),
            "sat": np.array([0.1, 0.9, 0.2]),
            "mat": np.array([0.15, 0.85, 0.1])
        }
        
        # Query vector (what we're looking for)
        query = embeddings["cat"]
        
        # Compute attention scores
        scores = {}
        for word, embedding in embeddings.items():
            score = np.dot(query, embedding)
            scores[word] = score
            
        print("Attention scores (unnormalized):")
        for word, score in scores.items():
            print(f"  {word}: {score:.4f}")
        
        # Apply softmax for probabilities
        scores_array = np.array(list(scores.values()))
        attention_weights = np.exp(scores_array) / np.sum(np.exp(scores_array))
        
        print("\nAttention weights (after softmax):")
        for word, weight in zip(scores.keys(), attention_weights):
            print(f"  {word}: {weight:.4f}")


class NeuralNetworkFromScratch:
    """A simple 2-layer neural network implemented from scratch."""
    
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        """Initialize network with random weights."""
        # Xavier initialization
        self.W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2.0 / input_size)
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * np.sqrt(2.0 / hidden_size)
        self.b2 = np.zeros((1, output_size))
        
        # For storing intermediate values
        self.cache = {}
        
    def relu(self, x):
        """ReLU activation function."""
        return np.maximum(0, x)
    
    def relu_derivative(self, x):
        """Derivative of ReLU."""
        return (x > 0).astype(float)
    
    def sigmoid(self, x):
        """Sigmoid activation function."""
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    def sigmoid_derivative(self, x):
        """Derivative of sigmoid."""
        s = self.sigmoid(x)
        return s * (1 - s)
    
    def forward(self, X):
        """Forward pass through the network."""
        # Layer 1
        self.cache['X'] = X
        self.cache['Z1'] = X @ self.W1 + self.b1
        self.cache['A1'] = self.relu(self.cache['Z1'])
        
        # Layer 2
        self.cache['Z2'] = self.cache['A1'] @ self.W2 + self.b2
        self.cache['A2'] = self.sigmoid(self.cache['Z2'])
        
        return self.cache['A2']
    
    def backward(self, y_true, learning_rate=0.01):
        """Backward pass - compute gradients and update weights."""
        m = y_true.shape[0]  # number of samples
        
        # Output layer gradients
        dZ2 = self.cache['A2'] - y_true  # derivative of cross-entropy + sigmoid
        dW2 = (1/m) * self.cache['A1'].T @ dZ2
        db2 = (1/m) * np.sum(dZ2, axis=0, keepdims=True)
        
        # Hidden layer gradients
        dA1 = dZ2 @ self.W2.T
        dZ1 = dA1 * self.relu_derivative(self.cache['Z1'])
        dW1 = (1/m) * self.cache['X'].T @ dZ1
        db1 = (1/m) * np.sum(dZ1, axis=0, keepdims=True)
        
        # Update weights
        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2
        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1
        
        return dW1, db1, dW2, db2
    
    def compute_loss(self, y_pred, y_true):
        """Binary cross-entropy loss."""
        m = y_true.shape[0]
        # Add small epsilon to prevent log(0)
        y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)
        loss = -(1/m) * np.sum(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        return loss


class ActivationFunctions:
    """Visualization of common activation functions."""
    
    @staticmethod
    def plot_activations():
        """Plot common activation functions and their derivatives."""
        x = np.linspace(-5, 5, 100)
        
        fig, axes = plt.subplots(2, 4, figsize=(15, 8))
        fig.suptitle('Activation Functions and Their Derivatives', fontsize=16)
        
        # ReLU
        relu = lambda x: np.maximum(0, x)
        relu_prime = lambda x: (x > 0).astype(float)
        
        axes[0, 0].plot(x, relu(x), 'b-', linewidth=2)
        axes[0, 0].set_title('ReLU')
        axes[0, 0].grid(True, alpha=0.3)
        axes[1, 0].plot(x, relu_prime(x), 'r-', linewidth=2)
        axes[1, 0].set_title('ReLU Derivative')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Sigmoid
        sigmoid = lambda x: 1 / (1 + np.exp(-x))
        sigmoid_prime = lambda x: sigmoid(x) * (1 - sigmoid(x))
        
        axes[0, 1].plot(x, sigmoid(x), 'b-', linewidth=2)
        axes[0, 1].set_title('Sigmoid')
        axes[0, 1].grid(True, alpha=0.3)
        axes[1, 1].plot(x, sigmoid_prime(x), 'r-', linewidth=2)
        axes[1, 1].set_title('Sigmoid Derivative')
        axes[1, 1].grid(True, alpha=0.3)
        
        # Tanh
        tanh = lambda x: np.tanh(x)
        tanh_prime = lambda x: 1 - np.tanh(x)**2
        
        axes[0, 2].plot(x, tanh(x), 'b-', linewidth=2)
        axes[0, 2].set_title('Tanh')
        axes[0, 2].grid(True, alpha=0.3)
        axes[1, 2].plot(x, tanh_prime(x), 'r-', linewidth=2)
        axes[1, 2].set_title('Tanh Derivative')
        axes[1, 2].grid(True, alpha=0.3)
        
        # GELU (approximation)
        gelu = lambda x: 0.5 * x * (1 + np.tanh(np.sqrt(2/np.pi) * (x + 0.044715 * x**3)))
        gelu_prime = lambda x: 0.5 * (1 + np.tanh(np.sqrt(2/np.pi) * (x + 0.044715 * x**3))) + \
                               0.5 * x * (1 - np.tanh(np.sqrt(2/np.pi) * (x + 0.044715 * x**3))**2) * \
                               np.sqrt(2/np.pi) * (1 + 3 * 0.044715 * x**2)
        
        axes[0, 3].plot(x, gelu(x), 'b-', linewidth=2)
        axes[0, 3].set_title('GELU')
        axes[0, 3].grid(True, alpha=0.3)
        axes[1, 3].plot(x, gelu_prime(x), 'r-', linewidth=2)
        axes[1, 3].set_title('GELU Derivative')
        axes[1, 3].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()


class GradientDescentOptimizers:
    """Different gradient descent optimization algorithms."""
    
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate
        self.optimizers = {
            'sgd': self.sgd,
            'momentum': self.momentum,
            'adam': self.adam
        }
        
    def sgd(self, params, grads, state=None):
        """Vanilla stochastic gradient descent."""
        for param, grad in zip(params, grads):
            param -= self.learning_rate * grad
        return params, state
    
    def momentum(self, params, grads, state=None, beta=0.9):
        """SGD with momentum."""
        if state is None:
            state = [np.zeros_like(p) for p in params]
            
        for i, (param, grad) in enumerate(zip(params, grads)):
            state[i] = beta * state[i] + (1 - beta) * grad
            param -= self.learning_rate * state[i]
            
        return params, state
    
    def adam(self, params, grads, state=None, beta1=0.9, beta2=0.999, epsilon=1e-8):
        """Adam optimizer."""
        if state is None:
            state = {
                'm': [np.zeros_like(p) for p in params],
                'v': [np.zeros_like(p) for p in params],
                't': 0
            }
        
        state['t'] += 1
        t = state['t']
        
        for i, (param, grad) in enumerate(zip(params, grads)):
            # Update biased first moment estimate
            state['m'][i] = beta1 * state['m'][i] + (1 - beta1) * grad
            # Update biased second raw moment estimate
            state['v'][i] = beta2 * state['v'][i] + (1 - beta2) * grad**2
            
            # Compute bias-corrected first moment estimate
            m_hat = state['m'][i] / (1 - beta1**t)
            # Compute bias-corrected second raw moment estimate
            v_hat = state['v'][i] / (1 - beta2**t)
            
            # Update parameters
            param -= self.learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)
            
        return params, state
    
    def compare_optimizers(self, iterations=1000):
        """Compare different optimizers on a simple function."""
        # Simple quadratic function: f(x,y) = x^2 + 5y^2
        def f(x, y):
            return x**2 + 5*y**2
        
        def grad_f(x, y):
            return np.array([2*x, 10*y])
        
        # Starting point
        start = np.array([5.0, 5.0])
        
        results = {}
        
        for name, optimizer in self.optimizers.items():
            point = start.copy()
            trajectory = [point.copy()]
            state = None
            
            for _ in range(iterations):
                grad = grad_f(point[0], point[1])
                [point], state = optimizer([point], [grad], state)
                trajectory.append(point.copy())
                
            results[name] = np.array(trajectory)
        
        # Plotting
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Contour plot
        x = np.linspace(-6, 6, 100)
        y = np.linspace(-6, 6, 100)
        X, Y = np.meshgrid(x, y)
        Z = f(X, Y)
        
        contours = ax.contour(X, Y, Z, levels=20, alpha=0.4)
        ax.clabel(contours, inline=True, fontsize=8)
        
        # Plot trajectories
        colors = ['red', 'blue', 'green']
        for (name, trajectory), color in zip(results.items(), colors):
            ax.plot(trajectory[:, 0], trajectory[:, 1], 
                   'o-', color=color, label=name.upper(), 
                   markersize=3, alpha=0.7)
        
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title('Comparison of Optimization Algorithms')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.show()


class PerformanceComparison:
    """Compare performance of different implementations."""
    
    @staticmethod
    def matrix_multiplication_benchmark():
        """Compare manual vs NumPy vs PyTorch matrix multiplication."""
        print("\n=== Matrix Multiplication Performance ===")
        
        sizes = [10, 50, 100, 200, 500]
        
        for size in sizes:
            A = np.random.randn(size, size)
            B = np.random.randn(size, size)
            
            # Manual implementation (very slow for large matrices)
            if size <= 100:
                start = time.time()
                C_manual = np.zeros((size, size))
                for i in range(size):
                    for j in range(size):
                        for k in range(size):
                            C_manual[i, j] += A[i, k] * B[k, j]
                manual_time = time.time() - start
            else:
                manual_time = float('inf')
            
            # NumPy
            start = time.time()
            C_numpy = A @ B
            numpy_time = time.time() - start
            
            # PyTorch (CPU)
            A_torch = torch.from_numpy(A)
            B_torch = torch.from_numpy(B)
            start = time.time()
            C_torch = A_torch @ B_torch
            torch_time = time.time() - start
            
            print(f"\nSize {size}x{size}:")
            if size <= 100:
                print(f"  Manual: {manual_time:.6f}s")
            print(f"  NumPy:  {numpy_time:.6f}s")
            print(f"  PyTorch: {torch_time:.6f}s")


def demonstrate_backpropagation():
    """Step-by-step backpropagation example."""
    print("\n=== Backpropagation Example ===")
    
    # Simple network: input(2) -> hidden(3) -> output(1)
    np.random.seed(42)
    
    # Initialize weights
    W1 = np.random.randn(2, 3) * 0.5
    b1 = np.zeros((1, 3))
    W2 = np.random.randn(3, 1) * 0.5
    b2 = np.zeros((1, 1))
    
    # Sample input and target
    X = np.array([[0.5, 0.8]])  # Single sample
    y_true = np.array([[1.0]])
    
    print("Initial weights:")
    print(f"W1:\n{W1}")
    print(f"W2:\n{W2}")
    
    # Forward pass
    print("\n--- Forward Pass ---")
    Z1 = X @ W1 + b1
    A1 = np.maximum(0, Z1)  # ReLU
    Z2 = A1 @ W2 + b2
    y_pred = 1 / (1 + np.exp(-Z2))  # Sigmoid
    
    print(f"Hidden layer output (A1): {A1}")
    print(f"Prediction (y_pred): {y_pred[0, 0]:.4f}")
    print(f"Target (y_true): {y_true[0, 0]}")
    
    # Compute loss
    loss = -y_true * np.log(y_pred) - (1 - y_true) * np.log(1 - y_pred)
    print(f"Loss: {loss[0, 0]:.4f}")
    
    # Backward pass
    print("\n--- Backward Pass ---")
    
    # Output layer
    dL_dy = -y_true/y_pred + (1-y_true)/(1-y_pred)
    dy_dZ2 = y_pred * (1 - y_pred)  # Sigmoid derivative
    dZ2 = dL_dy * dy_dZ2
    
    dW2 = A1.T @ dZ2
    db2 = dZ2
    
    print(f"dW2:\n{dW2}")
    
    # Hidden layer
    dA1 = dZ2 @ W2.T
    dZ1 = dA1 * (Z1 > 0)  # ReLU derivative
    
    dW1 = X.T @ dZ1
    db1 = dZ1
    
    print(f"dW1:\n{dW1}")
    
    # Update weights
    learning_rate = 0.1
    W1_new = W1 - learning_rate * dW1
    W2_new = W2 - learning_rate * dW2
    
    print(f"\nUpdated W1:\n{W1_new}")
    print(f"Updated W2:\n{W2_new}")


def train_xor_problem():
    """Train a network to solve the XOR problem."""
    print("\n=== Training XOR Problem ===")
    
    # XOR dataset
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[0], [1], [1], [0]])
    
    # Create and train network
    nn = NeuralNetworkFromScratch(input_size=2, hidden_size=4, output_size=1)
    
    losses = []
    
    for epoch in range(5000):
        # Forward pass
        y_pred = nn.forward(X)
        
        # Compute loss
        loss = nn.compute_loss(y_pred, y)
        losses.append(loss)
        
        # Backward pass
        nn.backward(y, learning_rate=0.5)
        
        if epoch % 1000 == 0:
            print(f"Epoch {epoch}, Loss: {loss:.4f}")
    
    # Final predictions
    print("\nFinal predictions:")
    final_pred = nn.forward(X)
    for i, (input_val, pred, true) in enumerate(zip(X, final_pred, y)):
        print(f"Input: {input_val}, Predicted: {pred[0]:.4f}, True: {true[0]}")
    
    # Plot loss curve
    plt.figure(figsize=(10, 6))
    plt.plot(losses)
    plt.title('Training Loss for XOR Problem')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    plt.show()


if __name__ == "__main__":
    # Run all demonstrations
    print("=" * 60)
    print("PREREQUISITES FOR TRANSFORMERS")
    print("=" * 60)
    
    # Linear algebra basics
    la = LinearAlgebraBasics()
    la.vector_operations()
    la.matrix_operations()
    la.attention_preview()
    
    # Activation functions
    print("\nPlotting activation functions...")
    ActivationFunctions.plot_activations()
    
    # Backpropagation
    demonstrate_backpropagation()
    
    # Train XOR
    train_xor_problem()
    
    # Optimizer comparison
    print("\nComparing optimizers...")
    opt = GradientDescentOptimizers(learning_rate=0.1)
    opt.compare_optimizers()
    
    # Performance comparison
    PerformanceComparison.matrix_multiplication_benchmark()
    
    print("\n" + "=" * 60)
    print("Congratulations! You've completed the prerequisites.")
    print("Next: Learn about sequence modeling and why we need attention!")
    print("=" * 60)