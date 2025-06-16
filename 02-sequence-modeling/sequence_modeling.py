"""
Introduction to Sequence Modeling
Implementing RNNs, LSTMs, and demonstrating their limitations.
"""

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict
import time
from IPython.display import clear_output


class VanillaRNN:
    """A vanilla RNN implemented from scratch in NumPy."""
    
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        # Initialize weights
        self.hidden_size = hidden_size
        
        # Xavier initialization
        self.Wxh = np.random.randn(hidden_size, input_size) * np.sqrt(2.0 / input_size)
        self.Whh = np.random.randn(hidden_size, hidden_size) * np.sqrt(2.0 / hidden_size)
        self.Why = np.random.randn(output_size, hidden_size) * np.sqrt(2.0 / hidden_size)
        
        self.bh = np.zeros((hidden_size, 1))
        self.by = np.zeros((output_size, 1))
        
        # For gradient computation
        self.last_inputs = None
        self.last_hiddens = None
        
    def forward(self, inputs: List[np.ndarray], h_prev: np.ndarray = None):
        """
        Forward pass through the RNN.
        
        Args:
            inputs: List of input vectors (each is column vector)
            h_prev: Initial hidden state
            
        Returns:
            outputs: List of output vectors
            hiddens: List of hidden states
        """
        if h_prev is None:
            h_prev = np.zeros((self.hidden_size, 1))
            
        hiddens = [h_prev]
        outputs = []
        
        for x in inputs:
            # h_t = tanh(Wxh @ x + Whh @ h_prev + bh)
            h = np.tanh(self.Wxh @ x + self.Whh @ hiddens[-1] + self.bh)
            hiddens.append(h)
            
            # y_t = Why @ h + by
            y = self.Why @ h + self.by
            outputs.append(y)
            
        # Store for backward pass
        self.last_inputs = inputs
        self.last_hiddens = hiddens
        
        return outputs, hiddens[1:]
    
    def backward(self, targets: List[np.ndarray], learning_rate: float = 0.01):
        """
        Backward pass using BPTT (Backpropagation Through Time).
        """
        # Initialize gradients
        dWxh = np.zeros_like(self.Wxh)
        dWhh = np.zeros_like(self.Whh)
        dWhy = np.zeros_like(self.Why)
        dbh = np.zeros_like(self.bh)
        dby = np.zeros_like(self.by)
        
        dh_next = np.zeros_like(self.last_hiddens[0])
        
        # Backward through time
        for t in reversed(range(len(self.last_inputs))):
            # Output layer gradients
            # Project hidden state to output space first
            y_pred = self.Why @ self.last_hiddens[t+1] + self.by
            dy = y_pred - targets[t]  # Gradient of MSE loss
            dWhy += dy @ self.last_hiddens[t+1].T
            dby += dy
            
            # Hidden layer gradients
            dh = self.Why.T @ dy + dh_next
            dh_raw = (1 - self.last_hiddens[t+1]**2) * dh  # tanh derivative
            
            # Parameter gradients
            dWxh += dh_raw @ self.last_inputs[t].T
            dWhh += dh_raw @ self.last_hiddens[t].T
            dbh += dh_raw
            
            # Gradient for next timestep
            dh_next = self.Whh.T @ dh_raw
            
        # Clip gradients to prevent explosion
        for dparam in [dWxh, dWhh, dWhy, dbh, dby]:
            np.clip(dparam, -5, 5, out=dparam)
            
        # Update parameters
        self.Wxh -= learning_rate * dWxh
        self.Whh -= learning_rate * dWhh
        self.Why -= learning_rate * dWhy
        self.bh -= learning_rate * dbh
        self.by -= learning_rate * dby
        
        return dh_next
    
    def generate_sequence(self, seed_input: np.ndarray, length: int):
        """Generate a sequence given a seed input."""
        h = np.zeros((self.hidden_size, 1))
        generated = []
        
        x = seed_input
        for _ in range(length):
            h = np.tanh(self.Wxh @ x + self.Whh @ h + self.bh)
            y = self.Why @ h + self.by
            generated.append(y)
            x = y  # Use output as next input
            
        return generated


class SimpleLSTM:
    """A simple LSTM implementation to show the gating mechanism."""
    
    def __init__(self, input_size: int, hidden_size: int):
        self.hidden_size = hidden_size
        
        # Combined weight matrix for all gates
        # [input_gate; forget_gate; cell_gate; output_gate]
        self.W = np.random.randn(4 * hidden_size, input_size + hidden_size) * 0.01
        self.b = np.zeros((4 * hidden_size, 1))
        
    def forward(self, x: np.ndarray, h_prev: np.ndarray, c_prev: np.ndarray):
        """
        Forward pass through LSTM cell.
        
        Args:
            x: Input vector (column)
            h_prev: Previous hidden state
            c_prev: Previous cell state
            
        Returns:
            h: New hidden state
            c: New cell state
        """
        # Concatenate input and previous hidden state
        combined = np.vstack([x, h_prev])
        
        # Compute all gates
        gates = self.W @ combined + self.b
        
        # Split gates
        i_gate = self._sigmoid(gates[:self.hidden_size])      # Input gate
        f_gate = self._sigmoid(gates[self.hidden_size:2*self.hidden_size])  # Forget gate
        c_tilde = np.tanh(gates[2*self.hidden_size:3*self.hidden_size])    # Candidate values
        o_gate = self._sigmoid(gates[3*self.hidden_size:])    # Output gate
        
        # Update cell state
        c = f_gate * c_prev + i_gate * c_tilde
        
        # Update hidden state
        h = o_gate * np.tanh(c)
        
        return h, c
    
    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))


class SequenceModelingDemos:
    """Demonstrations of sequence modeling concepts and problems."""
    
    @staticmethod
    def demonstrate_sequential_processing():
        """Show how RNNs must process sequentially."""
        print("=== Sequential Processing in RNNs ===\n")
        
        sequence_length = 10
        hidden_size = 4
        
        # Simulate RNN processing
        print("RNN Processing (Sequential):")
        total_time = 0
        for t in range(sequence_length):
            time.sleep(0.1)  # Simulate computation time
            total_time += 0.1
            print(f"Step {t+1}: Process token {t+1} (requires hidden state from step {t})")
        
        print(f"\nTotal time: {total_time:.1f}s (sequential)")
        
        print("\n" + "="*50 + "\n")
        print("Attention Processing (Parallel):")
        print("All tokens processed simultaneously!")
        time.sleep(0.1)  # All processed at once
        print("Total time: 0.1s (parallel)")
        
    @staticmethod
    def visualize_information_bottleneck():
        """Visualize the information bottleneck problem."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # RNN bottleneck
        sequence_lengths = [5, 10, 20, 50, 100]
        hidden_size = 128
        
        # Information capacity
        ax1.axhline(y=hidden_size, color='r', linestyle='--', label='Hidden state capacity')
        ax1.plot(sequence_lengths, sequence_lengths, 'b-', linewidth=2, label='Information to encode')
        ax1.fill_between(sequence_lengths, 0, hidden_size, alpha=0.3, color='green', label='Can encode')
        ax1.fill_between(sequence_lengths, hidden_size, sequence_lengths, alpha=0.3, color='red', 
                        label='Information lost', where=[s > hidden_size for s in sequence_lengths])
        
        ax1.set_xlabel('Sequence Length')
        ax1.set_ylabel('Information (arbitrary units)')
        ax1.set_title('RNN Information Bottleneck')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Attention - no bottleneck
        ax2.plot(sequence_lengths, sequence_lengths, 'g-', linewidth=2, 
                label='Information preserved')
        ax2.fill_between(sequence_lengths, 0, sequence_lengths, alpha=0.3, color='green')
        
        ax2.set_xlabel('Sequence Length')
        ax2.set_ylabel('Information (arbitrary units)')
        ax2.set_title('Attention: No Bottleneck')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def demonstrate_vanishing_gradients():
        """Demonstrate the vanishing gradient problem."""
        print("=== Vanishing Gradient Problem ===\n")
        
        # Simulate gradient flow through time
        timesteps = 20
        gradient = 1.0
        
        # Case 1: Vanishing gradients
        print("Case 1: Vanishing Gradients (gradient multiplier = 0.9)")
        gradients_vanish = []
        for t in range(timesteps):
            gradient *= 0.9
            gradients_vanish.append(gradient)
        
        # Case 2: Exploding gradients
        print("\nCase 2: Exploding Gradients (gradient multiplier = 1.1)")
        gradient = 1.0
        gradients_explode = []
        for t in range(timesteps):
            gradient *= 1.1
            gradients_explode.append(gradient)
        
        # Plot
        plt.figure(figsize=(12, 6))
        
        plt.subplot(1, 2, 1)
        plt.plot(gradients_vanish, 'b-', linewidth=2)
        plt.axhline(y=0.01, color='r', linestyle='--', label='Effectively zero')
        plt.xlabel('Timesteps')
        plt.ylabel('Gradient Magnitude')
        plt.title('Vanishing Gradients')
        plt.yscale('log')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        plt.plot(gradients_explode, 'r-', linewidth=2)
        plt.axhline(y=1000, color='r', linestyle='--', label='Gradient explosion')
        plt.xlabel('Timesteps')
        plt.ylabel('Gradient Magnitude')
        plt.title('Exploding Gradients')
        plt.yscale('log')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        print(f"\nAfter {timesteps} steps:")
        print(f"Vanishing: gradient = {gradients_vanish[-1]:.2e}")
        print(f"Exploding: gradient = {gradients_explode[-1]:.2e}")
    
    @staticmethod
    def compare_architectures():
        """Compare different sequence modeling architectures."""
        print("=== Architecture Comparison ===\n")
        
        # Create comparison data
        architectures = ['RNN', 'LSTM', 'GRU', 'Transformer']
        
        # Metrics (relative scores)
        metrics = {
            'Parallelization': [1, 1, 1, 10],
            'Long-range deps': [2, 5, 5, 10],
            'Training speed': [8, 4, 5, 9],
            'Memory usage': [10, 6, 7, 4],
            'Gradient flow': [2, 7, 7, 10]
        }
        
        # Create radar chart
        fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(projection='polar'))
        
        # Number of variables
        categories = list(metrics.keys())
        N = len(categories)
        
        # Compute angle for each axis
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]
        
        # Plot for each architecture
        colors = ['red', 'blue', 'green', 'purple']
        for idx, arch in enumerate(architectures):
            values = [metrics[cat][idx] for cat in categories]
            values += values[:1]
            
            ax.plot(angles, values, 'o-', linewidth=2, color=colors[idx], label=arch)
            ax.fill(angles, values, alpha=0.15, color=colors[idx])
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories)
        ax.set_ylim(0, 10)
        ax.set_title('Architecture Comparison\n(Higher is Better)', y=1.08)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        ax.grid(True)
        
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def sequence_to_sequence_demo():
        """Demonstrate sequence-to-sequence with bottleneck."""
        print("=== Sequence-to-Sequence Bottleneck ===\n")
        
        # Simulate encoding a sentence
        source_sentence = "The quick brown fox jumps over the lazy dog"
        words = source_sentence.split()
        
        print("Encoder RNN processing:")
        print(f"Source: '{source_sentence}'")
        print(f"Number of words: {len(words)}")
        
        # Simulate encoding
        hidden_size = 4
        print(f"\nEncoding into hidden state of size {hidden_size}:")
        
        for i, word in enumerate(words):
            print(f"Step {i+1}: Processing '{word}'...", end='')
            time.sleep(0.2)
            print(" ✓")
        
        print(f"\nFinal context vector: [0.23, -0.45, 0.67, -0.12] (size={hidden_size})")
        print("\n⚠️  Problem: Entire sentence compressed into 4 numbers!")
        
        # Show information loss
        info_per_word = 50  # Arbitrary units
        total_info = len(words) * info_per_word
        context_capacity = hidden_size * 10  # Arbitrary units
        
        print(f"\nInformation analysis:")
        print(f"- Information in source: ~{total_info} units")
        print(f"- Context vector capacity: ~{context_capacity} units")
        print(f"- Information lost: ~{max(0, total_info - context_capacity)} units ({max(0, total_info - context_capacity)/total_info*100:.1f}%)")
    
    @staticmethod
    def computational_complexity_analysis():
        """Analyze computational complexity of different approaches."""
        print("=== Computational Complexity Analysis ===\n")
        
        sequence_lengths = np.array([10, 50, 100, 500, 1000])
        
        # RNN complexity: O(n) sequential steps
        rnn_time = sequence_lengths
        
        # Attention complexity: O(n²) but parallelizable
        attention_compute = sequence_lengths ** 2
        # With parallelization (assuming perfect parallelization)
        attention_time = attention_compute / sequence_lengths  # O(n) with n processors
        
        plt.figure(figsize=(12, 5))
        
        # Time complexity
        plt.subplot(1, 2, 1)
        plt.plot(sequence_lengths, rnn_time, 'b-', linewidth=2, label='RNN (sequential)')
        plt.plot(sequence_lengths, attention_time, 'g-', linewidth=2, label='Attention (parallel)')
        plt.xlabel('Sequence Length')
        plt.ylabel('Time Steps')
        plt.title('Time Complexity (with parallelization)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Memory complexity
        plt.subplot(1, 2, 2)
        rnn_memory = sequence_lengths  # O(n)
        attention_memory = sequence_lengths ** 2  # O(n²)
        
        plt.plot(sequence_lengths, rnn_memory, 'b-', linewidth=2, label='RNN')
        plt.plot(sequence_lengths, attention_memory, 'g-', linewidth=2, label='Attention')
        plt.xlabel('Sequence Length')
        plt.ylabel('Memory Usage')
        plt.title('Memory Complexity')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.yscale('log')
        
        plt.tight_layout()
        plt.show()


class RNNLanguageModel:
    """A character-level RNN language model to demonstrate learning."""
    
    def __init__(self, vocab_size: int, hidden_size: int = 128):
        self.rnn = VanillaRNN(vocab_size, hidden_size, vocab_size)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        
    def train_on_text(self, text: str, epochs: int = 100):
        """Train the RNN on a text string."""
        # Create character mappings
        chars = list(set(text))
        self.char_to_idx = {ch: i for i, ch in enumerate(chars)}
        self.idx_to_char = {i: ch for ch, i in self.char_to_idx.items()}
        
        # Prepare training data
        inputs = []
        targets = []
        
        for i in range(len(text) - 1):
            # One-hot encode
            x = np.zeros((self.vocab_size, 1))
            y = np.zeros((self.vocab_size, 1))
            
            x[self.char_to_idx[text[i]]] = 1
            y[self.char_to_idx[text[i + 1]]] = 1
            
            inputs.append(x)
            targets.append(y)
        
        # Training loop
        losses = []
        for epoch in range(epochs):
            # Forward pass
            outputs, _ = self.rnn.forward(inputs)
            
            # Compute loss
            loss = 0
            for output, target in zip(outputs, targets):
                loss += np.sum((output - target) ** 2)
            losses.append(loss / len(outputs))
            
            # Backward pass
            self.rnn.backward(targets, learning_rate=0.01)
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Loss: {losses[-1]:.4f}")
        
        return losses
    
    def generate(self, seed_char: str, length: int = 100):
        """Generate text starting from a seed character."""
        # Start with seed
        x = np.zeros((self.vocab_size, 1))
        x[self.char_to_idx[seed_char]] = 1
        
        generated = seed_char
        h = np.zeros((self.hidden_size, 1))
        
        for _ in range(length):
            # Forward pass through RNN
            h = np.tanh(self.rnn.Wxh @ x + self.rnn.Whh @ h + self.rnn.bh)
            y = self.rnn.Why @ h + self.rnn.by
            
            # Sample from output distribution
            p = np.exp(y) / np.sum(np.exp(y))
            idx = np.random.choice(range(self.vocab_size), p=p.ravel())
            
            # Prepare next input
            x = np.zeros((self.vocab_size, 1))
            x[idx] = 1
            
            generated += self.idx_to_char[idx]
        
        return generated


def demonstrate_rnn_training():
    """Train a small RNN and show its behavior."""
    print("=== Training a Character-Level RNN ===\n")
    
    # Simple training text
    text = "hello world! " * 10
    vocab_size = len(set(text))
    
    print(f"Training text: '{text[:50]}...'")
    print(f"Vocabulary size: {vocab_size}")
    
    # Create and train model
    model = RNNLanguageModel(vocab_size, hidden_size=32)
    losses = model.train_on_text(text, epochs=50)
    
    # Generate some text
    print("\nGenerated text:")
    for seed in ['h', 'w', ' ']:
        generated = model.generate(seed, length=30)
        print(f"Seed '{seed}': {generated}")
    
    # Plot training loss
    plt.figure(figsize=(10, 6))
    plt.plot(losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('RNN Training Loss')
    plt.grid(True, alpha=0.3)
    plt.show()


def compare_sequence_lengths():
    """Show how RNN performance degrades with sequence length."""
    print("=== RNN Performance vs Sequence Length ===\n")
    
    # Simulate performance metrics
    lengths = [10, 20, 50, 100, 200, 500]
    
    # RNN accuracy drops with length
    rnn_accuracy = [0.95, 0.90, 0.75, 0.60, 0.45, 0.30]
    
    # Attention maintains performance
    attention_accuracy = [0.95, 0.94, 0.93, 0.92, 0.91, 0.90]
    
    plt.figure(figsize=(10, 6))
    plt.plot(lengths, rnn_accuracy, 'b-o', linewidth=2, markersize=8, label='RNN')
    plt.plot(lengths, attention_accuracy, 'g-o', linewidth=2, markersize=8, label='Attention')
    
    plt.xlabel('Sequence Length')
    plt.ylabel('Accuracy')
    plt.title('Model Performance vs Sequence Length')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Add annotations
    plt.annotate('Information bottleneck\nand gradient issues', 
                xy=(200, 0.45), xytext=(250, 0.60),
                arrowprops=dict(arrowstyle='->', color='red'),
                fontsize=10, color='red')
    
    plt.annotate('Direct connections\npreserve information', 
                xy=(200, 0.91), xytext=(250, 0.80),
                arrowprops=dict(arrowstyle='->', color='green'),
                fontsize=10, color='green')
    
    plt.ylim(0.2, 1.0)
    plt.show()


if __name__ == "__main__":
    # Run all demonstrations
    print("=" * 60)
    print("SEQUENCE MODELING DEMONSTRATIONS")
    print("=" * 60)
    
    # 1. Sequential processing
    SequenceModelingDemos.demonstrate_sequential_processing()
    print("\n" + "=" * 60 + "\n")
    
    # 2. Information bottleneck
    print("Visualizing information bottleneck...")
    SequenceModelingDemos.visualize_information_bottleneck()
    
    # 3. Vanishing gradients
    SequenceModelingDemos.demonstrate_vanishing_gradients()
    print("\n" + "=" * 60 + "\n")
    
    # 4. Architecture comparison
    print("Comparing architectures...")
    SequenceModelingDemos.compare_architectures()
    
    # 5. Seq2seq bottleneck
    SequenceModelingDemos.sequence_to_sequence_demo()
    print("\n" + "=" * 60 + "\n")
    
    # 6. Computational complexity
    print("Analyzing computational complexity...")
    SequenceModelingDemos.computational_complexity_analysis()
    
    # 7. Train a small RNN
    demonstrate_rnn_training()
    
    # 8. Performance vs sequence length
    compare_sequence_lengths()
    
    print("\n" + "=" * 60)
    print("Key Takeaways:")
    print("- RNNs process sequentially, preventing parallelization")
    print("- Information bottleneck limits long sequences")
    print("- Vanishing gradients prevent learning long-range dependencies")
    print("- Attention mechanisms solve these problems!")
    print("=" * 60)