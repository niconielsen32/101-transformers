"""
Scaling and Optimization
Implementation of model compression, quantization, optimized inference,
and deployment strategies for transformers.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import math
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass
from collections import OrderedDict, defaultdict
import matplotlib.pyplot as plt
from tqdm import tqdm
import psutil
import GPUtil


# Configuration Classes
@dataclass
class OptimizationConfig:
    """Configuration for model optimization."""
    model_name: str = "transformer"
    compression_method: str = "pruning"  # pruning, distillation, quantization
    target_sparsity: float = 0.5
    quantization_bits: int = 8
    batch_size: int = 32
    sequence_length: int = 512
    
    
@dataclass
class InferenceConfig:
    """Configuration for optimized inference."""
    use_kv_cache: bool = True
    use_flash_attention: bool = True
    dynamic_batching: bool = True
    max_batch_size: int = 64
    max_sequence_length: int = 2048
    dtype: str = "float16"  # float32, float16, int8


# Model Compression Techniques
class ModelCompressor:
    """Base class for model compression techniques."""
    
    def compress(self, model: nn.Module) -> nn.Module:
        """Compress the model."""
        raise NotImplementedError
        
    def get_compression_stats(self, model: nn.Module) -> Dict[str, float]:
        """Get compression statistics."""
        raise NotImplementedError


# Magnitude Pruning
class MagnitudePruning(ModelCompressor):
    """Magnitude-based weight pruning."""
    
    def __init__(self, sparsity: float = 0.5):
        self.sparsity = sparsity
        
    def compress(self, model: nn.Module) -> nn.Module:
        """Apply magnitude pruning to model."""
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                self._prune_linear(module)
            elif isinstance(module, nn.Conv2d):
                self._prune_conv2d(module)
                
        return model
    
    def _prune_linear(self, module: nn.Linear):
        """Prune linear layer weights."""
        weight = module.weight.data
        threshold = torch.quantile(torch.abs(weight), self.sparsity)
        mask = torch.abs(weight) > threshold
        module.weight.data = weight * mask
        
        # Store mask for structured pruning
        module.register_buffer('weight_mask', mask)
        
    def _prune_conv2d(self, module: nn.Conv2d):
        """Prune convolutional layer weights."""
        weight = module.weight.data
        threshold = torch.quantile(torch.abs(weight), self.sparsity)
        mask = torch.abs(weight) > threshold
        module.weight.data = weight * mask
        module.register_buffer('weight_mask', mask)
        
    def get_compression_stats(self, model: nn.Module) -> Dict[str, float]:
        """Calculate pruning statistics."""
        total_params = 0
        pruned_params = 0
        
        for name, module in model.named_modules():
            if hasattr(module, 'weight'):
                weight = module.weight.data
                total_params += weight.numel()
                pruned_params += (weight == 0).sum().item()
                
        return {
            'total_parameters': total_params,
            'pruned_parameters': pruned_params,
            'sparsity': pruned_params / total_params if total_params > 0 else 0,
            'compression_ratio': total_params / (total_params - pruned_params) if pruned_params < total_params else float('inf')
        }


# Structured Pruning
class StructuredPruning(ModelCompressor):
    """Structured pruning (channels, heads, etc.)."""
    
    def __init__(self, pruning_ratio: float = 0.3):
        self.pruning_ratio = pruning_ratio
        
    def compress(self, model: nn.Module) -> nn.Module:
        """Apply structured pruning."""
        # Prune attention heads
        self._prune_attention_heads(model)
        
        # Prune FFN dimensions
        self._prune_ffn_dimensions(model)
        
        return model
    
    def _prune_attention_heads(self, model: nn.Module):
        """Prune least important attention heads."""
        for name, module in model.named_modules():
            if 'attention' in name.lower() and hasattr(module, 'num_heads'):
                # Calculate head importance (simplified)
                num_heads = module.num_heads
                heads_to_prune = int(num_heads * self.pruning_ratio)
                
                if heads_to_prune > 0:
                    # Create head mask
                    head_mask = torch.ones(num_heads)
                    head_mask[:heads_to_prune] = 0
                    module.register_buffer('head_mask', head_mask)
                    
                    # Update num_heads
                    module.num_heads = num_heads - heads_to_prune
                    
    def _prune_ffn_dimensions(self, model: nn.Module):
        """Prune FFN hidden dimensions."""
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear) and 'intermediate' in name:
                out_features = module.out_features
                dims_to_prune = int(out_features * self.pruning_ratio)
                
                if dims_to_prune > 0:
                    # Create dimension mask
                    dim_mask = torch.ones(out_features)
                    dim_mask[:dims_to_prune] = 0
                    module.register_buffer('dim_mask', dim_mask)


# Knowledge Distillation
class KnowledgeDistillation:
    """Knowledge distillation for model compression."""
    
    def __init__(self, teacher_model: nn.Module, temperature: float = 3.0, 
                 alpha: float = 0.7):
        self.teacher = teacher_model
        self.temperature = temperature
        self.alpha = alpha
        
        # Freeze teacher
        for param in self.teacher.parameters():
            param.requires_grad = False
            
    def distillation_loss(self, student_outputs: torch.Tensor, 
                         teacher_outputs: torch.Tensor, 
                         targets: torch.Tensor) -> torch.Tensor:
        """Calculate distillation loss."""
        # Student loss
        student_loss = F.cross_entropy(student_outputs, targets)
        
        # Distillation loss
        student_soft = F.log_softmax(student_outputs / self.temperature, dim=-1)
        teacher_soft = F.softmax(teacher_outputs / self.temperature, dim=-1)
        distill_loss = F.kl_div(student_soft, teacher_soft, reduction='batchmean')
        distill_loss *= self.temperature ** 2
        
        # Combined loss
        total_loss = self.alpha * distill_loss + (1 - self.alpha) * student_loss
        
        return total_loss, student_loss, distill_loss


# Quantization
class ModelQuantizer:
    """Quantize model weights and activations."""
    
    def __init__(self, bits: int = 8, symmetric: bool = True):
        self.bits = bits
        self.symmetric = symmetric
        self.qmin = -(2 ** (bits - 1)) if symmetric else 0
        self.qmax = 2 ** (bits - 1) - 1 if symmetric else 2 ** bits - 1
        
    def quantize_tensor(self, tensor: torch.Tensor) -> Tuple[torch.Tensor, float, float]:
        """Quantize a tensor to specified bits."""
        # Calculate scale and zero point
        if self.symmetric:
            scale = tensor.abs().max() / self.qmax
            zero_point = 0
        else:
            scale = (tensor.max() - tensor.min()) / (self.qmax - self.qmin)
            zero_point = self.qmin - tensor.min() / scale
            
        # Quantize
        quantized = torch.round(tensor / scale + zero_point)
        quantized = torch.clamp(quantized, self.qmin, self.qmax)
        
        return quantized.to(torch.int8), scale, zero_point
    
    def dequantize_tensor(self, quantized: torch.Tensor, scale: float, 
                         zero_point: float) -> torch.Tensor:
        """Dequantize tensor back to float."""
        return (quantized.float() - zero_point) * scale
    
    def quantize_model(self, model: nn.Module) -> nn.Module:
        """Quantize entire model."""
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                # Quantize weights
                quantized_weight, scale, zero_point = self.quantize_tensor(module.weight.data)
                
                # Store quantization parameters
                module.register_buffer('quantized_weight', quantized_weight)
                module.register_buffer('weight_scale', torch.tensor(scale))
                module.register_buffer('weight_zero_point', torch.tensor(zero_point))
                
                # Replace forward method
                original_forward = module.forward
                
                def quantized_forward(x):
                    # Dequantize weight
                    weight = self.dequantize_tensor(
                        module.quantized_weight,
                        module.weight_scale,
                        module.weight_zero_point
                    )
                    return F.linear(x, weight, module.bias)
                
                module.forward = quantized_forward
                
        return model


# Dynamic Quantization
class DynamicQuantization:
    """Dynamic quantization for inference."""
    
    def __init__(self, dtype=torch.qint8):
        self.dtype = dtype
        
    def apply(self, model: nn.Module) -> nn.Module:
        """Apply dynamic quantization to model."""
        quantized_model = torch.quantization.quantize_dynamic(
            model,
            {nn.Linear, nn.LSTM, nn.GRU},
            dtype=self.dtype
        )
        return quantized_model


# Flash Attention Implementation (Simplified)
class FlashAttention(nn.Module):
    """Simplified Flash Attention for memory efficiency."""
    
    def __init__(self, hidden_size: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)
        
        self.qkv_proj = nn.Linear(hidden_size, 3 * hidden_size, bias=False)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, hidden_states: torch.Tensor, 
                attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass with Flash Attention algorithm."""
        batch_size, seq_len, _ = hidden_states.shape
        
        # QKV projection
        qkv = self.qkv_proj(hidden_states)
        qkv = qkv.reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, H, L, D]
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Flash Attention (simplified - real implementation requires CUDA kernels)
        if hidden_states.is_cuda and seq_len > 1024:
            # Use chunked attention for long sequences
            output = self._chunked_attention(q, k, v, chunk_size=256)
        else:
            # Standard attention for short sequences
            scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
            
            if attention_mask is not None:
                scores += attention_mask
                
            probs = F.softmax(scores, dim=-1)
            probs = self.dropout(probs)
            output = torch.matmul(probs, v)
            
        # Reshape and project
        output = output.transpose(1, 2).contiguous()
        output = output.reshape(batch_size, seq_len, self.hidden_size)
        output = self.o_proj(output)
        
        return output
    
    def _chunked_attention(self, q: torch.Tensor, k: torch.Tensor, 
                          v: torch.Tensor, chunk_size: int = 256) -> torch.Tensor:
        """Chunked attention computation for memory efficiency."""
        batch_size, num_heads, seq_len, head_dim = q.shape
        
        # Process attention in chunks
        output = torch.zeros_like(v)
        
        for i in range(0, seq_len, chunk_size):
            end_i = min(i + chunk_size, seq_len)
            q_chunk = q[:, :, i:end_i]
            
            # Initialize output chunk
            output_chunk = torch.zeros_like(q_chunk)
            max_score = torch.full((batch_size, num_heads, end_i - i, 1), 
                                  -float('inf'), device=q.device)
            sum_exp = torch.zeros_like(max_score)
            
            for j in range(0, seq_len, chunk_size):
                end_j = min(j + chunk_size, seq_len)
                k_chunk = k[:, :, j:end_j]
                v_chunk = v[:, :, j:end_j]
                
                # Compute scores for this chunk
                scores = torch.matmul(q_chunk, k_chunk.transpose(-2, -1)) * self.scale
                
                # Online softmax
                chunk_max = scores.max(dim=-1, keepdim=True).values
                new_max = torch.maximum(max_score, chunk_max)
                
                # Update statistics
                exp_diff = torch.exp(max_score - new_max)
                exp_scores = torch.exp(scores - new_max)
                
                sum_exp = sum_exp * exp_diff + exp_scores.sum(dim=-1, keepdim=True)
                output_chunk = output_chunk * exp_diff + torch.matmul(exp_scores, v_chunk)
                
                max_score = new_max
                
            # Normalize
            output[:, :, i:end_i] = output_chunk / sum_exp
            
        return output


# KV Cache for Efficient Generation
class KVCache:
    """Key-Value cache for autoregressive generation."""
    
    def __init__(self, max_batch_size: int, max_seq_length: int, 
                 num_layers: int, num_heads: int, head_dim: int):
        self.max_batch_size = max_batch_size
        self.max_seq_length = max_seq_length
        self.num_layers = num_layers
        
        # Pre-allocate cache
        cache_shape = (max_batch_size, num_heads, max_seq_length, head_dim)
        self.k_cache = [torch.zeros(cache_shape) for _ in range(num_layers)]
        self.v_cache = [torch.zeros(cache_shape) for _ in range(num_layers)]
        self.cache_lens = torch.zeros(max_batch_size, dtype=torch.long)
        
    def update(self, layer_idx: int, k: torch.Tensor, v: torch.Tensor, 
               batch_idx: Optional[torch.Tensor] = None):
        """Update cache with new key-value pairs."""
        if batch_idx is None:
            batch_idx = torch.arange(k.shape[0])
            
        seq_len = k.shape[2]
        
        # Update cache
        for i, idx in enumerate(batch_idx):
            cache_len = self.cache_lens[idx]
            self.k_cache[layer_idx][idx, :, cache_len:cache_len + seq_len] = k[i]
            self.v_cache[layer_idx][idx, :, cache_len:cache_len + seq_len] = v[i]
            self.cache_lens[idx] += seq_len
            
    def get(self, layer_idx: int, batch_idx: Optional[torch.Tensor] = None):
        """Retrieve cached key-value pairs."""
        if batch_idx is None:
            batch_idx = torch.arange(self.cache_lens.shape[0])
            
        # Get relevant cache entries
        k_cached = []
        v_cached = []
        
        for idx in batch_idx:
            cache_len = self.cache_lens[idx]
            k_cached.append(self.k_cache[layer_idx][idx, :, :cache_len])
            v_cached.append(self.v_cache[layer_idx][idx, :, :cache_len])
            
        return torch.stack(k_cached), torch.stack(v_cached)
    
    def clear(self, batch_idx: Optional[torch.Tensor] = None):
        """Clear cache for specified batch indices."""
        if batch_idx is None:
            self.cache_lens.zero_()
        else:
            self.cache_lens[batch_idx] = 0


# Dynamic Batching
class DynamicBatchingEngine:
    """Dynamic batching for efficient inference."""
    
    def __init__(self, model: nn.Module, max_batch_size: int = 32, 
                 max_wait_time: float = 0.01):
        self.model = model
        self.max_batch_size = max_batch_size
        self.max_wait_time = max_wait_time
        self.request_queue = []
        self.results = {}
        
    def add_request(self, request_id: str, input_ids: torch.Tensor):
        """Add request to processing queue."""
        self.request_queue.append({
            'id': request_id,
            'input_ids': input_ids,
            'timestamp': time.time()
        })
        
    def process_batch(self):
        """Process a batch of requests."""
        if not self.request_queue:
            return
            
        # Collect requests for batch
        batch = []
        current_time = time.time()
        
        while len(batch) < self.max_batch_size and self.request_queue:
            request = self.request_queue[0]
            
            # Check wait time
            if current_time - request['timestamp'] >= self.max_wait_time or \
               len(batch) == 0:
                batch.append(self.request_queue.pop(0))
            else:
                break
                
        if not batch:
            return
            
        # Pad inputs to same length
        max_length = max(req['input_ids'].shape[1] for req in batch)
        padded_inputs = []
        attention_masks = []
        
        for req in batch:
            input_ids = req['input_ids']
            pad_length = max_length - input_ids.shape[1]
            
            if pad_length > 0:
                padded = F.pad(input_ids, (0, pad_length), value=0)
                mask = F.pad(torch.ones_like(input_ids), (0, pad_length), value=0)
            else:
                padded = input_ids
                mask = torch.ones_like(input_ids)
                
            padded_inputs.append(padded)
            attention_masks.append(mask)
            
        # Stack batch
        batch_input = torch.cat(padded_inputs, dim=0)
        batch_mask = torch.cat(attention_masks, dim=0)
        
        # Run inference
        with torch.no_grad():
            outputs = self.model(batch_input, attention_mask=batch_mask)
            
        # Store results
        for i, req in enumerate(batch):
            self.results[req['id']] = outputs[i]
            
    def get_result(self, request_id: str) -> Optional[torch.Tensor]:
        """Get result for request ID."""
        return self.results.pop(request_id, None)


# Performance Monitoring
class PerformanceMonitor:
    """Monitor model performance metrics."""
    
    def __init__(self):
        self.metrics = defaultdict(list)
        self.start_time = None
        
    def start_inference(self):
        """Start timing inference."""
        self.start_time = time.time()
        
    def end_inference(self, batch_size: int, sequence_length: int):
        """End timing and record metrics."""
        if self.start_time is None:
            return
            
        elapsed = time.time() - self.start_time
        
        # Calculate metrics
        self.metrics['latency'].append(elapsed * 1000)  # ms
        self.metrics['throughput'].append(batch_size / elapsed)  # samples/sec
        self.metrics['tokens_per_second'].append(
            batch_size * sequence_length / elapsed
        )
        
        # Memory usage
        if torch.cuda.is_available():
            self.metrics['gpu_memory'].append(
                torch.cuda.max_memory_allocated() / 1e9  # GB
            )
            torch.cuda.reset_peak_memory_stats()
            
        # CPU usage
        self.metrics['cpu_percent'].append(psutil.cpu_percent())
        
        self.start_time = None
        
    def get_summary(self) -> Dict[str, float]:
        """Get performance summary."""
        summary = {}
        
        for metric, values in self.metrics.items():
            if values:
                summary[f'{metric}_mean'] = np.mean(values)
                summary[f'{metric}_p50'] = np.percentile(values, 50)
                summary[f'{metric}_p95'] = np.percentile(values, 95)
                summary[f'{metric}_p99'] = np.percentile(values, 99)
                
        return summary
    
    def plot_metrics(self):
        """Plot performance metrics."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Latency
        if 'latency' in self.metrics:
            axes[0, 0].plot(self.metrics['latency'])
            axes[0, 0].set_title('Inference Latency')
            axes[0, 0].set_xlabel('Request')
            axes[0, 0].set_ylabel('Latency (ms)')
            
        # Throughput
        if 'throughput' in self.metrics:
            axes[0, 1].plot(self.metrics['throughput'])
            axes[0, 1].set_title('Throughput')
            axes[0, 1].set_xlabel('Request')
            axes[0, 1].set_ylabel('Samples/sec')
            
        # GPU Memory
        if 'gpu_memory' in self.metrics:
            axes[1, 0].plot(self.metrics['gpu_memory'])
            axes[1, 0].set_title('GPU Memory Usage')
            axes[1, 0].set_xlabel('Request')
            axes[1, 0].set_ylabel('Memory (GB)')
            
        # Tokens per second
        if 'tokens_per_second' in self.metrics:
            axes[1, 1].plot(self.metrics['tokens_per_second'])
            axes[1, 1].set_title('Token Generation Speed')
            axes[1, 1].set_xlabel('Request')
            axes[1, 1].set_ylabel('Tokens/sec')
            
        plt.tight_layout()
        plt.show()


# Model Optimization Pipeline
class OptimizationPipeline:
    """Complete optimization pipeline for deployment."""
    
    def __init__(self, config: OptimizationConfig):
        self.config = config
        
    def optimize_model(self, model: nn.Module) -> Tuple[nn.Module, Dict[str, Any]]:
        """Apply optimization techniques to model."""
        stats = {}
        
        # Original model stats
        original_params = sum(p.numel() for p in model.parameters())
        stats['original_parameters'] = original_params
        
        # Apply pruning
        if 'pruning' in self.config.compression_method:
            pruner = MagnitudePruning(sparsity=self.config.target_sparsity)
            model = pruner.compress(model)
            stats['pruning'] = pruner.get_compression_stats(model)
            
        # Apply quantization
        if 'quantization' in self.config.compression_method:
            quantizer = ModelQuantizer(bits=self.config.quantization_bits)
            model = quantizer.quantize_model(model)
            stats['quantization'] = {
                'bits': self.config.quantization_bits,
                'compression_ratio': 32 / self.config.quantization_bits
            }
            
        # Optimize for inference
        model = self._optimize_for_inference(model)
        
        return model, stats
    
    def _optimize_for_inference(self, model: nn.Module) -> nn.Module:
        """Apply inference-specific optimizations."""
        model.eval()
        
        # Fuse operations
        if hasattr(torch.jit, 'fuse'):
            model = torch.jit.fuse(model)
            
        # TorchScript compilation (optional)
        # model = torch.jit.script(model)
        
        return model


# Benchmark Utilities
def benchmark_model(model: nn.Module, input_shape: Tuple[int, ...], 
                   num_runs: int = 100, warmup: int = 10) -> Dict[str, float]:
    """Benchmark model performance."""
    device = next(model.parameters()).device
    dummy_input = torch.randn(*input_shape).to(device)
    
    # Warmup
    for _ in range(warmup):
        with torch.no_grad():
            _ = model(dummy_input)
            
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        
    # Benchmark
    start_time = time.time()
    
    for _ in range(num_runs):
        with torch.no_grad():
            _ = model(dummy_input)
            
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        
    elapsed = time.time() - start_time
    
    return {
        'avg_latency_ms': (elapsed / num_runs) * 1000,
        'throughput': num_runs / elapsed,
        'total_time': elapsed
    }


def compare_optimization_methods(model: nn.Module, input_shape: Tuple[int, ...]):
    """Compare different optimization methods."""
    results = {}
    
    # Baseline
    print("Benchmarking baseline model...")
    results['baseline'] = benchmark_model(model, input_shape)
    
    # Pruning
    print("Benchmarking pruned model...")
    pruner = MagnitudePruning(sparsity=0.5)
    pruned_model = pruner.compress(model.clone())
    results['pruned'] = benchmark_model(pruned_model, input_shape)
    
    # Quantization
    print("Benchmarking quantized model...")
    quantizer = ModelQuantizer(bits=8)
    quantized_model = quantizer.quantize_model(model.clone())
    results['quantized'] = benchmark_model(quantized_model, input_shape)
    
    # Dynamic quantization
    print("Benchmarking dynamic quantized model...")
    dynamic_quant = DynamicQuantization()
    dynamic_model = dynamic_quant.apply(copy.deepcopy(model))
    results['dynamic_quant'] = benchmark_model(dynamic_model, input_shape)
    
    return results


# Example Usage
if __name__ == "__main__":
    print("=== Scaling and Optimization Demo ===\n")
    
    # Create a simple transformer model
    class SimpleTransformer(nn.Module):
        def __init__(self, hidden_size=768, num_layers=12, num_heads=12):
            super().__init__()
            self.embeddings = nn.Embedding(30000, hidden_size)
            self.layers = nn.ModuleList([
                nn.TransformerEncoderLayer(hidden_size, num_heads, hidden_size * 4)
                for _ in range(num_layers)
            ])
            self.norm = nn.LayerNorm(hidden_size)
            
        def forward(self, input_ids):
            x = self.embeddings(input_ids)
            for layer in self.layers:
                x = layer(x)
            return self.norm(x)
    
    # Create model
    model = SimpleTransformer(hidden_size=256, num_layers=4, num_heads=8)
    input_shape = (4, 128)  # batch_size, sequence_length
    
    print(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Test compression methods
    print("\n--- Testing Compression Methods ---")
    
    # Magnitude pruning
    import copy
    pruner = MagnitudePruning(sparsity=0.5)
    pruned_model = pruner.compress(copy.deepcopy(model))
    pruning_stats = pruner.get_compression_stats(pruned_model)
    
    print(f"\nPruning Results:")
    print(f"  Sparsity: {pruning_stats['sparsity']:.2%}")
    print(f"  Compression ratio: {pruning_stats['compression_ratio']:.2f}x")
    
    # Quantization
    quantizer = ModelQuantizer(bits=8)
    
    # Test quantization on a tensor
    test_tensor = torch.randn(10, 10)
    quantized, scale, zero_point = quantizer.quantize_tensor(test_tensor)
    dequantized = quantizer.dequantize_tensor(quantized, scale, zero_point)
    
    print(f"\nQuantization Test:")
    print(f"  Original range: [{test_tensor.min():.3f}, {test_tensor.max():.3f}]")
    print(f"  Quantized range: [{quantized.min()}, {quantized.max()}]")
    print(f"  Quantization error: {(test_tensor - dequantized).abs().mean():.6f}")
    
    # Benchmark different methods
    print("\n--- Benchmarking Optimization Methods ---")
    input_ids = torch.randint(0, 30000, input_shape)
    
    # Note: Simplified benchmarking for demo
    methods = {
        'Original': model,
        'Pruned (50%)': pruned_model,
    }
    
    for name, m in methods.items():
        start = time.time()
        with torch.no_grad():
            for _ in range(10):
                _ = m(input_ids)
        elapsed = time.time() - start
        print(f"{name}: {elapsed:.3f}s for 10 iterations")
    
    # Test Flash Attention
    print("\n--- Flash Attention Demo ---")
    flash_attn = FlashAttention(hidden_size=256, num_heads=8)
    hidden_states = torch.randn(2, 512, 256)
    
    output = flash_attn(hidden_states)
    print(f"Flash Attention input shape: {hidden_states.shape}")
    print(f"Flash Attention output shape: {output.shape}")
    
    # Test KV Cache
    print("\n--- KV Cache Demo ---")
    kv_cache = KVCache(
        max_batch_size=16,
        max_seq_length=2048,
        num_layers=4,
        num_heads=8,
        head_dim=32
    )
    
    # Simulate caching
    k = torch.randn(2, 8, 10, 32)  # [batch, heads, seq_len, head_dim]
    v = torch.randn(2, 8, 10, 32)
    
    kv_cache.update(layer_idx=0, k=k, v=v)
    print(f"Cached sequence length: {kv_cache.cache_lens}")
    
    # Performance monitoring
    print("\n--- Performance Monitoring ---")
    monitor = PerformanceMonitor()
    
    # Simulate inference requests
    for i in range(20):
        monitor.start_inference()
        time.sleep(0.01)  # Simulate inference
        monitor.end_inference(batch_size=4, sequence_length=128)
    
    summary = monitor.get_summary()
    print("\nPerformance Summary:")
    for metric, value in summary.items():
        if 'latency' in metric:
            print(f"  {metric}: {value:.2f} ms")
        else:
            print(f"  {metric}: {value:.2f}")
    
    print("\n✅ Scaling and optimization techniques demonstrated successfully!")