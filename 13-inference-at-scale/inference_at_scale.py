"""
Inference at Scale
Implementation of production serving architectures, batching strategies,
and optimization techniques for transformer model deployment.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import asyncio
import json
import hashlib
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from dataclasses import dataclass, field
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
import threading
import queue
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Configuration Classes
@dataclass
class ServingConfig:
    """Configuration for model serving."""
    model_name: str = "transformer"
    batch_size: int = 32
    max_sequence_length: int = 512
    num_workers: int = 4
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dtype: torch.dtype = torch.float16
    use_cache: bool = True
    cache_ttl: int = 3600
    
    
@dataclass
class BatchingConfig:
    """Configuration for batching strategies."""
    strategy: str = "dynamic"  # static, dynamic, continuous
    max_batch_size: int = 64
    max_wait_time_ms: float = 50.0
    bucket_sizes: List[int] = field(default_factory=lambda: [128, 256, 512, 1024])
    padding_strategy: str = "right"  # right, left, bucket


# Base Model Server
class ModelServer:
    """Base class for model serving."""
    
    def __init__(self, model_path: str, config: ServingConfig):
        self.config = config
        self.model = self._load_model(model_path)
        self.device = torch.device(config.device)
        self.model.to(self.device)
        self.model.eval()
        
    def _load_model(self, model_path: str) -> nn.Module:
        """Load model from path."""
        model = torch.load(model_path, map_location="cpu")
        if self.config.dtype == torch.float16:
            model.half()
        return model
        
    def predict(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Run inference on inputs."""
        with torch.no_grad():
            # Move inputs to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            outputs = self.model(**inputs)
            return outputs


# Replicated Model Server
class ReplicatedModelServer:
    """Server with multiple model replicas for scaling."""
    
    def __init__(self, model_path: str, num_replicas: int = 4):
        self.num_replicas = num_replicas
        self.models = []
        self.devices = []
        
        # Create replicas on different devices/streams
        for i in range(num_replicas):
            if torch.cuda.is_available() and torch.cuda.device_count() > 1:
                device = torch.device(f"cuda:{i % torch.cuda.device_count()}")
            else:
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                
            model = torch.load(model_path, map_location="cpu")
            model.to(device)
            model.eval()
            
            self.models.append(model)
            self.devices.append(device)
            
        self.current_replica = 0
        self.replica_lock = threading.Lock()
        
    def get_next_replica(self) -> Tuple[nn.Module, torch.device]:
        """Get next available replica (round-robin)."""
        with self.replica_lock:
            replica_id = self.current_replica
            self.current_replica = (self.current_replica + 1) % self.num_replicas
            
        return self.models[replica_id], self.devices[replica_id]
        
    async def predict_async(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Asynchronous prediction."""
        model, device = self.get_next_replica()
        
        # Move inputs to device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Run inference
        with torch.no_grad():
            outputs = model(**inputs)
            
        return outputs


# Static Batching
class StaticBatcher:
    """Fixed batch size batching."""
    
    def __init__(self, batch_size: int = 32):
        self.batch_size = batch_size
        self.queue = deque()
        self.lock = threading.Lock()
        
    def add_request(self, request_id: str, inputs: Dict[str, torch.Tensor]) -> None:
        """Add request to queue."""
        with self.lock:
            self.queue.append({
                'id': request_id,
                'inputs': inputs,
                'timestamp': time.time()
            })
            
    def get_batch(self) -> Optional[List[Dict[str, Any]]]:
        """Get a batch of requests."""
        with self.lock:
            if len(self.queue) >= self.batch_size:
                batch = []
                for _ in range(self.batch_size):
                    batch.append(self.queue.popleft())
                return batch
                
        return None
        
    def create_batch_inputs(self, batch: List[Dict[str, Any]]) -> Tuple[Dict[str, torch.Tensor], List[str]]:
        """Create batched inputs from requests."""
        # Find max length in batch
        max_length = max(req['inputs']['input_ids'].shape[1] for req in batch)
        
        # Pad and stack inputs
        input_ids_list = []
        attention_mask_list = []
        request_ids = []
        
        for req in batch:
            input_ids = req['inputs']['input_ids']
            attention_mask = req['inputs'].get('attention_mask', torch.ones_like(input_ids))
            
            # Pad to max length
            pad_length = max_length - input_ids.shape[1]
            if pad_length > 0:
                input_ids = F.pad(input_ids, (0, pad_length), value=0)
                attention_mask = F.pad(attention_mask, (0, pad_length), value=0)
                
            input_ids_list.append(input_ids)
            attention_mask_list.append(attention_mask)
            request_ids.append(req['id'])
            
        # Stack
        batched_inputs = {
            'input_ids': torch.cat(input_ids_list, dim=0),
            'attention_mask': torch.cat(attention_mask_list, dim=0)
        }
        
        return batched_inputs, request_ids


# Dynamic Batching
class DynamicBatcher:
    """Dynamic batching with timeout."""
    
    def __init__(self, config: BatchingConfig):
        self.config = config
        self.queue = deque()
        self.lock = threading.Lock()
        self.condition = threading.Condition(self.lock)
        
    def add_request(self, request_id: str, inputs: Dict[str, torch.Tensor]) -> None:
        """Add request to queue."""
        with self.lock:
            self.queue.append({
                'id': request_id,
                'inputs': inputs,
                'timestamp': time.time()
            })
            self.condition.notify()
            
    def get_batch(self, timeout_ms: Optional[float] = None) -> Optional[List[Dict[str, Any]]]:
        """Get batch with dynamic sizing."""
        timeout_ms = timeout_ms or self.config.max_wait_time_ms
        
        with self.lock:
            # Wait for requests or timeout
            if not self.queue:
                self.condition.wait(timeout_ms / 1000.0)
                
            if not self.queue:
                return None
                
            # Determine batch size
            current_time = time.time()
            batch = []
            
            while self.queue and len(batch) < self.config.max_batch_size:
                req = self.queue[0]
                
                # Check if we should wait for more requests
                wait_time = (current_time - req['timestamp']) * 1000
                if len(batch) == 0 or wait_time >= timeout_ms:
                    batch.append(self.queue.popleft())
                else:
                    break
                    
            return batch if batch else None
            
    def bucket_batch(self, batch: List[Dict[str, Any]]) -> Dict[int, List[Dict[str, Any]]]:
        """Group requests by sequence length buckets."""
        buckets = defaultdict(list)
        
        for req in batch:
            seq_len = req['inputs']['input_ids'].shape[1]
            
            # Find appropriate bucket
            bucket_size = min(
                b for b in self.config.bucket_sizes 
                if b >= seq_len
            )
            buckets[bucket_size].append(req)
            
        return dict(buckets)


# Continuous Batching
class ContinuousBatcher:
    """Continuous batching for generation tasks."""
    
    def __init__(self, max_batch_size: int = 32, block_size: int = 8):
        self.max_batch_size = max_batch_size
        self.block_size = block_size
        self.active_sequences = {}
        self.kv_cache = {}
        self.sequence_lock = threading.Lock()
        
    def add_sequence(self, seq_id: str, prompt: torch.Tensor, 
                    max_length: int = 100) -> None:
        """Add new sequence for generation."""
        with self.sequence_lock:
            self.active_sequences[seq_id] = {
                'input_ids': prompt,
                'generated_ids': [],
                'max_length': max_length,
                'done': False,
                'created_at': time.time()
            }
            
    def get_generation_batch(self) -> Tuple[Dict[str, torch.Tensor], List[str]]:
        """Get batch for next generation step."""
        with self.sequence_lock:
            # Collect active sequences
            batch_ids = []
            input_ids_list = []
            position_ids_list = []
            
            for seq_id, seq_data in list(self.active_sequences.items()):
                if seq_data['done']:
                    continue
                    
                # Check max length
                total_length = (seq_data['input_ids'].shape[-1] + 
                               len(seq_data['generated_ids']))
                if total_length >= seq_data['max_length']:
                    seq_data['done'] = True
                    continue
                    
                batch_ids.append(seq_id)
                
                # Get last token for generation
                if seq_data['generated_ids']:
                    last_token = seq_data['generated_ids'][-1]
                    input_ids = torch.tensor([[last_token]])
                else:
                    input_ids = seq_data['input_ids']
                    
                input_ids_list.append(input_ids)
                position_ids_list.append(torch.tensor([[total_length - 1]]))
                
            if not batch_ids:
                return {}, []
                
            # Create batch
            batch_inputs = {
                'input_ids': torch.cat(input_ids_list, dim=0),
                'position_ids': torch.cat(position_ids_list, dim=0)
            }
            
            # Add cached keys/values
            if self.kv_cache:
                batch_inputs['past_key_values'] = self._get_batch_kv_cache(batch_ids)
                
            return batch_inputs, batch_ids
            
    def update_sequences(self, batch_ids: List[str], 
                        outputs: Dict[str, torch.Tensor]) -> Dict[str, List[int]]:
        """Update sequences with generated tokens."""
        completed = {}
        
        with self.sequence_lock:
            for i, seq_id in enumerate(batch_ids):
                seq_data = self.active_sequences[seq_id]
                
                # Get next token
                logits = outputs['logits'][i, -1, :]
                next_token = torch.argmax(logits).item()
                
                # Update sequence
                seq_data['generated_ids'].append(next_token)
                
                # Check for EOS or max length
                if next_token == 2 or len(seq_data['generated_ids']) >= seq_data['max_length']:
                    seq_data['done'] = True
                    completed[seq_id] = seq_data['generated_ids']
                    
                # Update KV cache
                if 'past_key_values' in outputs:
                    self._update_kv_cache(seq_id, outputs['past_key_values'], i)
                    
        return completed
        
    def _get_batch_kv_cache(self, batch_ids: List[str]) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """Get KV cache for batch."""
        # Simplified - real implementation would handle variable lengths
        return self.kv_cache.get(batch_ids[0], [])
        
    def _update_kv_cache(self, seq_id: str, new_cache: List[Tuple[torch.Tensor, torch.Tensor]], 
                        batch_idx: int) -> None:
        """Update KV cache for sequence."""
        # Extract cache for this sequence
        seq_cache = []
        for layer_cache in new_cache:
            k, v = layer_cache
            seq_cache.append((
                k[batch_idx:batch_idx+1],
                v[batch_idx:batch_idx+1]
            ))
        self.kv_cache[seq_id] = seq_cache


# KV Cache Manager
class KVCacheManager:
    """Manage key-value cache for efficient generation."""
    
    def __init__(self, max_batch_size: int, max_seq_length: int,
                 num_layers: int, num_heads: int, head_dim: int,
                 dtype: torch.dtype = torch.float16):
        self.max_batch_size = max_batch_size
        self.max_seq_length = max_seq_length
        self.num_layers = num_layers
        self.dtype = dtype
        
        # Pre-allocate cache tensors
        cache_shape = (max_batch_size, num_heads, max_seq_length, head_dim)
        
        self.k_cache = [
            torch.zeros(cache_shape, dtype=dtype)
            for _ in range(num_layers)
        ]
        self.v_cache = [
            torch.zeros(cache_shape, dtype=dtype)
            for _ in range(num_layers)
        ]
        
        # Track usage
        self.cache_positions = torch.zeros(max_batch_size, dtype=torch.long)
        self.active_slots = set()
        self.free_slots = set(range(max_batch_size))
        self.slot_to_request = {}
        
    def allocate_slot(self, request_id: str) -> Optional[int]:
        """Allocate cache slot for request."""
        if not self.free_slots:
            return None
            
        slot = self.free_slots.pop()
        self.active_slots.add(slot)
        self.slot_to_request[slot] = request_id
        self.cache_positions[slot] = 0
        
        return slot
        
    def free_slot(self, request_id: str) -> None:
        """Free cache slot."""
        slot = None
        for s, rid in self.slot_to_request.items():
            if rid == request_id:
                slot = s
                break
                
        if slot is not None:
            self.active_slots.remove(slot)
            self.free_slots.add(slot)
            del self.slot_to_request[slot]
            self.cache_positions[slot] = 0
            
            # Clear cache entries
            for layer_idx in range(self.num_layers):
                self.k_cache[layer_idx][slot].zero_()
                self.v_cache[layer_idx][slot].zero_()
                
    def update_cache(self, slot: int, layer_idx: int,
                    k: torch.Tensor, v: torch.Tensor) -> None:
        """Update cache for slot."""
        seq_len = k.shape[2]
        pos = self.cache_positions[slot].item()
        
        self.k_cache[layer_idx][slot, :, pos:pos+seq_len] = k[0]
        self.v_cache[layer_idx][slot, :, pos:pos+seq_len] = v[0]
        
        self.cache_positions[slot] += seq_len
        
    def get_cache(self, slot: int) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """Get cache for slot."""
        pos = self.cache_positions[slot].item()
        cache = []
        
        for layer_idx in range(self.num_layers):
            k = self.k_cache[layer_idx][slot:slot+1, :, :pos]
            v = self.v_cache[layer_idx][slot:slot+1, :, :pos]
            cache.append((k, v))
            
        return cache


# Response Cache
class ResponseCache:
    """Cache for inference results."""
    
    def __init__(self, max_size: int = 10000, ttl: int = 3600):
        self.max_size = max_size
        self.ttl = ttl
        self.cache = {}
        self.access_times = {}
        self.lock = threading.Lock()
        
    def _get_key(self, inputs: Dict[str, Any]) -> str:
        """Generate cache key from inputs."""
        # Convert tensors to lists for hashing
        hashable_inputs = {}
        for k, v in inputs.items():
            if torch.is_tensor(v):
                hashable_inputs[k] = v.tolist()
            else:
                hashable_inputs[k] = v
                
        input_str = json.dumps(hashable_inputs, sort_keys=True)
        return hashlib.md5(input_str.encode()).hexdigest()
        
    def get(self, inputs: Dict[str, Any]) -> Optional[Any]:
        """Get cached result."""
        key = self._get_key(inputs)
        
        with self.lock:
            if key in self.cache:
                # Check TTL
                if time.time() - self.access_times[key] < self.ttl:
                    self.access_times[key] = time.time()
                    return self.cache[key]
                else:
                    # Expired
                    del self.cache[key]
                    del self.access_times[key]
                    
        return None
        
    def set(self, inputs: Dict[str, Any], outputs: Any) -> None:
        """Cache result."""
        key = self._get_key(inputs)
        
        with self.lock:
            # Evict oldest if at capacity
            if len(self.cache) >= self.max_size:
                oldest_key = min(self.access_times, key=self.access_times.get)
                del self.cache[oldest_key]
                del self.access_times[oldest_key]
                
            self.cache[key] = outputs
            self.access_times[key] = time.time()


# Model Load Balancer
class ModelLoadBalancer:
    """Load balance requests across model instances."""
    
    def __init__(self, model_servers: List[ModelServer]):
        self.servers = model_servers
        self.num_servers = len(model_servers)
        self.server_loads = [0] * self.num_servers
        self.server_latencies = [deque(maxlen=100) for _ in range(self.num_servers)]
        self.lock = threading.Lock()
        
    def select_server(self, strategy: str = "least_loaded") -> int:
        """Select server based on strategy."""
        with self.lock:
            if strategy == "round_robin":
                # Simple round-robin
                return int(time.time() * 1000) % self.num_servers
                
            elif strategy == "least_loaded":
                # Choose least loaded
                return self.server_loads.index(min(self.server_loads))
                
            elif strategy == "latency_aware":
                # Choose based on latency
                avg_latencies = []
                for latencies in self.server_latencies:
                    if latencies:
                        avg = sum(latencies) / len(latencies)
                    else:
                        avg = 0
                    avg_latencies.append(avg)
                    
                return avg_latencies.index(min(avg_latencies))
                
            else:
                return 0
                
    async def route_request(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Route request to selected server."""
        server_idx = self.select_server()
        
        # Update load
        with self.lock:
            self.server_loads[server_idx] += 1
            
        try:
            # Time request
            start_time = time.time()
            outputs = await self.servers[server_idx].predict_async(inputs)
            latency = time.time() - start_time
            
            # Update metrics
            with self.lock:
                self.server_latencies[server_idx].append(latency)
                
            return outputs
            
        finally:
            # Update load
            with self.lock:
                self.server_loads[server_idx] -= 1


# Auto-scaler
class AutoScaler:
    """Automatically scale model instances."""
    
    def __init__(self, min_instances: int = 1, max_instances: int = 10,
                 scale_up_threshold: float = 0.8, scale_down_threshold: float = 0.3):
        self.min_instances = min_instances
        self.max_instances = max_instances
        self.scale_up_threshold = scale_up_threshold
        self.scale_down_threshold = scale_down_threshold
        self.current_instances = min_instances
        self.metrics_history = deque(maxlen=100)
        
    def update_metrics(self, metrics: Dict[str, float]) -> None:
        """Update scaling metrics."""
        self.metrics_history.append({
            'timestamp': time.time(),
            'cpu_usage': metrics.get('cpu_usage', 0),
            'gpu_usage': metrics.get('gpu_usage', 0),
            'queue_size': metrics.get('queue_size', 0),
            'avg_latency': metrics.get('avg_latency', 0)
        })
        
    def should_scale(self) -> int:
        """Determine if scaling is needed."""
        if len(self.metrics_history) < 10:
            return 0
            
        # Calculate recent averages
        recent_metrics = list(self.metrics_history)[-10:]
        
        avg_cpu = sum(m['cpu_usage'] for m in recent_metrics) / len(recent_metrics)
        avg_gpu = sum(m['gpu_usage'] for m in recent_metrics) / len(recent_metrics)
        avg_queue = sum(m['queue_size'] for m in recent_metrics) / len(recent_metrics)
        avg_latency = sum(m['avg_latency'] for m in recent_metrics) / len(recent_metrics)
        
        # Scale up conditions
        if (avg_cpu > self.scale_up_threshold or 
            avg_gpu > self.scale_up_threshold or 
            avg_queue > 50 or 
            avg_latency > 100):
            if self.current_instances < self.max_instances:
                return 1  # Scale up
                
        # Scale down conditions
        if (avg_cpu < self.scale_down_threshold and 
            avg_gpu < self.scale_down_threshold and 
            avg_queue < 5 and 
            avg_latency < 20):
            if self.current_instances > self.min_instances:
                return -1  # Scale down
                
        return 0  # No scaling


# Model Router
class ModelRouter:
    """Route requests to appropriate models."""
    
    def __init__(self):
        self.models = {}
        self.routing_rules = {}
        
    def register_model(self, name: str, server: ModelServer,
                      rule: Callable[[Dict[str, Any]], bool]) -> None:
        """Register model with routing rule."""
        self.models[name] = server
        self.routing_rules[name] = rule
        
    def route(self, request: Dict[str, Any]) -> Optional[ModelServer]:
        """Route request to appropriate model."""
        for model_name, rule in self.routing_rules.items():
            if rule(request):
                return self.models[model_name]
                
        # Default model
        return self.models.get('default')
        
    def create_size_based_rules(self) -> Dict[str, Callable]:
        """Create routing rules based on input size."""
        return {
            'small': lambda req: req.get('sequence_length', 0) < 128,
            'medium': lambda req: 128 <= req.get('sequence_length', 0) < 512,
            'large': lambda req: req.get('sequence_length', 0) >= 512
        }


# Inference Engine
class InferenceEngine:
    """Complete inference engine with all optimizations."""
    
    def __init__(self, config: ServingConfig):
        self.config = config
        self.servers = []
        self.batcher = DynamicBatcher(BatchingConfig())
        self.cache = ResponseCache()
        self.kv_cache_manager = None
        self.metrics = defaultdict(list)
        self.executor = ThreadPoolExecutor(max_workers=config.num_workers)
        
    def add_server(self, server: ModelServer) -> None:
        """Add model server."""
        self.servers.append(server)
        
    async def process_request(self, request_id: str, 
                            inputs: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """Process single request."""
        # Check cache
        if self.config.use_cache:
            cached = self.cache.get(inputs)
            if cached is not None:
                logger.info(f"Cache hit for request {request_id}")
                return cached
                
        # Add to batch queue
        self.batcher.add_request(request_id, inputs)
        
        # Wait for batch processing
        result = await self._wait_for_result(request_id)
        
        # Cache result
        if self.config.use_cache and result is not None:
            self.cache.set(inputs, result)
            
        return result
        
    async def _wait_for_result(self, request_id: str, 
                             timeout: float = 30.0) -> Optional[Dict[str, Any]]:
        """Wait for request result."""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            if request_id in self.results:
                return self.results.pop(request_id)
            await asyncio.sleep(0.01)
            
        return None
        
    def run_batch_processor(self) -> None:
        """Run batch processing loop."""
        while True:
            batch = self.batcher.get_batch()
            if batch:
                self._process_batch(batch)
            else:
                time.sleep(0.001)
                
    def _process_batch(self, batch: List[Dict[str, Any]]) -> None:
        """Process a batch of requests."""
        start_time = time.time()
        
        # Create batched inputs
        batched_inputs, request_ids = self.batcher.create_batch_inputs(batch)
        
        # Select server (simplified - could use load balancer)
        server = self.servers[0]
        
        # Run inference
        outputs = server.predict(batched_inputs)
        
        # Split results
        for i, request_id in enumerate(request_ids):
            self.results[request_id] = {
                'logits': outputs['logits'][i],
                'hidden_states': outputs.get('hidden_states', [None])[i]
            }
            
        # Update metrics
        batch_time = time.time() - start_time
        self.metrics['batch_latency'].append(batch_time * 1000)
        self.metrics['batch_size'].append(len(batch))
        self.metrics['throughput'].append(len(batch) / batch_time)


# Performance Monitor
class PerformanceMonitor:
    """Monitor inference performance."""
    
    def __init__(self):
        self.metrics = defaultdict(lambda: deque(maxlen=1000))
        self.start_times = {}
        
    def start_request(self, request_id: str) -> None:
        """Start timing request."""
        self.start_times[request_id] = time.time()
        
    def end_request(self, request_id: str) -> None:
        """End timing and record metrics."""
        if request_id in self.start_times:
            latency = (time.time() - self.start_times[request_id]) * 1000
            self.metrics['request_latency'].append(latency)
            del self.start_times[request_id]
            
    def record_metric(self, name: str, value: float) -> None:
        """Record custom metric."""
        self.metrics[name].append(value)
        
    def get_summary(self) -> Dict[str, Dict[str, float]]:
        """Get metrics summary."""
        summary = {}
        
        for metric_name, values in self.metrics.items():
            if values:
                values_list = list(values)
                summary[metric_name] = {
                    'mean': np.mean(values_list),
                    'p50': np.percentile(values_list, 50),
                    'p95': np.percentile(values_list, 95),
                    'p99': np.percentile(values_list, 99),
                    'min': min(values_list),
                    'max': max(values_list)
                }
                
        return summary


# Example Usage
if __name__ == "__main__":
    print("=== Inference at Scale Demo ===\n")
    
    # Create dummy model for testing
    class DummyTransformer(nn.Module):
        def __init__(self, vocab_size=30000, hidden_size=768, num_layers=12):
            super().__init__()
            self.embeddings = nn.Embedding(vocab_size, hidden_size)
            self.layers = nn.ModuleList([
                nn.TransformerEncoderLayer(hidden_size, 8, hidden_size * 4)
                for _ in range(num_layers)
            ])
            self.lm_head = nn.Linear(hidden_size, vocab_size)
            
        def forward(self, input_ids, attention_mask=None, past_key_values=None):
            x = self.embeddings(input_ids)
            
            for layer in self.layers:
                x = layer(x)
                
            logits = self.lm_head(x)
            
            return {'logits': logits}
    
    # Test batching strategies
    print("--- Testing Batching Strategies ---\n")
    
    # Static batching
    static_batcher = StaticBatcher(batch_size=4)
    
    # Add requests
    for i in range(10):
        inputs = {
            'input_ids': torch.randint(0, 30000, (1, 128 + i * 10))
        }
        static_batcher.add_request(f"req_{i}", inputs)
    
    # Get batch
    batch = static_batcher.get_batch()
    if batch:
        print(f"Static batch size: {len(batch)}")
        batched_inputs, ids = static_batcher.create_batch_inputs(batch)
        print(f"Batched input shape: {batched_inputs['input_ids'].shape}")
    
    # Dynamic batching
    print("\n--- Dynamic Batching ---")
    dynamic_config = BatchingConfig(
        strategy="dynamic",
        max_batch_size=8,
        max_wait_time_ms=50.0
    )
    dynamic_batcher = DynamicBatcher(dynamic_config)
    
    # Simulate requests arriving at different times
    import threading
    
    def add_delayed_request(batcher, delay, req_id):
        time.sleep(delay)
        inputs = {'input_ids': torch.randint(0, 30000, (1, 256))}
        batcher.add_request(req_id, inputs)
    
    # Start threads
    for i in range(5):
        t = threading.Thread(
            target=add_delayed_request,
            args=(dynamic_batcher, i * 0.01, f"delayed_{i}")
        )
        t.start()
    
    time.sleep(0.1)
    batch = dynamic_batcher.get_batch()
    if batch:
        print(f"Dynamic batch collected {len(batch)} requests")
    
    # Test KV Cache
    print("\n--- KV Cache Manager ---")
    kv_cache = KVCacheManager(
        max_batch_size=16,
        max_seq_length=2048,
        num_layers=12,
        num_heads=12,
        head_dim=64
    )
    
    # Allocate slots
    slot1 = kv_cache.allocate_slot("request_1")
    slot2 = kv_cache.allocate_slot("request_2")
    print(f"Allocated slots: {slot1}, {slot2}")
    print(f"Free slots remaining: {len(kv_cache.free_slots)}")
    
    # Test continuous batching
    print("\n--- Continuous Batching ---")
    continuous_batcher = ContinuousBatcher(max_batch_size=4)
    
    # Add sequences
    for i in range(3):
        prompt = torch.randint(0, 30000, (1, 10))
        continuous_batcher.add_sequence(f"seq_{i}", prompt, max_length=50)
    
    # Get generation batch
    batch_inputs, batch_ids = continuous_batcher.get_generation_batch()
    print(f"Generation batch size: {len(batch_ids)}")
    print(f"Active sequences: {len(continuous_batcher.active_sequences)}")
    
    # Test caching
    print("\n--- Response Cache ---")
    cache = ResponseCache(max_size=100, ttl=60)
    
    # Cache some results
    test_input = {'input_ids': torch.tensor([[1, 2, 3, 4, 5]])}
    test_output = {'logits': torch.randn(1, 5, 30000)}
    
    cache.set(test_input, test_output)
    cached_result = cache.get(test_input)
    print(f"Cache hit: {cached_result is not None}")
    
    # Test auto-scaling
    print("\n--- Auto-scaling ---")
    scaler = AutoScaler(min_instances=1, max_instances=5)
    
    # Simulate high load
    for i in range(20):
        metrics = {
            'cpu_usage': 0.9 if i < 10 else 0.3,
            'gpu_usage': 0.85 if i < 10 else 0.25,
            'queue_size': 100 if i < 10 else 5,
            'avg_latency': 150 if i < 10 else 15
        }
        scaler.update_metrics(metrics)
    
    # Check scaling decision
    scale_decision = scaler.should_scale()
    print(f"Scaling decision: {'+1' if scale_decision > 0 else '-1' if scale_decision < 0 else '0'}")
    
    # Performance monitoring
    print("\n--- Performance Monitoring ---")
    monitor = PerformanceMonitor()
    
    # Simulate requests
    for i in range(100):
        req_id = f"perf_test_{i}"
        monitor.start_request(req_id)
        time.sleep(0.001 * (1 + i % 10))  # Variable latency
        monitor.end_request(req_id)
        monitor.record_metric('batch_size', 8 + (i % 8))
    
    # Get summary
    summary = monitor.get_summary()
    print("\nPerformance Summary:")
    for metric, stats in summary.items():
        print(f"\n{metric}:")
        for stat, value in stats.items():
            print(f"  {stat}: {value:.2f}")
    
    print("\n✅ Inference at scale techniques demonstrated successfully!")