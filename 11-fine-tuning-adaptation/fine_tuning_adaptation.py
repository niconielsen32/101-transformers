"""
Fine-tuning and Adaptation
Implementation of various fine-tuning methods including full fine-tuning,
LoRA, QLoRA, prompt tuning, and other parameter-efficient techniques.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass
from collections import OrderedDict
import matplotlib.pyplot as plt
from tqdm import tqdm


# Configuration Classes
@dataclass
class FineTuningConfig:
    """Configuration for fine-tuning."""
    model_name: str = "bert-base"
    task_type: str = "classification"  # classification, generation, qa
    num_labels: int = 2
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_steps: int = 500
    num_epochs: int = 3
    batch_size: int = 16
    max_length: int = 512
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    
    
@dataclass
class LoRAConfig:
    """Configuration for LoRA."""
    r: int = 16  # Rank
    lora_alpha: int = 32  # Scaling factor
    lora_dropout: float = 0.1
    target_modules: List[str] = None
    
    def __post_init__(self):
        if self.target_modules is None:
            self.target_modules = ["q_proj", "v_proj"]


# Base Classes
class FineTuningMethod:
    """Base class for fine-tuning methods."""
    
    def prepare_model(self, model: nn.Module) -> nn.Module:
        """Prepare model for fine-tuning."""
        raise NotImplementedError
        
    def get_optimizer_params(self, model: nn.Module) -> List[Dict]:
        """Get optimizer parameter groups."""
        raise NotImplementedError


# Full Fine-tuning
class FullFineTuning(FineTuningMethod):
    """Standard full fine-tuning of all parameters."""
    
    def __init__(self, config: FineTuningConfig):
        self.config = config
        
    def prepare_model(self, model: nn.Module) -> nn.Module:
        """All parameters are trainable."""
        for param in model.parameters():
            param.requires_grad = True
        return model
    
    def get_optimizer_params(self, model: nn.Module) -> List[Dict]:
        """Layer-wise learning rate decay."""
        # Group parameters by layer
        no_decay = ["bias", "LayerNorm.weight"]
        
        # Get layer groups
        layer_groups = self._get_layer_groups(model)
        
        optimizer_grouped_parameters = []
        
        for layer_id, layer_params in enumerate(layer_groups):
            # Calculate layer-wise lr decay
            lr_scale = 0.95 ** (len(layer_groups) - layer_id - 1)
            
            # Split into decay and no_decay groups
            decay_params = [p for n, p in layer_params if not any(nd in n for nd in no_decay)]
            no_decay_params = [p for n, p in layer_params if any(nd in n for nd in no_decay)]
            
            if decay_params:
                optimizer_grouped_parameters.append({
                    "params": decay_params,
                    "weight_decay": self.config.weight_decay,
                    "lr": self.config.learning_rate * lr_scale
                })
                
            if no_decay_params:
                optimizer_grouped_parameters.append({
                    "params": no_decay_params,
                    "weight_decay": 0.0,
                    "lr": self.config.learning_rate * lr_scale
                })
                
        return optimizer_grouped_parameters
    
    def _get_layer_groups(self, model: nn.Module) -> List[List[Tuple[str, nn.Parameter]]]:
        """Group parameters by layer."""
        layer_groups = []
        current_group = []
        current_layer_name = ""
        
        for name, param in model.named_parameters():
            layer_name = name.split('.')[0]
            
            if layer_name != current_layer_name and current_group:
                layer_groups.append(current_group)
                current_group = []
                current_layer_name = layer_name
                
            current_group.append((name, param))
            
        if current_group:
            layer_groups.append(current_group)
            
        return layer_groups


# LoRA Implementation
class LoRALayer(nn.Module):
    """Low-Rank Adaptation layer."""
    
    def __init__(self, in_features: int, out_features: int, 
                 rank: int = 16, alpha: int = 16, dropout: float = 0.0):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        self.dropout = nn.Dropout(dropout)
        
        # LoRA parameters
        self.lora_A = nn.Parameter(torch.zeros(in_features, rank))
        self.lora_B = nn.Parameter(torch.zeros(rank, out_features))
        
        # Initialize
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply LoRA transformation."""
        x = self.dropout(x)
        return (x @ self.lora_A @ self.lora_B) * self.scaling


class LoRALinear(nn.Module):
    """Linear layer with LoRA."""
    
    def __init__(self, base_layer: nn.Linear, rank: int = 16, 
                 alpha: int = 16, dropout: float = 0.0):
        super().__init__()
        self.base_layer = base_layer
        self.lora = LoRALayer(
            base_layer.in_features,
            base_layer.out_features,
            rank=rank,
            alpha=alpha,
            dropout=dropout
        )
        
        # Freeze base layer
        for param in self.base_layer.parameters():
            param.requires_grad = False
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass combining base layer and LoRA."""
        return self.base_layer(x) + self.lora(x)


class LoRAFineTuning(FineTuningMethod):
    """LoRA fine-tuning method."""
    
    def __init__(self, config: LoRAConfig):
        self.config = config
        
    def prepare_model(self, model: nn.Module) -> nn.Module:
        """Add LoRA layers to model."""
        # Freeze all parameters first
        for param in model.parameters():
            param.requires_grad = False
            
        # Add LoRA to target modules
        for name, module in model.named_modules():
            if any(target in name for target in self.config.target_modules):
                if isinstance(module, nn.Linear):
                    # Get parent module and attribute name
                    parent_name, attr_name = name.rsplit('.', 1)
                    parent_module = model
                    for part in parent_name.split('.'):
                        parent_module = getattr(parent_module, part)
                    
                    # Replace with LoRA layer
                    lora_layer = LoRALinear(
                        module,
                        rank=self.config.r,
                        alpha=self.config.lora_alpha,
                        dropout=self.config.lora_dropout
                    )
                    setattr(parent_module, attr_name, lora_layer)
                    
        return model
    
    def get_optimizer_params(self, model: nn.Module) -> List[Dict]:
        """Only LoRA parameters are trainable."""
        lora_params = []
        
        for name, param in model.named_parameters():
            if param.requires_grad:
                lora_params.append(param)
                
        return [{"params": lora_params}]
    
    def merge_weights(self, model: nn.Module) -> nn.Module:
        """Merge LoRA weights into base model."""
        for name, module in model.named_modules():
            if isinstance(module, LoRALinear):
                # Compute merged weight
                with torch.no_grad():
                    delta_w = module.lora.lora_A @ module.lora.lora_B * module.lora.scaling
                    module.base_layer.weight.data += delta_w.T
                    
        return model


# QLoRA Implementation
class QLoRALinear(nn.Module):
    """4-bit quantized linear layer with LoRA."""
    
    def __init__(self, base_layer: nn.Linear, rank: int = 16, 
                 alpha: int = 16, dropout: float = 0.0):
        super().__init__()
        self.in_features = base_layer.in_features
        self.out_features = base_layer.out_features
        
        # Quantize base layer weights to 4-bit
        self.quantized_weight, self.scales, self.zeros = self._quantize_weight(
            base_layer.weight.data
        )
        self.bias = base_layer.bias
        
        # LoRA parameters (in fp16)
        self.lora = LoRALayer(
            self.in_features,
            self.out_features,
            rank=rank,
            alpha=alpha,
            dropout=dropout
        )
        
    def _quantize_weight(self, weight: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Quantize weight to 4-bit NF4 format."""
        # Simplified 4-bit quantization (real QLoRA uses NF4)
        n_bits = 4
        abs_max = weight.abs().max()
        scales = abs_max / (2 ** (n_bits - 1) - 1)
        
        # Quantize
        quantized = torch.round(weight / scales).clamp(-8, 7).to(torch.int8)
        zeros = torch.zeros_like(scales)
        
        return quantized, scales, zeros
    
    def _dequantize_weight(self) -> torch.Tensor:
        """Dequantize weight back to fp16."""
        return self.quantized_weight.float() * self.scales
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with dequantized weights + LoRA."""
        # Dequantize weight
        weight = self._dequantize_weight()
        
        # Base computation
        output = F.linear(x, weight, self.bias)
        
        # Add LoRA
        output = output + self.lora(x)
        
        return output


# Adapter Modules
class AdapterModule(nn.Module):
    """Adapter module for parameter-efficient fine-tuning."""
    
    def __init__(self, hidden_size: int, adapter_size: int = 64, 
                 activation: str = "relu"):
        super().__init__()
        self.down_project = nn.Linear(hidden_size, adapter_size)
        self.up_project = nn.Linear(adapter_size, hidden_size)
        
        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "gelu":
            self.activation = nn.GELU()
        else:
            self.activation = nn.ReLU()
            
        # Initialize near identity
        nn.init.normal_(self.down_project.weight, std=1e-3)
        nn.init.zeros_(self.down_project.bias)
        nn.init.normal_(self.up_project.weight, std=1e-3)
        nn.init.zeros_(self.up_project.bias)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply adapter: residual connection with bottleneck."""
        return x + self.up_project(self.activation(self.down_project(x)))


class AdapterFineTuning(FineTuningMethod):
    """Adapter-based fine-tuning."""
    
    def __init__(self, adapter_size: int = 64):
        self.adapter_size = adapter_size
        
    def prepare_model(self, model: nn.Module) -> nn.Module:
        """Insert adapters into model."""
        # Freeze base model
        for param in model.parameters():
            param.requires_grad = False
            
        # Add adapters after self-attention and FFN
        for name, module in model.named_modules():
            if "attention" in name.lower() or "mlp" in name.lower() or "ffn" in name.lower():
                # Get hidden size
                hidden_size = None
                for child in module.modules():
                    if isinstance(child, nn.Linear):
                        hidden_size = child.out_features
                        break
                        
                if hidden_size:
                    # Add adapter
                    adapter = AdapterModule(hidden_size, self.adapter_size)
                    module.adapter = adapter
                    
        return model
    
    def get_optimizer_params(self, model: nn.Module) -> List[Dict]:
        """Only adapter parameters are trainable."""
        adapter_params = []
        
        for name, param in model.named_parameters():
            if "adapter" in name and param.requires_grad:
                adapter_params.append(param)
                
        return [{"params": adapter_params}]


# Prompt Tuning
class SoftPrompt(nn.Module):
    """Soft prompt for prompt tuning."""
    
    def __init__(self, n_tokens: int, embedding_dim: int):
        super().__init__()
        self.n_tokens = n_tokens
        self.embedding_dim = embedding_dim
        
        # Learnable prompt embeddings
        self.embeddings = nn.Parameter(
            torch.randn(n_tokens, embedding_dim) * 0.01
        )
        
    def forward(self, batch_size: int) -> torch.Tensor:
        """Expand prompt for batch."""
        return self.embeddings.unsqueeze(0).expand(batch_size, -1, -1)


class PromptTuning(FineTuningMethod):
    """Prompt tuning method."""
    
    def __init__(self, n_prompt_tokens: int = 20):
        self.n_prompt_tokens = n_prompt_tokens
        
    def prepare_model(self, model: nn.Module) -> nn.Module:
        """Add soft prompts to model."""
        # Freeze entire model
        for param in model.parameters():
            param.requires_grad = False
            
        # Get embedding dimension
        embedding_dim = None
        for module in model.modules():
            if isinstance(module, nn.Embedding):
                embedding_dim = module.embedding_dim
                break
                
        if embedding_dim is None:
            raise ValueError("Could not find embedding dimension")
            
        # Add soft prompt
        model.soft_prompt = SoftPrompt(self.n_prompt_tokens, embedding_dim)
        
        # Modify forward function to prepend prompt
        original_forward = model.forward
        
        def prompted_forward(input_ids, attention_mask=None, **kwargs):
            batch_size = input_ids.shape[0]
            
            # Get prompt embeddings
            prompt_embeds = model.soft_prompt(batch_size)
            
            # Get input embeddings
            input_embeds = model.get_input_embeddings()(input_ids)
            
            # Concatenate
            inputs_embeds = torch.cat([prompt_embeds, input_embeds], dim=1)
            
            # Update attention mask
            if attention_mask is not None:
                prompt_mask = torch.ones(
                    batch_size, self.n_prompt_tokens,
                    dtype=attention_mask.dtype,
                    device=attention_mask.device
                )
                attention_mask = torch.cat([prompt_mask, attention_mask], dim=1)
                
            # Forward with embeddings
            return original_forward(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                **kwargs
            )
            
        model.forward = prompted_forward
        
        return model
    
    def get_optimizer_params(self, model: nn.Module) -> List[Dict]:
        """Only prompt parameters are trainable."""
        return [{"params": [model.soft_prompt.embeddings]}]


# Prefix Tuning
class PrefixTuning(FineTuningMethod):
    """Prefix tuning for encoder-decoder models."""
    
    def __init__(self, prefix_length: int = 20, hidden_size: int = 768):
        self.prefix_length = prefix_length
        self.hidden_size = hidden_size
        
    def prepare_model(self, model: nn.Module) -> nn.Module:
        """Add prefix to model."""
        # Freeze model
        for param in model.parameters():
            param.requires_grad = False
            
        # Add prefix parameters
        n_layers = sum(1 for _ in model.modules() if isinstance(_, nn.TransformerEncoderLayer))
        
        # Prefix embeddings
        model.prefix_tokens = nn.Parameter(
            torch.randn(self.prefix_length, self.hidden_size)
        )
        
        # Reparameterization network
        model.prefix_mlp = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size * 2),
            nn.Tanh(),
            nn.Linear(self.hidden_size * 2, n_layers * 2 * self.hidden_size)
        )
        
        return model
    
    def get_optimizer_params(self, model: nn.Module) -> List[Dict]:
        """Prefix parameters only."""
        prefix_params = []
        
        for name, param in model.named_parameters():
            if "prefix" in name and param.requires_grad:
                prefix_params.append(param)
                
        return [{"params": prefix_params}]


# BitFit
class BitFit(FineTuningMethod):
    """Bias-term Fine-tuning."""
    
    def prepare_model(self, model: nn.Module) -> nn.Module:
        """Only bias terms are trainable."""
        for name, param in model.named_parameters():
            if "bias" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
                
        return model
    
    def get_optimizer_params(self, model: nn.Module) -> List[Dict]:
        """Only bias parameters."""
        bias_params = []
        
        for name, param in model.named_parameters():
            if param.requires_grad:
                bias_params.append(param)
                
        return [{"params": bias_params}]


# Training Utilities
class FineTuningTrainer:
    """Trainer for various fine-tuning methods."""
    
    def __init__(self, model: nn.Module, method: FineTuningMethod, 
                 config: FineTuningConfig):
        self.model = model
        self.method = method
        self.config = config
        
        # Prepare model
        self.model = self.method.prepare_model(self.model)
        
        # Count parameters
        self._count_parameters()
        
    def _count_parameters(self):
        """Count trainable and total parameters."""
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        print(f"\nParameter Summary:")
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"Trainable percentage: {trainable_params/total_params*100:.2f}%")
        
    def create_optimizer(self) -> torch.optim.Optimizer:
        """Create optimizer with appropriate parameter groups."""
        param_groups = self.method.get_optimizer_params(self.model)
        
        optimizer = torch.optim.AdamW(
            param_groups,
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        
        return optimizer
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> float:
        """Single training step."""
        self.model.train()
        
        # Forward pass
        outputs = self.model(**batch)
        loss = outputs.loss
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), 
            self.config.max_grad_norm
        )
        
        return loss.item()


# Analysis and Visualization
def compare_methods(model_size: int = 1000000):
    """Compare different fine-tuning methods."""
    
    methods = {
        'Full Fine-tuning': {'params': 1.0, 'memory': 1.0, 'quality': 1.0},
        'LoRA (r=16)': {'params': 0.01, 'memory': 0.1, 'quality': 0.98},
        'QLoRA': {'params': 0.01, 'memory': 0.05, 'quality': 0.97},
        'Adapters': {'params': 0.05, 'memory': 0.15, 'quality': 0.96},
        'Prompt Tuning': {'params': 0.001, 'memory': 0.01, 'quality': 0.90},
        'Prefix Tuning': {'params': 0.001, 'memory': 0.01, 'quality': 0.92},
        'BitFit': {'params': 0.001, 'memory': 0.1, 'quality': 0.85}
    }
    
    # Create comparison plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Parameters vs Quality
    params_pct = [v['params'] * 100 for v in methods.values()]
    quality = [v['quality'] * 100 for v in methods.values()]
    
    ax1.scatter(params_pct, quality, s=100)
    for i, method in enumerate(methods.keys()):
        ax1.annotate(method, (params_pct[i], quality[i]), 
                    xytext=(5, 5), textcoords='offset points')
    
    ax1.set_xlabel('Trainable Parameters (%)')
    ax1.set_ylabel('Relative Quality (%)')
    ax1.set_title('Parameter Efficiency vs Quality')
    ax1.set_xscale('log')
    ax1.grid(True, alpha=0.3)
    
    # Memory usage comparison
    method_names = list(methods.keys())
    memory_usage = [v['memory'] * 100 for v in methods.values()]
    
    bars = ax2.bar(range(len(method_names)), memory_usage)
    ax2.set_xticks(range(len(method_names)))
    ax2.set_xticklabels(method_names, rotation=45, ha='right')
    ax2.set_ylabel('Relative Memory Usage (%)')
    ax2.set_title('Memory Requirements by Method')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Color bars by efficiency
    colors = plt.cm.RdYlGn([(1 - m/100) for m in memory_usage])
    for bar, color in zip(bars, colors):
        bar.set_color(color)
    
    plt.tight_layout()
    plt.show()


def visualize_lora_weights(lora_A: torch.Tensor, lora_B: torch.Tensor):
    """Visualize LoRA weight matrices."""
    
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    
    # Matrix A
    im1 = ax1.imshow(lora_A.detach().numpy(), cmap='coolwarm', aspect='auto')
    ax1.set_title('LoRA Matrix A\n(Input → Rank)')
    ax1.set_xlabel('Rank dimension')
    ax1.set_ylabel('Input dimension')
    plt.colorbar(im1, ax=ax1)
    
    # Matrix B
    im2 = ax2.imshow(lora_B.detach().numpy(), cmap='coolwarm', aspect='auto')
    ax2.set_title('LoRA Matrix B\n(Rank → Output)')
    ax2.set_xlabel('Output dimension')
    ax2.set_ylabel('Rank dimension')
    plt.colorbar(im2, ax=ax2)
    
    # Combined effect (A @ B)
    combined = (lora_A @ lora_B).detach().numpy()
    im3 = ax3.imshow(combined, cmap='coolwarm', aspect='auto')
    ax3.set_title('Combined Effect\n(A @ B)')
    ax3.set_xlabel('Output dimension')
    ax3.set_ylabel('Input dimension')
    plt.colorbar(im3, ax=ax3)
    
    plt.tight_layout()
    plt.show()


# Example Usage
if __name__ == "__main__":
    print("=== Fine-tuning and Adaptation Methods Demo ===\n")
    
    # Compare methods
    compare_methods()
    
    # Demo LoRA
    print("\n--- LoRA Demo ---")
    
    # Create a simple linear layer
    linear = nn.Linear(768, 768)
    print(f"Original layer: {linear.weight.shape}")
    
    # Apply LoRA
    lora_config = LoRAConfig(r=16, lora_alpha=32)
    lora_layer = LoRALinear(linear, rank=lora_config.r, alpha=lora_config.lora_alpha)
    
    # Test forward pass
    x = torch.randn(2, 10, 768)
    output = lora_layer(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    
    # Count parameters
    original_params = sum(p.numel() for p in linear.parameters())
    lora_params = sum(p.numel() for p in lora_layer.parameters() if p.requires_grad)
    print(f"\nOriginal parameters: {original_params:,}")
    print(f"LoRA parameters: {lora_params:,}")
    print(f"Reduction: {(1 - lora_params/original_params)*100:.1f}%")
    
    # Visualize LoRA weights
    print("\n--- LoRA Weight Visualization ---")
    visualize_lora_weights(lora_layer.lora.lora_A, lora_layer.lora.lora_B)
    
    # Demo different methods
    print("\n--- Method Comparison ---")
    
    class SimpleModel(nn.Module):
        def __init__(self, hidden_size=768, num_layers=12):
            super().__init__()
            self.embeddings = nn.Embedding(30000, hidden_size)
            self.layers = nn.ModuleList([
                nn.TransformerEncoderLayer(hidden_size, 8, hidden_size*4)
                for _ in range(num_layers)
            ])
            self.pooler = nn.Linear(hidden_size, hidden_size)
            self.classifier = nn.Linear(hidden_size, 2)
            
        def forward(self, input_ids):
            x = self.embeddings(input_ids)
            for layer in self.layers:
                x = layer(x)
            pooled = x.mean(dim=1)
            return self.classifier(self.pooler(pooled))
    
    # Test different methods
    methods_to_test = [
        ("Full Fine-tuning", FullFineTuning(FineTuningConfig())),
        ("LoRA", LoRAFineTuning(LoRAConfig())),
        ("BitFit", BitFit()),
        ("Adapters", AdapterFineTuning(adapter_size=64)),
        ("Prompt Tuning", PromptTuning(n_prompt_tokens=10))
    ]
    
    print("\nParameter efficiency comparison:")
    print("-" * 60)
    
    for method_name, method in methods_to_test:
        model = SimpleModel()
        total_params = sum(p.numel() for p in model.parameters())
        
        try:
            model = method.prepare_model(model)
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            
            print(f"{method_name:20} | "
                  f"Trainable: {trainable_params:>10,} | "
                  f"Percentage: {trainable_params/total_params*100:>6.2f}%")
        except Exception as e:
            print(f"{method_name:20} | Error: {str(e)}")
    
    print("\n✅ Fine-tuning methods demonstrated successfully!")