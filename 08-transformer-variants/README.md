# Transformer Variants

Explore the diverse family of transformer architectures, from BERT to GPT to T5 and beyond. Each variant optimizes for different tasks and use cases.

## 🎯 Learning Objectives

By the end of this module, you will understand:
- BERT's bidirectional pre-training
- GPT's autoregressive generation
- T5's unified text-to-text framework
- Vision Transformers (ViT)
- Specialized architectures
- How to choose the right variant

## 📚 Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [BERT Family](#2-bert-family)
3. [GPT Family](#3-gpt-family)
4. [T5 and Seq2Seq](#4-t5-and-seq2seq)
5. [Vision Transformers](#5-vision-transformers)
6. [Efficient Variants](#6-efficient-variants)
7. [Specialized Architectures](#7-specialized-architectures)
8. [Comparison and Selection](#8-comparison-and-selection)

## 1. Architecture Overview

### 1.1 Three Main Paradigms

1. **Encoder-only (BERT)**: Bidirectional context, classification/understanding
2. **Decoder-only (GPT)**: Autoregressive generation, causal masking
3. **Encoder-Decoder (T5)**: Sequence-to-sequence tasks

### 1.2 Key Differences

| Feature | BERT | GPT | T5 |
|---------|------|-----|-----|
| Attention | Bidirectional | Causal | Encoder: Bidirectional, Decoder: Causal |
| Pre-training | MLM + NSP | Next token | Span corruption |
| Best for | Understanding | Generation | Any seq2seq |
| Parameters | Base: 110M | GPT-2: 1.5B | Base: 220M |

## 2. BERT Family

### 2.1 BERT (Bidirectional Encoder Representations from Transformers)

**Architecture**:
```python
class BERT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embeddings = BERTEmbeddings(config)
        self.encoder = nn.ModuleList([
            BERTLayer(config) for _ in range(config.num_hidden_layers)
        ])
        self.pooler = BERTPooler(config)
        
    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        # Bidirectional attention - can see all tokens
        embedding_output = self.embeddings(input_ids, token_type_ids)
        
        for layer in self.encoder:
            embedding_output = layer(embedding_output, attention_mask)
            
        pooled_output = self.pooler(embedding_output)
        return embedding_output, pooled_output
```

**Pre-training Tasks**:

1. **Masked Language Modeling (MLM)**:
```python
def create_mlm_data(tokens, mask_prob=0.15):
    labels = tokens.clone()
    
    # Create mask
    mask = torch.rand(tokens.shape) < mask_prob
    
    # 80% of time, replace with [MASK]
    indices = mask & (torch.rand(tokens.shape) < 0.8)
    tokens[indices] = tokenizer.mask_token_id
    
    # 10% of time, replace with random token
    indices = mask & (torch.rand(tokens.shape) < 0.9) & ~indices
    tokens[indices] = torch.randint(len(tokenizer), tokens.shape)[indices]
    
    # 10% of time, keep original
    
    return tokens, labels
```

2. **Next Sentence Prediction (NSP)**:
```python
def create_nsp_data(sentence_a, sentence_b, is_next=True):
    if not is_next:
        # Get random sentence
        sentence_b = random.choice(all_sentences)
    
    tokens = ['[CLS]'] + sentence_a + ['[SEP]'] + sentence_b + ['[SEP]']
    segment_ids = [0] * (len(sentence_a) + 2) + [1] * (len(sentence_b) + 1)
    
    return tokens, segment_ids, is_next
```

### 2.2 RoBERTa (Robustly Optimized BERT)

**Key improvements**:
- No NSP task (just MLM)
- Dynamic masking
- Larger batches
- More data and longer training

```python
class RoBERTa(BERT):
    def __init__(self, config):
        super().__init__(config)
        # Remove NSP head
        self.cls = RobertaLMHead(config)
        
    def forward(self, input_ids, attention_mask=None):
        # No token_type_ids
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        prediction_scores = self.cls(outputs[0])
        return prediction_scores
```

### 2.3 ALBERT (A Lite BERT)

**Efficiency techniques**:
1. **Factorized embedding parameterization**:
```python
class ALBERTEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.embedding_size)
        self.word_embeddings_projection = nn.Linear(config.embedding_size, config.hidden_size)
```

2. **Cross-layer parameter sharing**:
```python
class ALBERT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embeddings = ALBERTEmbeddings(config)
        # Single layer shared across all positions
        self.encoder_layer = ALBERTLayer(config)
        
    def forward(self, input_ids):
        hidden_states = self.embeddings(input_ids)
        
        for _ in range(self.config.num_hidden_layers):
            hidden_states = self.encoder_layer(hidden_states)
            
        return hidden_states
```

### 2.4 ELECTRA (Efficiently Learning an Encoder)

**Generator-Discriminator approach**:
```python
class ELECTRA(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.generator = BERTForMaskedLM(config.generator_config)
        self.discriminator = BERTForTokenClassification(config)
        
    def forward(self, input_ids):
        # Generator creates corrupted tokens
        masked_tokens, labels = mask_tokens(input_ids)
        generator_output = self.generator(masked_tokens)
        
        # Replace some tokens with generator predictions
        corrupted_tokens = replace_tokens(input_ids, generator_output)
        
        # Discriminator predicts which tokens are replaced
        discriminator_output = self.discriminator(corrupted_tokens)
        
        return discriminator_output
```

## 3. GPT Family

### 3.1 GPT (Generative Pre-trained Transformer)

**Architecture**:
```python
class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.token_embeddings = nn.Embedding(config.vocab_size, config.n_embd)
        self.position_embeddings = nn.Embedding(config.n_positions, config.n_embd)
        self.drop = nn.Dropout(config.dropout)
        
        self.blocks = nn.ModuleList([
            GPTBlock(config) for _ in range(config.n_layer)
        ])
        
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        
    def forward(self, input_ids, past_key_values=None):
        # Get embeddings
        token_embeds = self.token_embeddings(input_ids)
        position_ids = torch.arange(0, input_ids.size(-1))
        position_embeds = self.position_embeddings(position_ids)
        
        hidden_states = self.drop(token_embeds + position_embeds)
        
        # Apply transformer blocks with causal mask
        presents = []
        for block, past in zip(self.blocks, past_key_values or []):
            hidden_states, present = block(hidden_states, past=past)
            presents.append(present)
            
        hidden_states = self.ln_f(hidden_states)
        logits = self.head(hidden_states)
        
        return logits, presents
```

### 3.2 GPT-2 Improvements

- Layer normalization moved to input of each sub-block
- Additional layer normalization after final self-attention
- Modified initialization
- Larger scale (1.5B parameters)

### 3.3 GPT-3 and Beyond

**Scaling laws**:
```python
# Model configurations
GPT3_CONFIGS = {
    'small': {'n_layer': 12, 'n_head': 12, 'n_embd': 768},    # 125M
    'medium': {'n_layer': 24, 'n_head': 16, 'n_embd': 1024},  # 350M
    'large': {'n_layer': 24, 'n_head': 16, 'n_embd': 1536},   # 760M
    'xl': {'n_layer': 24, 'n_head': 24, 'n_embd': 2048},      # 1.3B
    '175B': {'n_layer': 96, 'n_head': 96, 'n_embd': 12288},   # 175B
}
```

**In-context learning**:
```python
def few_shot_prompt(examples, query):
    prompt = ""
    for example in examples:
        prompt += f"Input: {example['input']}\nOutput: {example['output']}\n\n"
    prompt += f"Input: {query}\nOutput:"
    return prompt
```

## 4. T5 and Seq2Seq

### 4.1 T5 (Text-to-Text Transfer Transformer)

**Unified framework**:
```python
class T5(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.encoder = T5Stack(config, is_decoder=False)
        self.decoder = T5Stack(config, is_decoder=True)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size)
        
    def forward(self, input_ids, decoder_input_ids):
        # Encode
        encoder_outputs = self.encoder(input_ids)
        
        # Decode with cross-attention to encoder
        decoder_outputs = self.decoder(
            decoder_input_ids,
            encoder_hidden_states=encoder_outputs
        )
        
        logits = self.lm_head(decoder_outputs)
        return logits
```

**Text-to-text examples**:
```python
# All tasks use same format
tasks = {
    'translation': "translate English to French: The house is wonderful.",
    'summarization': "summarize: <article text>",
    'classification': "sentiment: This movie is terrible.",
    'question_answering': "question: What is the capital? context: <text>"
}
```

### 4.2 BART (Bidirectional and Auto-Regressive Transformers)

**Denoising autoencoder**:
```python
def bart_noise_function(tokens):
    # Multiple noise types
    tokens = mask_tokens(tokens)          # Token masking
    tokens = delete_tokens(tokens)        # Token deletion  
    tokens = permute_sentences(tokens)    # Sentence permutation
    tokens = rotate_document(tokens)      # Document rotation
    tokens = infill_text(tokens)          # Text infilling
    return tokens
```

### 4.3 mT5 (Multilingual T5)

- Trained on 101 languages
- Same architecture as T5
- Uses SentencePiece tokenizer

## 5. Vision Transformers

### 5.1 ViT (Vision Transformer)

**Image patching**:
```python
class VisionTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, num_classes=1000):
        super().__init__()
        num_patches = (img_size // patch_size) ** 2
        
        self.patch_embed = nn.Conv2d(3, dim, kernel_size=patch_size, stride=patch_size)
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        
        self.transformer = Transformer(dim, depth, heads)
        self.head = nn.Linear(dim, num_classes)
        
    def forward(self, img):
        # Create patches
        x = self.patch_embed(img)  # (B, dim, H/P, W/P)
        x = x.flatten(2).transpose(1, 2)  # (B, num_patches, dim)
        
        # Add cls token
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        
        # Add positional embedding
        x = x + self.pos_embed
        
        # Apply transformer
        x = self.transformer(x)
        
        # Use cls token for classification
        x = x[:, 0]
        x = self.head(x)
        
        return x
```

### 5.2 DeiT (Data-efficient Image Transformers)

**Distillation token**:
```python
class DeiT(VisionTransformer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dist_token = nn.Parameter(torch.zeros(1, 1, self.dim))
        
    def forward(self, img):
        # Add both cls and distillation tokens
        x = self.patch_embed(img).flatten(2).transpose(1, 2)
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        dist_tokens = self.dist_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, dist_tokens, x), dim=1)
```

### 5.3 Swin Transformer

**Hierarchical architecture with shifted windows**:
```python
class SwinTransformerBlock(nn.Module):
    def __init__(self, dim, window_size=7, shift_size=0):
        super().__init__()
        self.window_size = window_size
        self.shift_size = shift_size
        self.attention = WindowAttention(dim, window_size)
        
    def forward(self, x):
        # Shift window
        if self.shift_size > 0:
            x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            
        # Partition into windows
        x_windows = window_partition(x, self.window_size)
        
        # Apply window attention
        attn_windows = self.attention(x_windows)
        
        # Merge windows
        x = window_reverse(attn_windows, self.window_size)
        
        # Reverse shift
        if self.shift_size > 0:
            x = torch.roll(x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
            
        return x
```

## 6. Efficient Variants

### 6.1 Linformer

**Linear complexity attention**:
```python
class LinformerAttention(nn.Module):
    def __init__(self, dim, seq_len, k=256):
        super().__init__()
        self.E = nn.Linear(seq_len, k)
        self.F = nn.Linear(seq_len, k)
        
    def forward(self, Q, K, V):
        # Project K and V to lower dimension
        K = self.E(K.transpose(-1, -2)).transpose(-1, -2)
        V = self.F(V.transpose(-1, -2)).transpose(-1, -2)
        
        # Now attention is O(nk) instead of O(n²)
        attention = torch.softmax(Q @ K.transpose(-1, -2) / sqrt(d_k), dim=-1)
        return attention @ V
```

### 6.2 Performer

**FAVOR+ attention**:
```python
def favor_attention(Q, K, V):
    # Random features approximation
    Q_prime = torch.exp(Q @ random_matrix - Q.norm(dim=-1, keepdim=True)**2 / 2)
    K_prime = torch.exp(K @ random_matrix - K.norm(dim=-1, keepdim=True)**2 / 2)
    
    # Linear attention
    KV = K_prime.transpose(-1, -2) @ V
    Z = K_prime.sum(dim=-2, keepdim=True)
    
    return Q_prime @ KV / (Q_prime @ Z)
```

### 6.3 Longformer

**Sliding window + global attention**:
```python
class LongformerAttention(nn.Module):
    def __init__(self, window_size=512):
        super().__init__()
        self.window_size = window_size
        
    def forward(self, hidden_states, is_global=None):
        # Local sliding window attention
        local_attn = sliding_window_attention(hidden_states, self.window_size)
        
        # Global attention for specific tokens
        if is_global is not None:
            global_attn = full_attention(hidden_states, is_global)
            return local_attn + global_attn
            
        return local_attn
```

## 7. Specialized Architectures

### 7.1 Retrieval-Augmented Models

**REALM**:
```python
class REALM(nn.Module):
    def __init__(self, encoder, retriever):
        super().__init__()
        self.encoder = encoder
        self.retriever = retriever
        
    def forward(self, query):
        # Retrieve relevant documents
        docs = self.retriever(query)
        
        # Encode query with each document
        scores = []
        for doc in docs:
            input_text = f"{query} [SEP] {doc}"
            score = self.encoder(input_text)
            scores.append(score)
            
        return scores
```

### 7.2 Multimodal Transformers

**CLIP**:
```python
class CLIP(nn.Module):
    def __init__(self):
        super().__init__()
        self.visual_encoder = VisionTransformer()
        self.text_encoder = TransformerEncoder()
        self.temperature = nn.Parameter(torch.ones(1))
        
    def forward(self, images, texts):
        # Encode both modalities
        image_features = self.visual_encoder(images)
        text_features = self.text_encoder(texts)
        
        # Normalize
        image_features = F.normalize(image_features, dim=-1)
        text_features = F.normalize(text_features, dim=-1)
        
        # Compute similarity
        logits = image_features @ text_features.T * self.temperature
        
        return logits
```

### 7.3 Sparse Transformers

**BigBird**:
```python
def bigbird_attention(Q, K, V, block_size=64):
    # Random attention
    random_attn = random_attention(Q, K, V, num_random_blocks=3)
    
    # Window attention  
    window_attn = sliding_window_attention(Q, K, V, window_size=3*block_size)
    
    # Global attention
    global_attn = global_tokens_attention(Q, K, V, num_global_tokens=2*block_size)
    
    return random_attn + window_attn + global_attn
```

## 8. Comparison and Selection

### 8.1 Task-Specific Recommendations

| Task | Recommended Model | Why |
|------|------------------|-----|
| Text Classification | BERT/RoBERTa | Bidirectional context |
| Text Generation | GPT-2/3 | Autoregressive |
| Translation | T5/mT5 | Seq2seq design |
| Summarization | BART/Pegasus | Denoising objective |
| Question Answering | BERT/ELECTRA | Bidirectional |
| Image Classification | ViT/Swin | Vision-specific |
| Multimodal | CLIP/ALIGN | Joint embedding |

### 8.2 Efficiency Considerations

| Model | Memory | Speed | Quality |
|-------|--------|-------|---------|
| BERT-Base | Medium | Fast | Good |
| GPT-3 | Very High | Slow | Excellent |
| ALBERT | Low | Fast | Good |
| Linformer | Low | Very Fast | Decent |
| T5-Base | High | Medium | Very Good |

### 8.3 Implementation Tips

```python
# Dynamic model selection
def select_model(task, constraints):
    if constraints['memory'] < 4:  # GB
        return 'DistilBERT'
    elif task == 'generation':
        return 'GPT2' if constraints['quality'] > 0.8 else 'GPT2-small'
    elif task == 'classification':
        return 'RoBERTa' if constraints['accuracy'] > 0.9 else 'BERT'
    else:
        return 'T5-base'
```

## 📊 Performance Benchmarks

| Model | GLUE Score | Parameters | Training Cost |
|-------|------------|------------|---------------|
| BERT-Base | 79.6 | 110M | $7K |
| RoBERTa-Large | 88.5 | 355M | $160K |
| ELECTRA-Large | 89.0 | 335M | $50K |
| T5-11B | 90.3 | 11B | $1.3M |
| GPT-3 | - | 175B | $4.6M |

## 🔍 Future Directions

1. **Mixture of Experts**: Conditional computation
2. **Retrieval Integration**: RAG, RETRO
3. **Multimodal Fusion**: Text + Vision + Audio
4. **Efficient Architectures**: Sub-quadratic attention
5. **Adaptive Computation**: Early exit, conditional layers

## 📝 Summary

The transformer family includes:
- **BERT**: Bidirectional understanding
- **GPT**: Autoregressive generation  
- **T5**: Unified text-to-text
- **ViT**: Vision applications
- **Efficient variants**: Linear complexity
- **Specialized models**: Task-specific optimizations

Choose based on:
- Task requirements
- Computational constraints
- Quality needs
- Deployment environment

## ➡️ Next Steps

Ready to explore cutting-edge LLM architectures? Head to [Topic 9: Modern LLM Architectures](../09-modern-llm-architectures/) to learn about GPT-4, LLaMA, and beyond!