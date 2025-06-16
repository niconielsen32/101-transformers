# Tokenization and Embeddings

The first step in any NLP pipeline! This module covers how text becomes numbers that transformers can process, from basic tokenization to modern subword methods.

## 🎯 Learning Objectives

By the end of this module, you will understand:
- Different tokenization strategies and their trade-offs
- Byte-Pair Encoding (BPE) algorithm
- WordPiece and SentencePiece tokenization
- Embedding initialization and training
- Handling out-of-vocabulary (OOV) tokens
- Position and token type embeddings

## 📚 Table of Contents

1. [Why Tokenization Matters](#1-why-tokenization-matters)
2. [Character vs Word vs Subword](#2-character-vs-word-vs-subword)
3. [Byte-Pair Encoding (BPE)](#3-byte-pair-encoding-bpe)
4. [WordPiece Tokenization](#4-wordpiece-tokenization)
5. [SentencePiece](#5-sentencepiece)
6. [Embeddings Deep Dive](#6-embeddings-deep-dive)
7. [Special Tokens and Vocabulary](#7-special-tokens-and-vocabulary)
8. [Implementation Examples](#8-implementation-examples)
9. [Best Practices](#9-best-practices)

## 1. Why Tokenization Matters

### 1.1 The Challenge

Computers work with numbers, not text. We need to convert:
```
"Hello, world!" → [15496, 11, 995, 0]
```

### 1.2 Key Considerations

1. **Vocabulary size**: Memory and computational cost
2. **OOV handling**: What about unseen words?
3. **Morphology**: How to handle word variations?
4. **Multilingual**: Different scripts and languages
5. **Efficiency**: Fast encoding/decoding

### 1.3 Impact on Model Performance

- **Too fine-grained** (character-level): Long sequences, harder to learn
- **Too coarse-grained** (word-level): Large vocabulary, OOV issues
- **Just right** (subword-level): Balance between both

## 2. Character vs Word vs Subword

### 2.1 Character-Level Tokenization

```python
"hello" → ['h', 'e', 'l', 'l', 'o']
```

**Pros**:
- Small vocabulary (~100 for English)
- No OOV issues
- Works for any language

**Cons**:
- Very long sequences
- Harder to capture meaning
- More computation needed

### 2.2 Word-Level Tokenization

```python
"Hello world" → ['Hello', 'world']
```

**Pros**:
- Intuitive units
- Shorter sequences
- Direct word embeddings

**Cons**:
- Huge vocabulary (100k+ words)
- OOV problem severe
- Doesn't handle morphology

### 2.3 Subword Tokenization

```python
"unhappiness" → ['un', 'happiness']
"preprocessing" → ['pre', 'process', 'ing']
```

**Pros**:
- Moderate vocabulary (10k-50k)
- Handles OOV by decomposition
- Captures morphology
- Good for multilingual

**Cons**:
- Less intuitive
- Variable token lengths
- Needs training

## 3. Byte-Pair Encoding (BPE)

### 3.1 The Algorithm

BPE builds vocabulary bottom-up by merging frequent pairs:

1. Start with character vocabulary
2. Count all adjacent pairs
3. Merge most frequent pair
4. Repeat until vocabulary size reached

### 3.2 Training Example

```
Initial: ['l', 'o', 'w', 'e', 'r']
Step 1: Merge 'e','r' → 'er'
Result: ['l', 'o', 'w', 'er']
Step 2: Merge 'l','o' → 'lo'
Result: ['lo', 'w', 'er']
Step 3: Merge 'lo','w' → 'low'
Result: ['low', 'er']
```

### 3.3 BPE Algorithm Implementation

```python
def train_bpe(texts, vocab_size):
    # Start with character vocabulary
    vocab = list(set(''.join(texts)))
    
    # Tokenize texts
    tokenized = [[c for c in text] for text in texts]
    
    while len(vocab) < vocab_size:
        # Count pairs
        pairs = {}
        for tokens in tokenized:
            for i in range(len(tokens)-1):
                pair = (tokens[i], tokens[i+1])
                pairs[pair] = pairs.get(pair, 0) + 1
        
        # Find most frequent
        best_pair = max(pairs, key=pairs.get)
        
        # Merge in vocabulary
        new_token = ''.join(best_pair)
        vocab.append(new_token)
        
        # Update tokenization
        for tokens in tokenized:
            i = 0
            while i < len(tokens)-1:
                if tokens[i] == best_pair[0] and tokens[i+1] == best_pair[1]:
                    tokens[i] = new_token
                    del tokens[i+1]
                else:
                    i += 1
    
    return vocab
```

### 3.4 Encoding with BPE

```python
def encode_bpe(text, merges):
    tokens = list(text)
    
    for pair, merged in merges:
        i = 0
        while i < len(tokens)-1:
            if tokens[i] == pair[0] and tokens[i+1] == pair[1]:
                tokens[i] = merged
                del tokens[i+1]
            else:
                i += 1
    
    return tokens
```

## 4. WordPiece Tokenization

### 4.1 How It Differs from BPE

WordPiece (used in BERT) maximizes likelihood instead of frequency:
- Chooses merges that maximize language model likelihood
- Uses special prefix ## for subwords

### 4.2 Example

```
"playing" → ['play', '##ing']
"unaffable" → ['un', '##aff', '##able']
```

### 4.3 Algorithm

```python
def wordpiece_tokenize(text, vocab, max_input_chars_per_word=100):
    output_tokens = []
    
    for token in whitespace_tokenize(text):
        chars = list(token)
        if len(chars) > max_input_chars_per_word:
            output_tokens.append('[UNK]')
            continue
            
        is_bad = False
        start = 0
        sub_tokens = []
        
        while start < len(chars):
            end = len(chars)
            cur_substr = None
            
            while start < end:
                substr = "".join(chars[start:end])
                if start > 0:
                    substr = "##" + substr
                if substr in vocab:
                    cur_substr = substr
                    break
                end -= 1
                
            if cur_substr is None:
                is_bad = True
                break
                
            sub_tokens.append(cur_substr)
            start = end
            
        if is_bad:
            output_tokens.append('[UNK]')
        else:
            output_tokens.extend(sub_tokens)
            
    return output_tokens
```

## 5. SentencePiece

### 5.1 Key Features

- Language-agnostic (no whitespace assumption)
- Treats input as raw stream
- Can reverse tokenization perfectly
- Includes BPE and Unigram models

### 5.2 Training

```python
import sentencepiece as spm

# Train model
spm.SentencePieceTrainer.train(
    input='corpus.txt',
    model_prefix='m',
    vocab_size=32000,
    model_type='bpe',  # or 'unigram'
    character_coverage=1.0,
    pad_id=0,
    unk_id=1,
    bos_id=2,
    eos_id=3,
)

# Use model
sp = spm.SentencePieceProcessor(model_file='m.model')
tokens = sp.encode("Hello world", out_type=str)
ids = sp.encode("Hello world", out_type=int)
```

### 5.3 Special Features

- **Sampling**: Multiple segmentations for regularization
- **NBest**: Get top-N tokenizations
- **Direct vocabulary control**: Add user-defined tokens

## 6. Embeddings Deep Dive

### 6.1 Token Embeddings

Convert token IDs to dense vectors:

```python
class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model
        
    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model)
```

### 6.2 Initialization Strategies

**Random Initialization**:
```python
# Uniform
nn.init.uniform_(embeddings.weight, -0.1, 0.1)

# Normal
nn.init.normal_(embeddings.weight, mean=0, std=0.02)

# Xavier/Glorot
nn.init.xavier_uniform_(embeddings.weight)
```

**Pre-trained Initialization**:
```python
# From Word2Vec, GloVe, FastText
pretrained_weights = load_pretrained_embeddings()
embedding.weight.data.copy_(pretrained_weights)
```

### 6.3 Position Embeddings

**Sinusoidal** (fixed):
```python
def sinusoidal_embeddings(max_len, d_model):
    pe = torch.zeros(max_len, d_model)
    position = torch.arange(0, max_len).unsqueeze(1)
    
    div_term = torch.exp(torch.arange(0, d_model, 2) *
                         -(math.log(10000.0) / d_model))
    
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    
    return pe
```

**Learned** (trainable):
```python
self.pos_embedding = nn.Embedding(max_position_embeddings, hidden_size)
```

### 6.4 Token Type Embeddings

For tasks with multiple sequences (e.g., BERT):

```python
class TokenTypeEmbedding(nn.Module):
    def __init__(self, type_vocab_size, d_model):
        super().__init__()
        self.embedding = nn.Embedding(type_vocab_size, d_model)
        
    def forward(self, token_type_ids):
        return self.embedding(token_type_ids)
```

### 6.5 Embedding Layer Norm and Dropout

```python
class TransformerEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.token_embeddings = nn.Embedding(config.vocab_size, config.d_model)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.d_model)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.d_model)
        
        self.LayerNorm = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.dropout)
        
    def forward(self, input_ids, token_type_ids=None, position_ids=None):
        seq_length = input_ids.size(1)
        
        if position_ids is None:
            position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
            
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)
            
        token_embeddings = self.token_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        
        embeddings = token_embeddings + position_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        
        return embeddings
```

## 7. Special Tokens and Vocabulary

### 7.1 Common Special Tokens

```python
SPECIAL_TOKENS = {
    '[PAD]': 0,    # Padding
    '[UNK]': 1,    # Unknown
    '[CLS]': 2,    # Classification/Start
    '[SEP]': 3,    # Separator
    '[MASK]': 4,   # Masked token (BERT)
    '<s>': 5,      # Start (GPT)
    '</s>': 6,     # End (GPT)
}
```

### 7.2 Building Vocabulary

```python
class Vocabulary:
    def __init__(self, special_tokens=None):
        self.token_to_id = {}
        self.id_to_token = {}
        
        # Add special tokens
        if special_tokens:
            for token in special_tokens:
                self.add_token(token)
                
    def add_token(self, token):
        if token not in self.token_to_id:
            idx = len(self.token_to_id)
            self.token_to_id[token] = idx
            self.id_to_token[idx] = token
            
    def encode(self, tokens):
        return [self.token_to_id.get(t, self.token_to_id['[UNK]']) for t in tokens]
        
    def decode(self, ids):
        return [self.id_to_token.get(i, '[UNK]') for i in ids]
```

### 7.3 Handling Multiple Languages

```python
# Language-specific tokens
LANG_TOKENS = {
    'en': '[EN]',
    'fr': '[FR]',
    'de': '[DE]',
    'es': '[ES]',
}

# Shared vocabulary with language indicators
text = f"{LANG_TOKENS['fr']} Bonjour le monde"
```

## 8. Implementation Examples

### 8.1 Complete BPE Implementation

```python
class BPETokenizer:
    def __init__(self, vocab_size=10000):
        self.vocab_size = vocab_size
        self.word_tokenize = re.compile(r'\w+|[^\w\s]+')
        
    def train(self, texts):
        # Get word frequencies
        word_freqs = {}
        for text in texts:
            words = self.word_tokenize.findall(text.lower())
            for word in words:
                word_freqs[word] = word_freqs.get(word, 0) + 1
        
        # Initialize with characters
        self.vocab = set()
        for word, freq in word_freqs.items():
            for char in word:
                self.vocab.add(char)
        
        # Split words into characters
        splits = {word: list(word) + ['</w>'] for word in word_freqs}
        
        # BPE iterations
        self.merges = []
        while len(self.vocab) < self.vocab_size:
            # Count pairs
            pair_freqs = {}
            for word, freq in word_freqs.items():
                split = splits[word]
                for i in range(len(split) - 1):
                    pair = (split[i], split[i + 1])
                    pair_freqs[pair] = pair_freqs.get(pair, 0) + freq
            
            # Most frequent pair
            if not pair_freqs:
                break
            best_pair = max(pair_freqs, key=pair_freqs.get)
            
            # Merge
            self.merges.append(best_pair)
            self.vocab.add(''.join(best_pair))
            
            # Update splits
            for word in splits:
                split = splits[word]
                new_split = []
                i = 0
                while i < len(split):
                    if i < len(split) - 1 and (split[i], split[i + 1]) == best_pair:
                        new_split.append(''.join(best_pair))
                        i += 2
                    else:
                        new_split.append(split[i])
                        i += 1
                splits[word] = new_split
        
        # Create final vocabulary
        self.token_to_id = {token: i for i, token in enumerate(sorted(self.vocab))}
        self.id_to_token = {i: token for token, i in self.token_to_id.items()}
```

### 8.2 Fast Tokenization with Tries

```python
class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_end = False
        self.token_id = None

class FastTokenizer:
    def __init__(self, vocab):
        self.root = TrieNode()
        self.build_trie(vocab)
        
    def build_trie(self, vocab):
        for token, token_id in vocab.items():
            node = self.root
            for char in token:
                if char not in node.children:
                    node.children[char] = TrieNode()
                node = node.children[char]
            node.is_end = True
            node.token_id = token_id
            
    def tokenize(self, text):
        tokens = []
        i = 0
        
        while i < len(text):
            node = self.root
            j = i
            last_match = None
            
            # Find longest match
            while j < len(text) and text[j] in node.children:
                node = node.children[text[j]]
                j += 1
                if node.is_end:
                    last_match = (j, node.token_id)
            
            if last_match:
                tokens.append(last_match[1])
                i = last_match[0]
            else:
                # Unknown character
                tokens.append(self.unk_token_id)
                i += 1
                
        return tokens
```

## 9. Best Practices

### 9.1 Vocabulary Size Selection

| Model Type | Typical Vocab Size | Reasoning |
|------------|-------------------|-----------|
| Small models | 8K-16K | Faster, less memory |
| Base models | 30K-50K | Good balance |
| Large models | 50K-100K | Better coverage |
| Multilingual | 100K-250K | Many languages |

### 9.2 Preprocessing Decisions

```python
def preprocess_text(text, lowercase=True, normalize_unicode=True):
    if normalize_unicode:
        text = unicodedata.normalize('NFKC', text)
    
    if lowercase:
        text = text.lower()
    
    # Remove control characters
    text = ''.join(ch for ch in text if unicodedata.category(ch)[0] != 'C')
    
    return text
```

### 9.3 Handling Numbers

```python
# Option 1: Individual digits
"2023" → ['2', '0', '2', '3']

# Option 2: Special tokens
"2023" → ['[NUM]']

# Option 3: Bucketing
"2023" → ['[NUM_1000-9999]']
```

### 9.4 Casing Strategies

```python
# Cased (preserves information)
"Hello World" → ['Hello', 'World']

# Uncased (smaller vocab)
"hello world" → ['hello', 'world']

# TrueCasing (smart casing)
"HELLO WORLD" → ['hello', 'world']  # All caps → lowercase
"Hello World" → ['Hello', 'World']  # Normal casing preserved
```

## 📊 Tokenizer Comparison

| Method | Vocab Size | OOV Handling | Speed | Use Case |
|--------|------------|--------------|-------|----------|
| Character | ~100 | Perfect | Fast | Any text |
| Word | 50K-200K | Poor | Fast | Known domain |
| BPE | 10K-50K | Good | Medium | General NLP |
| WordPiece | 10K-50K | Good | Medium | BERT models |
| SentencePiece | 10K-50K | Good | Fast | Multilingual |

## 🔍 Debugging Tokenization

```python
def debug_tokenization(tokenizer, text):
    tokens = tokenizer.tokenize(text)
    ids = tokenizer.convert_tokens_to_ids(tokens)
    
    print(f"Original: {text}")
    print(f"Tokens: {tokens}")
    print(f"IDs: {ids}")
    print(f"Reconstructed: {tokenizer.decode(ids)}")
    
    # Check for issues
    if '[UNK]' in tokens:
        print("⚠️ Unknown tokens found!")
    
    if len(tokens) > 100:
        print("⚠️ Very long tokenization!")
```

## 📝 Summary

Tokenization is the critical first step that affects everything downstream:
- **Subword tokenization** balances vocabulary size and coverage
- **BPE** builds vocabulary through frequency-based merging
- **Embeddings** convert discrete tokens to continuous representations
- **Special tokens** handle task-specific requirements
- **Preprocessing** choices significantly impact performance

Good tokenization enables models to:
- Handle any input text
- Generalize to unseen words
- Work across languages
- Maintain reasonable sequence lengths

## ➡️ Next Steps

Ready to train transformers effectively? Head to [Topic 7: Training Transformers](../07-training-transformers/) to learn about optimization strategies, learning rate schedules, and distributed training!