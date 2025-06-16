"""
Tokenization and Embeddings
Implementation of various tokenization methods and embedding strategies.
"""

import torch
import torch.nn as nn
import numpy as np
import math
from collections import Counter, defaultdict
import re
import unicodedata
from typing import List, Dict, Tuple, Optional
import json
import heapq
from tqdm import tqdm


class CharacterTokenizer:
    """Simple character-level tokenizer."""
    
    def __init__(self, vocab: Optional[List[str]] = None):
        if vocab is None:
            # Basic ASCII printable characters
            vocab = list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 !\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~")
        
        self.vocab = ['[PAD]', '[UNK]', '[CLS]', '[SEP]'] + vocab
        self.token_to_id = {token: i for i, token in enumerate(self.vocab)}
        self.id_to_token = {i: token for i, token in enumerate(self.vocab)}
        
    def tokenize(self, text: str) -> List[str]:
        return list(text)
    
    def encode(self, text: str) -> List[int]:
        tokens = self.tokenize(text)
        return [self.token_to_id.get(token, self.token_to_id['[UNK]']) for token in tokens]
    
    def decode(self, ids: List[int]) -> str:
        tokens = [self.id_to_token.get(id, '[UNK]') for id in ids]
        return ''.join(tokens).replace('[PAD]', '').replace('[CLS]', '').replace('[SEP]', '')


class WordTokenizer:
    """Word-level tokenizer with basic preprocessing."""
    
    def __init__(self, lowercase: bool = True, max_vocab_size: int = 50000):
        self.lowercase = lowercase
        self.max_vocab_size = max_vocab_size
        self.word_pattern = re.compile(r'\w+|[^\w\s]+')
        self.vocab = ['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]']
        self.token_to_id = {}
        self.id_to_token = {}
        
    def train(self, texts: List[str]):
        """Build vocabulary from texts."""
        word_freq = Counter()
        
        for text in texts:
            if self.lowercase:
                text = text.lower()
            words = self.word_pattern.findall(text)
            word_freq.update(words)
        
        # Keep most frequent words
        most_common = word_freq.most_common(self.max_vocab_size - len(self.vocab))
        self.vocab.extend([word for word, _ in most_common])
        
        # Build mappings
        self.token_to_id = {token: i for i, token in enumerate(self.vocab)}
        self.id_to_token = {i: token for i, token in enumerate(self.vocab)}
    
    def tokenize(self, text: str) -> List[str]:
        if self.lowercase:
            text = text.lower()
        return self.word_pattern.findall(text)
    
    def encode(self, text: str) -> List[int]:
        tokens = self.tokenize(text)
        return [self.token_to_id.get(token, self.token_to_id['[UNK]']) for token in tokens]
    
    def decode(self, ids: List[int]) -> str:
        tokens = [self.id_to_token.get(id, '[UNK]') for id in ids]
        return ' '.join(tokens)


class BPETokenizer:
    """Byte-Pair Encoding tokenizer implementation."""
    
    def __init__(self, vocab_size: int = 10000, min_frequency: int = 2):
        self.vocab_size = vocab_size
        self.min_frequency = min_frequency
        self.word_pattern = re.compile(r'\w+|[^\w\s]+')
        
        # Special tokens
        self.special_tokens = ['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]']
        self.vocab = set(self.special_tokens)
        self.merges = []
        self.token_to_id = {}
        self.id_to_token = {}
        
    def _get_word_frequencies(self, texts: List[str]) -> Dict[str, int]:
        """Count word frequencies in texts."""
        word_freq = Counter()
        for text in texts:
            words = self.word_pattern.findall(text.lower())
            word_freq.update(words)
        return dict(word_freq)
    
    def _get_pair_frequencies(self, splits: Dict[str, List[str]], 
                             word_freq: Dict[str, int]) -> Dict[Tuple[str, str], int]:
        """Count frequencies of adjacent pairs."""
        pair_freq = defaultdict(int)
        
        for word, freq in word_freq.items():
            split = splits[word]
            for i in range(len(split) - 1):
                pair = (split[i], split[i + 1])
                pair_freq[pair] += freq
                
        return dict(pair_freq)
    
    def train(self, texts: List[str]):
        """Train BPE on texts."""
        print("Training BPE tokenizer...")
        
        # Get word frequencies
        word_freq = self._get_word_frequencies(texts)
        
        # Initialize vocabulary with characters
        for word in word_freq:
            for char in word:
                self.vocab.add(char)
        self.vocab.add('</w>')  # End of word marker
        
        # Initialize splits (each word split into characters)
        splits = {word: list(word) + ['</w>'] for word in word_freq}
        
        # BPE iterations
        pbar = tqdm(total=self.vocab_size - len(self.vocab))
        
        while len(self.vocab) < self.vocab_size:
            # Get pair frequencies
            pair_freq = self._get_pair_frequencies(splits, word_freq)
            
            if not pair_freq:
                break
                
            # Find most frequent pair
            best_pair = max(pair_freq, key=pair_freq.get)
            
            if pair_freq[best_pair] < self.min_frequency:
                break
            
            # Add merge to vocabulary
            merged = ''.join(best_pair)
            self.vocab.add(merged)
            self.merges.append(best_pair)
            
            # Update splits
            for word in splits:
                split = splits[word]
                new_split = []
                i = 0
                
                while i < len(split):
                    if i < len(split) - 1 and (split[i], split[i + 1]) == best_pair:
                        new_split.append(merged)
                        i += 2
                    else:
                        new_split.append(split[i])
                        i += 1
                        
                splits[word] = new_split
            
            pbar.update(1)
        
        pbar.close()
        
        # Create token mappings
        sorted_vocab = sorted(list(self.vocab))
        self.token_to_id = {token: i for i, token in enumerate(sorted_vocab)}
        self.id_to_token = {i: token for i, token in enumerate(sorted_vocab)}
        
        print(f"Vocabulary size: {len(self.vocab)}")
        print(f"Number of merges: {len(self.merges)}")
    
    def tokenize(self, text: str) -> List[str]:
        """Tokenize text using learned BPE merges."""
        words = self.word_pattern.findall(text.lower())
        tokens = []
        
        for word in words:
            # Split word into characters
            word_tokens = list(word) + ['</w>']
            
            # Apply merges
            for merge in self.merges:
                i = 0
                while i < len(word_tokens) - 1:
                    if word_tokens[i] == merge[0] and word_tokens[i + 1] == merge[1]:
                        word_tokens[i] = ''.join(merge)
                        del word_tokens[i + 1]
                    else:
                        i += 1
            
            tokens.extend(word_tokens)
        
        return tokens
    
    def encode(self, text: str) -> List[int]:
        tokens = self.tokenize(text)
        return [self.token_to_id.get(token, self.token_to_id['[UNK]']) for token in tokens]
    
    def decode(self, ids: List[int]) -> str:
        tokens = [self.id_to_token.get(id, '[UNK]') for id in ids]
        text = ''.join(tokens)
        text = text.replace('</w>', ' ')
        return text.strip()
    
    def save(self, filepath: str):
        """Save tokenizer to file."""
        data = {
            'vocab_size': self.vocab_size,
            'min_frequency': self.min_frequency,
            'vocab': list(self.vocab),
            'merges': self.merges,
            'special_tokens': self.special_tokens
        }
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    @classmethod
    def load(cls, filepath: str):
        """Load tokenizer from file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        tokenizer = cls(vocab_size=data['vocab_size'], 
                       min_frequency=data['min_frequency'])
        tokenizer.vocab = set(data['vocab'])
        tokenizer.merges = [tuple(merge) for merge in data['merges']]
        tokenizer.special_tokens = data['special_tokens']
        
        # Rebuild mappings
        sorted_vocab = sorted(list(tokenizer.vocab))
        tokenizer.token_to_id = {token: i for i, token in enumerate(sorted_vocab)}
        tokenizer.id_to_token = {i: token for i, token in enumerate(sorted_vocab)}
        
        return tokenizer


class WordPieceTokenizer:
    """WordPiece tokenizer (used in BERT)."""
    
    def __init__(self, vocab_size: int = 30000, unk_token: str = '[UNK]'):
        self.vocab_size = vocab_size
        self.unk_token = unk_token
        self.vocab = [unk_token, '[PAD]', '[CLS]', '[SEP]', '[MASK]']
        self.token_to_id = {}
        self.id_to_token = {}
        
    def train(self, texts: List[str], max_word_length: int = 100):
        """Train WordPiece tokenizer."""
        # First, get character vocabulary
        char_counts = Counter()
        word_counts = Counter()
        
        for text in texts:
            words = text.lower().split()
            word_counts.update(words)
            for word in words:
                char_counts.update(word)
        
        # Add all characters to vocabulary
        for char in char_counts:
            if char not in self.vocab:
                self.vocab.append(char)
        
        # Initialize word splits
        word_splits = {}
        for word, count in word_counts.items():
            if len(word) <= max_word_length:
                # First character doesn't get ##
                word_splits[word] = [word[0]] + ['##' + c for c in word[1:]]
        
        # Iteratively build vocabulary
        while len(self.vocab) < self.vocab_size:
            # Score all possible subwords
            subword_scores = defaultdict(float)
            
            for word, split in word_splits.items():
                word_freq = word_counts[word]
                
                # Current likelihood
                current_score = sum(1.0 for token in split if token in self.vocab)
                
                # Try all possible merges
                for i in range(len(split) - 1):
                    # Create merged subword
                    if i == 0:
                        merged = split[i] + split[i + 1].replace('##', '')
                    else:
                        merged = split[i] + split[i + 1].replace('##', '')
                    
                    # New split after merge
                    new_split = split[:i] + [merged] + split[i + 2:]
                    
                    # New likelihood
                    new_score = sum(1.0 for token in new_split if token in self.vocab or token == merged)
                    
                    # Score improvement
                    score_delta = (new_score - current_score) * word_freq
                    subword_scores[merged] += score_delta
            
            if not subword_scores:
                break
            
            # Add best subword to vocabulary
            best_subword = max(subword_scores, key=subword_scores.get)
            self.vocab.append(best_subword)
            
            # Update word splits
            for word, split in word_splits.items():
                new_split = []
                i = 0
                
                while i < len(split):
                    if i < len(split) - 1:
                        if i == 0:
                            candidate = split[i] + split[i + 1].replace('##', '')
                        else:
                            candidate = split[i] + split[i + 1].replace('##', '')
                        
                        if candidate == best_subword:
                            new_split.append(best_subword)
                            i += 2
                            continue
                    
                    new_split.append(split[i])
                    i += 1
                
                word_splits[word] = new_split
        
        # Create mappings
        self.token_to_id = {token: i for i, token in enumerate(self.vocab)}
        self.id_to_token = {i: token for i, token in enumerate(self.vocab)}
    
    def tokenize(self, text: str, max_word_length: int = 100) -> List[str]:
        """Tokenize text using WordPiece."""
        words = text.lower().split()
        tokens = []
        
        for word in words:
            if len(word) > max_word_length:
                tokens.append(self.unk_token)
                continue
            
            # Try to tokenize word
            word_tokens = []
            start = 0
            
            while start < len(word):
                end = len(word)
                cur_substr = None
                
                while start < end:
                    substr = word[start:end]
                    if start > 0:
                        substr = '##' + substr
                    
                    if substr in self.token_to_id:
                        cur_substr = substr
                        break
                    
                    end -= 1
                
                if cur_substr is None:
                    word_tokens = [self.unk_token]
                    break
                
                word_tokens.append(cur_substr)
                start = end
            
            tokens.extend(word_tokens)
        
        return tokens
    
    def encode(self, text: str) -> List[int]:
        tokens = self.tokenize(text)
        return [self.token_to_id.get(token, self.token_to_id[self.unk_token]) for token in tokens]
    
    def decode(self, ids: List[int]) -> str:
        tokens = [self.id_to_token.get(id, self.unk_token) for id in ids]
        
        # Reconstruct text
        text = ''
        for token in tokens:
            if token.startswith('##'):
                text += token[2:]
            else:
                if text:
                    text += ' '
                text += token
        
        return text


# Embedding implementations
class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""
    
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        
        # Create div_term
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           -(math.log(10000.0) / d_model))
        
        # Apply sin/cos
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Add batch dimension
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Add positional encoding
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class LearnedPositionalEmbedding(nn.Module):
    """Learned positional embeddings."""
    
    def __init__(self, max_position_embeddings: int, hidden_size: int, dropout: float = 0.1):
        super().__init__()
        self.embedding = nn.Embedding(max_position_embeddings, hidden_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, position_ids: Optional[torch.Tensor] = None, 
                seq_length: Optional[int] = None) -> torch.Tensor:
        if position_ids is None:
            position_ids = torch.arange(seq_length, dtype=torch.long)
            
        embeddings = self.embedding(position_ids)
        return self.dropout(embeddings)


class TransformerEmbeddings(nn.Module):
    """Complete embedding layer for transformers."""
    
    def __init__(self, vocab_size: int, hidden_size: int, max_position_embeddings: int,
                 type_vocab_size: int = 2, layer_norm_eps: float = 1e-12,
                 dropout: float = 0.1, use_position_embeddings: bool = True):
        super().__init__()
        
        self.token_embeddings = nn.Embedding(vocab_size, hidden_size)
        self.use_position_embeddings = use_position_embeddings
        
        if use_position_embeddings:
            self.position_embeddings = nn.Embedding(max_position_embeddings, hidden_size)
            
        self.token_type_embeddings = nn.Embedding(type_vocab_size, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.dropout = nn.Dropout(dropout)
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize embedding weights."""
        # Token embeddings
        nn.init.normal_(self.token_embeddings.weight, mean=0.0, std=0.02)
        
        # Position embeddings
        if self.use_position_embeddings:
            nn.init.normal_(self.position_embeddings.weight, mean=0.0, std=0.02)
        
        # Token type embeddings
        nn.init.normal_(self.token_type_embeddings.weight, mean=0.0, std=0.02)
        
    def forward(self, input_ids: torch.Tensor, 
                token_type_ids: Optional[torch.Tensor] = None,
                position_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        
        seq_length = input_ids.size(1)
        
        # Get token embeddings
        token_embeds = self.token_embeddings(input_ids)
        
        # Get position embeddings
        if self.use_position_embeddings:
            if position_ids is None:
                position_ids = torch.arange(seq_length, dtype=torch.long, 
                                          device=input_ids.device)
                position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
            position_embeds = self.position_embeddings(position_ids)
        else:
            position_embeds = 0
        
        # Get token type embeddings
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)
        token_type_embeds = self.token_type_embeddings(token_type_ids)
        
        # Combine embeddings
        embeddings = token_embeds + position_embeds + token_type_embeds
        
        # Layer norm and dropout
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        
        return embeddings


# Utility functions
def visualize_tokenization(tokenizer, text: str):
    """Visualize how text is tokenized."""
    tokens = tokenizer.tokenize(text)
    ids = tokenizer.encode(text)
    decoded = tokenizer.decode(ids)
    
    print(f"Original text: {text}")
    print(f"Tokens ({len(tokens)}): {tokens}")
    print(f"Token IDs: {ids}")
    print(f"Decoded: {decoded}")
    
    # Character count
    char_count = len(text)
    token_count = len(tokens)
    print(f"\nCompression ratio: {char_count / token_count:.2f} chars/token")


def compare_tokenizers(text: str):
    """Compare different tokenization methods."""
    # Character tokenizer
    char_tokenizer = CharacterTokenizer()
    char_tokens = char_tokenizer.tokenize(text)
    
    # Word tokenizer
    word_tokenizer = WordTokenizer()
    word_tokenizer.train([text])
    word_tokens = word_tokenizer.tokenize(text)
    
    # Print comparison
    print(f"Original text ({len(text)} chars): {text}")
    print(f"\nCharacter tokens ({len(char_tokens)}): {char_tokens}")
    print(f"\nWord tokens ({len(word_tokens)}): {word_tokens}")
    
    # Note: BPE requires training on larger corpus
    print("\n(BPE and WordPiece require training on larger corpus)")


def demonstrate_embeddings():
    """Demonstrate different embedding types."""
    import matplotlib.pyplot as plt
    
    # Parameters
    d_model = 128
    max_len = 100
    
    # Sinusoidal positional encoding
    pe = torch.zeros(max_len, d_model)
    position = torch.arange(0, max_len).unsqueeze(1).float()
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                       -(math.log(10000.0) / d_model))
    
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    
    # Visualize
    plt.figure(figsize=(12, 8))
    
    # Heatmap of positional encodings
    plt.subplot(2, 2, 1)
    plt.imshow(pe[:50, :].numpy(), aspect='auto', cmap='RdBu')
    plt.colorbar()
    plt.xlabel('Dimension')
    plt.ylabel('Position')
    plt.title('Sinusoidal Positional Encoding')
    
    # Specific dimensions over positions
    plt.subplot(2, 2, 2)
    for dim in [0, 1, 10, 20]:
        plt.plot(pe[:, dim].numpy(), label=f'dim {dim}')
    plt.xlabel('Position')
    plt.ylabel('Encoding Value')
    plt.title('Positional Encoding by Dimension')
    plt.legend()
    
    # Show periodicity
    plt.subplot(2, 2, 3)
    dim = 10
    plt.plot(pe[:, dim].numpy())
    plt.xlabel('Position')
    plt.ylabel(f'Dimension {dim}')
    plt.title('Periodicity in Positional Encoding')
    plt.grid(True, alpha=0.3)
    
    # Dot product between positions (similarity)
    plt.subplot(2, 2, 4)
    similarity = torch.matmul(pe[:20, :], pe[:20, :].T)
    plt.imshow(similarity, cmap='RdBu')
    plt.colorbar()
    plt.xlabel('Position')
    plt.ylabel('Position')
    plt.title('Position Similarity (Dot Product)')
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    print("=== Tokenization and Embeddings Demo ===\n")
    
    # Test different tokenizers
    test_text = "Hello, world! Tokenization is fascinating."
    print("Comparing tokenizers:")
    compare_tokenizers(test_text)
    
    # Train and test BPE
    print("\n\n=== BPE Tokenizer Demo ===")
    corpus = [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is fascinating and powerful.",
        "Transformers revolutionized natural language processing.",
        "Tokenization is the first step in NLP pipelines.",
    ] * 10  # Repeat for more data
    
    bpe = BPETokenizer(vocab_size=100)
    bpe.train(corpus)
    
    test_texts = [
        "The quick brown fox",
        "Machine learning rocks!",
        "Unknown words like zoboomafoo",
    ]
    
    for text in test_texts:
        print(f"\nText: {text}")
        visualize_tokenization(bpe, text)
    
    # Demonstrate embeddings
    print("\n\n=== Embeddings Demo ===")
    demonstrate_embeddings()
    
    # Show embedding layer
    print("\n=== Transformer Embeddings Layer ===")
    embed_layer = TransformerEmbeddings(
        vocab_size=10000,
        hidden_size=768,
        max_position_embeddings=512,
        type_vocab_size=2
    )
    
    # Example input
    batch_size = 2
    seq_length = 10
    input_ids = torch.randint(0, 10000, (batch_size, seq_length))
    
    embeddings = embed_layer(input_ids)
    print(f"Input shape: {input_ids.shape}")
    print(f"Output shape: {embeddings.shape}")
    print(f"Embedding dimensionality: {embeddings.shape[-1]}")