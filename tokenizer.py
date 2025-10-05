# tokenizer.py
"""
Optimized Byte Pair Encoding (BPE) Tokenizer implementation.
BPE iteratively merges the most frequent pairs of tokens to build a vocabulary.
This version includes optimized encoding for faster processing.
"""

import re
from collections import Counter, defaultdict


class BPETokenizer:
    """
    Byte Pair Encoding tokenizer that learns subword units from text.
    
    How BPE works:
    1. Start with individual characters as base vocabulary
    2. Find the most frequent pair of tokens
    3. Merge this pair into a single token
    4. Repeat until desired vocabulary size is reached
    """
    
    def __init__(self, vocab_size=500, min_frequency=2):
        """
        Initialize BPE tokenizer.
        
        Args:
            vocab_size: Target size of vocabulary (number of tokens)
            min_frequency: Minimum frequency for a pair to be merged
        """
        self.vocab_size = vocab_size
        self.min_frequency = min_frequency
        self.merges = {}  # Dictionary storing merge operations {pair: merge_index}
        self.vocab = {}   # Final vocabulary mapping tokens to IDs
        self.inverse_vocab = {}  # Mapping IDs to tokens
        self.merge_order = []  # Ordered list of merges for fast encoding
        
    def get_stats(self, words):
        """
        Count frequency of adjacent token pairs in the corpus.
        
        Args:
            words: Dictionary mapping words to their frequencies
            
        Returns:
            Counter of token pairs and their frequencies
        """
        pairs = Counter()
        for word, freq in words.items():
            symbols = word.split()
            for i in range(len(symbols) - 1):
                pairs[(symbols[i], symbols[i + 1])] += freq
        return pairs
    
    def merge_vocab(self, pair, words):
        """
        Merge all occurrences of a token pair in the vocabulary.
        
        Args:
            pair: Tuple of two tokens to merge
            words: Dictionary mapping words to their frequencies
            
        Returns:
            Updated words dictionary with merged pairs
        """
        new_words = {}
        bigram = ' '.join(pair)
        replacement = ''.join(pair)
        
        for word in words:
            # Replace the pair with merged token
            new_word = word.replace(bigram, replacement)
            new_words[new_word] = words[word]
        
        return new_words
    
    def train(self, text):
        """
        Train BPE tokenizer on given text.
        
        Args:
            text: Training text corpus
        """
        # Step 1: Prepare initial vocabulary (character-level)
        # Split text into words and add end-of-word marker
        words = defaultdict(int)
        for word in text.split():
            # Add space between characters and </w> at end
            word_tokens = ' '.join(list(word)) + ' </w>'
            words[word_tokens] += 1
        
        # Step 2: Get all unique characters as base vocabulary
        vocab = set()
        for word in words.keys():
            vocab.update(word.split())
        
        # Step 3: Iteratively merge most frequent pairs
        num_merges = self.vocab_size - len(vocab)
        
        for i in range(num_merges):
            # Find most frequent pair
            pairs = self.get_stats(words)
            
            if not pairs:
                break
                
            best_pair = max(pairs, key=pairs.get)
            
            # Only merge if frequency meets threshold
            if pairs[best_pair] < self.min_frequency:
                break
            
            # Store the merge operation
            self.merges[best_pair] = i
            self.merge_order.append(best_pair)  # Keep ordered list
            
            # Apply merge to vocabulary
            words = self.merge_vocab(best_pair, words)
            
            # Add new merged token to vocabulary
            vocab.add(''.join(best_pair))
            
            if (i + 1) % 50 == 0:
                print(f"Merge {i + 1}/{num_merges}: {best_pair} (freq: {pairs[best_pair]})")
        
        # Step 4: Create final token-to-ID mapping
        self.vocab = {token: idx for idx, token in enumerate(sorted(vocab))}
        self.inverse_vocab = {idx: token for token, idx in self.vocab.items()}
        
        print(f"\nBPE Training Complete!")
        print(f"Final vocabulary size: {len(self.vocab)}")
        print(f"Number of merges learned: {len(self.merges)}")
    
    def tokenize_word(self, word):
        """
        Apply BPE merges to a single word efficiently.
        
        Args:
            word: Word to tokenize (string)
            
        Returns:
            List of tokens
        """
        # Start with character-level tokens
        tokens = list(word) + ['</w>']
        
        # Apply merges in the order they were learned
        for pair in self.merge_order:
            i = 0
            while i < len(tokens) - 1:
                if (tokens[i], tokens[i + 1]) == pair:
                    # Merge the pair
                    tokens = tokens[:i] + [''.join(pair)] + tokens[i + 2:]
                else:
                    i += 1
        
        return tokens
    
    def encode(self, text):
        """
        Encode text to list of token IDs efficiently.
        
        Args:
            text: Text to encode
            
        Returns:
            List of integer token IDs
        """
        # Split into words and process each word
        words = text.split()
        all_tokens = []
        
        # Process in batches for progress indication
        total_words = len(words)
        batch_size = 10000
        
        for i in range(0, total_words, batch_size):
            batch = words[i:i + batch_size]
            
            for word in batch:
                # Tokenize word
                word_tokens = self.tokenize_word(word)
                all_tokens.extend(word_tokens)
                
                # Add space token between words
                if word != words[-1]:  # Not the last word
                    all_tokens.append(' ')
            
            # Show progress
            if (i + batch_size) % 50000 == 0:
                progress = min((i + batch_size) / total_words * 100, 100)
                print(f"Encoding progress: {progress:.1f}%")
        
        # Map tokens to IDs
        token_ids = [self.vocab.get(token, 0) for token in all_tokens]
        
        return token_ids
    
    def decode(self, ids):
        """
        Decode list of token IDs back to text.
        
        Args:
            ids: List of integer token IDs
            
        Returns:
            Decoded text string
        """
        tokens = [self.inverse_vocab.get(idx, '') for idx in ids]
        # Join tokens and clean up end-of-word markers and spaces
        text = ''.join(tokens).replace('</w>', ' ').strip()
        return text
    
    def get_vocab_size(self):
        """Return the size of the vocabulary."""
        return len(self.vocab)