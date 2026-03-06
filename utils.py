# Helper functions (text cleaning, tokenization, wordindex mapping)

import re
import numpy as np

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s.,!?]', '', text)
    return text

def create_vocab(text):
    words = text.split()
    vocab = sorted(set(words))
    word_to_idx = {w: i for i, w in enumerate(vocab)}
    idx_to_word = {i: w for i, w in enumerate(vocab)}
    return word_to_idx, idx_to_word

def text_to_sequences(text, word_to_idx, seq_length=5):
    words = text.split()
    sequences = []
    for i in range(len(words)-seq_length):
        seq = words[i:i+seq_length+1]
        sequences.append([word_to_idx[w] for w in seq])
    return sequences
    