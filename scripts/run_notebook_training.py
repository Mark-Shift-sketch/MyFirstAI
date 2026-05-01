"""
Run a small, quick training pass using project modules (for notebook fallback).
"""
from pathlib import Path
import sys
# ensure repo root is importable
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from model.lstm_model import SimpleLSTM
from core.utils import clean_text, create_vocab, text_to_sequences


def main():
    # locate data file relative to the repository root (script's parent parent)
    data_path = Path(__file__).resolve().parents[1] / 'data' / 'corpus.txt'
    if not data_path.exists():
        raise FileNotFoundError(f'Data file not found: {data_path}')

    text = clean_text(data_path.read_text(encoding='utf-8'))
    word_to_idx, idx_to_word = create_vocab(text)
    vocab_size = len(word_to_idx)
    seq_length = 5
    sequences = text_to_sequences(text, word_to_idx, seq_length)
    sequences = np.array(sequences)
    if len(sequences) == 0:
        raise ValueError('No sequences generated from text — check corpus and seq_length')

    max_samples = 5000
    sequences = sequences[:max_samples]
    x = torch.tensor(sequences[:, :-1], dtype=torch.long)
    y = torch.tensor(sequences[:, -1], dtype=torch.long)

    model = SimpleLSTM(vocab_size)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.005)

    epochs = 20
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        output, _ = model(x)
        last_output = output[:, -1, :]
        loss = criterion(last_output, y)
        loss.backward()
        optimizer.step()
        print(f'epoch {epoch+1}/{epochs}, loss={loss.item():.4f}')

    save_dir = Path(__file__).resolve().parents[1] / 'models'
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / 'script_lstm_weights.pth'
    torch.save(model.state_dict(), str(save_path))
    print(f'Saved model to {save_path}')


if __name__ == '__main__':
    main()
