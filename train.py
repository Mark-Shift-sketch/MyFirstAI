# Script to train the model on dataset\

import torch
import torch.nn as nn
import torch.optim as optim
from core.utils import clean_text, create_vocab, text_to_sequences
from model.lstm_model import SimpleLSTM
import numpy as np

# Load and clean data
with open("data/corpus.txt", "r", encoding="utf-8") as f:
    text = f.read()

text = clean_text(text)
word_to_idx, idx_to_word = create_vocab(text)
vocab_size = len(word_to_idx)
seq_length = 5

sequences = text_to_sequences(text, word_to_idx, seq_length)
sequences = np.array(sequences)

# split sequences into x (input and y target)
x = sequences[:, :-1]
y = sequences[:, -1]

x = torch.tensor(x, dtype=torch.long)
y = torch.tensor(y, dtype=torch.long)

# Model, loss, optimizer

model = SimpleLSTM(vocab_size)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Training
epochs = 100
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()

    output, _ = model(x)

    # Use only last timestep for prediction
    last_output = output[:, -1, :]   # shape: (batch, vocab)

    loss = criterion(last_output, y)
    loss.backward()
    optimizer.step()

    if (epoch+1) % 100 == 0:
        print(f"epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

# save model
torch.save(model.state_dict(), "model.lstm_weights.pth")
print("Training complete and model save")