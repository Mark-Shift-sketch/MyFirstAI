#Defines the LSTM model class


import torch
import torch.nn as nn

class SimpleLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim=256, hidden_dim=512, num_layers=2, dropout=0.3):
        super(SimpleLSTM, self).__init__()

        # Embedding
        self.embedding = nn.Embedding(vocab_size, embed_dim)

        # LSTM Layer
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=num_layers, batch_first=True, dropout=dropout)

        # Output layer
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hidden=None):
        x = self.embedding(x)              # shape: (batch, seq_len, embed_dim)

        out, hidden = self.lstm(x, hidden) # shape: (batch, seq_len, hidden_dim)

        out = self.fc(out)                 # shape: (batch, seq_len, vocab_size)

        return out, hidden
