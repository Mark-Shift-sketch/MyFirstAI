#Defines the LSTM model class

import torch
import torch.nn as nn
"""
class SimpleLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim=128,hidden_dim=128):
        super(SimpleLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        
def forward(self, x, hidden=None):
    x = self.embedding(x)
    out, hidden = self.lstm(x, hidden)
    out = self.fc(out)
    return out, hidden
"""

import torch
import torch.nn as nn

class SimpleLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=128):
        super(SimpleLSTM, self).__init__()

        # Embedding
        self.embedding = nn.Embedding(vocab_size, embed_dim)

        # LSTM Layer
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)

        # Output layer
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hidden=None):
        x = self.embedding(x)              # shape: (batch, seq_len, embed_dim)

        out, hidden = self.lstm(x, hidden) # shape: (batch, seq_len, hidden_dim)

        out = self.fc(out)                 # shape: (batch, seq_len, vocab_size)

        return out, hidden
