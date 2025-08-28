import torch
import torch.nn as nn

class BiGRU(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim, n_layers, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.gru = nn.GRU(embed_dim, hidden_dim, num_layers=n_layers,
                          batch_first=True, bidirectional=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim*2, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        emb = self.embedding(x)
        out, h = self.gru(emb)
        last = torch.cat((h[-2,:,:], h[-1,:,:]), dim=1)
        return self.sigmoid(self.fc(last)).squeeze(1)
