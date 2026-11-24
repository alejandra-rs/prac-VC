import numpy as np
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence


class DrowsinessGRU(nn.Module):
    def __init__(self, mean, std, hidden, layers, dropout, n_classes):
        super().__init__()
        self.register_buffer("mean", torch.tensor(mean, dtype=torch.float32))
        self.register_buffer("std", torch.tensor(std, dtype=torch.float32))
        self.gru = nn.GRU(len(mean), hidden, layers, batch_first=True, dropout=dropout)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden, n_classes))

    def forward(self, x):
        self.gru.flatten_parameters()
        _, hidden = self.gru((x - self.mean) / self.std)
        return self.head(hidden[-1])


class VideoGRU(DrowsinessGRU):
    def forward(self, x):
        steps = x[..., -1].sum(1).long().cpu()
        packed = pack_padded_sequence((x[..., :-1] - self.mean) / self.std, steps, batch_first=True, enforce_sorted=False)
        _, hidden = self.gru(packed)
        return self.head(hidden[-1])


class SequencePredictor:
    def __init__(self, model, batch_size=256):
        self.model = model.eval()
        self.batch_size = batch_size

    @torch.no_grad()
    def predict(self, sequences):
        device = self.model.mean.device
        predictions = []
        for i in range(0, len(sequences), self.batch_size):
            batch = torch.from_numpy(sequences[i:i + self.batch_size]).to(device)
            predictions.append(self.model(batch).argmax(1).cpu().numpy())
        return np.concatenate(predictions)
