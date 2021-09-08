import torch
import torch.nn as nn
from decoder import DecoderCup
from encoder import Encoder

class TransUnet(nn.Module):
    def __init__(self, hidden_size=768, heads=12):
        self.encoder = Encoder(hidden_size, heads)
        self.decoder = DecoderCup(hidden_size)

    def forward(self, x):
        hidden_layers, hidden_state = self.encoder(x)
        y = self.decoder(hidden_state, hidden_layers[0], hidden_layers[1], hidden_layers[2])
        return y