import torch
import torch.nn as nn
import torch.nn.functional as F

from resnet import ResNet
from transformers import Transformer


class Embedding(nn.Module):
    """
    Construct the embeddings from patch, position embeddings.
    """
    def __init__(self, hidden_size, image_size=(224, 224), patch_size=(16, 16)):
        super(Embedding, self).__init__()
        self.hidden_size = hidden_size
        grid_size = (image_size[0]//patch_size[0], image_size[1]//patch_size[1])  #(14x14)
        self.embedding = nn.Linear(1024, self.hidden_size)
        position = torch.arange(0, 196, 1)
        self.p = F.one_hot(position, num_classes=196)
        self.position_embedding = nn.Linear(196, self.hidden_size)

    def forward(self, x):
        x = self.patch_embedding(x)
        x = x.permute(0, 2, 1)  # (1, 1024, 196) -> (1, 196, 1024)
        latents = x.shape[-1]
        x = x.view(-1, latents) # (bx196, 1024)
        x = self.embedding(x)  # (bx196, hidden_size)
        x = x.reshape(-1, 196, self.hidden_size)  # (1, 196, hidden_size)

        x_p = self.position_embedding(self.p)  # （196, hidden_size)

        output = x + x_p  # (b, 196, hidden_size)

        return output


class Encoder(nn.Module):
    def __init__(self, hidden_size, heads):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.resnet = ResNet([3, 4, 6, 3])
        self.embedding = Embedding(hidden_size)
        self.transformer1 = Transformer(hidden_size, heads)
        self.transformer2 = Transformer(hidden_size, heads)
        self.transformer3 = Transformer(hidden_size, heads)

        # self.fc = nn.Linear()

    def forward(self, x):
        b, c, h, w = x.shape
        x1, x2, x3, x4 = self.resnet(x)
        z = self.embedding(x4)
        z1 = self.transformer1(z)
        z2 = self.transformer2(z1)
        z3 = self.transformer3(z2)

        z = z.reshape(b, 14, 14, self.hidden_size)
        z = z.permute(0, 3, 14, 14)  # (b, hidden_size(D), 14, 14)

        return [x1, x2, x3], z
