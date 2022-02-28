"""Implementation of different utility functions for adapter layers."""
import torch
import torch.nn as nn
from transformers.activations import get_activation


class GEGLU(nn.Module):
    def __init__(self):
        super().__init__()
        self.gelu = nn.GELU()

    def forward(self, x):
        x1, x2 = torch.chunk(x, 2, -1)
        return self.gelu(x1) * x2



class Activations(nn.Module):
    def __init__(self, activation_type):
        super().__init__()
        if activation_type == "geglu":
            self.f = GEGLU()
        else:
            self.f = get_activation(activation_type)

    def forward(self, x):
        return self.f(x)
