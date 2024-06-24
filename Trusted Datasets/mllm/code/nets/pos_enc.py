import torch
import torch.nn as nn

# Positional Encoding Layer
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1280):
        super(PositionalEncoding, self).__init__()
        self.encoding = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        self.encoding[:, 0::2] = torch.sin(position * div_term)
        self.encoding[:, 1::2] = torch.cos(position * div_term)
        self.encoding = self.encoding.unsqueeze(0).detach()  # make sure it’s detached and not requiring gradients

    def forward(self, x):
        if x.size(1) > self.encoding.size(1):
            raise ValueError(f"Input sequence length {x.size(1)} is greater than the maximum length {self.encoding.size(1)} for positional encoding.")
        return x + self.encoding[:, :x.size(1)].to(x.device)  # move to the same device as x
