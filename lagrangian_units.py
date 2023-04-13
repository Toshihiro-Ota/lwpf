import torch
import torch.nn as nn

class ReLU(nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(x)
        mask = torch.heaviside(x, torch.tensor([0.0]).to(x.device)).detach()
        return x, mask

class PFU(nn.Module):
    def __init__(self, bias = None, mu = None, sigma = 1.0):
        super().__init__()
        self.square_sigma = float(sigma) ** 2
        self.relu = nn.ReLU()

        if hasattr(self, mu):
            self.mu = getattr(self, mu)
        else:
            self.mu_value = float(mu)
            self.mu = lambda x: self.mu_value

        if bias == 'mu':
            self.bias = lambda x: x
        else:
            self.bias_value = float(bias)
            self.bias =lambda x: self.bias_value

    def dmedian(self, x):
        return x.median()

    def forward_forget(self, x, a, sgn = 1.):
        h = self.relu(sgn*(x - a))
        mask = torch.heaviside(h, torch.tensor([0.0]).to(h.device)).detach()
        x = mask * x + (1. - mask) * self.bias(a)
        return x, mask

    def forward(self, x):
        eps = 0.
        if self.training:
            eps = torch.randn(1).to(x.device) * self.square_sigma
        x, mask = self.forward_forget(x, self.mu(x) + eps)
        return x, mask
