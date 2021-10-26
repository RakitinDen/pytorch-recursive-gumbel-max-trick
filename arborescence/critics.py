import torch
import torch.nn as nn


class RELAXCritic(nn.Module):

    def __init__(self, n, hidden_dim):
        super(RELAXCritic, self).__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(n * n, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, z):
        assert (z.ndimension() == 2) or (z.ndimension() == 3)

        return self.net(z)
