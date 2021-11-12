import torch
import torch.nn as nn

class RELAXCritic(nn.Module):

    def __init__(self, d, hidden_dim):
        super(RELAXCritic, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(d, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, z):
        assert (z.ndimension() == 1) or (z.ndimension() == 2)

        return self.net(z.squeeze())
