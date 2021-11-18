import torch
import torch.nn as nn

class RELAXCritic(nn.Module):
    def __init__(self, dim, hidden_dim):
        super(RELAXCritic, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, exp):
        assert (exp.ndimension() == 2)
        return self.net(exp)
