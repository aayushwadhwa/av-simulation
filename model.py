import torch
import torch.nn as nn
import torch.nn.functional as F

class Generator(nn.Module):
    def __init__(self, num_inputs, hidden_size, num_output):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(num_inputs, hidden_size), nn.Tanh(),
            nn.Linear(hidden_size, hidden_size), nn.Tanh(),
            nn.Linear(hidden_size, num_output)
        )
        self.model.train()
    def forward(self, x):
        return self.model(x)

class Discriminator(nn.Module):
    def __init__(self, num_inputs, hidden_size, num_output) -> None:
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(num_inputs, hidden_size), nn.Tanh(),
            nn.Linear(hidden_size, hidden_size), nn.Tanh(),
            nn.Linear(hidden_size, num_output)
        )
        self.optimizer = torch.optim.Adam(self.model.parameters())
        self.model.train()

    def update_disc(self, dataloader):
        self.train()
        for data in dataloader:
            res = self.model(data)
            loss = F.binary_cross_entropy_with_logits(
                res, torch.ones(res.size()))
            # Add in the other loss term
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
