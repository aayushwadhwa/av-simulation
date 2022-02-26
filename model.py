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
        self.optimizer = torch.optim.Adam(self.model.parameters())
        self.model.train()
    
    def forward(self, x):
        return self.model(x)

    def update_gen(self, disc, x):
        output = disc(x)
        loss = F.binary_cross_entropy_with_logits(
            output, torch.ones(output.size()))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

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

    def forward(self, x):
        return self.model(x)

    def update_disc(self, real_data, storage, gen_data=None, idx=0):
        self.train()
        res = self.model(real_data)
        real_loss = F.binary_cross_entropy_with_logits(
            res, torch.ones(res.size()))
        fake_loss = 0
        res = self.model(gen_data)
        fake_loss = F.binary_cross_entropy_with_logits(
                res, torch.zeros(res.size()))

        # Add in the other loss term
        loss = (real_loss + fake_loss)
        storage.disc_loss.append(loss) # change it to loss array per epoch
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()
