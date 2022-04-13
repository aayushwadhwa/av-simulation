import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import autograd

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
    
    def update_gen_2(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

class Discriminator(nn.Module):
    def __init__(self, num_inputs, hidden_size, num_output, device) -> None:
        super(Discriminator, self).__init__()
        self.returns = None
        self.model = nn.Sequential(
            nn.Linear(num_inputs, hidden_size), nn.Tanh(),
            nn.Linear(hidden_size, hidden_size), nn.Tanh(),
            nn.Linear(hidden_size, num_output)
        ).to(device)
        self.device = device
        self.optimizer = torch.optim.Adam(self.model.parameters())
        self.model.train()

    def forward(self, x):
        return self.model(x)

    def update_disc(self, expert_loader, rollouts):
        self.train()
        policy_data_generator = rollouts.feed_forward_generator(mini_batch_size=expert_loader.batch_size)
        loss = expert_loss_t = policy_loss_t = 0
        n = 0
        for expert_batch, policy_batch in zip(expert_loader,
                                              policy_data_generator):
            
            expert_data = expert_batch[0]
            policy_data = policy_batch[-1]

            policy_d = self.model(policy_data)
            expert_d = self.model(expert_data)

            expert_loss = F.binary_cross_entropy_with_logits(
                expert_d,
                torch.ones(expert_d.size()).to(self.device))
            policy_loss = F.binary_cross_entropy_with_logits(
                policy_d,
                torch.zeros(policy_d.size()).to(self.device))
            
            gail_loss = expert_loss + policy_loss
            grad_pen = self.compute_grad_pen_combined(expert_data, policy_data)

            loss += (gail_loss + grad_pen).item()
            expert_loss_t += expert_loss.item()
            policy_loss_t += policy_loss.item()
            n += 1
            self.optimizer.zero_grad()
            (gail_loss + grad_pen).backward()
            self.optimizer.step()
        
        return loss / n, expert_loss_t / n, policy_loss_t / n
    
    def compute_grad_pen_combined(self,
                                  expert_combined,
                                  policy_combined,
                                  lambda_=10.
                                  ):
        alpha = torch.rand(expert_combined.size(0), 1)
        alpha = alpha.expand_as(expert_combined).to(expert_combined.device)
        mixup_data = alpha * expert_combined + (1 - alpha) * policy_combined
        mixup_data.requires_grad = True

        disc = self.model(mixup_data)
        ones = torch.ones(disc.size()).to(disc.device)
        grad = autograd.grad(
            outputs=disc,
            inputs=mixup_data,
            grad_outputs=ones,
            create_graph=True,
            retain_graph=True,
            only_inputs=True)[0]

        grad_pen = lambda_ * (grad.norm(2, dim=1) - 1).pow(2).mean()
        return grad_pen

    
    def get_reward(self, d, gamma, masks, delta_param, offset = 0.0):
        with torch.no_grad():
            d = self.model(d)
            s = torch.sigmoid(d)
            reward = (s + 1e-7).log() - (1 - s + 1e-7).log() + offset + delta_param
            if self.returns is None:
                self.returns = reward.clone()
            else:
                self.returns = self.returns * gamma * masks + reward
            return reward, self.returns, d
