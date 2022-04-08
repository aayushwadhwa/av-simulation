import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module

class FixedNormal(torch.distributions.Normal):
    def log_probs(self, actions):
        return super().log_prob(actions).sum(-1, keepdim=True)

    def entropy(self):
        return super().entropy().sum(-1)

    def mode(self):
        return self.mean


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class SplitPolicy(nn.Module):
    def __init__(self, num_inputs):
        super(SplitPolicy, self).__init__()
        self.base = SplitPolicyBaseNew(num_inputs)
        self.dist = StateDiagGaussianNew()

    def forward(self, masks):
        # not used
        raise NotImplementedError

    def act(self, inputs, masks):
        value, actor_mean = self.base(inputs, masks)
        dist = self.dist(actor_mean)
        delta = dist.sample((1,2))
        delta_log_probs = dist.log_probs(delta)

        return value, delta, delta_log_probs

    def get_value(self, inputs, masks):
        value, _ = self.base(inputs, masks)
        return value

    def evaluate_actions(self, inputs, masks, action=None):
        value, actor_mean = self.base(inputs, masks)
        dist = self.dist(actor_mean)
        delta_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()

        return value, delta_log_probs, dist_entropy
    
    def get_mean_std(self):
        return self.dist.mean, self.dist.std


class SplitPolicyBaseNew(nn.Module):
    def __init__(self, num_inputs, hidden_size=64):
        
        # Neural Net for delta
        super(SplitPolicyBaseNew, self).__init__()
        
        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), np.sqrt(2))
        init_final_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                                     constant_(x, 0))

        self.actor_delta = nn.Sequential(
            init_(nn.Linear(num_inputs, hidden_size)), nn.Tanh(),
            init_(nn.Linear(hidden_size, hidden_size)), nn.Tanh())

        self.critic_full = nn.Sequential(
            init_(nn.Linear(num_inputs, hidden_size)), nn.Tanh(),
            init_(nn.Linear(hidden_size, hidden_size)), nn.Tanh(),
            init_final_(nn.Linear(hidden_size, 1)))

        self.train()

    def forward(self, x, masks):
        # x = inputs.clone()
        value = self.critic_full(x)
        action_feat = self.actor_delta(x)
        return value, action_feat


class StateDiagGaussianNew(nn.Module):
    def __init__(self, hidden_size=64, num_feet=1):
        super(StateDiagGaussianNew, self).__init__()
        self.hidden_size = hidden_size
        self.delta_mean = nn.Linear(hidden_size, 1 * num_feet)
        self.delta_logstd = nn.Linear(hidden_size, 1 * num_feet)
        self.mean = None
        self.std = None

    def forward(self, input):
        # input = x.detach()
        delta_feat = input[:, :self.hidden_size]
        delta_mean = self.delta_mean(delta_feat)
        delta_logstd = self.delta_logstd(delta_feat)
        self.mean = delta_mean
        self.std = delta_logstd.exp()
        return FixedNormal(delta_mean, delta_logstd.exp())