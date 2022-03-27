import torch
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler, SequentialSampler

class RolloutStorage(object):
    def __init__(self, num_steps, num_processes, obs_shape, feat_len=0):
        self.obs = torch.zeros(num_steps + 1, num_processes, obs_shape)
        self.obs_feat = torch.zeros(num_steps + 1, num_processes, feat_len)
        self.rewards = torch.zeros(num_steps, num_processes, 1)
        self.value_preds = torch.zeros(num_steps + 1, num_processes, 1)
        self.returns = torch.zeros(num_steps + 1, num_processes, 1)
        self.delta_log_probs = torch.zeros(num_steps, num_processes, 1)
        self.delta = torch.zeros(num_steps, num_processes, 1)
        self.bad_masks = torch.ones(num_steps + 1, num_processes, 1)
        self.num_steps = num_steps
        self.step = 0

    def to(self, device):
        self.obs = self.obs.to(device)
        self.obs_feat = self.obs_feat.to(device)
        self.rewards = self.rewards.to(device)
        self.value_preds = self.value_preds.to(device)
        self.returns = self.returns.to(device)
        self.delta = self.delta.to(device)
        self.delta_log_probs = self.delta_log_probs.to(device)
        self.bad_masks = self.bad_masks.to(device)

    def insert(self, obs, delta, delta_log_prob, value_pred, bad_masks, obs_feat=None):
        self.obs[self.step + 1].copy_(obs)
        if obs_feat is not None:
            self.obs_feat[self.step + 1].copy_(obs_feat)
        self.value_preds[self.step].copy_(value_pred)
        # self.rewards[self.step].copy_(rewards)
        self.step = (self.step + 1) % self.num_steps
        self.delta[self.step].copy_(delta)
        self.delta_log_probs[self.step].copy_(delta_log_prob)
        self.bad_masks[self.step + 1].copy_(bad_masks)


    def after_update(self):
        self.obs[0].copy_(self.obs[-1])
        self.obs_feat[0].copy_(self.obs_feat[-1])
        self.bad_masks[0].copy_(self.bad_masks[-1])

    def compute_returns(self,
                        next_value,
                        gamma):
                # self.value_preds[-1] = next_value
                # gae = 0
                # for step in reversed(range(self.rewards.size(0))):
                #     delta = self.rewards[step] + gamma * self.value_preds[
                #         step + 1] - self.value_preds[step]
                #     gae = delta + gamma * 0.95 * gae
                #     gae = gae * self.bad_masks[step + 1]
                #     self.returns[step] = gae + self.value_preds[step]

                self.returns[-1] = next_value
                for step in reversed(range(self.rewards.size(0))):
                    self.returns[step] = (self.returns[step + 1] * \
                                          gamma + self.rewards[step]) * self.bad_masks[step + 1] \
                                         + (1 - self.bad_masks[step + 1]) * self.value_preds[step]
    
    def feed_forward_generator(self, advantages = None, mini_batch_size=None):
        num_steps = self.rewards.size()[0]
        num_steps, num_processes = self.rewards.size()[0:2]
        batch_size = num_processes * num_steps
        
        sampler = BatchSampler(
            SubsetRandomSampler(range(batch_size)),
            mini_batch_size,
            drop_last=True)
        
        for indices in sampler:
            obs_batch = self.obs[:-1].view(-1, *self.obs.size()[2:])[indices]
            obs_feat_batch = self.obs_feat[:-1].view(-1, *self.obs_feat.size()[2:])[indices]
            next_obs_feat_batch = self.obs_feat[1:].view(-1, *self.obs_feat.size()[2:])[indices]
            return_batch = self.returns[:-1].view(-1, 1)[indices]
            old_delta_log_probs_batch = self.delta_log_probs.view(-1, 1)[indices]
            delta_batch = self.delta.view(-1, self.delta.size(-1))[indices]
            value_preds_batch = self.value_preds[:-1].view(-1, 1)[indices]
            if advantages is None:
                adv_targ = None
            else:
                adv_targ = advantages.view(-1, 1)[indices]

            yield obs_batch, delta_batch, value_preds_batch, return_batch, old_delta_log_probs_batch, adv_targ, obs_feat_batch, next_obs_feat_batch
