import torch
import torch.nn as nn
import torch.optim as optim

class PPO():
    def __init__(self,
                 actor_critic,
                 value_loss_coef = 0.5,
                 entropy_coef = 0.01,
                 num_mini_batch = 32,
                 clip_param = 0.2,
                 symmetry_coef=0,
                 ppo_epoch = 10,
                 lr=3e-4,
                 eps=1e-5,
                 max_grad_norm=0.5,
                 use_clipped_value_loss=True,
                 mirror_obs=None,
                 mirror_act=None):

        self.actor_critic = actor_critic

        self.clip_param = clip_param
        self.ppo_epoch = ppo_epoch
        self.num_mini_batch = num_mini_batch

        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef

        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss

        self.optimizer = optim.Adam(actor_critic.parameters(), lr=lr, eps=eps)

        self.symmetry_coef = symmetry_coef
        self.mirror_obs = mirror_obs
        self.mirror_act = mirror_act

        self.is_cuda = next(actor_critic.parameters()).is_cuda

    def update(self, rollouts):
        advantages = rollouts.returns[:-1] - rollouts.value_preds[:-1]
        advantages = (advantages - advantages.mean()) / (
                advantages.std() + 1e-5)

        value_loss_epoch = 0
        action_loss_epoch = 0
        dist_entropy_epoch = 0
        # Check Advantage
        for e in range(self.ppo_epoch):
            data_generator = rollouts.feed_forward_generator(advantages, self.num_mini_batch)
            for sample in data_generator:
                obs_batch, delta_batch, value_preds_batch, return_batch, masks_batch, old_log_probs, adv_targ, _, _ = sample
                # Reshape to do in a single forward pass for all steps
                values, delta_log_probs, dist_entropy = self.actor_critic.evaluate_actions(obs_batch, masks_batch, delta_batch)
                # value_loss = 0.5 * (return_batch - values).pow(2).mean()

                value_pred_clipped = value_preds_batch + \
                                         (values - value_preds_batch).clamp(-self.clip_param, self.clip_param)
                value_losses = (values - return_batch).pow(2)
                value_losses_clipped = (
                            value_pred_clipped - return_batch).pow(2)
                value_loss = 0.5 * torch.max(value_losses,
                                                 value_losses_clipped).mean()

                ratio = torch.exp(delta_log_probs -
                                  old_log_probs)
                surr1 = ratio * adv_targ
                surr2 = torch.clamp(ratio, 1.0 - self.clip_param,
                                    1.0 + self.clip_param) * adv_targ
                action_loss = -torch.min(surr1, surr2).mean()


                self.optimizer.zero_grad()
                (value_loss * self.value_loss_coef + action_loss - dist_entropy * self.entropy_coef).backward()

                torch.nn.utils.clip_grad_norm_(self.actor_critic.parameters(),
                                         self.max_grad_norm)
                self.optimizer.step()
                value_loss_epoch += value_loss.item()
                action_loss_epoch += action_loss.item()
                dist_entropy_epoch += dist_entropy.item()
        
        num_updates = self.ppo_epoch * self.num_mini_batch

        value_loss_epoch /= num_updates
        action_loss_epoch /= num_updates
        dist_entropy_epoch /= num_updates

        return value_loss_epoch, action_loss_epoch, dist_entropy_epoch
