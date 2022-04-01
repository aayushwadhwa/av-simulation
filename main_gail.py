from load import Load
from model import Discriminator, Generator
import torch
from torch.utils.data import DataLoader, TensorDataset
from ppo import PPO
from running_mean_std import RunningMeanStd
from split_policy import SplitPolicy
from storage import RolloutStorage
from equations import Leader, Follower
import numpy as np
import matplotlib.pyplot as plt

# TRAJECTORY_LOAD_PATH="./av-simulation/data_without_noise"
TRAJECTORY_LOAD_PATH="./data"
TRAJECTORY_LENGTH = 100
TRAJECTORIES_NUM = 15 # MAX 90
BATCH_SIZE = 100
HIDDEN_SIZE = 128
GAMMA = 0.99
PROCESSES = 1
STEPS = 1000
NUM_ITR = 500
LAMBDA = 0.1

if __name__ == "__main__":
    torch.autograd.set_detect_anomaly(True)
    device = torch.device("cpu")
    load = Load()
    data = load.get_data(TRAJECTORY_LOAD_PATH, TRAJECTORIES_NUM, TRAJECTORY_LENGTH) # Data in the form of SAS, features = 9
    gen_data = data.iloc[:, 0:5] # Generate only needs curr state and action
    disc_data = torch.Tensor(data.values)
    gen_data = torch.Tensor(gen_data.values)
    num_inputs_disc, num_inputs_gen = disc_data.shape[1], gen_data.shape[1]

    # convert to 3D
    removed = gen_data.shape[0] % PROCESSES
    if removed:
        gen_data = gen_data[: -removed, :]
    gen_batch = int(gen_data.shape[0] / PROCESSES)
    gen_data = gen_data.reshape(gen_batch, PROCESSES, num_inputs_gen)


    dataset = TensorDataset(disc_data)
    dataloader = DataLoader(dataset=dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=True)
    disc = Discriminator(num_inputs=num_inputs_disc, num_output=1, hidden_size=HIDDEN_SIZE, device=device)
    
    # convert disc data
    if removed:
        disc_data = disc_data[: -removed, :]
    disc_data = disc_data.reshape(gen_batch, PROCESSES, num_inputs_disc)

    obs = gen_data[0]
    num_steps = STEPS
    rollouts = RolloutStorage(num_steps, PROCESSES, num_inputs_gen, num_inputs_disc)
    rollouts.obs[0].copy_(gen_data[0])
    actor_critic = SplitPolicy(num_inputs_gen)
    agent = PPO(actor_critic)

    leader, follower = Leader(), Follower()
    
    ret_rms = RunningMeanStd(shape=())
    disc_loss = []
    delta_predition = []
    returns_per_epoch = []
    d_avg = []
    for ab in range(NUM_ITR):
        print(ab)
        for step in range(num_steps):
            with torch.no_grad():
                value, delta_pred, delta_log_probs = actor_critic.act(rollouts.obs[step], rollouts.masks[step])
            updated_gen_data = rollouts.obs[step]
            # Only update follower state as we are calculating delta
            leader_state, follower_state, vref = updated_gen_data[:, 0:2], updated_gen_data[:, 2:4], updated_gen_data[:, -1:]
            next_leader_state = leader.step(leader_state.detach().numpy(), vref.detach().numpy())
            follower_state = follower.step(np.array(follower_state), leader_state.detach().numpy(), delta_pred.detach().numpy())
            next_sa = torch.cat((torch.Tensor(next_leader_state), torch.Tensor(follower_state), gen_data[step + 1, :, -1:]), 1)
            obs_feat = torch.cat((disc_data[step, :, :-4], next_sa[:, :-1]), 1)
            masks = torch.Tensor([[0.0] if state[1] > 45 and state[3] > 25 else [1.0] for state in next_sa])
            bad_mask = torch.Tensor([[1.0] for pred in delta_pred])
            rollouts.insert(next_sa, delta_pred, delta_log_probs, value, masks, bad_mask, obs_feat)

        print(rollouts.delta[-10:, :])
        print(torch.mean(rollouts.delta))
        delta_predition.append(torch.mean(rollouts.delta).item())
    
        print("Mean, Std", actor_critic.get_mean_std())

        with torch.no_grad():
            next_value = actor_critic.get_value(rollouts.obs[-1], rollouts.masks[-1]).detach()
        
        gail_epochs = 1
        d_loss = []
        for _ in range(gail_epochs):
            loss, _, _ = disc.update_disc(dataloader, rollouts)
            d_loss.append(loss)

        num_of_dones = (1.0 - rollouts.masks).sum().cpu().numpy() \
            + PROCESSES / 2

        num_of_expert_dones = (num_steps * PROCESSES) / TRAJECTORY_LENGTH

        d_sa = 1 - num_of_dones / (num_of_dones + num_of_expert_dones)

        r_sa = np.log(d_sa) - np.log(1 - d_sa) # Keep alive bonous
        d_val = []
        for step in range(num_steps):
            delta_param =  torch.tanh(rollouts.delta[step])

            rollouts.rewards[step], returns = disc.get_reward( rollouts.obs_feat[step + 1], GAMMA, rollouts.masks[step], delta_param, offset=-r_sa)

            ret_rms.update(returns.view(-1).cpu().numpy())
            rews = rollouts.rewards[step].view(-1).cpu().numpy()
            rews = np.clip(rews / np.sqrt(ret_rms.var + 1e-7),
                        -10.0, 10.0)
            # print(ret_rms.var)    # just one number
            rollouts.rewards[step] = torch.Tensor(rews).view(-1, 1)
        
        rollouts.compute_returns(next_value, GAMMA)
        
        value_loss, action_loss, dist_entropy = agent.update(rollouts)

        avg_returns = torch.sum(rollouts.returns).item() / len(rollouts.returns)
        disc_loss.append(sum(d_loss) / len(d_loss))
        returns_per_epoch.append(avg_returns)
        print("Reward::", avg_returns)
        rollouts.after_update()

    fig, axs = plt.subplots(3)
    axs[0].plot(returns_per_epoch)
    axs[0].set_title("Returns per epoch")
    axs[1].plot(disc_loss)
    axs[1].set_title("Disc Loss")
    axs[2].plot(delta_predition)
    axs[2].set_title("Delta Prediction")
    plt.savefig('books_read.png')

