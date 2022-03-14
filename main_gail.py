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

TRAJECTORY_LOAD_PATH="./data"
TRAJECTORY_LENGTH = 100
TRAJECTORIES_NUM = 2 # MAX 90
BATCH_SIZE = 20
HIDDEN_SIZE = 128
GAMMA = 0.99
PROCESSES = 4
STEPS = 150
NUM_ITR = 50

if __name__ == "__main__":
    torch.autograd.set_detect_anomaly(True)
    device = torch.device("cpu")
    load = Load()
    data = load.get_data(TRAJECTORY_LOAD_PATH, TRAJECTORIES_NUM) # Data in the form of SAS, features = 9
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

    for _ in range(NUM_ITR):
        for step in range(num_steps):
            with torch.no_grad():
                value, delta_pred, delta_log_probs = actor_critic.act(rollouts.obs[step])
            updated_gen_data = rollouts.obs[step]
            # Only update fol lower state as we are calculating delta
            leader_state, follower_state = updated_gen_data[:, 0:2], updated_gen_data[:, 2:4]
            follower_state = follower.step(np.array(follower_state), leader_state.detach().numpy(), delta_pred.detach().numpy())
            next_obs = torch.cat((gen_data[step + 1, :, :-2], torch.Tensor(follower_state[:, -1:]), gen_data[step + 1, :, -1:]), 1)
            obs_feat = torch.cat((disc_data[step, :, :-1], torch.Tensor(follower_state[:, -1:])), 1)
            rollouts.insert(next_obs, delta_pred, delta_log_probs, value, obs_feat)
        print(rollouts.delta)
        with torch.no_grad():
            next_value = actor_critic.get_value(rollouts.obs[-1]).detach()
            
        gail_epochs = 5
        for _ in range(gail_epochs):
            disc.update_disc(dataloader, rollouts)


        for step in range(num_steps):
            rollouts.rewards[step], returns = disc.get_reward( rollouts.obs_feat[step + 1], GAMMA)
            

            ret_rms.update(returns.view(-1).cpu().numpy())
            rews = rollouts.rewards[step].view(-1).cpu().numpy()
            rews = np.clip(rews / np.sqrt(ret_rms.var + 1e-7),
                        -10.0, 10.0)
            # print(ret_rms.var)    # just one number
            rollouts.rewards[step] = torch.Tensor(rews).view(-1, 1)

        rollouts.compute_returns(next_value, GAMMA)
        
        value_loss, action_loss, dist_entropy = agent.update(rollouts)

        rollouts.after_update()

