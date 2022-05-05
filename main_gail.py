from load import Load
from model import Discriminator, Generator
import torch
from torch.utils.data import DataLoader, TensorDataset
from ppo import PPO
from running_mean_std import RunningMeanStd
from split_policy import SplitPolicy
from storage import RolloutStorage
from equations import Leader, Follower, NewSystem
import numpy as np
import matplotlib.pyplot as plt

# from torch.utils.tensorboard import SummaryWriter
# writer = SummaryWriter()

# TRAJECTORY_LOAD_PATH="./av-simulation/data_without_noise"
TRAJECTORY_LOAD_PATH="./data"
TRAJECTORY_LENGTH = 110
TRAJECTORIES_NUM = 29 # MAX 90
BATCH_SIZE = 50
HIDDEN_SIZE = 128
GAMMA = 0.99
PROCESSES = 1
STEPS = 1400
NUM_ITR = 200
LAMBDA = 0.1
PREDICTION_STEPS = 50

def predict(actor_critic, obs, deltas):
    for step in range(PREDICTION_STEPS):
        with torch.no_grad():
            _, delta_pred, _ = actor_critic.act(obs[step], torch.Tensor([[1.0]]))
        updated_gen_data = obs[step]
        delta_pred = delta_pred.reshape(1,2)
        prev_state = updated_gen_data[:, 0:2]
        next_sa = newSystem.step(prev_state, delta_pred)
        next_sa = torch.tensor(next_sa)
        # masks = torch.Tensor([[1.0] for _ in next_sa])
        obs[step + 1].copy_(next_sa)
        deltas[step].copy_(delta_pred)


if __name__ == "__main__":
    torch.autograd.set_detect_anomaly(True)
    device = torch.device("cpu")
    load = Load()
    data = load.get_data(TRAJECTORY_LOAD_PATH, TRAJECTORIES_NUM) # Data in the form of SAS, features = 9
    gen_data = data.iloc[:, 0:3] # Generate only needs curr state and action
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
    newSystem = NewSystem()
    
    ret_rms = RunningMeanStd(shape=())
    disc_loss = []
    delta_predition = []
    returns_per_epoch = []
    d_avg = []
    for ab in range(NUM_ITR):
        print(ab)
        # rollouts.obs[0].copy_(gen_data[0])
        for step in range(num_steps):
            with torch.no_grad():
                value, delta_pred, delta_log_probs = actor_critic.act(rollouts.obs[step], rollouts.masks[step])
            updated_gen_data = rollouts.obs[step]
            # delta_pred = delta_pred.reshape(1,2)
            # delta_log_probs = delta_log_probs.reshape(1,2)
            prev_state = updated_gen_data[:, 0:2]
            next_s = newSystem.step(prev_state, delta_pred)
            # Only update follower state as we are calculating delta
            # leader_state, follower_state, vref = updated_gen_data[:, 0:2], updated_gen_data[:, 2:4], updated_gen_data[:, -1:]
            # next_leader_state = leader.step(leader_state.detach().numpy(), vref.detach().numpy())
            # follower_state = follower.step(np.array(follower_state), leader_state.detach().numpy(), delta_pred.detach().numpy())
            # next_sa = torch.cat((torch.Tensor(next_leader_state), torch.Tensor(follower_state), gen_data[step + 1, :, -1:]), 1)
            next_a = disc_data[step][:, 2]
            next_sa = torch.cat((next_s, next_a.reshape(1,1)), 1)
            obs_feat = torch.cat((updated_gen_data, next_s), 1)
            masks = torch.Tensor([[1.0] for _ in next_sa])
            bad_mask = torch.Tensor([[1.0] for _ in delta_pred])
            rollouts.insert(next_sa, delta_pred, delta_log_probs, value, masks, bad_mask, obs_feat)

        print(rollouts.delta[-10:, :])
        print(torch.mean(rollouts.delta))
        delta_predition.append(torch.mean(rollouts.delta).item())
    
        print("Mean, Std", actor_critic.get_mean_std())
        std = actor_critic.get_mean_std()[1]

        # if std[0][0] < 1:
        #     break

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
            # delta_param =  torch.tanh(rollouts.delta[step])
            r_sa = 0 # no keep alive bonus
            rollouts.rewards[step], returns = disc.get_reward( rollouts.obs_feat[step + 1], GAMMA, rollouts.masks[step], delta_param = 0, offset=-r_sa)

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
    
    obs = torch.zeros(PREDICTION_STEPS + 1, PROCESSES, num_inputs_gen)
    deltas = torch.zeros(PREDICTION_STEPS + 1, PROCESSES, 2)
    obs[0].copy_(gen_data[0])
    predict(actor_critic, obs, deltas)

    plt.figure(1)
    obs = obs.squeeze().cpu().numpy()
    x1, x2 = obs[:, 0], obs[:, 1]
    print("x1::", x1)
    print("x2::", x2)
    plt.plot(x1, x2)
    plt.savefig("1.png")

    plt.figure(3)
    # deltas = deltas.squeeze().cpu().numpy()
    # c1, c2 = deltas[:, 0], deltas[:, 1]
    fig, axs = plt.subplots(2)
    axs[0].plot(x1)
    axs[0].set_title("State 1 (x1)")
    axs[1].plot(x2)
    axs[1].set_title("State 2(x2)")
    plt.savefig("3.png")


    plt.figure(4)
    deltas = deltas.squeeze().cpu().numpy()
    c1, c2 = deltas[:, 0], deltas[:, 1]
    fig, axs = plt.subplots(2)
    axs[0].plot(c1)
    axs[0].set_title("Parameter 1 (C1)")
    axs[1].plot(c2)
    axs[1].set_title("Parameter 2 (C2)")
    plt.savefig("4.png")



    print(returns_per_epoch)
    print(disc_loss)
    plt.figure(2)
    fig, axs = plt.subplots(2)
    axs[0].plot(returns_per_epoch)
    axs[0].set_title("Returns per epoch")
    axs[1].plot(disc_loss)
    axs[1].set_title("Disc Loss")
    plt.savefig('2.png')

    

