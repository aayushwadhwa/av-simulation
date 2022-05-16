from load import Load
from model import Discriminator, Generator
import torch
from torch.utils.data import DataLoader, TensorDataset
from ppo import PPO
from split_policy import SplitPolicy
from storage import RolloutStorage
from equations import Leader, Follower
import numpy as np

TRAJECTORY_LOAD_PATH="./data"
TRAJECTORY_LENGTH = 100
TRAJECTORIES_NUM = 20 # MAX 90
BATCH_SIZE = 80
HIDDEN_SIZE = 128
EPOCHS = 1000

if __name__ == "__main__":
    load = Load()
    data = load.get_data(TRAJECTORY_LOAD_PATH, TRAJECTORIES_NUM) # Data in the form of SAS, features = 9
    gen_data = data.iloc[:, 0:5] # Generate only needs curr state and action
    disc_data = torch.Tensor(data.values)
    gen_data = torch.Tensor(gen_data.values)
    num_inputs_disc, num_inputs_gen = disc_data.shape[1], gen_data.shape[1]
    dataset = TensorDataset(disc_data, gen_data)
    dataloader = DataLoader(dataset=dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=True)
    disc = Discriminator(num_inputs=num_inputs_disc, num_output=1, hidden_size=HIDDEN_SIZE)
    
    actor_critic = SplitPolicy(num_inputs_gen)
    agent = PPO(actor_critic)

    leader, follower = Leader(), Follower()

    for epoch in range(EPOCHS):
        updated_gen_data = None
        next_state = None
        g_loss, d_loss, prediction = [], [], []
        for idx, (disc_data, gen_data) in enumerate(dataloader):
            if idx == 0:
                delta_pred, _, log_probs = actor_critic.act(gen_data)
                updated_gen_data = gen_data
            else:
                # Thing about new gen data to pass to disc and gen
                updated_gen_data = torch.cat((next_state, gen_data[:, -1:]), 1)
                delta_pred, _, log_probs = actor_critic.act(updated_gen_data)
            
            # Only update fol lower state as we are calculating delta
            leader_state, follower_state = updated_gen_data[:, 0:2], updated_gen_data[:, 2:4]
            follower_next_state = follower.step(np.array(follower_state), leader_state.detach().numpy(), delta_pred.detach().numpy())

            gen_disc_data = torch.cat((disc_data[:, :-2], torch.Tensor(follower_next_state)), 1)
            # disc_loss = disc.update_disc(disc_data, storage, idx=idx, gen_data=gen_disc_data)
            
            with torch.no_grad():
                next_value = actor_critic.act(gen_disc_data).detach()
            
            gail_epochs = 5
            for _ in gail_epochs:
                disc.update_disc(disc_data, storage, idx=idx, gen_data=gen_disc_data)


            reward, returns = disc.get_reward(gen_disc_data)

            
            # gen_loss = np.asarray(g_l)
            # gen_loss = torch.tensor(gen_loss, requires_grad=True)
            # # print(f"D: {disc_loss}, G: {gen_loss.item()}, Delta: f{delta_pred}")
            # gen.update_gen_2(gen_loss)
            # next_state = torch.cat((disc_data[:, -4:-2], torch.Tensor(follower_next_state)), 1)
            # g_loss.append(gen_loss.item())
            # d_loss.append(disc_loss)
            # prediction.append(torch.mean(delta_pred))
        
        # storage.generate_loss.append(sum(g_loss)/len(g_loss))
        # storage.disciminator_loss.append(sum(d_loss)/len(d_loss))
        # storage.delta.append(sum(prediction)/len(prediction))
        # if epoch % 20 == 0:
        #     print("EPOCH", epoch)
        #     storage.print_last()
        #     storage.reset()

