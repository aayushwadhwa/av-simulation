from load import Load
from model import Discriminator, Generator
import torch
from torch.utils.data import DataLoader, TensorDataset
from storage import StorageNode
from equations import Leader, Follower
import numpy as np

TRAJECTORY_LOAD_PATH="./data"
TRAJECTORY_LENGTH = 100
TRAJECTORIES_NUM = 20 # MAX 90
BATCH_SIZE = 100
HIDDEN_SIZE = 128
EPOCHS = 100

if __name__ == "__main__":
    load = Load()
    data = load.get_data(TRAJECTORY_LOAD_PATH, TRAJECTORIES_NUM) # Data in the form of SAS, features = 9
    gen_data = data.iloc[:, 0:5] # Generate only needs curr state and action
    disc_data = torch.Tensor(data.values)
    gen_data = torch.Tensor(gen_data.values)
    num_inputs_disc, num_inputs_gen = disc_data.shape[1], gen_data.shape[1]
    dataset = TensorDataset(disc_data, gen_data)
    dataloader = DataLoader(
        dataset=dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

    disc = Discriminator(num_inputs=num_inputs_disc, num_output=1, hidden_size=HIDDEN_SIZE)
    gen = Generator(num_inputs=num_inputs_gen, num_output=1, hidden_size=HIDDEN_SIZE)

    leader, follower = Leader(), Follower()
    storage = StorageNode()
    
    for epoch in range(EPOCHS):
        updated_gen_data = None
        next_state = None
        print("EPOCH", epoch)
        g_loss, d_loss, prediction = [], [], []
        for idx, (disc_data, gen_data) in enumerate(dataloader):
            """
            Train both generate and discriminator:
            1. Predict "delta" by passing the curr state and action to generator
            2. Calculate next state by passing delta to the follower equation along with curr leader and follower state
            3. Pass the original and new generated SAS pair to disciminator
            4. Train the disciminator
            5. Train the generator adversarially
            """
            if idx == 0:
                delta_pred = gen(gen_data)
                updated_gen_data = gen_data
            else:
                # Thing about new gen data to pass to disc and gen
                updated_gen_data = torch.cat((next_state, gen_data[:, -1:]), 1)
                delta_pred = gen(updated_gen_data)
            
            # Only update follower state as we are calculating delta
            leader_state, follower_state = updated_gen_data[:, 0:2], updated_gen_data[:, 2:4]
            follower_next_state = follower.step(np.array(follower_state), leader_state.detach().numpy(), delta_pred.detach().numpy())

            gen_disc_data = torch.cat((disc_data[:, :-2], torch.Tensor(follower_next_state)), 1)
            disc_loss = disc.update_disc(disc_data, storage, idx=idx, gen_data=gen_disc_data)

            """
            reward = np.asarray(np.log(disc_loss))
            print(f"Loss: {disc_loss}, Reward: {reward}, Delta: {torch.mean(delta_pred)}")
            # reward = torch.from_numpy(reward)
            reward = torch.tensor(reward, requires_grad=True)
            gen.update_gen(reward)
            # print(disc_data)
            next_state = torch.cat((disc_data[:, -4:-2], torch.Tensor(follower_next_state)), 1)
            # print("NExt State",next_state)
            # asdf
            """

            gen_loss = gen.update_gen(disc, gen_disc_data.detach())
            next_state = torch.cat((disc_data[:, -4:-2], torch.Tensor(follower_next_state)), 1)
            g_loss.append(gen_loss)
            d_loss.append(disc_loss)
            prediction.append(torch.mean(delta_pred))
        
        storage.generate_loss.append(sum(g_loss)/len(g_loss))
        storage.disciminator_loss.append(sum(d_loss)/len(d_loss))
        storage.delta.append(sum(prediction)/len(prediction))
        storage.printLast()

