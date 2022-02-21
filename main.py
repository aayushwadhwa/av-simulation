from load import Load
from model import Discriminator, Generator
import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset

TRAJECTORY_LOAD_PATH="./data"
TRAJECTORY_LENGTH = 100
TRAJECTORIES_NUM = 1 # MAX 90
BATCH_SIZE = 20
HIDDEN_SIZE = 128
EPOCHS = 100

if __name__ == "__main__":
    load = Load()
    data = load.get_data(TRAJECTORY_LOAD_PATH, TRAJECTORIES_NUM)
    x = torch.Tensor(data.values)
    num_inputs = x.shape[0]
    dataset = TensorDataset(x)
    dataloader = DataLoader(
        dataset=dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    disc = Discriminator(num_inputs=num_inputs, num_output=1, hidden_size=HIDDEN_SIZE)
    gen = Generator(num_inputs=num_inputs, num_output=1, hidden_size=HIDDEN_SIZE)
    
        


