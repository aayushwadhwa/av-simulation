# The code is based on the link:
# https://machinelearningmastery.com/how-to-develop-lstm-models-for-multi-step-time-series-forecasting-of-household-power-consumption/

import data_prep as Data
from numpy import split
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, TensorDataset
import matplotlib.pyplot as plt

SEQUENCE_LEN = 10
EPOCHS = 50
INPUT_DIM = 6
HIDDEN_DIM = 200
LSTM_LAYERS = 1
LEARNING_RATE = 10e-4
BATCH_SIZE = 10


class Model(nn.Module):
    def __init__(self, n_features, n_outputs):
        super(Model, self).__init__()
        self.lstm = nn.LSTM(n_features, HIDDEN_DIM,
                            LSTM_LAYERS, batch_first=True)
        self.linear1 = nn.Linear(HIDDEN_DIM, 100)
        self.act = nn.Tanh()
        self.linear2 = nn.Linear(100, n_outputs)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        out = self.linear1(lstm_out)
        out = self.act(out)
        out = self.linear2(out)
        return out


def split_dataset(data):
    train, test_dataset = data[1:-499], data[-499:-(SEQUENCE_LEN - 1)]
    train = np.array(split(train, len(train)/SEQUENCE_LEN))
    test = np.array(split(test_dataset, len(test_dataset)/SEQUENCE_LEN))
    return train, test, test_dataset


def to_supervised(data, n_input, n_out=SEQUENCE_LEN):
    data = data.reshape((data.shape[0]*data.shape[1], data.shape[2]))
    X, y = list(), list()
    in_start = 0
    for _ in range(len(data)):
        in_end = in_start + n_input
        out_end = in_end + n_out
        if out_end <= len(data):
            x_input = data[in_start:in_end, :]
            X.append(x_input)
            y.append(data[in_end:out_end, :])
        in_start += 1
    return np.array(X), np.array(y)


def get_dataloader(data_x, data_y, batch=10, shuffle=False):
    x = torch.Tensor(data_x)
    y = torch.Tensor(data_y)
    dataset = TensorDataset(x, y)
    dataloader = DataLoader(
        dataset=dataset, batch_size=batch, shuffle=shuffle)
    return dataloader


def get_model(train, n_input):
    x, y = to_supervised(train, n_input)
    n_timesteps, n_features, n_outputs = x.shape[1], x.shape[2], y.shape[2]
    model = Model(n_features, n_outputs)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    return model, optimizer, loss_fn, x, y


def forecast(model, data, n_input, steps=10):
    data = np.array(data)
    data = data.reshape((data.shape[0]*data.shape[1], data.shape[2]))
    # retrieve last observations for input data
    input_x = data[-n_input:, :]
    # reshape into [1, n_input, n]
    input_x = input_x.reshape((1, input_x.shape[0], input_x.shape[1]))
    # forecast the next week
    predictions = torch.zeros((steps, input_x.shape[2]))
    input_x = torch.Tensor(input_x)
    for i in range(steps):
        model.eval()
        yhat = model(input_x)
        predictions[i] = yhat[0][0]
        temp = input_x[0][1:,:]  # Remove first row
        temp = torch.cat((temp, yhat[0][:1,:]), 0) # Add predicted row
        input_x = temp.reshape((1, temp.shape[0], temp.shape[1]))
    return predictions


def train_model(dataloader, model, optimizer, loss_fn):
    avg_loss = []
    for _ in range(EPOCHS):
        cum_loss = 0
        for x_batch, y_batch in dataloader:
            # print(x_batch.shape)
            model.train()
            yhat = model(x_batch)
            loss = loss_fn(yhat, y_batch)
            cum_loss += loss
            model.zero_grad()
            loss.backward()
            optimizer.step()
            # optimizer.zero_grad()
        print(cum_loss / len(dataloader))
        avg_loss.append(cum_loss / len(dataloader))
    return avg_loss


def evaluate_predictions(data, predictions):
    fig, axs = plt.subplots(3, 2)
    plot_axis = [(0,0), (0,1), (1,0), (1,1), (2,0), (2,1)]
    plots = ["v_leader", "x_leader",
                    "v_follower", "x_follower", "v_human", "x_human"]
    for i, (x,y) in enumerate(plot_axis):
        axs[x, y].plot(data[:, i], label="Data")
        axs[x, y].plot(predictions[:, i], label="Predicted")
        axs[x,y].set_title(plots[i])
    plt.legend()
    plt.show()


if __name__ == "__main__":
    data = Data.DataPrep()
    dataset = data.get_data()
    train, test, test_dataset = split_dataset(dataset.values)
    model, optimizer, loss_fn, train_x, train_y = get_model(
        train, SEQUENCE_LEN)
    train_dataloader = get_dataloader(train_x, train_y, BATCH_SIZE)
    train_loss = train_model(train_dataloader, model, optimizer, loss_fn)
    prediction_steps = 300
    predictions = forecast(model, train, SEQUENCE_LEN, prediction_steps)
    evaluate_predictions(test_dataset[:prediction_steps], predictions.detach().numpy())
    # test_x, test_x = to_supervised(test, SEQUENCE_LEN)
    # Pass one sequence and generate dataloader

    # test_dataloader = get_dataloader(test_x, test_x, BATCH_SIZE)

    # plt.plot(train_loss)
    # plt.show()
