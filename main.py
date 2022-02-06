from data_prep import data, DataPrep
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, TensorDataset
import matplotlib.pyplot as plt


device = 'cuda' if torch.cuda.is_available() else 'cpu'
hidden_size = 64
epochs = 20
# model = nn.Sequential(
#     nn.Linear(4, 15),
#     nn.ReLU(),
#     nn.Linear(15,15),
#     nn.ReLU(),
#     nn.Linear(15, 2),
# )

model = nn.Sequential(
    nn.Linear(6, hidden_size), nn.Tanh(),
    nn.Linear(hidden_size, hidden_size), nn.Tanh(),
    nn.Linear(hidden_size, 6))

loss_fn = torch.nn.MSELoss()
learning_rate = 1e-4
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


def get_dataloader(data_x, data_y, batch=1199, shuffle=False):
    x = torch.Tensor(data_x.values)
    y = torch.Tensor(data_y.values)
    dataset = TensorDataset(x, y)
    dataloader = DataLoader(
        dataset=dataset, batch_size=batch, shuffle=shuffle)
    return dataloader


def train(x_batch, y_batch):
    y_pred = model(x_batch)
    loss = loss_fn(y_pred, y_batch)
    model.zero_grad()
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    return loss.item()


def main():
    d = DataPrep()
    x, y = d.get_data(train=True, validation=False)
    train_loader = get_dataloader(x, y)
    val_x, val_y = d.get_data(train=False, validation=True)
    val_loader = get_dataloader(val_x, val_y)

    all_losses = {"train": [], "validation": []}

    for epoch in range(epochs):
        print(f"\nTraining... Epoch:{epoch}")
        losses = []
        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            loss = train(x_batch, y_batch)
            losses.append(loss)
        avg = sum(losses) / len(losses)
        all_losses["train"].append(avg)
        print(f"{epoch + 1} epochs, avg. loss {avg}")

        print("Validating...")
        with torch.no_grad():
            val_losses = []
            for x_val, y_val in val_loader:
                x_val = x_val.to(device)
                y_val = y_val.to(device)

                model.eval()

                y_pred = model(x_val)
                val_loss = loss_fn(y_pred, y_val)
                val_losses.append(val_loss)
            
            avg = sum(val_losses) / len(val_losses)
            all_losses["validation"].append(avg)
            print(f"Avg. Validation loss {avg}")

    print("\nTesting...")
    x_test, y_test = d.get_data(train=False, validation=False)
    test_loader = get_dataloader(
        x_test, y_test, batch=len(x_test), shuffle=True)
    test_loss = 0
    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            model.eval()

            y_pred = model(x_batch)
            test_loss = loss_fn(y_pred, y_batch)
    print("Test Loss = ", test_loss)
    plt.plot(all_losses["train"], label="Avg. training loss")
    plt.plot(all_losses["validation"], label="Avg. validation loss")
    # plt.plot(epochs, test_loss, label="Test Loss")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
