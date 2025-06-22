import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# LSTM Model Definition
class LSTMPredictor(nn.Module):
    def __init__(self, n_hidden=64):
        super().__init__()
        self.n_hidden = n_hidden
        self.lstm1 = nn.LSTMCell(1, self.n_hidden)
        self.lstm2 = nn.LSTMCell(self.n_hidden, self.n_hidden)
        self.linear = nn.Linear(self.n_hidden, 1)

    def forward(self, x, future=0):
        outputs = []
        n_samples = x.size(0)

        h_t = torch.zeros(n_samples, self.n_hidden, dtype=torch.float32, device=x.device)
        c_t = torch.zeros(n_samples, self.n_hidden, dtype=torch.float32, device=x.device)
        h_t2 = torch.zeros(n_samples, self.n_hidden, dtype=torch.float32, device=x.device)
        c_t2 = torch.zeros(n_samples, self.n_hidden, dtype=torch.float32, device=x.device)

        for input_t in x.split(1, dim=1):
            h_t, c_t = self.lstm1(input_t, (h_t, c_t))
            h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))
            output = self.linear(h_t2)
            outputs.append(output)

        for _ in range(future):
            h_t, c_t = self.lstm1(output, (h_t, c_t))
            h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))
            output = self.linear(h_t2)
            outputs.append(output)

        return torch.cat(outputs, dim=1)


# === Utility Functions ===

def load_and_preprocess_data(dataframe: pd.DataFrame, time_instant):
    # 🔹 Filter rows based on Entry Time
    df = dataframe[dataframe["Entry Time"] <= time_instant].copy()

    # 🔹 Drop the Entry Time column and convert to float
    data = df.drop(columns=["Entry Time"]).values.astype(np.float32)
    data = data.T  # Shape: (num_lanes, num_data_points)

    # 🔹 Normalize
    mean = data.mean(axis=1, keepdims=True)
    std = data.std(axis=1, keepdims=True)
    data_norm = (data - mean) / std

    # 🔹 Prepare x and y
    x = data_norm[:, :-1]
    y = data_norm[:, 1:]

    return torch.from_numpy(x), torch.from_numpy(y), mean, std


def train_model(model, train_input, train_target, n_steps=10, lr=1e-4):
    criterion = nn.MSELoss()
    optimizer = optim.LBFGS(model.parameters(), lr=lr)

    def closure():
        optimizer.zero_grad()
        out = model(train_input)
        loss = criterion(out, train_target)
        loss.backward()
        return loss

    train_loss = []
    for step in range(n_steps):
        loss = optimizer.step(closure)
        train_loss.append(loss.item())
        print(f"Step {step + 1}/{n_steps}, Train Loss: {loss.item():.6f}")

    return model, train_loss


def predict(model, train_input, mean, std, future=60):
    with torch.no_grad():
        pred = model(train_input, future=future)
        pred = pred.cpu().numpy()
        pred_denorm = pred * std + mean
    return pred_denorm


def plot_predictions(pred_denorm, future, num_lanes):
    plt.figure(figsize=(16, 10))
    lane_names = [f"Lane {i + 1}" for i in range(num_lanes)]
    colors = ['r', 'g', 'b', 'm', 'c', 'y', 'k'][:num_lanes]

    n = pred_denorm.shape[1] - future
    rows = (num_lanes + 1) // 2
    cols = 2

    for i in range(num_lanes):
        plt.subplot(rows, cols, i + 1)
        plt.title(lane_names[i])
        plt.xlabel("Time (minutes)")
        plt.ylabel("Vehicle Count")
        plt.plot(np.arange(n), pred_denorm[i, :n], colors[i], label="Past")
        plt.plot(np.arange(n, n + future), pred_denorm[i, n:], colors[i] + ":", label="Predicted")
        plt.grid(True)
        plt.legend()

    plt.tight_layout()
    plt.show()

def main(dataframe: pd.DataFrame, time_instant, future):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_input, train_target, mean, std = load_and_preprocess_data(dataframe, time_instant) 

    train_input = train_input.to(device)
    train_target = train_target.to(device)

    model = LSTMPredictor().to(device)
    model, _ = train_model(model, train_input, train_target)

    pred_denorm = predict(model, train_input, mean, std, future=future)

    # num_lanes = pred_denorm.shape[0]
    # plot_predictions(pred_denorm, future, num_lanes)

    return pred_denorm
