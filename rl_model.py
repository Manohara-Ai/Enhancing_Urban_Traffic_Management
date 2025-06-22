import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
from rl_env import SignalPlan

class Linear_QNet(nn.Module):

    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        # Define the first linear layer with input size and hidden size
        self.linear1 = nn.Linear(input_size, hidden_size)
        # Define the second linear layer with hidden size and output size
        self.linear2 = nn.Linear(hidden_size, output_size)


    def forward(self, x):
        # Apply the first linear transformation followed by ReLU activation
        x = F.relu(self.linear1(x))
        # Apply the second linear transformation to produce the output
        x = self.linear2(x)
        return x


    def save(self, file_name='model.pth'):
        # Create a directory to save the model if it doesn't exist
        model_folder_path = './Model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        # Save the model state dictionary to the specified file
        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)


class QTrainer:

    def __init__(self, model, lr, gamma):
        # Learning rate for the optimizer
        self.lr = lr
        # Discount factor for future rewards
        self.gamma = gamma
        # Initialize the model
        self.model = model
        # Use Adam optimizer for training
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        # Mean Squared Error loss for training
        self.criterion = nn.MSELoss()

    def get_action_indices(self, action: SignalPlan, num_lanes: int):
        indices = []

        for i, signal in enumerate(action.signals):
            base = i * 3
            for j, v in enumerate(signal):  # j = 0 (L), 1 (S), 2 (R)
                if v == 1:
                    indices.append(base + j)

        # duration is the last output
        indices.append(3 * num_lanes)
        return indices

    def train_step(self, state, action, reward, next_state, done):
        state = torch.tensor(state, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        reward = torch.tensor(reward, dtype=torch.float)

        # If action is a structured SignalPlan, no need to convert to tensor
        if len(state.shape) == 1:
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done,)
            action = [action]  # wrap SignalPlan in list

        pred = self.model(state)
        target = pred.clone()

        for idx in range(len(done)):
            Q_new = reward[idx]
            if not done[idx]:
                Q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx]))

            # Update all indices corresponding to the structured action
            action_indices = self.get_action_indices(action[idx], num_lanes=len([f for f in os.listdir("train_simulation") if os.path.isfile(os.path.join("train_simulation", f))]))  # Adjust num_lanes as needed
            for i in action_indices:
                target[idx][i] = Q_new

        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()
        self.optimizer.step()