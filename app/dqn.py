import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import numpy as np
import pandas as pd

class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.fc3 = nn.Linear(256, 128)
        self.bn3 = nn.BatchNorm1d(128)
        self.fc4 = nn.Linear(128, 128)
        self.bn4 = nn.BatchNorm1d(128)
        self.fc5 = nn.Linear(128, output_dim)

    def forward(self, x):
        x = torch.relu(self.bn1(self.fc1(x)))
        x = torch.relu(self.bn2(self.fc2(x)))
        x = torch.relu(self.bn3(self.fc3(x)))
        x = torch.relu(self.bn4(self.fc4(x)))
        x = self.fc5(x)
        return x

class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)

    def push(self, state, action, next_state, reward):
        self.memory.append((state, action, next_state, reward))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class DQNAgent:
    def __init__(self, state_dim, action_dims, gamma=0.99, lr=0.001, batch_size=64, memory_capacity=10000, device='cpu'):
        self.state_dim = state_dim
        self.action_dims = action_dims
        self.gamma = gamma
        self.batch_size = batch_size
        self.device = device

        self.policy_net = DQN(state_dim, self.calculate_output_dim(action_dims)).to(self.device)
        self.target_net = DQN(state_dim, self.calculate_output_dim(action_dims)).to(self.device)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.memory = ReplayMemory(memory_capacity)

        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01

    def calculate_output_dim(self, action_dims):
        replace_actions = np.prod(action_dims["replace"])
        swap_actions = np.prod(action_dims["swap"])
        return replace_actions + swap_actions

    def select_action(self, state, epsilon):
        if random.random() < epsilon:
            if random.random() < 0.5:
                action_type = 0
                replace_action = np.array([random.randrange(dim) for dim in self.action_dims["replace"]])
                return np.concatenate(([action_type], replace_action))
            else:
                action_type = 1
                swap_action = np.array([random.randrange(dim) for dim in self.action_dims["swap"]])
                return np.concatenate(([action_type], swap_action))
        else:
            self.policy_net.eval()  # Set the network to evaluation mode
            with torch.no_grad():
                if state.dim() == 1:
                    state = state.unsqueeze(0)  # Add batch dimension
                q_values = self.policy_net(state)
            self.policy_net.train()  # Set the network back to training mode
            action_index = torch.argmax(q_values).item()
            if action_index < np.prod(self.action_dims["replace"]):
                action_type = 0
                replace_action = np.unravel_index(action_index, self.action_dims["replace"])
                return np.concatenate(([action_type], replace_action))
            else:
                action_type = 1
                swap_index = action_index - np.prod(self.action_dims["replace"])
                swap_action = np.unravel_index(swap_index, self.action_dims["swap"])
                return np.concatenate(([action_type], swap_action))



    def optimize_model(self):
        if len(self.memory) < self.batch_size:
            return None

        batch = self.memory.sample(self.batch_size)
        state_batch, action_batch, next_state_batch, reward_batch = zip(*batch)

        state_batch = torch.FloatTensor(np.array(state_batch)).to(self.device)
        next_state_batch = torch.FloatTensor(np.array(next_state_batch)).to(self.device)
        reward_batch = torch.FloatTensor(reward_batch).to(self.device)

        action_indices = []
        for action in action_batch:
            action_type = int(action[0])
            if action_type == 0:
                index = np.ravel_multi_index(action[1:], self.action_dims["replace"])
            else:
                swap_index = np.ravel_multi_index(action[1:], self.action_dims["swap"])
                index = np.prod(self.action_dims["replace"]) + swap_index
            action_indices.append(index)

        action_indices = torch.LongTensor(action_indices).to(self.device)

        current_q_values = self.policy_net(state_batch).view(-1, self.calculate_output_dim(self.action_dims))
        current_q_values = current_q_values.gather(1, action_indices.view(-1, 1)).squeeze()

        with torch.no_grad():
            next_state_actions = self.policy_net(next_state_batch).view(-1, self.calculate_output_dim(self.action_dims)).argmax(1)
            max_next_q_values = self.target_net(next_state_batch).view(-1, self.calculate_output_dim(self.action_dims)).gather(1, next_state_actions.view(-1, 1)).squeeze()
            expected_q_values = reward_batch + self.gamma * max_next_q_values

        loss = nn.SmoothL1Loss()(current_q_values, expected_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        self.optimizer.step()

        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        return loss.item()

    def update_target_net(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def predict(self, state):
        self.policy_net.eval()
        with torch.no_grad():
            if state.dim() == 1:
                state = state.unsqueeze(0)  # Add batch dimension
            q_values = self.policy_net(state)
        self.policy_net.train()
        action_index = torch.argmax(q_values).item()
        if action_index < np.prod(self.action_dims["replace"]):
            action_type = 0
            replace_action = np.unravel_index(action_index, self.action_dims["replace"])
            return np.concatenate(([action_type], replace_action))
        else:
            action_type = 1
            swap_index = action_index - np.prod(self.action_dims["replace"])
            swap_action = np.unravel_index(swap_index, self.action_dims["swap"])
            return np.concatenate(([action_type], swap_action))

    def generate_predictions(self, env, initial_state, num_steps):
        state = torch.tensor(initial_state.flatten(), dtype=torch.float32).to(self.device)
        predictions = []

        for _ in range(num_steps):
            action = self.predict(state)
            next_state, reward, done, _ = env.step(action)
            action_log = self.get_human_readable_action_log(action, reward)
            predictions.append({
                "State": state.cpu().numpy(),
                "Action": action,
                "Next State": next_state.flatten(),
                "Reward": reward,
                "Log": action_log
            })
            state = torch.tensor(next_state.flatten(), dtype=torch.float32).to(self.device)
            if done:
                break

        return predictions

    def get_human_readable_action_log(self, action, reward):
        action_type = "Replace" if action[0] == 0 else "Swap"
        if action[0] == 0:
            truck_idx, tire_idx = action[1], action[2]
            log = f"{action_type} action on Truck {truck_idx} Tire {tire_idx} with reward {reward}."
        else:
            truck_idx, other_truck_idx, tire_idx, other_tire_idx = action[1], action[2], action[3], action[4]
            log = f"{action_type} action between Truck {truck_idx} Tire {tire_idx} and Truck {other_truck_idx} Tire {other_tire_idx} with reward {reward}."
        return log

    def save_predictions_to_excel(self, predictions, file_path):
        df = pd.DataFrame(predictions)
        df.to_excel(file_path, index=False)

    def save_detailed_log_to_excel(self, initial_state, predictions, file_path):
        detailed_log = []
        for prediction in predictions:
            for truck_idx in range(len(initial_state)):
                log_entry = {
                    "Truck": truck_idx,
                    "Initial State": initial_state[truck_idx],
                    "Action": prediction["Log"],
                    "Next State": prediction["Next State"][truck_idx],
                    "Reward": prediction["Reward"]
                }
                detailed_log.append(log_entry)

        df = pd.DataFrame(detailed_log)
        df.to_excel(file_path, index=False)
        
    def save_detailed_log_to_dict(self, initial_state, predictions):
        detailed_log = []
        for prediction in predictions:
            for truck_idx in range(len(initial_state)):
                log_entry = {
                    "Truck": truck_idx,
                    "Initial State": initial_state[truck_idx].tolist(),
                    "Action": prediction["Log"],
                    "Next State": prediction["Next State"][truck_idx].tolist(),
                    "Reward": prediction["Reward"]
                }
                detailed_log.append(log_entry)
        return detailed_log