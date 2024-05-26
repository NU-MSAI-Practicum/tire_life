import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import numpy as np

# DQN network
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)  # Increased the number of neurons
        self.fc2 = nn.Linear(256, 256)        # Increased the number of neurons
        self.fc3 = nn.Linear(256, 128)        # Added an extra layer
        self.fc4 = nn.Linear(128, 128)        # Added an extra layer
        self.fc5 = nn.Linear(128, output_dim) # Final output layer

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        x = self.fc5(x)
        return x

# Replay memory
class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)

    def push(self, state, action, next_state, reward):
        self.memory.append((state, action, next_state, reward))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

# DQN agent
class DQNAgent:
    def __init__(self, state_dim, action_dims, gamma=0.99, lr=0.001, batch_size=64, memory_capacity=10000):
        self.state_dim = state_dim
        self.action_dims = action_dims
        self.gamma = gamma
        self.batch_size = batch_size

        self.policy_net = DQN(state_dim, self.calculate_output_dim(action_dims))
        self.target_net = DQN(state_dim, self.calculate_output_dim(action_dims))
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.memory = ReplayMemory(memory_capacity)

        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
    
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
            with torch.no_grad():
                q_values = self.policy_net(torch.FloatTensor(state))
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
            return

        batch = self.memory.sample(self.batch_size)
        state_batch, action_batch, next_state_batch, reward_batch = zip(*batch)

        state_batch = torch.FloatTensor(np.array(state_batch))
        next_state_batch = torch.FloatTensor(np.array(next_state_batch))
        reward_batch = torch.FloatTensor(reward_batch)

        # Convert actions to indices and handle different action types
        action_indices = []
        for action in action_batch:
            action_type = int(action[0])
            if action_type == 0:
                index = np.ravel_multi_index(action[1:], self.action_dims["replace"])
            else:
                swap_index = np.ravel_multi_index(action[1:], self.action_dims["swap"])
                index = np.prod(self.action_dims["replace"]) + swap_index
            action_indices.append(index)
        
        action_indices = torch.LongTensor(action_indices)

        current_q_values = self.policy_net(state_batch).view(-1, self.calculate_output_dim(self.action_dims))
        current_q_values = current_q_values.gather(1, action_indices.view(-1, 1)).squeeze()

        max_next_q_values = self.target_net(next_state_batch).view(-1, self.calculate_output_dim(self.action_dims)).max(1)[0]
        expected_q_values = reward_batch + self.gamma * max_next_q_values

        loss = nn.MSELoss()(current_q_values, expected_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
    
    def action_to_index(self, actions):
        indices = []
        for action in actions:
            action_type = int(action[0])
            if action_type == 0:
                index = np.ravel_multi_index(action[1:], self.action_dims["replace"])
            else:
                swap_index = np.ravel_multi_index(action[1:], self.action_dims["swap"])
                index = np.prod(self.action_dims["replace"]) + swap_index
            indices.append(index)
        return indices

    def update_target_net(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())