import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import numpy as np

# Dueling DQN network
class DuelingDQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DuelingDQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        
        # Dueling streams
        self.value_stream = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        self.advantage_stream = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))

        value = self.value_stream(x)
        advantage = self.advantage_stream(x)

        # Ensure the advantage mean calculation is on the correct dimension
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        return q_values

# Prioritized Replay Memory
class PrioritizedReplayMemory:
    def __init__(self, capacity, alpha=0.6):
        self.capacity = capacity
        self.alpha = alpha
        self.memory = deque(maxlen=capacity)
        self.priorities = deque(maxlen=capacity)

    def push(self, state, action, next_state, reward):
        max_priority = max(self.priorities, default=1.0)
        self.memory.append((state, action, next_state, reward))
        self.priorities.append(max_priority)

    def sample(self, batch_size, beta=0.4):
        if len(self.memory) == 0:
            return [], [], []

        priorities = np.array(self.priorities)
        if len(priorities) != len(self.memory):
            priorities = priorities[:len(self.memory)]

        probabilities = priorities ** self.alpha
        probabilities /= probabilities.sum()

        indices = np.random.choice(len(self.memory), batch_size, p=probabilities)
        samples = [self.memory[idx] for idx in indices]

        total = len(self.memory)
        weights = (total * probabilities[indices]) ** (-beta)
        weights /= weights.max()
        weights = torch.FloatTensor(weights)

        return samples, weights, indices

    def update_priorities(self, indices, priorities):
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority

    def __len__(self):
        return len(self.memory)

# DQN Agent
class DQNAgent:
    def __init__(self, state_dim, action_dims, gamma=0.99, lr=0.001, batch_size=64, memory_capacity=10000, alpha=0.6, beta_start=0.4, beta_frames=10000):
        self.state_dim = state_dim
        self.action_dims = action_dims
        self.gamma = gamma
        self.batch_size = batch_size
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.beta = beta_start

        self.policy_net = DuelingDQN(state_dim, self.calculate_output_dim(action_dims))
        self.target_net = DuelingDQN(state_dim, self.calculate_output_dim(action_dims))
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.memory = PrioritizedReplayMemory(memory_capacity, alpha)

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
                state = torch.FloatTensor(state).unsqueeze(0)  # Ensure state is 2D
                q_values = self.policy_net(state)
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

    def optimize_model(self, frame_idx):
        if len(self.memory) < self.batch_size:
            return

        beta = min(1.0, self.beta_start + frame_idx * (1.0 - self.beta_start) / self.beta_frames)
        batch, weights, indices = self.memory.sample(self.batch_size, beta)

        if not batch:  # Skip if batch is empty
            return

        state_batch, action_batch, next_state_batch, reward_batch = zip(*batch)

        state_batch = torch.FloatTensor(np.array(state_batch))
        next_state_batch = torch.FloatTensor(np.array(next_state_batch))
        reward_batch = torch.FloatTensor(reward_batch)
        weights = torch.FloatTensor(weights)

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

        with torch.no_grad():
            next_state_actions = self.policy_net(next_state_batch).view(-1, self.calculate_output_dim(self.action_dims)).argmax(1)
            max_next_q_values = self.target_net(next_state_batch).view(-1, self.calculate_output_dim(self.action_dims)).gather(1, next_state_actions.view(-1, 1)).squeeze()
            expected_q_values = reward_batch + self.gamma * max_next_q_values

        td_errors = (current_q_values - expected_q_values).abs().detach().numpy()
        self.memory.update_priorities(indices, td_errors + 1e-5)

        loss = (weights * nn.MSELoss(reduction='none')(current_q_values, expected_q_values)).mean()

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
