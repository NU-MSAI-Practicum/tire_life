import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque, namedtuple
import numpy as np

# Replay memory with Prioritized Experience Replay
class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)
        self.priorities = deque(maxlen=capacity)
        self.capacity = capacity

    def push(self, state, action, next_state, reward, td_error):
        self.memory.append((state, action, next_state, reward))
        self.priorities.append(td_error)

    def sample(self, batch_size, alpha=0.6):
        priorities = np.array(self.priorities) ** alpha
        probabilities = priorities / np.sum(priorities)
        indices = np.random.choice(len(self.memory), batch_size, p=probabilities)
        batch = [self.memory[i] for i in indices]
        return batch, indices

    def update_priorities(self, indices, td_errors):
        for idx, td_error in zip(indices, td_errors):
            self.priorities[idx] = td_error

    def __len__(self):
        return len(self.memory)

class DuelingDQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DuelingDQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc_value = nn.Linear(128, 128)
        self.fc_adv = nn.Linear(128, 128)
        self.value = nn.Linear(128, 1)
        self.advantage = nn.Linear(128, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        value = torch.relu(self.fc_value(x))
        adv = torch.relu(self.fc_adv(x))
        value = self.value(value)
        advantage = self.advantage(adv)
        
        # print(f"value shape: {value.shape}, advantage shape: {advantage.shape}")
        
        # Ensure value and advantage have the correct dimensions
        if advantage.dim() == 1:
            advantage = advantage.unsqueeze(0)
        return value + (advantage - advantage.mean(dim=1, keepdim=True))


# DQN agent
class DQNAgent:
    def __init__(self, state_dim, action_dims, gamma=0.99, lr=0.001, batch_size=64, memory_capacity=10000):
        self.state_dim = state_dim
        self.action_dims = action_dims
        self.gamma = gamma
        self.batch_size = batch_size

        self.policy_net = DuelingDQN(state_dim, np.prod(action_dims))
        self.target_net = DuelingDQN(state_dim, np.prod(action_dims))
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.memory = ReplayMemory(memory_capacity)

        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

    def select_action(self, state, epsilon):
        if random.random() < epsilon:
            return np.array([random.randrange(dim) for dim in self.action_dims])  # Random action
        else:
            with torch.no_grad():
                q_values = self.policy_net(torch.FloatTensor(state)).reshape(self.action_dims)
                action_index = torch.argmax(q_values).item()
                return np.unravel_index(action_index, self.action_dims)

    def optimize_model(self):
        if len(self.memory) < self.batch_size:
            return

        batch, indices = self.memory.sample(self.batch_size)
        state_batch, action_batch, next_state_batch, reward_batch = zip(*batch)

        state_batch = torch.FloatTensor(np.array(state_batch))
        action_batch = np.array(action_batch)
        next_state_batch = torch.FloatTensor(np.array(next_state_batch))
        reward_batch = torch.FloatTensor(reward_batch)

        current_q_values = self.policy_net(state_batch).view(-1, np.prod(self.action_dims))
        action_indices = np.ravel_multi_index(action_batch.T, self.action_dims)
        current_q_values = current_q_values.gather(1, torch.LongTensor(action_indices).view(-1, 1)).squeeze()

        with torch.no_grad():
            max_next_q_values = self.target_net(next_state_batch).view(-1, np.prod(self.action_dims)).max(1)[0]

        expected_q_values = reward_batch + self.gamma * max_next_q_values

        td_errors = (expected_q_values - current_q_values).abs().detach().cpu().numpy()
        self.memory.update_priorities(indices, td_errors)

        loss = nn.MSELoss()(current_q_values, expected_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1)  # Gradient clipping
        self.optimizer.step()

    def update_target_net(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())
