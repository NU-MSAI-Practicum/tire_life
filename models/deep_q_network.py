import gym
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque

class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )

    def forward(self, x):
        return self.net(x)

def select_action(state, epsilon, action_space, policy_net):
    if random.random() > epsilon:
        with torch.no_grad():
            return policy_net(state).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[random.randrange(action_space)]], dtype=torch.long)

def optimize_model(memory, batch_size, policy_net, target_net, optimizer):
    if len(memory) < batch_size:
        return
    transitions = random.sample(memory, batch_size)
    batch = list(zip(*transitions))

    states = torch.cat(batch[0])
    actions = torch.cat(batch[1])
    rewards = torch.cat(batch[2])
    next_states = torch.cat(batch[3])
    dones = torch.cat(batch[4])

    current_q_values = policy_net(states).gather(1, actions)
    next_q_values = target_net(next_states).max(1)[0].detach()
    expected_q_values = (next_q_values * 0.99) * (1 - dones) + rewards

    loss = nn.MSELoss()(current_q_values, expected_q_values.unsqueeze(1))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

def main():
    env = gym.make('CartPole-v1')
    input_dim = env.observation_space.shape[0]
    output_dim = env.action_space.n
    
    policy_net = DQN(input_dim, output_dim)
    target_net = DQN(input_dim, output_dim)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters())
    memory = deque(maxlen=10000)
    episodes = 500
    batch_size = 64
    epsilon = 0.9
    epsilon_decay = 0.995
    min_epsilon = 0.05

    for episode in range(episodes):
        state = torch.tensor([env.reset()], dtype=torch.float)
        for t in range(1000):
            action = select_action(state, epsilon, output_dim, policy_net)
            next_state, reward, done, _ = env.step(action.item())
            reward = torch.tensor([reward], dtype=torch.float)
            next_state = torch.tensor([next_state], dtype=torch.float)
            done = torch.tensor([done], dtype=torch.float)
            
            memory.append((state, action, reward, next_state, done))
            state = next_state
            
            optimize_model(memory, batch_size, policy_net, target_net, optimizer)
            
            if done:
                break
        
        epsilon = max(min_epsilon, epsilon * epsilon_decay)
        if episode % 50 == 0:
            target_net.load_state_dict(policy_net.state_dict())

    env.close()

if __name__ == "__main__":
    main()
