import os
import numpy as np
import pandas as pd
import random
from collections import deque
from copy import deepcopy

import torch
import torch.nn as nn
import torch.optim as optim

from dqn import DQN

def train_dqn(env, summary_file, episodes=1000, gamma=0.99, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995, batch_size=64, target_update_freq=10, max_steps_per_episode=200):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    state_dim = env.observation_space.shape[0] * env.observation_space.shape[1]
    action_dim = np.prod(env.action_space.nvec)  # Calculate the total number of possible actions
    q_network = DQN(state_dim, action_dim).to(device)
    target_network = DQN(state_dim, action_dim).to(device)
    target_network.load_state_dict(q_network.state_dict())
    optimizer = optim.Adam(q_network.parameters())
    criterion = nn.MSELoss()
    memory = deque(maxlen=2000)

    summary_logs = []

    print(f"{'Episode Number':<15} {'Total Reward':<15} {'Optimal State':<15} {'Total Actions':<15}")

    for episode in range(1, episodes + 1):
        state = env.reset().flatten()
        initial_state = deepcopy(state)  # Capture the initial state right after reset
        total_reward = 0
        done = False
        total_actions = 0  # Track the total number of actions for the episode
        step_count = 0  # Track the number of steps within the episode

        while not done and step_count < max_steps_per_episode:
            step_count += 1
            total_actions += 1
            if np.random.rand() <= epsilon:
                action = [np.random.randint(dim) for dim in env.action_space.nvec]
            else:
                with torch.no_grad():
                    q_values = q_network(torch.FloatTensor(state).unsqueeze(0).to(device))
                action_idx = np.argmax(q_values.cpu().numpy())
                action = np.unravel_index(action_idx, env.action_space.nvec)

            next_state, reward, done, _ = env.step(action)
            next_state = next_state.flatten()
            memory.append((state, action, reward, next_state, done))
            state = next_state
            total_reward += reward

            if len(memory) >= batch_size:
                batch = random.sample(memory, batch_size)
                states, actions, rewards, next_states, dones = zip(*batch)
                states = torch.FloatTensor(np.array(states)).to(device)  # Convert list of arrays to a single numpy array
                actions = torch.LongTensor([np.ravel_multi_index(action, env.action_space.nvec) for action in actions]).to(device)
                rewards = torch.FloatTensor(rewards).to(device)
                next_states = torch.FloatTensor(np.array(next_states)).to(device)  # Convert list of arrays to a single numpy array
                dones = torch.FloatTensor(dones).to(device)

                q_values = q_network(states)
                next_q_values = target_network(next_states)
                target_q_values = rewards + gamma * next_q_values.max(1)[0] * (1 - dones)

                loss = criterion(q_values.gather(1, actions.unsqueeze(1)).squeeze(), target_q_values)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        epsilon = max(epsilon_min, epsilon * epsilon_decay)

        if episode % target_update_freq == 0:
            target_network.load_state_dict(q_network.state_dict())

        # Save logs
        env.initial_state = np.copy(initial_state).reshape(env.state.shape)  # Ensure initial state is in correct shape
        env.save_logs(episode)

        print(f"{episode:<15} {total_reward:<15} {env.is_optimal_state():<15} {total_actions:<15}")
        
        # Log summary for the episode
        summary_logs.append({
            "Episode Number": episode,
            "Total Reward": total_reward,
            "Optimal State Reached?": env.is_optimal_state(),
            "Total Actions": total_actions
        })

    # Save the trained model
    torch.save(q_network.state_dict(), os.path.join(os.path.dirname(summary_file), 'dqn_model.pth'))
    
    # Save the summary logs to a CSV file
    summary_df = pd.DataFrame(summary_logs)
    summary_df.to_csv(summary_file, index=False)

def evaluate_agent(env, q_network, episodes=100):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    total_rewards = []
    for episode in range(episodes):
        state = env.reset().flatten()
        total_reward = 0
        done = False
        while not done:
            with torch.no_grad():
                q_values = q_network(torch.FloatTensor(state).unsqueeze(0).to(device))
            action_idx = np.argmax(q_values.cpu().numpy())
            action = np.unravel_index(action_idx, env.action_space.nvec)
            next_state, reward, done, _ = env.step(action)
            state = next_state.flatten()
            total_reward += reward
        total_rewards.append(total_reward)
        print(f"Evaluation Episode {episode + 1}: Total Reward: {total_reward}")
    
    avg_reward = np.mean(total_rewards)
    print(f"Average Reward over {episodes} Evaluation Episodes: {avg_reward}")
    return avg_reward
