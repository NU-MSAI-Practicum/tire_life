# train.py

import os
import torch
import pandas as pd
import numpy as np
from env import TruckFleetEnv
from dqn.dqn_v2 import DQNAgent
from copy import deepcopy
from utils import plot_rewards, plot_losses

def train(logs_folder, metrics_folder, model_folder):
    num_episodes = 25000
    epsilon_start = 1.0
    epsilon_end = 0.001
    epsilon_decay = 3000
    gamma = 0.1
    lr = 0.0005
    batch_size = 128
    memory_capacity = 500000

    num_trucks = 2
    num_tires_per_truck = 10
    health_threshold = 0.09
    max_steps = 20

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    env = TruckFleetEnv(num_trucks, num_tires_per_truck, health_threshold, max_steps)
    state_dim = env.num_trucks * env.num_tires_per_truck
    action_dims = {
        "replace": [num_trucks, num_tires_per_truck],
        "swap": [num_trucks, num_trucks, 2, num_tires_per_truck - 2]
    }

    agent = DQNAgent(state_dim, action_dims, gamma, lr, batch_size, memory_capacity, device)

    all_action_logs = []
    rewards = []
    losses = []
    episode_data = []

    csv_file_path = os.path.join(metrics_folder, 'episode_timestep_reward.csv')
    if not os.path.exists(csv_file_path):
        # Create the CSV file and write the header if it doesn't exist
        with open(csv_file_path, mode='w') as f:
            f.write("Episode,Timestep,Reward\n")

    for episode in range(num_episodes):
        initial_state = deepcopy(env.reset())
        state = initial_state.flatten()
        state = torch.tensor(state, dtype=torch.float32).to(device)
        total_reward = 0
        episode_losses = []

        for t in range(max_steps):
            epsilon = epsilon_end + (epsilon_start - epsilon_end) * np.exp(-episode / epsilon_decay)
            action = agent.select_action(state, epsilon)
            next_state, reward, done, _ = env.step(action)
            next_state = next_state.flatten()
            next_state = torch.tensor(next_state, dtype=torch.float32).to(device)

            agent.memory.push(state.cpu(), action, next_state.cpu(), reward)
            state = next_state
            total_reward += reward

            loss = agent.optimize_model()
            if loss is not None:
                episode_losses.append(loss)

            if done:
                break


        episode_data.append([episode, t + 1, total_reward])  # Collect episode-timestep-reward data

        # Write episode data to CSV after each episode
        with open(csv_file_path, mode='a') as f:
            f.write(f"{episode},{t + 1},{total_reward}\n")

        if episode % 10 == 0:
            agent.update_target_net()

        print(f"Episode {episode} - Total reward: {total_reward} - TimeStep {t + 1}")
        action_logs = env.get_action_log()
        final_state = deepcopy(env.state)

        episode_log = {
            'Episode': episode,
            'Initial State': initial_state,
            'Final State': final_state,
            'Total Reward': total_reward,
            'Action Logs': action_logs,
            'Number of Actions': [len(log) for log in action_logs]
        }
        all_action_logs.append(episode_log)

        episode_df = pd.DataFrame({
            'Truck': [f'Truck {i}' for i in range(env.num_trucks)],
            'Initial State': [initial_state[i] for i in range(env.num_trucks)],
            'Final State': [final_state[i] for i in range(env.num_trucks)],
            'Number of Actions': [len(action_logs[i]) for i in range(env.num_trucks)],
            'Actions': [action_logs[i] for i in range(env.num_trucks)]
        })
        episode_df.to_excel(os.path.join(logs_folder, f'episode_{episode}_log.xlsx'), index=False)

        rewards.append(total_reward)
        if episode_losses:
            losses.append(np.mean(episode_losses))

    plot_rewards(rewards, os.path.join(metrics_folder, 'rewards_plot.png'))
    plot_losses(losses, os.path.join(metrics_folder, 'losses_plot.png'))

    torch.save(agent.policy_net.state_dict(), os.path.join(model_folder, 'dqn_truck_fleet.pth'))
    print("Training completed and model saved.")
