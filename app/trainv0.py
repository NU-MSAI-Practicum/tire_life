import os
import torch
import pandas as pd
import numpy as np
from env import TruckFleetEnv
# from dqn_agent import DQNAgent
from dqn_agent import DQNAgent

from datetime import datetime
from copy import deepcopy
import matplotlib.pyplot as plt


def train(logs_folder):

    # Training loop
    num_episodes = 1000
    epsilon_start = 0.5
    epsilon_end = 0.001
    epsilon_decay = 10

    env = TruckFleetEnv(mean=0.5, std=0.1)
    agent = DQNAgent(state_dim=env.num_trucks * env.num_tires_per_truck, action_dims=[3, env.num_trucks, env.num_tires_per_truck, env.num_tires_per_truck])

    # To store the action logs for all episodes
    all_action_logs = []

    for episode in range(num_episodes):
        initial_state = deepcopy(env.reset())
        state = initial_state.flatten()
        total_reward = 0

        for t in range(100):  # Limit number of steps per episode
            epsilon = epsilon_end + (epsilon_start - epsilon_end) * np.exp(-1. * episode / epsilon_decay)
            action = agent.select_action(state, epsilon)
            next_state, reward, done, _ = env.step(action)
            next_state = next_state.flatten()

            agent.memory.push(state, action, next_state, reward)
            state = next_state
            total_reward += reward

            agent.optimize_model()

            if done:
                print(f"Episode {episode} finished after {t+1} timesteps with total reward {total_reward}")
                break

        if episode % 10 == 0:
            agent.update_target_net()

        print(f"Episode {episode} - Total reward: {total_reward}")
        action_logs = env.get_action_log()
        final_state = deepcopy(env.state)

        # Log details for this episode
        episode_log = {
            'Episode': episode,
            'Initial State': initial_state,
            'Final State': final_state,
            'Total Reward': total_reward,
            'Action Logs': action_logs,
            'Number of Actions': [len(log) for log in action_logs]
        }
        all_action_logs.append(episode_log)

        # Save details for this episode to an Excel file
        episode_df = pd.DataFrame({
            'Truck': [f'Truck {i}' for i in range(env.num_trucks)],
            'Initial State': [initial_state[i] for i in range(env.num_trucks)],
            'Final State': [final_state[i] for i in range(env.num_trucks)],
            'Number of Actions': [len(action_logs[i]) for i in range(env.num_trucks)],
            'Actions': [action_logs[i] for i in range(env.num_trucks)]
        })
        episode_df.to_excel(os.path.join(logs_folder, f'episode_{episode}_log.xlsx'), index=False)

    # Save the trained model
    torch.save(agent.policy_net.state_dict(), 'app/models/dqn_truck_fleet.pth')
    print("Training completed and model saved.")