import os
import torch
import pandas as pd
import numpy as np
from agent.env import TruckFleetEnv
from agent.dqn.dueling_dqn import DQNAgent
from copy import deepcopy

def train(logs_folder):
    # Training loop
    num_episodes = 5000  # Increased number of episodes
    epsilon_start = 1.0  # Higher starting epsilon for more exploration
    epsilon_end = 0.01  # Slightly higher ending epsilon to maintain exploration
    epsilon_decay = 200  # Slower decay rate

    num_trucks = 2
    num_tires_per_truck = 10
    health_threshold = 0.09
    max_steps = 200

    env = TruckFleetEnv(num_trucks, num_tires_per_truck, health_threshold, max_steps)
    agent = DQNAgent(
        state_dim=env.num_trucks * env.num_tires_per_truck,
        action_dims={
            "replace": [num_trucks, num_tires_per_truck],
            "swap": [num_trucks, 2, num_trucks, num_tires_per_truck - 2]
        }
    )

    # To store the action logs for all episodes
    all_action_logs = []

    frame_idx = 0
    for episode in range(num_episodes):
        initial_state = deepcopy(env.reset())
        state = initial_state.flatten()
        total_reward = 0

        for t in range(max_steps):  # Limit number of steps per episode
            epsilon = epsilon_end + (epsilon_start - epsilon_end) * np.exp(-1. * frame_idx / epsilon_decay)
            action = agent.select_action(state, epsilon)
            next_state, reward, done, _ = env.step(action)
            next_state = next_state.flatten()

            agent.memory.push(state, action, next_state, reward)
            state = next_state
            total_reward += reward

            agent.optimize_model(frame_idx)
            frame_idx += 1

            if done:
                # print(f"Episode {episode} finished after {t+1} timesteps with total reward {total_reward}")
                break

        if episode % 10 == 0:
            agent.update_target_net()

        print(f"Episode {episode} - Time Step: {t+1} - Total reward: {total_reward}")
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