import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from env import TruckFleetEnv
from double_dqn import DQNAgent
from copy import deepcopy
from datetime import datetime
import shutil

def train(logs_folder):
    num_episodes = 5000  # Further increase the number of episodes
    epsilon_start = 1.0
    epsilon_end = 0.05  # Adjust the final epsilon for a better balance between exploration and exploitation
    epsilon_decay = 1000  # Fine-tune the decay rate for epsilon

    num_trucks = 2
    num_tires_per_truck = 10
    health_threshold = 0.09
    max_steps = 250  # Reduced number of steps per episode

    env = TruckFleetEnv(num_trucks, num_tires_per_truck, health_threshold, max_steps)

    agent = DQNAgent(
        state_dim=env.num_trucks * env.num_tires_per_truck,
        action_dims={
            "replace": [num_trucks, num_tires_per_truck],
            "swap": [num_trucks, 2, num_trucks, num_tires_per_truck - 2]
        }
    )

    all_action_logs = []
    rewards_per_episode = []
    losses_per_step = []

    for episode in range(num_episodes):
        initial_state = deepcopy(env.reset())
        state = initial_state.flatten()
        total_reward = 0

        for t in range(max_steps):  # Use max_steps defined earlier
            epsilon = epsilon_end + (epsilon_start - epsilon_end) * np.exp(-1. * episode / epsilon_decay)
            action = agent.select_action(state, epsilon)
            next_state, reward, done, _ = env.step(action)
            next_state = next_state.flatten()

            agent.memory.push(state, action, next_state, reward)
            state = next_state
            total_reward += reward

            loss = agent.optimize_model()
            if loss is not None:
                losses_per_step.append(loss)

            if done:
                break

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
        rewards_per_episode.append(total_reward)

        episode_df = pd.DataFrame({
            'Truck': [f'Truck {i}' for i in range(env.num_trucks)],
            'Initial State': [initial_state[i] for i in range(env.num_trucks)],
            'Final State': [final_state[i] for i in range(env.num_trucks)],
            'Number of Actions': [len(action_logs[i]) for i in range(env.num_trucks)],
            'Actions': [action_logs[i] for i in range(env.num_trucks)]
        })
        episode_df.to_excel(os.path.join(logs_folder, f'episode_{episode}_log.xlsx'), index=False)

    torch.save(agent.policy_net.state_dict(), 'app/models/dqn_truck_fleet.pth')
    print("Training completed and model saved.")

    # Plotting rewards per episode
    plt.figure(figsize=(12, 6))
    plt.plot(rewards_per_episode, label='Total Reward per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Total Reward per Episode')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Plotting loss per optimization step
    plt.figure(figsize=(12, 6))
    plt.plot(losses_per_step, label='Loss per Step')
    plt.xlabel('Optimization Step')
    plt.ylabel('Loss')
    plt.title('Loss per Optimization Step')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    current_date = datetime.now().strftime('%Y-%m-%d')
    current_time = datetime.now().strftime('%H-%M-%S')
    logs_folder = f'logs/{current_date}/{current_time}'

    if os.path.exists('logs'):
        shutil.rmtree('logs')
    os.makedirs(logs_folder)

    train(logs_folder)
