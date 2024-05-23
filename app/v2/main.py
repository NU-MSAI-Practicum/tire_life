import os
import shutil
import torch
import pandas as pd
import numpy as np
from datetime import datetime
from copy import deepcopy
import matplotlib.pyplot as plt

from dqn import *
from train import *
from env import TruckMaintenanceEnv

# Disable OpenMP parallelism
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'

# Set PyTorch to use single-threaded execution
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

def main():
    # Create a new logs folder by date and time
    current_date = datetime.now().strftime('%Y-%m-%d')
    current_time = datetime.now().strftime('%H-%M-%S')
    logs_folder = f'./app/v2/logs/{current_date}/{current_time}'
    os.makedirs(logs_folder)

    # CSV file for all episodes
    summary_file = os.path.join(logs_folder, 'training_summary.csv')

    # Initialize the environment
    max_trucks = 3
    health_threshold = 0.09
    env = TruckMaintenanceEnv(logs_folder, max_trucks, health_threshold)
    env.log_folder = logs_folder

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Train the DQN agent
    train_dqn(env, summary_file=summary_file, episodes=1000, gamma=0.99, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995, batch_size=64, target_update_freq=10)

    # Initialize DQN model for evaluation
    state_dim = env.observation_space.shape[0] * env.observation_space.shape[1]
    action_dim = np.prod(env.action_space.nvec)
    q_network = DQN(state_dim, action_dim).to(device)
    q_network.load_state_dict(torch.load(os.path.join(logs_folder, 'dqn_model.pth')))

    # Evaluate the agent
    avg_reward = evaluate_agent(env, q_network, episodes=100)
    print(f"Average Reward during evaluation: {avg_reward}")

if __name__ == "__main__":
    main()
