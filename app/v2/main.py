import os
import shutil
import torch
import pandas as pd
import numpy as np
from datetime import datetime
from copy import deepcopy
import matplotlib.pyplot as plt

from env import TruckMaintenanceEnv
from util import *

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

    # Initialize the environment
    env = TruckMaintenanceEnv(max_trucks=5, health_threshold=0.09, log_folder=logs_folder)
    
    # CSV file for all episodes
    summary_file = os.path.join(logs_folder, 'training_summary.csv')
    summary_logs = []

    # Number of episodes
    num_episodes = 10

    for episode in range(1, num_episodes + 1):
        log_file = os.path.join(logs_folder, f'episode_{episode}_log.txt')
        total_actions = test_environment(env, num_steps=20, log_file=log_file, episode_num=episode)
        
        # Log summary for the episode
        summary_logs.append({
            "Episode Number": episode,
            "Total Reward": env.total_reward,
            "Total Actions": total_actions,
            "Optimal State Reached?": env.is_optimal_state()
        })

    # Save the summary logs to a CSV file
    summary_df = pd.DataFrame(summary_logs)
    summary_df.to_csv(summary_file, index=False)

if __name__ == "__main__":
    main()