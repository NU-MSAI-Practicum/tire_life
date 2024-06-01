import torch
import numpy as np
import os
from oldapp.agent.env import TruckFleetEnv  # Adjusted import
from agents.dqn.dqn_agents.dqn import DQNAgent  # Adjusted import
import pandas as pd
from copy import deepcopy

# Disable OpenMP parallelism
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'


class Predictor:
    def __init__(self, model_path):
        
        # self.env = TruckFleetEnv(mean=0.5, std=0.1)
        num_trucks=2
        num_tires_per_truck=10
        health_threshold=0.09
        max_steps=200
        self.env = TruckFleetEnv(num_trucks,num_tires_per_truck,health_threshold,max_steps)
        self.agent = DQNAgent(state_dim=self.env.num_trucks * self.env.num_tires_per_truck, action_dims=[2, self.env.num_trucks, self.env.num_tires_per_truck, self.env.num_trucks, self.env.num_tires_per_truck])
        self.agent.policy_net.load_state_dict(torch.load(model_path))
        self.agent.policy_net.eval()

    def predict(self, state, log_file_path='app/logs/prediction_log.xlsx'):
        self.env.reset(initial_state=state)
        initial_state = deepcopy(self.env.state)
        state = state.flatten()
        
        for _ in range(100):
            with torch.no_grad():
                action = self.agent.select_action(state, epsilon=0)  # No exploration during prediction
            state, _, done, _ = self.env.step(action)
            state = state.flatten()
            if done:
                break
        
        final_state = deepcopy(self.env.state)

        # Save action log to a file
        action_logs = self.env.get_action_log()
        episode_log = {
            'Initial State': initial_state,
            'Final State': final_state,
            'Action Logs': action_logs,
            'Number of Actions': [len(log) for log in action_logs]
        }

        # Debug print
        print("Saving log file to:", log_file_path)

        # Log details for this episode
        episode_df = pd.DataFrame({
            'Truck': [f'Truck {i}' for i in range(self.env.num_trucks)],
            'Initial State': [initial_state[i] for i in range(self.env.num_trucks)],
            'Final State': [final_state[i] for i in range(self.env.num_trucks)],
            'Number of Actions': [len(action_logs[i]) for i in range(self.env.num_trucks)],
            'Actions': [action_logs[i] for i in range(self.env.num_trucks)]
        })
        episode_df.to_excel(log_file_path, index=False)

        # Check if file exists after saving
        if not os.path.exists(log_file_path):
            print("Error: Log file was not saved.")
        else:
            print("Log file saved successfully.")

        return initial_state, final_state, log_file_path