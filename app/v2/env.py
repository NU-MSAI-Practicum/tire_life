import gym
from gym import spaces
import numpy as np
from copy import deepcopy
import os
import pandas as pd

class TruckMaintenanceEnv(gym.Env):
    def __init__(self, log_folder, max_trucks, health_threshold):
        super(TruckMaintenanceEnv, self).__init__()
        self.max_trucks = max_trucks
        self.max_tires_per_truck = 10
        self.health_threshold = health_threshold
        self.log_folder = log_folder
        self.total_reward = 0
        self.action_count = np.zeros(self.max_trucks, dtype=int)
        self.reward_count = np.zeros(self.max_trucks, dtype=int)
        self.action_log = [[] for _ in range(self.max_trucks)]

        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(self.max_trucks, self.max_tires_per_truck), dtype=np.float32)
        self.action_space = spaces.MultiDiscrete([3, self.max_trucks, self.max_tires_per_truck, self.max_trucks, self.max_tires_per_truck])

        self.state = None
        self.initial_state = None
        self.reset()
    
    def reset(self):
        self.total_reward = 0
        self.action_count = np.zeros(self.max_trucks, dtype=int)
        self.reward_count = np.zeros(self.max_trucks, dtype=int)
        self.action_log = [[] for _ in range(self.max_trucks)]

        self.state = np.zeros((self.max_trucks, self.max_tires_per_truck), dtype=np.float32)
        for truck in range(self.max_trucks):
            self.state[truck] = np.round(np.random.uniform(low=0.0, high=1.0, size=self.max_tires_per_truck), 2)

        self.initial_state = deepcopy(self.state)

        return self.state

    def step(self, action):
        action_type, truck1, tire1, truck2, tire2 = action
        action_valid = False
        action_description = ""

        if action_type == 0:  # No action
            reward = 0
            action_description = "Valid action: No action"
        elif action_type == 1:  # Replace tire in truck1 at position tire1
            if self.state[truck1][tire1] <= self.health_threshold:
                self.state[truck1][tire1] = 1.0
                reward = 10
                action_valid = True
                action_description = f"Valid action: Replace tire in truck {truck1} at position {tire1}"
            else:
                reward = -1
                action_description = f"Invalid action: Replace tire in truck {truck1} at position {tire1}"
        elif action_type == 2:  # Swap tires
            if truck1 == truck2 and tire1 == tire2:
                reward = -1  # Invalid swap: same tire
                action_description = f"Invalid action: Swap tire {tire1} in truck {truck1} with itself"
            else:
                if (tire1 < 2 and tire2 < 2) or (tire1 >= 2 and tire2 >= 2):  # Invalid swap between rear/drive or steer tires
                    reward = -1
                    action_description = f"Invalid action: Swap tire {tire1} in truck {truck1} with tire {tire2} in truck {truck2}"
                else:
                    # Calculate weighted RCP sum before swap
                    weighted_rcp_before = self.calculate_weighted_rcp()
                    
                    # Perform the swap
                    self.state[truck1][tire1], self.state[truck2][tire2] = self.state[truck2][tire2], self.state[truck1][tire1]
                    
                    # Round the swapped values to 2 decimal places
                    self.state[truck1][tire1] = round(self.state[truck1][tire1], 2)
                    self.state[truck2][tire2] = round(self.state[truck2][tire2], 2)
                    
                    # Calculate weighted RCP sum after swap
                    weighted_rcp_after = self.calculate_weighted_rcp()
                    
                    # Reward if the weighted RCP increases
                    reward = 10 if weighted_rcp_after > weighted_rcp_before else -1
                    action_valid = reward == 10
                    action_description = f"{'Valid' if action_valid else 'Invalid'} action: Swap tire {tire1} in truck {truck1} with tire {tire2} in truck {truck2}"
        else:
            reward = -1
            action_description = "Invalid action type"
        
        self.action_count[truck1] += 1
        self.reward_count[truck1] += reward
        self.action_log[truck1].append(action_description)
        if action_type == 2 and truck1 != truck2:
            self.action_count[truck2] += 1
            self.reward_count[truck2] += reward
            self.action_log[truck2].append(action_description)
        
        self.total_reward += reward
        
        if self.is_optimal_state():
            done = True
            reward += 100
            self.total_reward += 100
        else:
            done = False
        
        return self.state, reward, done, {}

    def calculate_weighted_rcp(self):
        steer_weight = 0.8
        rear_drive_weight = 0.2
        
        steer_rcp = self.state[:, :2].sum() * steer_weight
        rear_drive_rcp = self.state[:, 2:].sum() * rear_drive_weight
        
        total_rcp = steer_rcp + rear_drive_rcp
        total_tires = self.state[:, :2].size * steer_weight + self.state[:, 2:].size * rear_drive_weight
        
        return total_rcp / total_tires
    
    def is_optimal_state(self):
        # Check if all tires are above the threshold
        if not np.all(self.state >= self.health_threshold):
            return False
        
        # Check if steer tire health is greater than rear/drive tire health for each truck
        for truck in range(self.max_trucks):
            steer_tires = self.state[truck, :2]
            rear_drive_tires = self.state[truck, 2:]
            for steer_tire in steer_tires:
                if steer_tire <= rear_drive_tires.max():
                    return False
        
        return True
    
    def save_logs(self, episode_num):
        # Create a DataFrame for the logs
        logs = {
            "Truck ID": [],
            "Truck Initial State": [],
            "Truck Final State": [],
            "Action Count": [],
            "Reward Count": [],
            "Action Log": []
        }
        
        for truck in range(self.max_trucks):
            logs["Truck ID"].append(truck)
            logs["Truck Initial State"].append([round(val, 2) for val in self.initial_state[truck]])
            logs["Truck Final State"].append([round(val, 2) for val in self.state[truck]])
            logs["Action Count"].append(self.action_count[truck])
            logs["Reward Count"].append(self.reward_count[truck])
            logs["Action Log"].append(self.action_log[truck])

        logs_df = pd.DataFrame(logs)
        
        # Save the DataFrame to an Excel file
        log_file = os.path.join(self.log_folder, f"episode_{episode_num}.xlsx")
        logs_df.to_excel(log_file, index=False)
    
    def render(self):
        state_str = '\n'.join(['Truck {}: {}'.format(i, [round(val, 2) for val in row]) for i, row in enumerate(self.state)])
        print(state_str)
    
def test_environment(env, num_steps=100, log_file="environment_log.txt", episode_num=1):
    state = env.reset()
    initial_state = deepcopy(state)  # Capture the initial state right after reset
    print(f"Initial State for Episode {episode_num}:")
    
    num_trucks = np.count_nonzero(np.any(env.state != 0, axis=1))  # Number of active trucks
    
    total_actions = 0  # Track the total number of actions for the episode
    for step in range(num_steps):
        total_actions += 1
        # Generate a random action
        action_type = np.random.randint(0, 3)
        truck1 = np.random.randint(0, num_trucks)
        tire1 = np.random.randint(0, env.max_tires_per_truck)
        
        if action_type == 1:  # Replace tire action
            action = [action_type, truck1, tire1, 0, 0]
        else:  # Swap tires action or no action
            truck2 = np.random.randint(0, num_trucks)
            tire2 = np.random.randint(0, env.max_tires_per_truck)
            action = [action_type, truck1, tire1, truck2, tire2]
        
        state, reward, done, _ = env.step(action)
        
        with open(log_file, 'a') as f:
            f.write(f"Step {step + 1}: Action = {action}, Reward = {reward}, Done = {done}\n")
        
        if done:
            with open(log_file, 'a') as f:
                f.write("Optimal state achieved, stopping test.\n")
            break
    
    env.initial_state = initial_state  # Set the captured initial state before saving logs
    env.save_logs(episode_num)
    return total_actions  # Return the total number of actions for this episode
