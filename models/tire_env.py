import gym
from gym import spaces
import pandas as pd
import numpy as np
import random

class TireOptimizationEnv(gym.Env):
    """Custom Environment for optimizing tire usage on trucks using Gym"""
    metadata = {'render.modes': ['console']}

    def __init__(self, data):
        super(TireOptimizationEnv, self).__init__()
        self.df = data
        self.current_step = 0
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(low=0, high=1, shape=(10,), dtype=np.float32)
        self.actions_taken = []  # Initialize an empty list to store actions


    def reset(self):
        self.state = self.df.iloc[self.current_step].values[1:]
        self.current_truck_id = self.df.iloc[self.current_step].values[0]
        self.current_step += 1
        self.actions_taken = []  # Reset actions taken at the start of each new episode
        return self.state


    def step(self, action):
        reward = 0
        done = False

        # Current tire states
        steer_left, steer_right = self.state[0], self.state[1]
        other_tires = self.state[2:]
        tire_positions = ['steertireleft', 'steertireright', 'drive1tireleft', 'drive1tireright', 'drive2tireleft', 'drive2tireright', 'rear1tireleft', 'rear1tireright', 'rear2tireleft', 'rear2tireright']

        # Calculate penalty for inaction based on tire conditions
        if action == 0:
            num_critical_tires = sum(1 for t in self.state if t < 0.09)
            reward -= 5 * num_critical_tires  # Smaller, scaled penalty
            self.actions_taken.append("No action taken")

        # Check for terminal state
        if all(tire >= 0.09 for tire in self.state) and max(other_tires) > max(steer_left, steer_right):
            done = True  # Optimal state reached
            reward += 100

        # Action logic
        if action == 1 and not done:  # Replace steer tires if needed
            for idx, (pos, tire) in enumerate(zip(tire_positions[2:], other_tires)):
                if tire > steer_left or tire > steer_right:
                    if tire > steer_left:
                        self.actions_taken.append(f"{pos} swapped with steer_left")
                        steer_left = tire  # Update steer_left
                    if tire > steer_right:
                        self.actions_taken.append(f"{pos} swapped with steer_right")
                        steer_right = tire  # Update steer_right
                    reward += 25  # Reward for each successful swap

        elif action == 2 and not done:  # Replace bad tires
            for i, (pos, tire) in enumerate(zip(tire_positions, self.state)):
                if tire < 0.09:
                    self.state[i] = 1.0
                    self.actions_taken.append(f"replaced {pos} with new tire")
                    reward += 10  # Reward for replacing each critical tire

        # Update the state
        self.state[0], self.state[1] = steer_left, steer_right
        self.state[2:] = other_tires
        if self.current_step >= len(self.df):
            done = True

        return self.state, reward, done, {'actions_taken': self.actions_taken}

    def render(self, mode='console'):
        if mode == 'console':
            print(f'Truck ID {self.current_truck_id}: Current tire states: {self.state}')
            if self.actions_taken:  # Use self.actions_taken here
                print("Actions taken:", self.actions_taken)


def main():
    data = pd.read_csv('data/rf_training/final_data1.csv')
    env = TireOptimizationEnv(data)
    for episode in range(len(data)):
        state = env.reset()
        done = False
        while not done:
            action = env.action_space.sample()  # Random sampling should be replaced by RL agent's decision
            state, reward, done, info = env.step(action)
            env.render()
        


if __name__ == "__main__":
    main()
