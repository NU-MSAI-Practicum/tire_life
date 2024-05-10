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

    def reset(self):
        self.state = self.df.iloc[self.current_step].values[1:]
        self.current_truck_id = self.df.iloc[self.current_step].values[0]
        self.current_step += 1
        return self.state

    def step(self, action):
        reward = 0
        done = False
        actions_taken = []

        # Current tire states
        steer_left, steer_right = self.state[0], self.state[1]
        other_tires = self.state[2:]
        tire_positions = ['steertireleft','steertireright','drive1tireleft','drive1tireright','drive2tireleft','drive2tireright','rear1tireleft','rear1tireright','rear2tireleft','rear2tireright']

        # Check for terminal state
        if all(tire >= 0.09 for tire in self.state) and np.argsort(other_tires)[-1:] > max(steer_left, steer_right):
            done = True  # Optimal state reached
            reward += 100
            return self.state, reward, done, {'actions_taken': actions_taken}
        elif action == 0 and not done:  # No action
            reward -= 50

        elif action == 1 and not done:  # Replace steer tires if needed
            sorted_indexes = np.argsort(other_tires)
            max_idx1, max_idx2 = sorted_indexes[-1], sorted_indexes[-2]
            best_tires = [other_tires[max_idx1], other_tires[max_idx2]]
            changed = False

            if best_tires[0] > steer_left:
                actions_taken.append(f"{tire_positions[max_idx1 + 2]} swapped with steer_left")
                other_tires[max_idx1], steer_left = steer_left, best_tires[0]  # Swap values
                changed = True

            if best_tires[1] > steer_right:
                actions_taken.append(f"{tire_positions[max_idx2 + 2]} swapped with steer_right")
                other_tires[max_idx2], steer_right = steer_right, best_tires[1]  # Swap values
                changed = True

            if changed:
                reward += 50
            else:
                reward -= 10  # Penalty for ineffective action

        elif action == 2 and not done:  # Replace bad tires
            for i in range(len(self.state)):
                if self.state[i] < 0.09:
                    self.state[i] = 1.0
                    actions_taken.append(f"replaced {tire_positions[i]} with new tire")
            reward += 5 * len(actions_taken)

        self.state[0], self.state[1] = steer_left, steer_right
        self.state[2:] = other_tires
        if self.current_step >= len(self.df):
            done = True

        return self.state, reward, done, {'actions_taken': actions_taken}

    def render(self, mode='console'):
        print(f'Truck ID {self.current_truck_id}: Current tire states: {self.state}')

def main():
    data = pd.read_csv('data/rf_training/final_data1.csv')
    env = TireOptimizationEnv(data)
    for episode in range(len(data)):
        state = env.reset()
        done = False
        while not done:
            action = env.action_space.sample()  # This should be replaced by your RL agent's decision
            state, reward, done, info = env.step(action)
            env.render()
            if info['actions_taken']:
                print("Actions taken:", info['actions_taken'])

if __name__ == "__main__":
    main()
