import gym
from gym import spaces
import pandas as pd
import numpy as np
import logging
import torch

# Disable parallelism in terminal before starting
# export OMP_NUM_THREADS=1
# export MKL_NUM_THREADS=1


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class TireOptimizationEnv(gym.Env):
    """Custom Environment for optimizing tire usage on trucks using Gym"""
    metadata = {'render.modes': ['console']}

    def __init__(self, data):
        super(TireOptimizationEnv, self).__init__()
        self.df = data
        self.current_step = 0
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(low=0, high=1, shape=(10,), dtype=np.float32)
        self.actions_taken = []

    def reset(self):
        if self.current_step >= len(self.df):
            self.current_step = 0
        self.state = self.df.iloc[self.current_step].values[1:]
        self.current_truck_id = self.df.iloc[self.current_step].values[0]
        self.actions_taken = []
        return self.state


    def step(self, action):
        done = False
        reward = 0
        steer_left, steer_right = self.state[0], self.state[1]
        other_tires = self.state[2:]
        tire_positions = self.df.columns[1:]

        if action == 0:
            num_critical_tires = sum(1 for t in self.state if t < 0.09)
            reward -= 5 * num_critical_tires
            self.actions_taken.append("No action")

        elif action == 1:
            min_steer_idx = np.argmin([steer_left, steer_right])
            max_other_idx = np.argmax(other_tires)
            min_steer_value = min(steer_left, steer_right)
            max_other_value = other_tires[max_other_idx]

            if max_other_value > min_steer_value:
                if min_steer_idx == 0:
                    steer_left, other_tires[max_other_idx] = max_other_value, min_steer_value
                    self.actions_taken.append( f"steer_left swapped with {tire_positions[max_other_idx+2]}")
                else:
                    steer_right, other_tires[max_other_idx] = max_other_value, min_steer_value
                    self.actions_taken.append(f"steer_right swapped with {tire_positions[max_other_idx+2]}")
                reward += 25

        elif action == 2 and not done:  # Replace bad tires
            for i, (pos, tire) in enumerate(zip(tire_positions, self.state)):
                if tire < 0.09:
                    self.state[i] = 1.0
                    self.actions_taken.append(f"replaced {pos} with new tire")
                    reward += 10  # Reward for replacing each critical tire

        self.state[0], self.state[1] = steer_left, steer_right
        self.state[2:] = other_tires

        if all(tire >= 0.09 for tire in self.state) and max(other_tires) <= min(steer_left, steer_right):
            done = True
            reward += 100

        if not done:
            #logging.info(f'Truck ID {self.current_truck_id}: Current tire states: {self.state}')
            return self.state, reward, False, {}
        else:
            logging.info(f'Truck ID {self.current_truck_id}: Final tire states: {self.state} Actions Taken: {self.actions_taken}')
            #check if it is the last row of the dataframe
            if self.current_step + 1 < len(self.df):
                self.current_step += 1
                self.state = self.df.iloc[self.current_step].values[1:]
                self.current_truck_id = self.df.iloc[self.current_step].values[0]
            return self.state, reward, True, {}

    def render(self, mode='console'):
        if mode == 'console':
            logging.info(f'Truck ID {self.current_truck_id}: Current tire states: {self.state}')
            
#function to save the model
def save_model(model, filename):
    torch.save(model.state_dict(), filename)


def main():
    data = pd.read_csv('data/rf_training/final_data.csv')
    env = TireOptimizationEnv(data)
    for episode in range(5):
        state = env.reset()
        done = False
        while not done:
            # First, check for critical tires to replace (Action 2)
            critical_tires = any(tire < 0.09 for tire in state)
            if critical_tires:
                action = 2
                state, reward, done, info = env.step(action)
                # Optionally render the state post-action
                #env.render()
            
            # Next, if no critical tires or after replacing, check for beneficial swaps (Action 1)
            if not critical_tires or not done:  # Proceed if the episode is not marked as done
                steer_left, steer_right = state[0], state[1]
                other_tires = state[2:]
                min_steer_idx = np.argmin([steer_left, steer_right])
                max_other_idx = np.argmax(other_tires)
                min_steer_value = min(steer_left, steer_right)
                max_other_value = max(other_tires)

                if max_other_value > min_steer_value:
                    action = 1
                else:
                    # If no beneficial swap and no critical tires left to replace, end the episode
                    action = 0
                    done = True

                state, reward, done, info = env.step(action)
                #env.render()
            
            # End the episode if no actions are necessary
            if action == 0:
                done = True

        # Reset to new state for the next episode
        state = env.reset()

if __name__ == "__main__":
    main()
