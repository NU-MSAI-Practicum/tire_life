import gym
from gym import spaces
import numpy as np

import gym
from gym import spaces
import numpy as np

class TruckFleetEnv(gym.Env):
    def __init__(self, num_trucks=2, num_tires_per_truck=10, health_threshold=0.09, max_steps=200):
        super(TruckFleetEnv, self).__init__()

        self.num_trucks = num_trucks
        self.num_tires_per_truck = num_tires_per_truck
        self.health_threshold = health_threshold
        self.max_steps = max_steps

        # # Action space: (action_type, truck_idx, tire_idx, other_tire_idx/other_truck_idx)
        # self.action_space = spaces.MultiDiscrete([3, num_trucks, num_tires_per_truck, max(num_tires_per_truck, num_trucks)])

        # Action space: (action_type, truck_idx, tire_idx, other_truck_idx, other_tire_idx)
        self.action_space = spaces.MultiDiscrete([2, num_trucks, num_tires_per_truck, num_trucks, num_tires_per_truck - 2])

        self.observation_space = spaces.Box(
            low=0, high=1, shape=(num_trucks, num_tires_per_truck), dtype=np.float32
        )

        self.reset()

    def reset(self, initial_state=None):
        if initial_state is not None:
            self.state = initial_state
        else:
            self.state = np.random.rand(self.num_trucks, self.num_tires_per_truck)
        self.action_log = [[] for _ in range(self.num_trucks)]  # Reset the action log
        self.current_step = 0
        self.optimal_state_achieved = False  # Track if optimal state is achieved
        return self.state

    def step(self, action):
        action_type, truck_idx, tire_idx, other_truck_idx, other_tire_idx = action
        invalid_action = False

        reward = 0

        if self.optimal_state_achieved:
            # reward = self.calculate_reward(invalid_action, penalty=True)
            reward -= 100
            self.current_step += 1
            done = self.current_step >= self.max_steps
            return self.state, reward, done, {}

        if action_type == 0:  # Replace tire
            if self.state[truck_idx][tire_idx] <= self.health_threshold:
                self.state[truck_idx][tire_idx] = 1
                self.action_log[truck_idx].append(f"Replaced tire {tire_idx} on truck {truck_idx}")
                reward += 10
            else:
                invalid_action = True
                self.action_log[truck_idx].append(f"Invalid tried to replace healthy tire {tire_idx} on truck {truck_idx}")
                reward -= 100
        # elif action_type == 1:  # Swap within truck
        #     swap_idx = other_tire_idx
        #     if tire_idx < 2 and swap_idx >= 2 and self.state[truck_idx][tire_idx] < self.state[truck_idx][swap_idx]:
        #         self.state[truck_idx][tire_idx], self.state[truck_idx][swap_idx] = self.state[truck_idx][swap_idx], self.state[truck_idx][tire_idx]
        #         self.action_log[truck_idx].append(f"Swapped steer tire {tire_idx} with healthier non-steer tire {swap_idx}")
        #         reward += 10
        #     else:
        #         invalid_action = True
        #         self.action_log[truck_idx].append(f"Invalid swap of tire {tire_idx} with tire {swap_idx}")
        #         reward -= 10
        elif action_type == 1:  # Swap tires
            if truck_idx == other_truck_idx:  # Swap within the same truck
                swap_idx = other_tire_idx
                if tire_idx < 2 and swap_idx >= 2 and self.state[truck_idx][tire_idx] < self.state[truck_idx][swap_idx]:
                    self.state[truck_idx][tire_idx], self.state[truck_idx][swap_idx] = self.state[truck_idx][swap_idx], self.state[truck_idx][tire_idx]
                    self.action_log[truck_idx].append(f"Swapped steer tire {tire_idx} with non-steer tire {swap_idx} within truck {truck_idx}")
                    reward += 10
                else:
                    invalid_action = True
                    self.state[truck_idx][tire_idx], self.state[truck_idx][swap_idx] = self.state[truck_idx][swap_idx], self.state[truck_idx][tire_idx]
                    self.action_log[truck_idx].append(f"Invalid swap of tire {tire_idx} with tire {swap_idx} within truck {truck_idx}")
                    reward -= 500
            else:  # Swap between different trucks
                if tire_idx < 2 and swap_idx >= 2 and self.state[truck_idx][tire_idx] < self.state[other_truck_idx][swap_idx]:
                    self.state[truck_idx][tire_idx], self.state[other_truck_idx][swap_idx] = self.state[other_truck_idx][swap_idx], self.state[truck_idx][tire_idx]
                    self.action_log[truck_idx].append(f"Swapped steer tire {tire_idx} on truck {truck_idx} with non-steer tire {swap_idx} on truck {other_truck_idx}")
                    reward += 10
                else:
                    invalid_action = True
                    self.state[truck_idx][tire_idx], self.state[other_truck_idx][swap_idx] = self.state[other_truck_idx][swap_idx], self.state[truck_idx][tire_idx]
                    self.action_log[truck_idx].append(f"Invalid swap of tire {tire_idx} with tire {swap_idx} between trucks {truck_idx} and {other_truck_idx}")
                    reward -= 500


        self.current_step += 1
       
       #rewards for achieving/moving towards goal
        self.optimal_state_achieved = self.is_optimal_state()
        if self.optimal_state_achieved == True:
            reward += 10000
        # else:
        #     steer_positions = self.state[:, :2]
        #     other_positions = self.state[:, 2:]
        #     # Check if all steer tires on a truck are healthier than its non-steer tires
        #     for i in range(self.num_trucks):
        #         steer_tires_healthier = all(steer_tire >= max(other_positions[i]) for steer_tire in steer_positions[i])
        #         if steer_tires_healthier:
        #             reward += 10
        #     # # Penalize if tires tires are below the health thresholdx
            # if np.all(self.state <= self.health_threshold):
            #     reward -= 10
                        
        done = self.optimal_state_achieved or self.current_step >= self.max_steps

        # reward = self.calculate_reward(invalid_action)

        return self.state, reward, done, {}

    def calculate_reward(self, invalid_action, penalty=False):
        steer_positions = self.state[:, :2]
        other_positions = self.state[:, 2:]

        # Positive rewards for healthy tires
        # reward = np.sum(steer_positions) * 2 + np.sum(other_positions)
        # Check if all steer tires on a truck are healthier than its non-steer tires
        for i in range(self.num_trucks):
            steer_tires_healthier = all(steer_tire > max(other_positions[i]) for steer_tire in steer_positions[i])
            if steer_tires_healthier:
                reward += 50

        # # Additional rewards for steer tires above threshold
        # reward += np.sum(steer_positions >= self.health_threshold) * 0.5
                
        # Additional reward if all tires on all trucks are above the health threshold
        if np.all(self.state >= self.health_threshold):
            reward += 50

        # Penalty for invalid actions
        if invalid_action:
            reward -= 1

        # Penalty for having any tire below the health threshold
        if np.any(self.state < self.health_threshold):
            reward -= 10

        # Bonus for achieving optimal state
        if self.is_optimal_state():
            reward += 1000

        # Large penalty for taking actions after achieving the optimal state
        if penalty:
            reward -= 100

        return reward

    def is_optimal_state(self):
        steer_positions = self.state[:, :2]
        other_positions = self.state[:, 2:]

        all_steers_healthier = True
        for i in range(self.num_trucks):
            for steer_tire in steer_positions[i]:
                if not np.all(steer_tire >= other_positions[i]):
                    all_steers_healthier = False
                    break
            if not all_steers_healthier:
                break

        return np.all(self.state >= self.health_threshold) and all_steers_healthier

    def render(self, mode='human'):
        print(self.state)

    def close(self):
        pass

    def get_action_log(self):
        return self.action_log
