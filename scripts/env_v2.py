import gym
from gym import spaces
import numpy as np
from copy import deepcopy

class TruckFleetEnv(gym.Env):
    def __init__(self, num_trucks=2, num_tires_per_truck=10, health_threshold=0.09, mean=0.5, std=0.1, max_steps=1000):
        super(TruckFleetEnv, self).__init__()

        self.num_trucks = num_trucks
        self.num_tires_per_truck = num_tires_per_truck
        self.health_threshold = health_threshold
        self.mean = mean
        self.std = std
        self.max_steps = max_steps

        # Action space: (action_type, truck_idx, tire_idx, other_tire_idx/other_truck_idx)
        self.action_space = spaces.MultiDiscrete([3, num_trucks, num_tires_per_truck, max(num_tires_per_truck, num_trucks)])
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(num_trucks, num_tires_per_truck), dtype=np.float32
        )

        self.reset()

    def reset(self):
        self.state = np.random.rand(self.num_trucks, self.num_tires_per_truck)
        self.action_log = [[] for _ in range(self.num_trucks)]  # Reset the action log
        self.current_step = 0
        self.optimal_state_achieved = False  # Track if optimal state is achieved
        return self.state

    def step(self, action):
        previous_state = deepcopy(self.state)  # Save the previous state using deep copy
        action_type, truck_idx, tire_idx, other_idx = action
        invalid_action = False

        if self.optimal_state_achieved:
            reward = self.calculate_reward(previous_state, invalid_action)
            self.current_step += 1
            done = self.current_step >= self.max_steps
            return self.state, reward, done, {}

        if action_type == 0:  # Replace tire
            if self.state[truck_idx][tire_idx] < self.health_threshold:
                self.state[truck_idx][tire_idx] = 1
                self.action_log[truck_idx].append(f"Replaced tire {tire_idx}")
            else:
                invalid_action = True
                self.action_log[truck_idx].append(f"Attempted to replace healthy tire {tire_idx}")
        elif action_type == 1:  # Swap within truck
            swap_idx = other_idx
            if tire_idx < 2 and swap_idx >= 2 and self.state[truck_idx][tire_idx] < self.state[truck_idx][swap_idx]:
                if self.state[truck_idx][tire_idx] < self.health_threshold:
                    invalid_action = True
                    self.action_log[truck_idx].append(f"Invalid swap of steer tire {tire_idx} with non-steer tire {swap_idx} when steer tire below threshold")
                else:
                    self.state[truck_idx][tire_idx], self.state[truck_idx][swap_idx] = self.state[truck_idx][swap_idx], self.state[truck_idx][tire_idx]
                    self.action_log[truck_idx].append(f"Swapped steer tire {tire_idx} with healthier non-steer tire {swap_idx}")
            else:
                invalid_action = True
                self.action_log[truck_idx].append(f"Invalid swap of tire {tire_idx} with tire {swap_idx}")
        elif action_type == 2:  # Swap with another truck
            other_truck_idx = other_idx // self.num_tires_per_truck
            other_tire_idx = other_idx % self.num_tires_per_truck
            if truck_idx != other_truck_idx and tire_idx < 2 and other_tire_idx >= 2 and self.state[truck_idx][tire_idx] < self.state[other_truck_idx][other_tire_idx]:
                if self.state[truck_idx][tire_idx] < self.health_threshold:
                    invalid_action = True
                    self.action_log[truck_idx].append(f"Invalid swap of steer tire {tire_idx} with non-steer tire {other_tire_idx} on truck {other_truck_idx} when steer tire below threshold")
                else:
                    self.state[truck_idx][tire_idx], self.state[other_truck_idx][other_tire_idx] = self.state[other_truck_idx][other_tire_idx], self.state[truck_idx][tire_idx]
                    self.action_log[truck_idx].append(f"Swapped steer tire {tire_idx} with healthier non-steer tire {other_tire_idx} on truck {other_truck_idx}")
            else:
                invalid_action = True
                self.action_log[truck_idx].append(f"Invalid swap of tire {tire_idx} with tire {other_tire_idx} on truck {other_truck_idx}")

        reward = self.calculate_reward(previous_state, invalid_action)
        self.current_step += 1
        self.optimal_state_achieved = self.is_optimal_state()
        done = self.optimal_state_achieved or self.current_step >= self.max_steps

        return self.state, reward, done, {}

    def calculate_reward(self, previous_state, invalid_action):
        reward = 0
        # Calculate the weighted average RCP before the action
        previous_steer_positions = previous_state[:, :2]
        previous_other_positions = previous_state[:, 2:]
        previous_weighted_avg_rcp = (0.8 * np.sum(previous_steer_positions) + 0.2 * np.sum(previous_other_positions))

        # Calculate the weighted average RCP after the action
        steer_positions = self.state[:, :2]
        other_positions = self.state[:, 2:]
        current_weighted_avg_rcp = (0.8 * np.sum(steer_positions) + 0.2 * np.sum(other_positions))

        # Compute the reward as the change in weighted average RCP
        if current_weighted_avg_rcp > previous_weighted_avg_rcp:
            reward += 10

        # # Penalty for invalid actions
        # if invalid_action:
        #     reward -= 1

        # # Penalty for having any tire below the health threshold
        # num_tires_below_threshold = np.sum(self.state < self.health_threshold)
        # if num_tires_below_threshold > 0:
        #     reward -= 1

        
        # # Penalty if any steer tire is less than any rear/drive tire
        # for steer_tire in steer_positions.flatten():
        #     if np.any(steer_tire < other_positions):
        #         reward -= 1
        #         break

        # Bonus for achieving optimal state
        if self.is_optimal_state():
            reward += 100

        return reward


    def is_optimal_state(self):
        steer_positions = self.state[:, :2].flatten()
        other_positions = self.state[:, 2:].flatten()

        # Check if all steer tires are healthier than all rear/drive tires
        all_steers_healthier = np.all(steer_positions[:, None] >= other_positions)

        # Check if all tires are above the health threshold
        all_above_threshold = np.all(self.state >= self.health_threshold)

        return all_above_threshold and all_steers_healthier

    def render(self, mode='human'):
        print(self.state)

    def close(self):
        pass

    def get_action_log(self):
        return self.action_log
