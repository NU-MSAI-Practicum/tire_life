import gym
from gym import spaces
import numpy as np

class TruckFleetEnv(gym.Env):
    def __init__(self, num_trucks, num_tires_per_truck, health_threshold, max_steps):
        super(TruckFleetEnv, self).__init__()

        self.num_trucks = num_trucks
        self.num_tires_per_truck = num_tires_per_truck
        self.health_threshold = health_threshold
        self.max_steps = max_steps

        # # Action space: (action_type, truck_idx, tire_idx, other_tire_idx/other_truck_idx)
        # self.action_space = spaces.MultiDiscrete([3, num_trucks, num_tires_per_truck, max(num_tires_per_truck, num_trucks)])

        # # Action space: (action_type, truck_idx, tire_idx, other_truck_idx, other_tire_idx)
        # self.action_space = spaces.MultiDiscrete([2, num_trucks, num_tires_per_truck, num_trucks, num_tires_per_truck - 2])

         # Define separate action spaces for replace and swap actions
        self.action_space = spaces.Dict({
            "replace": spaces.MultiDiscrete([num_trucks, num_tires_per_truck]),
            "swap": spaces.MultiDiscrete([num_trucks, 2, num_trucks, num_tires_per_truck - 2])
        })

        self.observation_space = spaces.Box(
            low=0, high=1, shape=(num_trucks, num_tires_per_truck), dtype=np.float32
        )

        self.reset()

    def reset(self, initial_state=None):
        if initial_state is not None:
            self.state = np.round(initial_state, 2)
        else:
            self.state = np.round(np.random.rand(self.num_trucks, self.num_tires_per_truck), 2)
        self.action_log = [[] for _ in range(self.num_trucks)]  # Reset the action log
        self.current_step = 0
        self.optimal_state_achieved = False  # Track if optimal state is achieved
        return self.state

    def step(self, action):

        action_type = action[0]

        reward = 0

        if self.optimal_state_achieved:
            reward -= 50
            self.current_step += 1
            done = self.current_step >= self.max_steps
            return self.state, reward, done, {}

        if action_type == 0:  # Replace tire
            truck_idx, tire_idx = action[1], action[2]
            if self.state[truck_idx][tire_idx] <= self.health_threshold:
                self.state[truck_idx][tire_idx] = 1
                self.action_log[truck_idx].append(f"Valid Replace: Truck {truck_idx} / Tire {tire_idx}")
                reward += 5
            else:
                self.action_log[truck_idx].append(f"Invalid Replace: Truck {truck_idx} / Tire {tire_idx}")
                # self.state[truck_idx][tire_idx] = 1
                reward -= 10
        elif action_type == 1:  # Swap tires
            truck_idx, other_truck_idx, tire_idx, other_tire_idx = action[1], action[2], action[3], action[4]
            swap_idx = other_tire_idx
            if truck_idx == other_truck_idx:  # Swap within the same truck
                if tire_idx < 2 and swap_idx >= 2 and self.state[truck_idx][tire_idx] < self.state[truck_idx][swap_idx]:
                    self.state[truck_idx][tire_idx], self.state[truck_idx][swap_idx] = self.state[truck_idx][swap_idx], self.state[truck_idx][tire_idx]
                    self.action_log[truck_idx].append(f"Valid Swap: Truck {truck_idx} - Tire {tire_idx} / Tire {swap_idx}")
                    reward += 5
                else:
                    # self.state[truck_idx][tire_idx], self.state[truck_idx][swap_idx] = self.state[truck_idx][swap_idx], self.state[truck_idx][tire_idx]
                    self.action_log[truck_idx].append(f"Invalid Swap: Truck {truck_idx} - Tire {tire_idx} / Tire {swap_idx}")
                    reward -= 10
            else:  # Swap between different trucks
                if tire_idx < 2 and swap_idx >= 2 and self.state[truck_idx][tire_idx] < self.state[other_truck_idx][swap_idx]:
                    self.state[truck_idx][tire_idx], self.state[other_truck_idx][swap_idx] = self.state[other_truck_idx][swap_idx], self.state[truck_idx][tire_idx]
                    self.action_log[truck_idx].append(f"Valid Swap: Truck {truck_idx} - Tire {tire_idx} / Truck {other_truck_idx} - Tire {swap_idx}")
                    reward += 5
                else:
                    # self.state[truck_idx][tire_idx], self.state[other_truck_idx][swap_idx] = self.state[other_truck_idx][swap_idx], self.state[truck_idx][tire_idx]
                    # self.action_log[truck_idx].append(f"Invalid Swap: Truck {truck_idx} - Tire {tire_idx} / Truck {other_truck_idx} - Tire {swap_idx}")
                    reward -= 10


        self.current_step += 1
       
       #rewards for achieving/moving towards goal
        self.optimal_state_achieved = self.is_optimal_state()
        if self.optimal_state_achieved == True:
            reward += 100
        # else:
        #     steer_positions = self.state[:, :2]
        #     other_positions = self.state[:, 2:]
            
        #     for i in range(self.num_trucks):
        #         steer_tires = steer_positions[i]
        #         other_tires = other_positions[i]
        #         if np.all(steer_tires[:, np.newaxis] >= other_tires):
        #             reward += 1 * len(steer_tires)

        #     low_health_tires = (self.state <= self.health_threshold).sum()
        #     reward -= 10 * low_health_tires
                        
        done = self.optimal_state_achieved or self.current_step >= self.max_steps

        return self.state, reward, done, {}

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

        return np.all(self.state > self.health_threshold) and all_steers_healthier

    def render(self, mode='human'):
        print(self.state)

    def close(self):
        pass

    def get_action_log(self):
        return self.action_log
