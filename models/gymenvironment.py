import gym
from gym import spaces
import numpy as np

class TireManagementEnv(gym.Env):
    """
    Custom Environment for managing tire health across a fleet of trucks.
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, trucks_data):
        super(TireManagementEnv, self).__init__()
        self.trucks_data = trucks_data  # Data containing truck tire information
        self.n_trucks = len(trucks_data)
        self.num_tires_per_truck = 10  # Assuming each truck has 10 tires

        # Define action space
        # Actions are represented as tuples (truck_id, tire_index, action_type)
        # action_type: 0 = no action, 1 = replace with new, 2 = swap within truck
        self.action_space = spaces.MultiDiscrete([self.n_trucks, self.num_tires_per_truck, 3])

        # Define observation space (health of each tire)
        low = np.zeros(self.n_trucks * self.num_tires_per_truck)
        high = np.ones(self.n_trucks * self.num_tires_per_truck)
        self.observation_space = spaces.Box(low, high, dtype=np.float32)

        # Initialize state
        self.state = self.initialize_state()

    def step(self, action):
        truck_id, tire_idx, action_type = action
        reward = 0
        done = False
        info = {}

        if action_type != 0:  # If action is not 'no action'
            reward += self.apply_action(truck_id, tire_idx, action_type)

        self.state = self.get_state()
        done = self.check_if_done()

        return np.array(self.state), reward, done, info

    def reset(self):
        self.state = self.initialize_state()
        return np.array(self.state)

    def render(self, mode='human'):
        # Optional: Print current tire health for visualization
        print("Current tire health status:")
        for idx, truck in enumerate(self.trucks_data):
            print(f"Truck {idx + 1}: {['%.2f' % tire['health'] for tire in truck['tires']]}")

    def close(self):
        pass

    def apply_action(self, truck_id, tire_idx, action_type):
        truck = self.trucks_data[truck_id]
        tire = truck['tires'][tire_idx]
        reward = 0

        if action_type == 1:  # Replace with a new tire
            if tire['health'] < 0.09:  # Only replace if tire is critically low
                tire['health'] = 1.0  # Setting health to 100%
                reward = 20  # Positive reward for replacing a critically low tire
            else:
                reward = -10  # Penalty for replacing a tire that is not critically low
        elif action_type == 2:  # Swap tire
            if tire['health'] < 0.09:  # Consider swap if tire health is critically low
                best_tire_idx, best_tire = max(enumerate(truck['tires']), key=lambda x: x[1]['health'])
                if best_tire['health'] > tire['health']:
                    # Swap the tires
                    truck['tires'][tire_idx], truck['tires'][best_tire_idx] = truck['tires'][best_tire_idx], truck['tires'][tire_idx]
                    reward = 15  # Reward for improving the condition of a critically low tire
                else:
                    reward = -20  # Penalty if no better tire available for swap
            else:
                reward = -5  # Penalty for attempting unnecessary swap
        else:
            if tire['health'] < 0.09:
                reward = -30  # High penalty for not taking action on critically low tire
        return reward


    def check_if_done(self):
        # Determine if the episode should end
        return False

    def initialize_state(self):
        # Initialize the state from trucks_data
        return [tire['health'] for truck in self.trucks_data for tire in truck['tires']]

    def get_state(self):
        # Update the state from trucks_data
        return [tire['health'] for truck in self.trucks_data for tire in truck['tires']]

# Example data (simplified for clarity)
trucks_data = [
    {
        "truck_id": 1,
        "maintenance_location_id": 123,
        "available_for_swapping": False,
        "tires": [{"position": "steer", "health": 0.8}, {"position": "drive", "health": 0.7}, {"position": "rear", "health": 0.6}]
    },
    # Add more trucks as needed
]

env = TireManagementEnv(trucks_data=trucks_data)
state = env.reset()
done = False

while not done:
    action = env.action_space.sample()  # Randomly sample an action
    next_state, reward, done, info = env.step(action)
    if done:
        break

env.close()
