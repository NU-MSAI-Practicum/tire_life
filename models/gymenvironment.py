import gym
from gym import spaces
import numpy as np

class TireManagementEnv(gym.Env):
    """
    Custom Environment for an RL agent that will make tire rotations decision across a fleet of trucks. 
    
    The goal of the agent is to maximize the steer tire health measured over a cummulative weighted average of the tire RCP's of all trucks part of the episode. 
    
    The agent has 3 possible actions - install a new tire (all new tires are assigned to the lowest steer positions and the old tire is cascaded down),  
    swap tires within the same truck, or do nothing. 
    (There is a future experiment to swap tires between trucks not part of the current code.) 
    
    The episode ends when <TO BE DEFINED>

    Before each episode the agent is reset with new truck data with no correlation to the previous episode. The count of trucks in each episode may vary, 
    but the action and observation space is set constant with a max_truck limit. 

    """

    metadata = {'render.modes': ['human']}

    def __init__(self, max_trucks=10):
        
        super(TireManagementEnv, self).__init__()

        self.current_num_trucks = None
        self.state = None
        
        # Defines the capacity of the maintenance location so as to keep the action space constant as DQN cannot handle dynamic. If current number of trucks are lesser 
        # than max_trucks the step function will be modified with a Maximum Limit with Masking to handle the empty action/observation space.

        self.max_trucks = max_trucks

        # Action space: [Select Truck (0 to max_trucks-1), Action Type (0 or 1), Select Tire Position (0 to 7) for swapping]
        # When the action type is 0 (do nothing): the agent will perform no action, the third dimension will be ingnored.
        # When the action type is 1 (replace a tire): Only the truck selection is relevant. The third dimension (selecting the tire for switching) will be ignored.
        # When the action type is 2 (switch tire positions): All three dimensions are relevant, the agent must decide the truck and the specific rear or drive 
        # tire to swap with the steer tire.
        self.action_space = spaces.MultiDiscrete([self.max_trucks, 3, 8])

        # Observation Space shape=(self.max_trucks, 10): defines the shape of the array that represents the observation space. It is set to have 
        # self.max_trucks rows and 10 columns. Each row corresponds to a truck, and each of the 10 columns corresponds to the tire positions on that truck. 
        self.observation_space = spaces.Box(low=0, high=1, shape=(self.max_trucks, 10), dtype=np.float32)

    def reset(self, initial_state):
        
        # The 'initial_state' should be a (n, 10) array where 'n' is the number of trucks in the current episode.
        
        if initial_state.shape[1] != 10:
            raise ValueError("Each truck must have exactly 10 tires.")
        
        self.state = initial_state
        
        # Pad the state array for non-existent trucks if fewer trucks are provided
        if initial_state.shape[0] < self.max_trucks:
            padding = np.zeros((self.max_trucks - initial_state.shape[0], 10))
            self.state = np.vstack([self.state, padding])
        return self.state

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

    # def initialize_state(self):
    #     # Initialize the state from trucks_data
    #     return [tire['health'] for truck in self.trucks_data for tire in truck['tires']]

    # def get_state(self):
    #     # Update the state from trucks_data
    #     return [tire['health'] for truck in self.trucks_data for tire in truck['tires']]

# # Example data (simplified for clarity)
# trucks_data = [
#     {
#         "truck_id": 1,
#         "maintenance_location_id": 123,
#         "available_for_swapping": False,
#         "tires": [{"position": "steer", "health": 0.8}, {"position": "drive", "health": 0.7}, {"position": "rear", "health": 0.6}]
#     },
#     # Add more trucks as needed
# ]
