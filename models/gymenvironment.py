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

        # Action space: [Select Action Type (0 or 1), Truck (0 to max_trucks-1), Select Steer Tire Position (0 to 1), Select Rear/Drive Tire Position (0 to 7)]
        # When the action type is 0 (do nothing): the agent will perform no action.
        # When the action type is 1 (replace a tire): All dimension are relevant - (action_type, truck ID, steer pos to receive new tire, rear/drive pos to receive old tire)
        # When the action type is 2 (switch tire positions): All dimension are relevant - (action_type, truck ID, steer pos for rear/drive tire, rear/drive pos for steer tire)
        self.action_space = spaces.MultiDiscrete([3, self.max_trucks, 2, 8])

        # Observation Space shape=(self.max_trucks, 10): defined as a tuple. eg. (0, array([0.03633198, 0.42370757], dtype=float32)). The Discrete type holds the truck IDs
        # followed by a Box array of size 10 in range 0 to 1 to represent tireRCPs.
        self.observation_space = spaces.Tuple((
            spaces.Discrete(self.max_trucks),  # to represent truck IDs
            spaces.Box(low=0, high=1, shape=(self.max_trucks, 10), dtype=np.float32) # to represent tire RCPs
        ))

    def reset(self, truck_ids, tire_rcps):

        """
        Reset the environment with externally provided truck IDs and tire RCP values at the beginning of each episode.
        Args:
            truck_ids (array-like): Array of truck IDs.
            tire_rcps (2D array-like): Array where each row corresponds to the RCPs of the tires on a truck.
    
         Returns:
            tuple: The initial state of the environment, consisting of truck IDs and their tire RCPs.
        """

        if len(truck_ids) != tire_rcps.shape[0]:
            raise ValueError("The number of truck IDs must match the number of rows in tire RCPs.")
        
        self.current_num_trucks = len(truck_ids)  # Update the current number of trucks

        truck_ids_padded = -1 * np.ones(self.max_trucks, dtype=np.int32)  # Use -1 to indicate no truck
        tire_rcps_padded = np.zeros((self.max_trucks, 10))

        truck_ids_padded[:self.current_num_trucks] = truck_ids
        tire_rcps_padded[:self.current_num_trucks, :] = tire_rcps

        self.state = (truck_ids_padded, tire_rcps_padded)
        
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
