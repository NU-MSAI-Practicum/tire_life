import numpy as np
from copy import deepcopy

def log_state(env, log_file, mode='a'):
    state_str = '\n'.join(['Truck {}: {}'.format(i, [round(val, 2) for val in row]) for i, row in enumerate(env.state)])
    with open(log_file, mode) as f:
        f.write(state_str + '\n')

def test_environment(env, num_steps=100, log_file="environment_log.txt", episode_num=1):
    state = env.reset()
    initial_state = deepcopy(state)  # Capture the initial state right after reset
    print(f"Initial State for Episode {episode_num}:")
    log_state(env, log_file, mode='w')
    
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
        log_state(env, log_file, mode='a')
        
        if done:
            with open(log_file, 'a') as f:
                f.write("Optimal state achieved, stopping test.\n")
            break
    
    env.initial_state = initial_state  # Set the captured initial state before saving logs
    env.save_logs(episode_num)
    return total_actions  # Return the total number of actions for this episode