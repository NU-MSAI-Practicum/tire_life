from gymenvironment import *
from dqn import *
import torch
import torch.optim as optim
from collections import deque


def main():

    env = TireManagementEnv()
    state = env.reset()
    
    input_dim = env.observation_space.shape[0]
    output_dim = env.action_space.n
    
    policy_net = DQN(input_dim, output_dim)
    target_net = DQN(input_dim, output_dim)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters())
    memory = deque(maxlen=10000)
    episodes = 500
    batch_size = 64
    epsilon = 0.9
    epsilon_decay = 0.995
    min_epsilon = 0.05

    for episode in range(episodes):

        # to define truck_data which will select random trucks from the data set to reset the environment each time.
        state = torch.tensor([env.reset(truck_data)], dtype=torch.float)
        for t in range(1000):
            action = select_action(state, epsilon, output_dim, policy_net)
            next_state, reward, done, _ = env.step(action.item())
            reward = torch.tensor([reward], dtype=torch.float)
            next_state = torch.tensor([next_state], dtype=torch.float)
            done = torch.tensor([done], dtype=torch.float)
            
            memory.append((state, action, reward, next_state, done))
            state = next_state
            
            optimize_model(memory, batch_size, policy_net, target_net, optimizer)
            
            if done:
                break
        
        epsilon = max(min_epsilon, epsilon * epsilon_decay)
        if episode % 50 == 0:
            target_net.load_state_dict(policy_net.state_dict())

    env.close()

if __name__ == "__main__":
    main()
