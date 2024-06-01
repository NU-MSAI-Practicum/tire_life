import matplotlib.pyplot as plt

def plot_rewards(rewards, filename):
    plt.figure(figsize=(10, 5))
    plt.plot(rewards, label='Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Episode Rewards Over Time')
    plt.legend()
    plt.savefig(filename)
    plt.close()

def plot_losses(losses, filename):
    plt.figure(figsize=(10, 5))
    plt.plot(losses, label='Losses')
    plt.xlabel('Episode')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Time')
    plt.legend()
    plt.savefig(filename)
    plt.close()

def plot_optimal_states(optimal_states, filename):
    plt.figure(figsize=(10, 5))
    plt.plot(optimal_states, label='Optimal States')
    plt.xlabel('100 Episode Blocks')
    plt.ylabel('Percentage of Optimal States Achieved')
    plt.title('Optimal States Achieved Over Time')
    plt.legend()
    plt.savefig(filename)
    plt.close()

def plot_valid_actions(valid_actions, filename):
    plt.figure(figsize=(10, 5))
    plt.plot(valid_actions, label='Valid Actions')
    plt.xlabel('Episode')
    plt.ylabel('Percentage of Valid Actions')
    plt.title('Valid Actions Per Episode Over Time')
    plt.legend()
    plt.savefig(filename)
    plt.close()
