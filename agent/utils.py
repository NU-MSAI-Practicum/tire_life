# utils.py
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

def plot_optimal_state_achievements(achievements, filename):
    plt.figure(figsize=(10, 5))
    plt.plot(achievements, label='Optimal State Achieved')
    plt.xlabel('Episode')
    plt.ylabel('Optimal State Achieved (1 = Yes, 0 = No)')
    plt.title('Optimal State Achievements Over Episodes')
    plt.legend()
    plt.savefig(filename)
    plt.close()
