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
