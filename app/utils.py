import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def process_excel(df, num_trucks, num_tires_per_truck):
    expected_size = num_trucks * num_tires_per_truck
    actual_size = df.size
    if actual_size != expected_size:
        raise ValueError(f"Excel file size ({actual_size}) does not match the expected size ({expected_size}) for {num_trucks} trucks and {num_tires_per_truck} tires per truck.")
    initial_state = df.to_numpy().reshape(num_trucks, num_tires_per_truck)
    return initial_state

def generate_initial_state(num_trucks, num_tires_per_truck):
    # Generates a random initial state
    return np.round(np.random.rand(num_trucks, num_tires_per_truck), 2)

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
