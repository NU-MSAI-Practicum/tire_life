import os
import torch
import numpy as np
from env import TruckFleetEnv
from dqn import DQNAgent

def predict_and_save(logs_folder, model_path, num_trucks, num_tires_per_truck, health_threshold, max_steps, num_steps, initial_state, output_file, detailed_log_file):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    initial_state = np.array(initial_state).reshape(num_trucks, num_tires_per_truck)

    env = TruckFleetEnv(num_trucks, num_tires_per_truck, health_threshold, max_steps)
    env.reset(initial_state=initial_state)
    
    state_dim = env.num_trucks * env.num_tires_per_truck
    action_dims = {
        "replace": [num_trucks, num_tires_per_truck],
        "swap": [num_trucks, num_trucks, 2, num_tires_per_truck - 2]
    }

    agent = DQNAgent(state_dim, action_dims, device=device)
    agent.policy_net.load_state_dict(torch.load(model_path, map_location=device))
    agent.policy_net.eval()

    predictions = agent.generate_predictions(env, initial_state, num_steps)
    agent.save_predictions_to_excel(predictions, output_file)
    agent.save_detailed_log_to_excel(initial_state, predictions, detailed_log_file)

    # Log actions verbally
    for prediction in predictions:
        print(prediction['Log'])

    print(f"Predictions saved to {output_file}")
    print(f"Detailed log saved to {detailed_log_file}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Truck Tire Management Prediction")
    parser.add_argument('--logs_folder', type=str, default="./app/logs", help="Path to the logs folder")
    parser.add_argument('--model_path', type=str, default="./app/models/dqn_truck_fleet.pth", help="Path to the trained model")
    parser.add_argument('--num_trucks', type=int, default=2, help="Number of trucks")
    parser.add_argument('--num_tires_per_truck', type=int, default=10, help="Number of tires per truck")
    parser.add_argument('--health_threshold', type=float, default=0.09, help="Health threshold")
    parser.add_argument('--max_steps', type=int, default=20, help="Maximum steps per episode")
    parser.add_argument('--num_steps', type=int, default=20, help="Number of steps for prediction")
    parser.add_argument('--initial_state', type=float, nargs='+', help="Initial state values")
    parser.add_argument('--output_file', type=str, default="app/logs/predictions.xlsx", help="Output Excel file name")
    parser.add_argument('--detailed_log_file', type=str, default="app/logs/detailed_log.xlsx", help="Detailed log Excel file name")

    args = parser.parse_args()

    predict_and_save(args.logs_folder, args.model_path, args.num_trucks, args.num_tires_per_truck, args.health_threshold, args.max_steps, args.num_steps, args.initial_state, args.output_file, args.detailed_log_file)