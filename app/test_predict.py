import sys
import os
import numpy as np
import pandas as pd

# Add the app directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

from predict import Predictor



def test_predictor():
    # Path to your trained model
    model_path = 'app/models/dqn_truck_fleet.pth'

    # Initialize the predictor
    predictor = Predictor(model_path=model_path)

    # Sample state input
    state = np.array([
        [0.01, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.9],
        [0.01, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.9, 0.5]
    ])

    # Predict and log the actions
    log_file_path = 'app/logs/prediction_log.xlsx'
    initial_state, final_state, log_file_path = predictor.predict(state, log_file_path)

    # Print the initial and final states
    print("Initial State:")
    print(initial_state)
    print("\nFinal State:")
    print(final_state)

    # Print the content of the log file
    if os.path.exists(log_file_path):
        log_df = pd.read_excel(log_file_path)
        print("\nAction Log:")
        print(log_df)
    else:
        print("Log file not found!")

if __name__ == "__main__":
    test_predictor()
