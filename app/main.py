# main.py
import os
import torch
from datetime import datetime
from train import *

# Disable OpenMP parallelism
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'

# Set PyTorch to use single-threaded execution
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

def main():
    # Create new logs folder by date and time
    current_date = datetime.now().strftime('%Y-%m-%d')
    current_time = datetime.now().strftime('%H-%M-%S')
    
    experiment_folder = './experiments'
    date_folder = os.path.join(experiment_folder, current_date)
    time_folder = os.path.join(date_folder, current_time)
    
    logs_folder = os.path.join(time_folder, 'logs')
    metrics_folder = os.path.join(time_folder, 'metrics')
    model_folder = os.path.join(time_folder, 'metrics')

    os.makedirs(logs_folder, exist_ok=True)
    os.makedirs(metrics_folder, exist_ok=True)

    train(logs_folder, metrics_folder,model_folder)

    print("All details are stored in: ", time_folder)

if __name__ == "__main__":
    main()
