import os
import shutil
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
    # Create a new logs folder by date and time
    current_date = datetime.now().strftime('%Y-%m-%d')
    current_time = datetime.now().strftime('%H-%M-%S')
    logs_folder = f'logs/{current_date}/{current_time}'

    if os.path.exists('logs'):
        shutil.rmtree('logs')
    os.makedirs(logs_folder)

    train(logs_folder)


if __name__ == "__main__":
    main()