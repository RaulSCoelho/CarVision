from typing import Callable, Dict, List, Optional
import numpy as np
import pandas as pd
from scipy.io import loadmat
from torch.utils.data import DataLoader
from utils.time import loading_message

ColsType = Dict[str, str]
FilterConditionType = Optional[Callable[[dict], np.ndarray]]

def mat_to_frame(path_to_mat_file: str, data_key: str, cols: ColsType, filter: FilterConditionType = None) -> pd.DataFrame:
    """
    Convert a .mat file to a pandas DataFrame.

    Args:
    - path_to_mat_file (str): Path to the .mat file.
    - data_key (str): Key to the data in the .mat file.
    - cols (Dict[str, str]): Dictionary of column names for the DataFrame.

    Returns:
    - df (pd.DataFrame): DataFrame containing the data.
    """
    # Load data from .mat file
    data = loadmat(path_to_mat_file)

    # Extract data using the specified key
    data = data[data_key]

    # Apply filter if provided
    if filter is not None:
        data = data[filter(data)]

    # Convert data to a DataFrame
    df = pd.DataFrame({ col_name: data[col_key].squeeze() for col_name, col_key in cols.items() })

    return df

def mat_to_list(path_to_mat_file: str, data_key: str) -> List[str]:
    """
    Convert a .mat file to a list of strings.

    Args:
    - path_to_mat_file (str): Path to the .mat file.
    - data_key (str): Key to the data in the .mat file.

    Returns:
    - data (list): List containing the data as strings.
    """
    # Load data from .mat file
    data = loadmat(path_to_mat_file)
    # Extract data using the specified key
    data_array = data[data_key].squeeze()
    # Convert numpy array to a list of strings
    data_list = [str(item[0]) for item in data_array]

    return data_list

@loading_message("Calculating mean and std")
def get_mean_and_std(loader: DataLoader):
    mean = 0.
    std = 0.
    total_images_count = 0
    for images, _ in loader:
        batch_samples = images.size(0)
        images = images.view(batch_samples, images.size(1), -1)
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)
        total_images_count += batch_samples

    mean /= total_images_count
    std /= total_images_count

    return mean, std
