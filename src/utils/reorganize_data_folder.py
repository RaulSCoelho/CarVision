import os
import re
import shutil
import random
import logging
import pandas as pd
from typing import List

def remove_empty_dirs(source_dir):
    # Get list of folders in source directory
    folders = os.listdir(source_dir)

    for folder in folders:
        if not os.listdir(os.path.join(source_dir, folder)):
            shutil.rmtree(os.path.join(source_dir, folder))

def split_data(source_dir, percentage):
    # Get list of folders in source directory
    folders = os.listdir(source_dir)

    # Filtra os itens que não são "test" ou "train"
    folders = [item for item in folders if item not in ("test", "train")]

    train_dir = os.path.join(source_dir, "train")
    test_dir = os.path.join(source_dir, "test")

    # Create train and test directories if they don't exist
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    for folder in folders:
        folder_dir = os.path.join(source_dir, folder)

        # Get list of images in current folder
        images = os.listdir(folder_dir)

        # Create subdirectories in train and test directories
        os.makedirs(os.path.join(train_dir, folder), exist_ok=True)
        os.makedirs(os.path.join(test_dir, folder), exist_ok=True)

        # Calculate number of images to move to test directory
        num_train_images = int(len(images) * percentage)

        # Shuffle images
        random.shuffle(images)

        # Move images to train and test directories
        for i, image in enumerate(images):
            src = os.path.join(folder_dir, image)
            dest = os.path.join(train_dir if i < num_train_images else test_dir, folder, image)
            try:
                shutil.move(src, dest)
            except FileNotFoundError as e:
                logging.warning(f"File not found: {src}")
            except Exception as e:
                logging.error(f"Error moving file: {src} -> {dest}. Reason: {e}")

    print("Data splitting complete.")

def reorganize_data_folder(data_folder_dir: str, annotations_csv_dir: str, image_col: str, class_col: str, classes: List[str]):
    """
    Reorganizes images in the data folder based on class labels provided in a CSV file using pandas.

    Args:
        data_folder_dir (str): Directory containing the images.
        annotations_csv_dir (str): Path to the CSV file containing image annotations.
        image_col (str): Name of the column containing image names in the CSV file.
        class_col (str): Name of the column containing class labels (indices) in the CSV file.
        classes (List[str]): List of class labels.

    Returns:
        None
    """
    # Validate input arguments
    if not os.path.isdir(data_folder_dir):
        raise ValueError("Invalid data folder directory.")
    if not os.path.isfile(annotations_csv_dir):
        raise ValueError("Invalid annotations CSV file.")

    # Replace invalid characters in class names
    valid_classes = [replace_invalid_chars(class_name) for class_name in classes]

    # Read CSV file using pandas
    df = pd.read_csv(annotations_csv_dir)

    # Check if every class index inside the DataFrame is inside the classes array
    for class_index in df[class_col]:
        if class_index not in range(1, len(valid_classes) + 1):
            raise ValueError(f"Invalid class index '{class_index}' found in DataFrame. It is not in the classes array.")

    # Create folders for each class
    for class_name in valid_classes:
        class_dir = os.path.join(data_folder_dir, class_name)
        os.makedirs(class_dir, exist_ok=True)

    # Apply lambda function to move files based on class labels
    df.apply(lambda row: move_image(data_folder_dir, row[image_col], valid_classes[int(row[class_col]) - 1], valid_classes), axis=1)

def move_image(data_folder_dir: str, image_name: str, class_name: str, classes: List[str]):
    # Check if class index is valid
    if class_name in classes:
        src = os.path.join(data_folder_dir, image_name)
        dest = os.path.join(data_folder_dir, class_name, image_name)
        try:
            shutil.move(src, dest)
        except FileNotFoundError as e:
            logging.warning(f"File not found: {src}")
        except Exception as e:
            logging.error(f"Error moving file: {src} -> {dest}. Reason: {e}")
    else:
        logging.warning(f"Invalid class '{class_name}' for image '{image_name}'")

def replace_invalid_chars(class_name):
    # Define a pattern to match invalid characters
    invalid_chars_pattern = r'[\/\\:*?"<>|]'
    # Replace invalid characters with underscores and print if replacement occurs
    modified_name = re.sub(invalid_chars_pattern, '_', class_name)

    if modified_name != class_name:
        print(f"Invalid name: {class_name} -> {modified_name}")

    return modified_name
