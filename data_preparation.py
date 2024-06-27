import os
import json
import csv
import random
from sklearn.model_selection import train_test_split

def split_dataset(img_dir, dataset_name, test_size=0.2, seed=42):
    """
    Splits the dataset into training and testing sets and saves the split information.
    
    Parameters:
    - img_dir: Directory containing images.
    - dataset_name: Name of the dataset.
    - test_size: Proportion of the dataset to include in the test split.
    - seed: Random seed for reproducibility.
    
    Returns:
    - train_files: List of training label files.
    - test_files: List of testing label files.
    """
    # Set the random seed for reproducibility
    random.seed(seed)
    
    # List all label files in the image directory
    label_files = [f for f in os.listdir(img_dir) if f.endswith('.json')]
    
    # Split the label files into training and testing sets
    train_files, test_files = train_test_split(label_files, test_size=test_size, random_state=seed)

    # Directory to save the split information
    split_dir = "/home/deamoon_uw_nn/split_dir/"
    os.makedirs(split_dir, exist_ok=True)
    
    # Path to save the split JSON file
    split_file = os.path.join(split_dir, f"{dataset_name}_split.json")
    split_data = {'train': train_files, 'test': test_files}
    
    # Save the split data to a JSON file
    with open(split_file, 'w') as f:
        json.dump(split_data, f)

    print(f"Training & Testing data successfully split into {split_file}")

    return train_files, test_files
