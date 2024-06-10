import os
import json
import csv  # Add this import at the beginning

def split_dataset(img_dir, label_dir, dataset_name, output_dir, test_size=0.2, seed=42):
    """
    Splits the dataset into training and testing sets, saves the split information,
    and writes the splits to CSV files.

    Parameters:
    - img_dir: Directory containing images.
    - label_dir: Directory containing labels.
    - dataset_name: Name of the dataset.
    - output_dir: Directory to save the split CSV files.
    - test_size: Proportion of the dataset to include in the test split.
    - seed: Random seed for reproducibility.

    Returns:
    - train_files: List of training label files.
    - test_files: List of testing label files.
    """
    random.seed(seed)
    label_files = [f for f in os.listdir(label_dir) if f.endswith('.json')]
    train_files, test_files = train_test_split(label_files, test_size=test_size, random_state=seed)

    # Save the split
    split_dir = "/home/deamoon_uw_nn/split_dir/"
    os.makedirs(split_dir, exist_ok=True)
    split_file = os.path.join(split_dir, f"{dataset_name}_split.json")
    split_data = {'train': train_files, 'test': test_files}
    with open(split_file, 'w') as f:
        json.dump(split_data, f)

    # Save the splits to CSV
    os.makedirs(output_dir, exist_ok=True)
    train_csv_path = os.path.join(output_dir, f"{dataset_name}_train_split.csv")
    test_csv_path = os.path.join(output_dir, f"{dataset_name}_test_split.csv")
    
    with open(train_csv_path, mode='w', newline='') as train_csv:
        writer = csv.writer(train_csv)
        writer.writerow(['filename'])
        for file in train_files:
            writer.writerow([file])
    
    with open(test_csv_path, mode='w', newline='') as test_csv:
        writer = csv.writer(test_csv)
        writer.writerow(['filename'])
        for file in test_files:
            writer.writerow([file])
    
    print(f"Training split saved to {train_csv_path}")
    print(f"Testing split saved to {test_csv_path}")

    return train_files, test_files

# Example usage
# if __name__ == "__main__":
#     dataset_name = "polyhipes"  # Example dataset name
#     img_dir = "/home/deamoon_uw_nn/DATASET/polyhipes/images"  # Example image directory
#     label_dir = "/home/deamoon_uw_nn/DATASET/polyhipes/labels"  # Example label directory
#     output_dir = "./splits"  # Directory to save split CSV files

#     # Split the dataset
#     train_files, test_files = split_dataset(img_dir, label_dir, dataset_name, output_dir)
