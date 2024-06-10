import argparse
import os
from data_preparation import split_dataset
from train_model import train_on_dataset
from evaluate_model import evaluate_model
from inference import run_inference  # Assuming you have an inference script or function

def main():
    parser = argparse.ArgumentParser(
        description="Pipeline for preparing data, training, evaluating, and running inference on models."
    )

    parser.add_argument(
        '--task', type=str, required=True, choices=['prepare', 'train', 'evaluate', 'inference'],
        help="Task to perform:\n"
             "- 'prepare': Prepare the dataset by splitting into train and test sets.\n"
             "- 'train': Train a model on the dataset.\n"
             "- 'evaluate': Evaluate the trained model on the test set.\n"
             "- 'inference': Run inference on new data using the trained model."
    )
    parser.add_argument(
        '--dataset_name', type=str, required=True,
        help="Name of the dataset to use (e.g., 'polyhipes')."
    )
    parser.add_argument(
        '--visualize', action='store_true',
        help="Flag to visualize results during evaluation and inference. Saves visualizations of predictions."
    )
    
    args = parser.parse_args()

    img_dir = "/path/to/images"  # Set the fixed path for image directory
    output_dir = "./output"  # Set the fixed path for output directory

    if args.task == 'prepare':
        print(f"Preparing dataset {args.dataset_name}...")
        split_dataset(img_dir, args.dataset_name, output_dir)

    elif args.task == 'train':
        print(f"Training model on dataset {args.dataset_name}...")
        train_on_dataset(args.dataset_name, output_dir)
        
    elif args.task == 'evaluate':
        print(f"Evaluating model on dataset {args.dataset_name}...")
        evaluate_model(args.dataset_name, output_dir, args.visualize)
        
    elif args.task == 'inference':
        print(f"Running inference on dataset {args.dataset_name}...")
        run_inference(args.dataset_name, output_dir, args.visualize)

if __name__ == "__main__":
    main()
