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
        '--img_dir', type=str, required=True,
        help="Directory containing the images for the dataset."
    )
    parser.add_argument(
        '--label_dir', type=str, required=True,
        help="Directory containing the label files for the dataset."
    )
    parser.add_argument(
        '--output_dir', type=str, required=True,
        help="Directory to save outputs such as model checkpoints, split files, evaluation results, and inference outputs."
    )
    parser.add_argument(
        '--model_path', type=str, default=None,
        help="Path to a pre-trained model (.pth file). Required for 'evaluate' and 'inference' tasks."
    )
    parser.add_argument(
        '--visualize', action='store_true',
        help="Flag to visualize results during evaluation and inference. Saves visualizations of predictions."
    )
    
    args = parser.parse_args()

    if args.task == 'prepare':
        print(f"Preparing dataset {args.dataset_name}...")
        split_dataset(args.img_dir, args.label_dir, args.dataset_name, args.output_dir)

    elif args.task == 'train':
        print(f"Training model on dataset {args.dataset_name}...")
        train_on_dataset(args.dataset_name, dataset_info)
        
    elif args.task == 'evaluate':
        if not args.model_path:
            raise ValueError("Model path is required for evaluation.")
        print(f"Evaluating model on dataset {args.dataset_name}...")
        evaluate_model(args.dataset_name, args.model_path, args.output_dir, args.visualize)
        
    elif args.task == 'inference':
        if not args.model_path:
            raise ValueError("Model path is required for inference.")
        print(f"Running inference on dataset {args.dataset_name}...")
        run_inference(args.selected_dataset, dataset_info)

if __name__ == "__main__":

    
    # Available datasets
    dataset_info = {
        "polyhipes": ("/home/deamoon_uw_nn/DATASET/polyhipes/", "/home/deamoon_uw_nn/DATASET/polyhipes/", ["throat", "pore"])
    }

    # Call training
    selected_dataset = "polyhipes"  # User-selected dataset


    
     # dataset_name = "polyhipes"  # Example dataset name
#     model_path = "./trained_models/polyhipes/model_final.pth"  # Example path to model
#     output_dir = "./evaluation_results"  # Directory to save evaluation results and visualizations

#     # Evaluate the model
#     metrics = evaluate_model(dataset_name, model_path, output_dir, visualize=True)
#     print(metrics)
    
    main()
