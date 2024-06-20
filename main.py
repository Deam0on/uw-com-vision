import argparse
import os
import shutil
from datetime import datetime, timedelta
from pathlib import Path
from data_preparation import split_dataset
from train_model import train_on_dataset
from evaluate_model import evaluate_model
from inference import run_inference
import json
import time

def download_data_from_bucket():
    """
    Download data from a Google Cloud Storage bucket to a local directory.
    
    Parameters:
    - bucket_url: URL of the Google Cloud Storage bucket.
    - local_dir: Local directory to store downloaded data.
    """
    dirpath = Path('/home/deamoon_uw_nn') / 'DATASET'
    if dirpath.exists() and dirpath.is_dir():
        shutil.rmtree(dirpath)

    os.system("gsutil -m cp -r gs://uw-com-vision/DATASET /home/deamoon_uw_nn")

def upload_data_to_bucket():
    """
    Upload data from local directories to a Google Cloud Storage bucket.

    Parameters:
    - local_dirs: List of local directories or files to upload.
    - bucket_url: URL of the Google Cloud Storage bucket.
    """
    # Create a directory with the current time-date stamp
    time_offset = timedelta(hours=2)
    timestamp = (datetime.now() + time_offset).strftime("%Y%m%d_%H%M%S")
    archive_path = f"gs://uw-com-vision/Archive/{timestamp}/"

    # Upload the specified directories or files to the bucket
    os.system(f"gsutil -m cp -r /home/deamoon_uw_nn/*.png {archive_path}")
    os.system(f"gsutil -m cp -r /home/deamoon_uw_nn/*.csv {archive_path}")
    os.system(f"gsutil -m cp -r /home/deamoon_uw_nn/output/ {archive_path}")

def read_eta_data():
    if os.path.exists(ETA_FILE):
        with open(ETA_FILE, 'r') as file:
            return json.load(file)
    else:
        return {
            "prepare": {"average_time": 300},
            "evaluate": {"average_time": 1800},
            "inference": {"average_time_per_image": 5, "buffer": 1.1}
        }

def update_eta_data(task, elapsed_time):
    data = read_eta_data()
    if task in data:
        if 'average_time' in data[task]:
            current_avg = data[task]['average_time']
            data[task]['average_time'] = (current_avg + elapsed_time) / 2
        elif 'average_time_per_image' in data[task]:
            avg_time_per_image = data[task]['average_time_per_image']
            num_images = elapsed_time['num_images']
            data[task]['average_time_per_image'] = (avg_time_per_image + elapsed_time['total_time'] / num_images) / 2
    else:
        data[task] = {"average_time": elapsed_time}
    
    with open(ETA_FILE, 'w') as file:
        json.dump(data, file, indent=2)

def estimate_eta(task, num_images=0):
    data = read_eta_data()
    if task == 'inference':
        avg_time_per_image = data.get(task, {}).get('average_time_per_image', 1)
        buffer = data.get(task, {}).get('buffer', 1)
        return avg_time_per_image * num_images * buffer
    else:
        return data.get(task, {}).get('average_time', 60)

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
        '--visualize', action='store_true', default=False,
        help="Flag to visualize results during evaluation and inference. Saves visualizations of predictions. Default is False."
    )
    parser.add_argument(
        '--download', action='store_true', default=True,
        help="Flag to download data from Google Cloud Storage before executing the task. Default is False."
    )
    parser.add_argument(
        '--upload', action='store_true', default=True,
        help="Flag to upload results to Google Cloud Storage after executing the task. Default is True."
    )
    parser.add_argument(
    '--threshold', type=float, default=0.65,
    help="Threshold for model ROI heads score."
    )
    
    args = parser.parse_args()

    os.system("gsutil -m cp -r gs://uw-com-vision/dataset_info.json /home/deamoon_uw_nn/uw-com-vision")

    img_dir = os.path.join("/home/deamoon_uw_nn/DATASET", args.dataset_name)  # Set the fixed path for image directory
    output_dir = "/home/deamoon_uw_nn/split_dir"  # Set the fixed path for output directory
    ETA_FILE = '/home/deamoon_uw_nn/uw-com-vision/eta_data.json'

    if args.download:
        print(f"Downloading data for dataset {args.dataset_name} from bucket...")
        download_data_from_bucket()

    # Estimate and display ETA before task execution
    if args.task == 'inference':
        num_images = len([f for f in os.listdir(img_dir) if f.endswith('.tif')])
        eta = estimate_eta('inference', num_images)
    else:
        eta = estimate_eta(args.task)

    print(f"Estimated Time to Complete: {str(timedelta(seconds=eta))}")
    start_time = time.time()

    if args.task == 'prepare':
        print(f"Preparing dataset {args.dataset_name}...")
        split_dataset(img_dir, args.dataset_name)

    elif args.task == 'train':
        print(f"Training model on dataset {args.dataset_name}...")
        train_on_dataset(args.dataset_name, output_dir)
        
    elif args.task == 'evaluate':
        print(f"Evaluating model on dataset {args.dataset_name}...")
        evaluate_model(args.dataset_name, output_dir, args.visualize)
        
    elif args.task == 'inference':
        print(f"Running inference on dataset {args.dataset_name}...")
        os.system("rm -f *.png")
        os.system("rm -f *.csv")
        os.system("rm -f *.jpg")
        # run_inference(args.dataset_name, output_dir, args.visualize)
        run_inference(args.dataset_name, output_dir, args.visualize, args.threshold)

 elapsed_time = time.time() - start_time

if args.task != 'inference':
    update_eta_data(args.task, elapsed_time)
else:
    update_eta_data(args.task, {'total_time': elapsed_time, 'num_images': num_images})
    
    if args.upload:
        print(f"Uploading results for dataset {args.dataset_name} to bucket...")
        upload_data_to_bucket()

if __name__ == "__main__":
    main()
