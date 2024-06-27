import argparse
import os
import shutil
from datetime import datetime, timedelta
from pathlib import Path
import json
from data_preparation import split_dataset
from train_model import train_on_dataset
from evaluate_model import evaluate_model
from inference import run_inference

ETA_FILE = '/home/deamoon_uw_nn/uw-com-vision/eta_data.json'

def download_data_from_bucket():
    """
    Download data from a Google Cloud Storage bucket to a local directory.
    
    Returns:
    - float: Time taken to download data in seconds.
    """
    download_start_time = datetime.now()
    dirpath = Path('/home/deamoon_uw_nn') / 'DATASET'
    if dirpath.exists() and dirpath.is_dir():
        shutil.rmtree(dirpath)

    os.system("gsutil -m cp -r gs://uw-com-vision/DATASET /home/deamoon_uw_nn")
    download_end_time = datetime.now()

    return (download_end_time - download_start_time).total_seconds()

def upload_data_to_bucket():
    """
    Upload data from local directories to a Google Cloud Storage bucket.
    
    Returns:
    - float: Time taken to upload data in seconds.
    """
    upload_start_time = datetime.now()
    time_offset = timedelta(hours=2)
    timestamp = (datetime.now() + time_offset).strftime("%Y%m%d_%H%M%S")
    archive_path = f"gs://uw-com-vision/Archive/{timestamp}/"

    os.system(f"gsutil -m cp -r /home/deamoon_uw_nn/*.png {archive_path}")
    os.system(f"gsutil -m cp -r /home/deamoon_uw_nn/*.csv {archive_path}")
    os.system(f"gsutil -m cp -r /home/deamoon_uw_nn/output/ {archive_path}")
    upload_end_time = datetime.now()

    return (upload_end_time - upload_start_time).total_seconds()

def read_eta_data():
    """
    Read ETA data from a JSON file.
    
    Returns:
    - dict: ETA data.
    """
    if os.path.exists(ETA_FILE):
        with open(ETA_FILE, 'r') as file:
            return json.load(file)
    else:
        return {
            "prepare": {"average_time": 300},
            "evaluate": {"average_time": 1800},
            "inference": {"average_time_per_image": 5, "buffer": 1.1},
            "download": {"average_time": 60},
            "upload": {"average_time": 60}
        }

def update_eta_data(task, time_taken, num_images=0):
    """
    Update ETA data with new timings.

    Parameters:
    - task: Task name (e.g., 'inference', 'prepare').
    - time_taken: Time taken for the task.
    - num_images: Number of images processed (relevant for 'inference' task).
    """
    data = read_eta_data()

    if task == 'inference':
        avg_time_per_image = time_taken / max(num_images, 1)
        current_avg = data.get(task, {}).get('average_time_per_image', avg_time_per_image)
        buffer = data.get(task, {}).get('buffer', 1.1)

        new_avg_time_per_image = (current_avg + avg_time_per_image) / 2
        data[task] = {"average_time_per_image": new_avg_time_per_image, "buffer": buffer}
    else:
        current_avg = data.get(task, {}).get('average_time', time_taken)

        new_avg_time = (current_avg + time_taken) / 2
        data[task] = {"average_time": new_avg_time}

    with open(ETA_FILE, 'w') as file:
        json.dump(data, file, indent=2)

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
        '--threshold', type=float, default=0.65,
        help="Threshold for inference. Default is 0.65."
    )
    parser.add_argument(
        '--visualize', action='store_true', default=False,
        help="Flag to visualize results during evaluation and inference. Saves visualizations of predictions. Default is False."
    )
    parser.add_argument(
        '--download', action='store_true', default=True,
        help="Flag to download data from Google Cloud Storage before executing the task. Default is True."
    )
    parser.add_argument(
        '--upload', action='store_true', default=True,
        help="Flag to upload results to Google Cloud Storage after executing the task. Default is True."
    )

    args = parser.parse_args()

    os.system("gsutil -m cp -r gs://uw-com-vision/dataset_info.json /home/deamoon_uw_nn/uw-com-vision")

    img_dir = os.path.join("/home/deamoon_uw_nn/DATASET", args.dataset_name)
    output_dir = "/home/deamoon_uw_nn/split_dir"

    total_start_time = datetime.now()
    download_time_taken = 0
    upload_time_taken = 0

    print(f"Running task: {args.task} on dataset: {args.dataset_name}")  # Debug: print task and dataset
    
    if args.download:
        print(f"Downloading data for dataset {args.dataset_name} from bucket...")
        download_time_taken = download_data_from_bucket()

    if args.task == 'prepare':
        print(f"Preparing dataset {args.dataset_name}...")
        task_start_time = datetime.now()
        split_dataset(img_dir, args.dataset_name)
        task_end_time = datetime.now()

    elif args.task == 'train':
        print(f"Training model on dataset {args.dataset_name}...")
        train_on_dataset(args.dataset_name, output_dir)

    elif args.task == 'evaluate':
        print(f"Evaluating model on dataset {args.dataset_name}...")
        task_start_time = datetime.now()
        evaluate_model(args.dataset_name, output_dir, args.visualize)
        task_end_time = datetime.now()

    elif args.task == 'inference':
        print(f"Running inference on dataset {args.dataset_name}...")

        os.system("rm -f *.png")
        os.system("rm -f *.csv")
        os.system("rm -f *.jpg")

        num_images = len([f for f in os.listdir(img_dir) if f.endswith(('.tif', '.png', '.jpg'))])

        task_start_time = datetime.now()
        run_inference(args.dataset_name, output_dir, args.visualize)
        task_end_time = datetime.now()

        inference_time_taken = (task_end_time - task_start_time).total_seconds()
        update_eta_data('inference', inference_time_taken, num_images)

    total_end_time = datetime.now()
    total_time_taken = (total_end_time - total_start_time).total_seconds()
    
    if args.upload:
        print(f"Uploading results for dataset {args.dataset_name} to bucket...")
        upload_time_taken = upload_data_to_bucket()

    if args.task != 'inference':
        update_eta_data(args.task, total_time_taken)

    if args.download:
        update_eta_data('download', download_time_taken)
    if args.upload:
        update_eta_data('upload', upload_time_taken)

if __name__ == "__main__":
    main()
