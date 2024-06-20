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
import subprocess

ETA_FILE = '/home/deamoon_uw_nn/uw-com-vision/eta_data.json'

def run_command_real_time(command):
    """
    Execute a shell command and yield output line by line in real-time.

    Parameters:
    - command: Command to execute.

    Yields:
    - str: Output line.
    """
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    while True:
        output = process.stdout.readline()
        error = process.stderr.readline()
        if output == '' and error == '' and process.poll() is not None:
            break
        if output:
            yield 'stdout', output.strip()
        if error:
            yield 'stderr', error.strip()
    return_code = process.poll()
    if return_code:
        yield 'error', f"Command failed with exit status {return_code}"

def download_data_from_bucket():
    """
    Download data from a Google Cloud Storage bucket to a local directory.
    """
    download_start_time = datetime.now()
    dirpath = Path('/home/deamoon_uw_nn') / 'DATASET'
    if dirpath.exists() and dirpath.is_dir():
        shutil.rmtree(dirpath)

    os.system("gsutil -m cp -r gs://uw-com-vision/DATASET /home/deamoon_uw_nn")
    download_end_time = datetime.now()

    # Return the download time
    return (download_end_time - download_start_time).total_seconds()

def upload_data_to_bucket():
    """
    Upload data from local directories to a Google Cloud Storage bucket.
    """
    upload_start_time = datetime.now()
    # Create a directory with the current time-date stamp
    time_offset = timedelta(hours=2)
    timestamp = (datetime.now() + time_offset).strftime("%Y%m%d_%H%M%S")
    archive_path = f"gs://uw-com-vision/Archive/{timestamp}/"

    # Upload the specified directories or files to the bucket
    os.system(f"gsutil -m cp -r /home/deamoon_uw_nn/*.png {archive_path}")
    os.system(f"gsutil -m cp -r /home/deamoon_uw_nn/*.csv {archive_path}")
    os.system(f"gsutil -m cp -r /home/deamoon_uw_nn/output/ {archive_path}")
    upload_end_time = datetime.now()

    # Return the upload time
    return (upload_end_time - upload_start_time).total_seconds()

def read_eta_data():
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
    data = read_eta_data()

    if task == 'inference':
        # Calculate average time per image
        avg_time_per_image = time_taken / max(num_images, 1)
        current_avg = data.get(task, {}).get('average_time_per_image', avg_time_per_image)
        buffer = data.get(task, {}).get('buffer', 1.1)

        # Update average time per image
        new_avg_time_per_image = (current_avg + avg_time_per_image) / 2
        data[task] = {"average_time_per_image": new_avg_time_per_image, "buffer": buffer}
    else:
        current_avg = data.get(task, {}).get('average_time', time_taken)

        # Update average time for task
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
        help="Flag to download data from Google Cloud Storage before executing the task. Default is False."
    )
    parser.add_argument(
        '--upload', action='store_true', default=True,
        help="Flag to upload results to Google Cloud Storage after executing the task. Default is True."
    )
    
    args = parser.parse_args()

    os.system("gsutil -m cp -r gs://uw-com-vision/dataset_info.json /home/deamoon_uw_nn/uw-com-vision")

    img_dir = os.path.join("/home/deamoon_uw_nn/DATASET", args.dataset_name)  # Set the fixed path for image directory
    output_dir = "/home/deamoon_uw_nn/split_dir"  # Set the fixed path for output directory

    # Start time for the total operation
    total_start_time = datetime.now()
    download_time_taken = 0
    upload_time_taken = 0
    
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
        command = f"python3 train_model.py --dataset_name {args.dataset_name} --output_dir {output_dir}"
        for output_type, output_line in run_command_real_time(command):
            if output_type == 'stdout':
                print(output_line)
            elif output_type == 'stderr':
                print(f"Error: {output_line}")

    elif args.task == 'evaluate':
        print(f"Evaluating model on dataset {args.dataset_name}...")
        task_start_time = datetime.now()
        evaluate_model(args.dataset_name, output_dir, args.visualize)
        task_end_time = datetime.now()

    elif args.task == 'inference':
        print(f"Running inference on dataset {args.dataset_name}...")

        # Remove old inference results
        os.system("rm -f *.png")
        os.system("rm -f *.csv")
        os.system("rm -f *.jpg")

        # Count the number of images after downloading
        num_images = len([f for f in os.listdir(img_dir) if f.endswith(('.tif', '.png', '.jpg'))])

        # Start time for the inference task
        task_start_time = datetime.now()
        run_inference(args.dataset_name, output_dir, args.visualize)
        task_end_time = datetime.now()

        # Update ETA for inference
        inference_time_taken = (task_end_time - task_start_time).total_seconds()
        update_eta_data('inference', inference_time_taken, num_images)

    # End time for the total operation
    total_end_time = datetime.now()
    total_time_taken = (total_end_time - total_start_time).total_seconds()
    
    if args.upload:
        print(f"Uploading results for dataset {args.dataset_name} to bucket...")
        upload_time_taken = upload_data_to_bucket()

    # Update ETA for other tasks
    if args.task != 'inference':
        update_eta_data(args.task, total_time_taken)

    # Update ETA for download and upload times
    if args.download:
        update_eta_data('download', download_time_taken)
    if args.upload:
        update_eta_data('upload', upload_time_taken)

if __name__ == "__main__":
    main()


# import argparse
# import os
# import shutil
# from datetime import datetime, timedelta
# from pathlib import Path
# from data_preparation import split_dataset
# from train_model import train_on_dataset
# from evaluate_model import evaluate_model
# from inference import run_inference
# import json
# import time

# ETA_FILE = '/home/deamoon_uw_nn/uw-com-vision/eta_data.json'

# def download_data_from_bucket():
#     """
#     Download data from a Google Cloud Storage bucket to a local directory.
#     """
#     dirpath = Path('/home/deamoon_uw_nn') / 'DATASET'
#     if dirpath.exists() and dirpath.is_dir():
#         shutil.rmtree(dirpath)

#     os.system("gsutil -m cp -r gs://uw-com-vision/DATASET /home/deamoon_uw_nn")

# def upload_data_to_bucket():
#     """
#     Upload data from local directories to a Google Cloud Storage bucket.
#     """
#     # Create a directory with the current time-date stamp
#     time_offset = timedelta(hours=2)
#     timestamp = (datetime.now() + time_offset).strftime("%Y%m%d_%H%M%S")
#     archive_path = f"gs://uw-com-vision/Archive/{timestamp}/"

#     # Upload the specified directories or files to the bucket
#     os.system(f"gsutil -m cp -r /home/deamoon_uw_nn/*.png {archive_path}")
#     os.system(f"gsutil -m cp -r /home/deamoon_uw_nn/*.csv {archive_path}")
#     os.system(f"gsutil -m cp -r /home/deamoon_uw_nn/output/ {archive_path}")

# def read_eta_data():
#     if os.path.exists(ETA_FILE):
#         with open(ETA_FILE, 'r') as file:
#             return json.load(file)
#     else:
#         return {
#             "prepare": {"average_time": 300},
#             "evaluate": {"average_time": 1800},
#             "inference": {"average_time_per_image": 5, "buffer": 1.1}
#         }

# def update_eta_data(task, time_taken, num_images=0):
#     data = read_eta_data()

#     if task == 'inference':
#         # Calculate average time per image
#         avg_time_per_image = time_taken / max(num_images, 1)
#         current_avg = data.get(task, {}).get('average_time_per_image', avg_time_per_image)
#         buffer = data.get(task, {}).get('buffer', 1.1)

#         # Update average time per image
#         new_avg_time_per_image = (current_avg + avg_time_per_image) / 2
#         data[task] = {"average_time_per_image": new_avg_time_per_image, "buffer": buffer}
#     else:
#         current_avg = data.get(task, {}).get('average_time', time_taken)

#         # Update average time for task
#         new_avg_time = (current_avg + time_taken) / 2
#         data[task] = {"average_time": new_avg_time}

#     with open(ETA_FILE, 'w') as file:
#         json.dump(data, file, indent=2)

# def estimate_eta(task, num_images=0):
#     data = read_eta_data()
#     if task == 'inference':
#         avg_time_per_image = data.get(task, {}).get('average_time_per_image', 1)
#         buffer = data.get(task, {}).get('buffer', 1)
#         return avg_time_per_image * num_images * buffer
#     else:
#         return data.get(task, {}).get('average_time', 60)

# def main():
#     parser = argparse.ArgumentParser(
#         description="Pipeline for preparing data, training, evaluating, and running inference on models."
#     )

#     parser.add_argument(
#         '--task', type=str, required=True, choices=['prepare', 'train', 'evaluate', 'inference'],
#         help="Task to perform:\n"
#              "- 'prepare': Prepare the dataset by splitting into train and test sets.\n"
#              "- 'train': Train a model on the dataset.\n"
#              "- 'evaluate': Evaluate the trained model on the test set.\n"
#              "- 'inference': Run inference on new data using the trained model."
#     )
#     parser.add_argument(
#         '--dataset_name', type=str, required=True,
#         help="Name of the dataset to use (e.g., 'polyhipes')."
#     )
#     parser.add_argument(
#         '--visualize', action='store_true', default=False,
#         help="Flag to visualize results during evaluation and inference. Saves visualizations of predictions. Default is False."
#     )
#     parser.add_argument(
#         '--download', action='store_true', default=True,
#         help="Flag to download data from Google Cloud Storage before executing the task. Default is False."
#     )
#     parser.add_argument(
#         '--upload', action='store_true', default=True,
#         help="Flag to upload results to Google Cloud Storage after executing the task. Default is True."
#     )
#     parser.add_argument(
#     '--threshold', type=float, default=0.65,
#     help="Threshold for model ROI heads score."
#     )
    
#     args = parser.parse_args()

#     os.system("gsutil -m cp -r gs://uw-com-vision/dataset_info.json /home/deamoon_uw_nn/uw-com-vision")

#     img_dir = os.path.join("/home/deamoon_uw_nn/DATASET", args.dataset_name)  # Set the fixed path for image directory
#     output_dir = "/home/deamoon_uw_nn/split_dir"  # Set the fixed path for output directory

#     # Estimate and display ETA before task execution
#     if args.task == 'inference':
#         num_images = len([f for f in os.listdir(img_dir) if f.endswith('.tif')])
#         eta = estimate_eta('inference', num_images)
#     else:
#         eta = estimate_eta(args.task)

#     # Start time for the total operation
#     total_start_time = datetime.now()
    
#     if args.download:
#         print(f"Downloading data for dataset {args.dataset_name} from bucket...")
#         download_data_from_bucket()

#     if args.task == 'prepare':
#         print(f"Preparing dataset {args.dataset_name}...")
#         task_start_time = datetime.now()
#         split_dataset(img_dir, args.dataset_name)
#         task_end_time = datetime.now()

#     elif args.task == 'train':
#         print(f"Training model on dataset {args.dataset_name}...")
#         train_on_dataset(args.dataset_name, output_dir)

#     elif args.task == 'evaluate':
#         print(f"Evaluating model on dataset {args.dataset_name}...")
#         task_start_time = datetime.now()
#         evaluate_model(args.dataset_name, output_dir, args.visualize)
#         task_end_time = datetime.now()

#     elif args.task == 'inference':
#         print(f"Running inference on dataset {args.dataset_name}...")

#         # Remove old inference results
#         os.system("rm -f *.png")
#         os.system("rm -f *.csv")
#         os.system("rm -f *.jpg")

#         # Count the number of images after downloading
#         num_images = len([f for f in os.listdir(img_dir) if f.endswith(('.tif', '.png', '.jpg'))])

#         # Start time for the inference task
#         task_start_time = datetime.now()
#         run_inference(args.dataset_name, output_dir, args.visualize)
#         task_end_time = datetime.now()

#         # Update ETA for inference
#         inference_time_taken = (task_end_time - task_start_time).total_seconds()
#         update_eta_data('inference', inference_time_taken, num_images)

#     # End time for the total operation
#     total_end_time = datetime.now()
#     total_time_taken = (total_end_time - total_start_time).total_seconds()
    
#     if args.upload:
#         print(f"Uploading results for dataset {args.dataset_name} to bucket...")
#         upload_data_to_bucket()

#     # Update ETA for other tasks
#     if args.task != 'inference':
#         update_eta_data(args.task, total_time_taken)

# if __name__ == "__main__":
#     main()
