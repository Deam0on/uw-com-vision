import streamlit as st
import subprocess
import os
import json
from google.cloud import storage
from datetime import datetime
from google.api_core import page_iterator
from io import BytesIO
from PIL import Image
import time
import re  # Regular expressions for parsing ETA

# Absolute path to main.py
MAIN_SCRIPT_PATH = '/home/deamoon_uw_nn/uw-com-vision/main.py'

# GCS bucket details
GCS_BUCKET_NAME = 'uw-com-vision'
GCS_DATASET_FOLDER = 'DATASET'
GCS_INFERENCE_FOLDER = 'DATASET/INFERENCE'
GCS_ARCHIVE_FOLDER = 'Archive'
GCS_DATASET_INFO_PATH = 'dataset_info.json'

def _item_to_value(iterator, item):
    return item

def list_directories(bucket_name, prefix):
    if prefix and not prefix.endswith('/'):
        prefix += '/'

    extra_params = {
        "projection": "noAcl",
        "prefix": prefix,
        "delimiter": '/'
    }

    gcs = storage.Client()

    path = "/b/" + bucket_name + "/o"

    iterator = page_iterator.HTTPIterator(
        client=gcs,
        api_request=gcs._connection.api_request,
        path=path,
        items_key='prefixes',
        item_to_value=_item_to_value,
        extra_params=extra_params,
    )

    return [x for x in iterator]

# Function to run shell commands with ETA
def run_command_with_eta(command, task):
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    eta_placeholder = st.empty()
    progress_bar = st.progress(0)

    for line in iter(process.stdout.readline, ''):
        st.text(line.strip())  # Display output line by line

        # Check for ETA in the output for the training task
        if task == "train":
            match = re.search(r'ETA: (\d+:\d+:\d+)', line)
            if match:
                eta = match.group(1)
                eta_placeholder.text(f"Estimated Time Remaining: {eta}")
            else:
                # Update progress bar if ETA is not found
                match_progress = re.search(r'\[.*?(\d+)%\]', line)
                if match_progress:
                    progress = int(match_progress.group(1)) / 100.0
                    progress_bar.progress(int(progress * 100))

    stdout, stderr = process.communicate()
    process.stdout.close()
    process.stderr.close()
    progress_bar.progress(100)
    eta_placeholder.text("Task completed.")
    return stdout, stderr

# Function to list .png files in a GCS folder
def list_png_files_in_gcs_folder(bucket_name, folder):
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blobs = bucket.list_blobs(prefix=folder)
    return [blob for blob in blobs if blob.name.endswith('.png')]

# Function to list specific .csv files in a GCS folder
def list_specific_csv_files_in_gcs_folder(bucket_name, folder):
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blobs = bucket.list_blobs(prefix=folder)
    return [
        blob for blob in blobs
        if blob.name.endswith('results_x_pred_1.csv') or blob.name.endswith('results_x_pred_0.csv')
    ]

# Function to check if stderr contains errors
def contains_errors(stderr):
    error_keywords = ['error', 'failed', 'exception', 'traceback', 'critical']
    return any(keyword in stderr.lower() for keyword in error_keywords)

# Function to load dataset names from JSON in GCS
def load_dataset_names_from_gcs():
    client = storage.Client()
    bucket = client.bucket(GCS_BUCKET_NAME)
    blob = bucket.blob(GCS_DATASET_INFO_PATH)
    
    # Try to download the file
    try:
        data = json.loads(blob.download_as_bytes())
    except Exception as e:
        # If the file does not exist, initialize with an empty dictionary
        st.warning(f"dataset_info.json not found. Initializing a new one.")
        data = {}
        save_dataset_names_to_gcs(data)
    
    return data

# Function to save dataset names to JSON in GCS
def save_dataset_names_to_gcs(data):
    client = storage.Client()
    bucket = client.bucket(GCS_BUCKET_NAME)
    blob = bucket.blob(GCS_DATASET_INFO_PATH)
    blob.upload_from_string(json.dumps(data, indent=2), content_type='application/json')
    st.write("Dataset info updated in GCS.")

# Function to upload files to GCS
def upload_files_to_gcs(bucket_name, target_folder, files, overwrite):
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    
    # If overwriting, delete all existing blobs in the target folder
    if overwrite:
        blobs = bucket.list_blobs(prefix=target_folder)
        for blob in blobs:
            blob.delete()
        st.write(f"Existing files in '{target_folder}' have been deleted.")
    
    for file in files:
        blob = bucket.blob(f"{target_folder}/{file.name}")
        blob.upload_from_file(file)
        st.write(f"Uploaded {file.name} to {target_folder}")

# Initialize session state
if 'show_errors' not in st.session_state:
    st.session_state.show_errors = False
if 'stderr' not in st.session_state:
    st.session_state.stderr = ""
if 'folders' not in st.session_state:
    st.session_state.folders = []
if 'show_images' not in st.session_state:
    st.session_state.show_images = False
if 'datasets' not in st.session_state:
    st.session_state.datasets = load_dataset_names_from_gcs()
if 'confirm_delete' not in st.session_state:
    st.session_state.confirm_delete = False

# Streamlit interface
st.title("PaCE Neural Network Control Panel")

# Task selection
st.header("Script controls")
use_new_data = st.checkbox("Use new data from bucket", value=False)

# Slider for detection threshold
st.header("Detection Threshold")
detection_threshold = st.slider("Set detection threshold", min_value=0.0, max_value=1.0, value=0.5, step=0.01)

new_dataset = st.checkbox("New dataset")
if new_dataset:
    new_dataset_name = st.text_input("Enter new dataset name")
    if new_dataset_name:
        path1 = f"/home/deamoon_uw_nn/DATASET/{new_dataset_name}/"
        path2 = path1
        new_classes = st.text_input("Enter classes (comma separated)")
        if st.button("Add Dataset"):
            classes = [cls.strip() for cls in new_classes.split(',')] if new_classes else []
            if new_dataset_name and classes:
                st.session_state.datasets[new_dataset_name] = [path1, path2, classes]
                save_dataset_names_to_gcs(st.session_state.datasets)
                st.success(f"Dataset '{new_dataset_name}' added.")
            else:
                st.warning("Please enter a valid dataset name and classes.")
                
task = st.selectbox("Select Task", ["prepare", "train", "evaluate", "inference"])
dataset_name = st.selectbox("Dataset Name", list(st.session_state.datasets.keys()))

# Align checkbox and button to the right side
col1, col2 = st.columns([3, 1])
with col1:
    # Execute task
    if st.button("Run Task"):
        visualize_flag = "--visualize"  # Always true
        upload_flag = "--upload"  # Always true
        download_flag = "--download" if use_new_data else ""
        threshold_flag = f"--threshold {detection_threshold}"

        command = f"python3 {MAIN_SCRIPT_PATH} --task {task} --dataset_name {dataset_name} {visualize_flag} {download_flag} {upload_flag} {threshold_flag}"
        st.info(f"Running: {command}")
        
        with st.spinner('Running task...'):
            stdout, stderr = run_command_with_eta(command, task)

        st.text(stdout)

        st.session_state.stderr = stderr  # Store stderr in session state

        # Reset the show_errors state if there are new errors
        if stderr and contains_errors(stderr):
            st.session_state.show_errors = True
        else:
            st.success(f"{task.capitalize()} task completed successfully!")

with col2:
    confirm_deletion = st.checkbox("Confirm Deletion")
    if st.button("Remove Dataset"):
        if confirm_deletion:
            del st.session_state.datasets[dataset_name]
            save_dataset_names_to_gcs(st.session_state.datasets)
            st.success(f"Dataset '{dataset_name}' deleted.")
            st.session_state.confirm_delete = False  # Automatically uncheck after deletion
            st.experimental_rerun()  # Refresh to reflect deletion
        else:
            st.warning("Please check the confirmation box to delete the dataset.")

# Conditionally show the upload section
if use_new_data:
    st.header("Upload Files to GCS")
    upload_folder = st.selectbox(
        "Select folder to upload to",
        [f"{GCS_DATASET_FOLDER}/{dataset_name}", GCS_INFERENCE_FOLDER]
    )
    overwrite = st.checkbox("Overwrite existing data in the folder")
    uploaded_files = st.file_uploader(
        "Choose files to upload",
        accept_multiple_files=True
    )
    if st.button("Upload Files") and uploaded_files:
        with st.spinner('Uploading files...'):
            upload_files_to_gcs(GCS_BUCKET_NAME, upload_folder, uploaded_files, overwrite)
        st.success("Files uploaded successfully.")

# Show errors and warnings
if st.session_state.show_errors:
    if st.button("Hide Errors and Warnings"):
        st.session_state.show_errors = False
    else:
        if contains_errors(st.session_state.stderr):
            st.error(st.session_state.stderr)
else:
    if st.session_state.stderr and st.button("Show Errors and Warnings"):
        st.session_state.show_errors = True

# List folders in the GCS bucket
st.header("Google Cloud Storage")
if 'folders' not in st.session_state or not st.session_state.folders:
    st.session_state.folders = list_directories(GCS_BUCKET_NAME, GCS_ARCHIVE_FOLDER)

if st.session_state.folders:
    folder_dropdown = st.selectbox("Select Folder", st.session_state.folders, format_func=lambda x: x.strip('/'))
else:
    st.write("No folders found in the GCS bucket.")

# Button to show inference images
if st.button("Show Inference Images") and st.session_state.folders:
    st.session_state.show_images = True

# Display images if available
if st.session_state.show_images:
    st.write(f"Displaying images from folder: {folder_dropdown}")
    image_files = list_png_files_in_gcs_folder(GCS_BUCKET_NAME, folder_dropdown)
    if image_files:
        for blob in image_files:
            img_bytes = blob.download_as_bytes()
            img = Image.open(BytesIO(img_bytes))
            st.image(img, caption=os.path.basename(blob.name))
    else:
        st.write("No images found in the selected folder.")

    # Button to download specific CSV files
    csv_files = list_specific_csv_files_in_gcs_folder(GCS_BUCKET_NAME, folder_dropdown)
    if csv_files:
        for blob in csv_files:
            csv_bytes = blob.download_as_bytes()
            csv_name = os.path.basename(blob.name)
            if csv_name == 'results_x_pred_1.csv':
                download_name = 'results_pores.csv'
            elif csv_name == 'results_x_pred_0.csv':
                download_name = 'results_throats.csv'
            else:
                continue  # Skip files that don't match the specific names
            st.download_button(
                label=f"Download {download_name}",
                data=csv_bytes,
                file_name=download_name,
                mime='text/csv'
            )
    else:
        st.write("No specific CSV files found in the selected folder.")