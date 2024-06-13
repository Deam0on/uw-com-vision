import streamlit as st
import subprocess
import os
import json
from google.cloud import storage
from datetime import datetime
from google.api_core import page_iterator
from io import BytesIO
from PIL import Image

# Absolute path to main.py
MAIN_SCRIPT_PATH = '/home/deamoon_uw_nn/uw-com-vision/main.py'

# Path to dataset info file
DATASET_INFO_PATH = './uw-com-vision/dataset_info.json'

# GCS bucket details
GCS_BUCKET_NAME = 'uw-com-vision'
GCS_DATASET_FOLDER = 'DATASET'
GCS_INFERENCE_FOLDER = 'INFERENCE'
GCS_ARCHIVE_FOLDER = 'Archive'

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

# Function to run shell commands
def run_command(command):
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    return result.stdout, result.stderr

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

# Load dataset names from dataset_info.json
def load_dataset_names():
    with open(DATASET_INFO_PATH, 'r') as f:
        data = json.load(f)
    return list(data.keys())

# Function to upload files to GCS
def upload_files_to_gcs(bucket_name, target_folder, files):
    client = storage.Client()
    bucket = client.bucket(bucket_name)
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

# Streamlit interface
st.title("Neural Network Control Panel")

# Task selection
st.header("Run Neural Network Script")
task = st.selectbox("Select Task", ["prepare", "train", "evaluate", "inference"])
dataset_name = st.selectbox("Dataset Name", load_dataset_names())
use_new_data = st.checkbox("Use new data from bucket", value=False)

# Conditionally show the upload section
if use_new_data:
    st.header("Upload Files to GCS")
    upload_folder = st.selectbox(
        "Select folder to upload to",
        [f"{GCS_DATASET_FOLDER}/{dataset_name}", GCS_INFERENCE_FOLDER]
    )
    uploaded_files = st.file_uploader(
        "Choose files to upload",
        accept_multiple_files=True
    )
    if st.button("Upload Files") and uploaded_files:
        upload_files_to_gcs(GCS_BUCKET_NAME, upload_folder, uploaded_files)

# Execute task
if st.button("Run Task"):
    visualize_flag = "--visualize"  # Always true
    upload_flag = "--upload"  # Always true
    download_flag = "--download" if use_new_data else ""

    command = f"python3 {MAIN_SCRIPT_PATH} --task {task} --dataset_name {dataset_name} {visualize_flag} {download_flag} {upload_flag}"
    st.info(f"Running: {command}")
    stdout, stderr = run_command(command)
    st.text(stdout)

    st.session_state.stderr = stderr  # Store stderr in session state

    # Reset the show_errors state if there are new errors
    if stderr:
        st.session_state.show_errors = True
    else:
        st.success(f"{task.capitalize()} task completed successfully!")

# Show errors and warnings
if st.session_state.show_errors:
    if st.button("Hide Errors and Warnings"):
        st.session_state.show_errors = False
    else:
        if contains_errors(st.session_state.stderr):
            st.error(st.session_state.stderr)
        else:
            st.warning(st.session_state.stderr)
else:
    if st.session_state.stderr:
        if st.button("Show Errors and Warnings"):
            st.session_state.show_errors = True

# List folders in the GCS bucket
st.header("Google Cloud Storage")
if not st.session_state.folders:
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
