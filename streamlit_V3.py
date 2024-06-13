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
GCS_FOLDER = 'Archive'

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

# Function to list .csv files in a GCS folder
def list_csv_files_in_gcs_folder(bucket_name, folder):
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blobs = bucket.list_blobs(prefix=folder)
    return [blob for blob in blobs if blob.name.endswith('.csv')]

# Function to check if stderr contains errors
def contains_errors(stderr):
    error_keywords = ['error', 'failed', 'exception', 'traceback', 'critical']
    return any(keyword in stderr.lower() for keyword in error_keywords)

# Load dataset names from dataset_info.json
def load_dataset_names():
    with open(DATASET_INFO_PATH, 'r') as f:
        data = json.load(f)
    return list(data.keys())

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
download = st.checkbox("Download Data", value=False)

# Execute task
if st.button("Run Task"):
    visualize_flag = "--visualize"  # Always true
    upload_flag = "--upload"  # Always true
    download_flag = "--download" if download else ""

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
    st.session_state.folders = list_directories(GCS_BUCKET_NAME, GCS_FOLDER)

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

    # Button to download CSV files
    csv_files = list_csv_files_in_gcs_folder(GCS_BUCKET_NAME, folder_dropdown)
    if csv_files:
        for blob in csv_files:
            csv_bytes = blob.download_as_bytes()
            csv_name = os.path.basename(blob.name)
            st.download_button(
                label=f"Download {csv_name}",
                data=csv_bytes,
                file_name=csv_name,
                mime='text/csv'
            )
    else:
        st.write("No CSV files found in the selected folder.")
