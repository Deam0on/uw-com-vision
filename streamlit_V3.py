import streamlit as st
import subprocess
import os
import json
from google.cloud import storage
from datetime import datetime

# Absolute path to main.py
MAIN_SCRIPT_PATH = '/home/deamoon_uw_nn/uw-com-vision/main.py'

# Directory to list files on the VM
VM_DIR = '/home/deamoon_uw_nn/DATASET/INFERENCE/UPLOAD'

# Path to dataset info file
DATASET_INFO_PATH = './uw-com-vision/dataset_info.json'

# GCS bucket details
GCS_BUCKET_NAME = 'uw-com-vision'
GCS_FOLDER = 'Archive'

# Function to run shell commands
def run_command(command):
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    return result.stdout, result.stderr

# Function to list files in a directory on the VM
def list_files_on_vm(directory):
    try:
        return os.listdir(directory)
    except FileNotFoundError:
        return []

# Function to list folders in a GCS bucket
def list_folders_in_bucket(bucket_name, folder):
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blobs = bucket.list_blobs(prefix=folder, delimiter='/')
    return [blob.name for blob in blobs if blob.name.endswith('/')]

# Function to list .png files in a GCS folder
def list_png_files_in_gcs_folder(bucket_name, folder):
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blobs = bucket.list_blobs(prefix=folder)
    return [blob.name for blob in blobs if blob.name.endswith('.png')]

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

# Streamlit interface
st.title("Neural Network Control Panel")

# Task selection
st.header("Run Neural Network Script")
task = st.selectbox("Select Task", ["prepare", "train", "evaluate", "inference"])
dataset_name = st.selectbox("Dataset Name", load_dataset_names())
visualize = st.checkbox("Visualize Results", value=False)
download = st.checkbox("Download Data", value=False)
upload = st.checkbox("Upload Results", value=True)

# Execute task
if st.button("Run Task"):
    visualize_flag = "--visualize" if visualize else ""
    download_flag = "--download" if download else ""
    upload_flag = "--upload" if upload else ""

    command = f"python3 {MAIN_SCRIPT_PATH} --task {task} --dataset_name {dataset_name} {visualize_flag} {download_flag} {upload_flag}"
    st.info(f"Running: {command}")
    stdout, stderr = run_command(command)
    st.text(stdout)

    st.session_state.stderr = stderr  # Store stderr in session state
    
    # Reset the show_errors state
    st.session_state.show_errors = False
    
    if stderr:
        st.session_state.show_errors = True
    else:
        st.success(f"{task.capitalize()} task completed successfully!")

# Show errors and warnings
if st.session_state.show_errors:
    if st.button("Show Errors and Warnings"):
        if contains_errors(st.session_state.stderr):
            st.error(st.session_state.stderr)
        else:
            st.warning(st.session_state.stderr)

# List files on the VM
st.header("Files on VM")
vm_files = list_files_on_vm(VM_DIR)
if vm_files:
    for file in vm_files:
        st.write(file)
else:
    st.write("No files found in the specified directory on the VM.")

# List folders in the GCS bucket
st.header("Google Cloud Storage")
folders = list_folders_in_bucket(GCS_BUCKET_NAME, GCS_FOLDER)

# Rename folders for readability
folder_names = {folder: datetime.strptime(folder.split('/')[1], "%Y%m%d_%H%M%S").strftime("%d %B %Y, %H:%M:%S") for folder in folders}
folder_dropdown = st.selectbox("Select Folder", list(folder_names.keys()), format_func=lambda x: folder_names[x])

# Display images if available
if st.button("Show Inference Images"):
    st.write(f"Displaying images from folder: {folder_names[folder_dropdown]}")
    image_files = list_png_files_in_gcs_folder(GCS_BUCKET_NAME, folder_dropdown)
    for img_file in image_files:
        img_url = f"https://storage.googleapis.com/{GCS_BUCKET_NAME}/{img_file}"
        st.image(img_url, caption=os.path.basename(img_file))
