import streamlit as st
import subprocess
import os
import json

from google.cloud import storage

# Absolute path to main.py
MAIN_SCRIPT_PATH = '/home/deamoon_uw_nn/uw-com-vision/main.py'

# Directory to list files on the VM
VM_DIR = '/home/deamoon_uw_nn/DATASET/INFERENCE/UPLOAD'

# Path to dataset info file
DATASET_INFO_PATH = './uw-com-vision/dataset_info.json'

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

# Function to list files in a GCS bucket
def list_files_in_bucket(bucket_name, folder):
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blobs = bucket.list_blobs(prefix=folder)
    return [blob.name for blob in blobs]

# Function to check if stderr contains errors
def contains_errors(stderr):
    error_keywords = ['error', 'failed', 'exception', 'traceback', 'critical']
    return any(keyword in stderr.lower() for keyword in error_keywords)

# Load dataset names from dataset_info.json
def load_dataset_names():
    with open(DATASET_INFO_PATH, 'r') as f:
        data = json.load(f)
    return list(data.keys())

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
    
    # Display buttons for error and warning messages
    if stderr:
        if st.button("Show Errors and Warnings"):
            if contains_errors(stderr):
                st.error(stderr)
            else:
                st.warning(stderr)
    else:
        st.success(f"{task.capitalize()} task completed successfully!")

# List files on the VM
st.header("Files on VM")
vm_files = list_files_on_vm(VM_DIR)
if vm_files:
    for file in vm_files:
        st.write(file)
else:
    st.write("No files found in the specified directory on the VM.")

# Display images if available
if task == 'inference':
    if st.button("Show Inference Images"):
        for img_file in os.listdir(VM_DIR):
            if img_file.endswith('.png'):
                st.image(os.path.join(VM_DIR, img_file), caption=img_file)
