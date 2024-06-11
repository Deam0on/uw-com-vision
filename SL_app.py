# app.py

import streamlit as st
import subprocess
import os

# GitHub Repository
REPO_URL = "https://github.com/Deam0on/uw-com-vision"

# Function to pull the latest code from GitHub
def pull_from_github():
    os.system("git pull")

# Function to upload files to Google Cloud Storage bucket
def upload_to_gcs(files, bucket_name):
    for file in files:
        os.system(f"gsutil cp {file.name} gs://{bucket_name}/{file.name}")

# Function to run NN scripts
def run_nn_script(task, dataset_name, visualize):
    visualize_flag = "--visualize" if visualize else ""
    command = f"python main.py --task {task} --dataset_name {dataset_name} {visualize_flag}"
    result = subprocess.run(command, shell=True, capture_output=True)
    return result.stdout

# Streamlit interface
st.title("Neural Network Control Panel")

st.header("Pull the latest code from GitHub")
if st.button("Pull from GitHub"):
    pull_from_github()
    st.success("Pulled the latest code from GitHub.")

st.header("Upload Images to Google Cloud Storage")
uploaded_files = st.file_uploader("Choose images", accept_multiple_files=True)
bucket_name = st.text_input("GCS Bucket Name")
if st.button("Upload to GCS") and uploaded_files and bucket_name:
    upload_to_gcs(uploaded_files, bucket_name)
    st.success("Uploaded files to GCS.")

st.header("Run Neural Network Script")
task = st.selectbox("Select Task", ["prepare", "train", "evaluate", "inference"])
dataset_name = st.text_input("Dataset Name")
visualize = st.checkbox("Visualize Results", value=False)
if st.button("Run Script"):
    output = run_nn_script(task, dataset_name, visualize)
    st.text(output.decode("utf-8"))
