import streamlit as st
import subprocess
import os
import json
from google.cloud import storage
from datetime import datetime, timedelta
from google.api_core import page_iterator
from io import BytesIO
from PIL import Image
import time
import zipfile
from tempfile import TemporaryDirectory


# Add these lines at the beginning of the script
ADMIN_PASSWORD = "deamoon_uw_nn"

# Absolute path to main.py
MAIN_SCRIPT_PATH = '/home/deamoon_uw_nn/uw-com-vision/main.py'

# GCS bucket details
GCS_BUCKET_NAME = 'uw-com-vision'
GCS_DATASET_FOLDER = 'DATASET'
GCS_INFERENCE_FOLDER = 'DATASET/INFERENCE'
GCS_ARCHIVE_FOLDER = 'Archive'
GCS_DATASET_INFO_PATH = 'dataset_info.json'

def create_zip_from_gcs(bucket_name, folder, zip_name="archive.zip"):
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blobs = bucket.list_blobs(prefix=folder)
    
    with TemporaryDirectory() as tempdir:
        zip_path = os.path.join(tempdir, zip_name)
        with zipfile.ZipFile(zip_path, 'w') as zipf:
            for blob in blobs:
                if blob.name.endswith('.png') or blob.name.endswith('results_x_pred_1.csv') or blob.name.endswith('results_x_pred_0.csv'):
                    file_path = os.path.join(tempdir, os.path.basename(blob.name))
                    blob.download_to_filename(file_path)
                    zipf.write(file_path, os.path.basename(blob.name))
        with open(zip_path, "rb") as f:
            bytes_zip = f.read()
    return bytes_zip


def check_password():
    def password_entered():
        if st.session_state["password"] == ADMIN_PASSWORD:
            st.session_state["password_correct"] = True
            del st.session_state["password"]
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        # First run, show input for password
        st.text_input("Password", type="password", on_change=password_entered, key="password")
        return False
    elif not st.session_state["password_correct"]:
        st.error("Incorrect password")
        return False
    else:
        return True

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
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    stdout, stderr = [], []
    while process.poll() is None:
        output = process.stdout.readline()
        if output:
            stdout.append(output.strip())
    process.stdout.close()
    stderr_output = process.stderr.read()
    process.stderr.close()
    return '\n'.join(stdout), stderr_output, process.returncode == 0

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

def format_and_sort_folders(folders):
    """
    Format folder names from 'YYYYMMDD_HHMMSS' to a more readable format
    and sort them from newest to oldest.

    Parameters:
    - folders: List of folder names.

    Returns:
    - List of tuples (original_name, formatted_name), sorted from newest to oldest.
    """
    formatted_folders = []
    for folder in folders:
        # Strip trailing slashes and extract the timestamp part
        folder = folder.rstrip('/')
        try:
            # Parse and format the timestamp
            timestamp = datetime.strptime(folder.split('/')[-1], '%Y%m%d_%H%M%S')
            formatted_name = timestamp.strftime('%B %d, %Y %H:%M:%S')
            formatted_folders.append((folder, formatted_name))
        except ValueError:
            # If parsing fails, keep the original folder name
            formatted_folders.append((folder, folder))

    # Sort by the timestamp (newest first)
    formatted_folders.sort(key=lambda x: x[1], reverse=True)
    return formatted_folders

# Define a function to read and calculate ETA
def estimate_eta(task, num_images=0):
    data = read_eta_data()
    if task == 'inference':
        avg_time_per_image = data.get(task, {}).get('average_time_per_image', 1)
        buffer = data.get(task, {}).get('buffer', 1)
        task_eta = avg_time_per_image * num_images * buffer
    else:
        task_eta = data.get(task, {}).get('average_time', 60)
    
    download_eta = data.get('download', {}).get('average_time', 60)
    upload_eta = data.get('upload', {}).get('average_time', 60)
    
    return download_eta, task_eta, upload_eta

# Define a function to read ETA data
def read_eta_data():
    ETA_FILE = '/home/deamoon_uw_nn/uw-com-vision/eta_data.json'
    if os.path.exists(ETA_FILE):
        with open(ETA_FILE, 'r') as file:
            return json.load(file)

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

# Wrap the dataset creation and deletion sections with password check
if check_password():
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
                
# task = st.selectbox("Select Task", ["prepare", "train", "evaluate", "inference"])

# Define the task mapping
task_mapping = {
    "inference": "MEASUREMENT",
    "evaluate": "MODEL EVALUATE",
    "prepare": "DATA PREPARATION FOR TRAINING",
    "train": "NEW MODEL TRAINING"
}

# Change the task order and create a reverse mapping for the command
tasks_order = ["inference", "evaluate", "prepare", "train"]
reverse_task_mapping = {v: k for k, v in task_mapping.items()}

# Update task selection with user-friendly names
task_display_names = [task_mapping[task] for task in tasks_order]
selected_task_display = st.selectbox("Select Task", task_display_names)
task = reverse_task_mapping[selected_task_display]

dataset_name = st.selectbox("Dataset Name", list(st.session_state.datasets.keys()))

threshold = st.slider(
    "Select Detection Threshold",
    min_value=0.0,
    max_value=1.0,
    value=0.65,
    step=0.01,
    help="Adjust the detection threshold for the model."
)

# Align checkbox and button to the right side
# col1, col2 = st.columns([3, 1])
# with col1:
# Update the task execution button code to handle the combined ETA
if st.button("Run Task"):
    visualize_flag = "--visualize"  # Always true
    upload_flag = "--upload"  # Always true
    download_flag = "--download"
    
    # Calculate ETAs
    download_eta, task_eta, upload_eta = estimate_eta(task)
    total_eta = download_eta + task_eta + upload_eta
    
    command = f"python3 {MAIN_SCRIPT_PATH} --task {task} --dataset_name {dataset_name} --threshold {threshold} {visualize_flag} {download_flag} {upload_flag}"
    st.info(f"Running: {command}")

    def update_progress_bar_and_countdown(start_time, eta, phase):
        end_time = start_time + eta
        while time.time() < end_time:
            elapsed_time = time.time() - start_time
            remaining_time = end_time - time.time()
            hours, remainder = divmod(remaining_time, 3600)
            minutes, seconds = divmod(remainder, 60)
            countdown_str = f"{int(hours):02}:{int(minutes):02}:{int(seconds):02}"
            countdown_placeholder.text(f"{phase} - Time Remaining: {countdown_str}")
            progress_percentage = min(elapsed_time / total_eta, 1.0)
            progress_bar.progress(progress_percentage)
            time.sleep(1)
            if phase == "Task in progress" and not process.poll() is None:
                # If task phase and the command finished, exit the loop
                break

    with st.spinner('Running task...'):
        progress_bar = st.progress(0)
        countdown_placeholder = st.empty()
        start_time = time.time()

        # Download Phase
        update_progress_bar_and_countdown(start_time, download_eta, "Downloading")

        # Task Phase
        task_start_time = time.time()  # Reset start time for task phase
        process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        update_progress_bar_and_countdown(task_start_time, task_eta, "Task in progress")

        stdout, stderr = process.communicate()

        # Upload Phase
        start_time = time.time()  # Reset start time for upload phase
        update_progress_bar_and_countdown(start_time, upload_eta, "Uploading")

        # Ensure the progress bar is full at the end
        progress_bar.progress(100)
        countdown_placeholder.text("Task Completed")

    st.text(stdout)
    st.session_state.stderr = stderr  # Store stderr in session state

    # Reset the show_errors state if there are new errors
    if stderr:
        st.session_state.show_errors = True
    else:
        st.success(f"{task.capitalize()} task completed successfully!")

    # confirm_deletion = st.checkbox("Confirm Deletion")
    # if st.button("Remove Dataset"):
    #     if confirm_deletion:
    #         del st.session_state.datasets[dataset_name]
    #         save_dataset_names_to_gcs(st.session_state.datasets)
    #         st.success(f"Dataset '{dataset_name}' deleted.")
    #         st.session_state.confirm_delete = False  # Automatically uncheck after deletion
    #         st.experimental_rerun()  # Refresh to reflect deletion
    #     else:
    #         st.warning("Please check the confirmation box to delete the dataset.")

# Conditionally show the upload section
# Mapping for upload folder display names
upload_folder_mapping = {
    f"{GCS_DATASET_FOLDER}/{dataset_name}": "TRAINING DATA",
    GCS_INFERENCE_FOLDER: "MEASUREMENT DATA"
}
reverse_upload_folder_mapping = {v: k for k, v in upload_folder_mapping.items()}
# Conditionally show the upload section
if use_new_data:
    st.header("Upload Files to GCS")
    upload_folder_display_names = [upload_folder_mapping[f"{GCS_DATASET_FOLDER}/{dataset_name}"], upload_folder_mapping[GCS_INFERENCE_FOLDER]]
    selected_upload_folder_display = st.selectbox(
        "Select folder to upload to",
        upload_folder_display_names
    )
    upload_folder = reverse_upload_folder_mapping[selected_upload_folder_display]
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
            st.warning(st.session_state.stderr)
else:
    if st.session_state.stderr and st.button("Show Errors and Warnings"):
        st.session_state.show_errors = True

# List folders in the GCS bucket
st.header("Google Cloud Storage")
if 'folders' not in st.session_state or not st.session_state.folders:
    st.session_state.folders = list_directories(GCS_BUCKET_NAME, GCS_ARCHIVE_FOLDER)

# Apply formatting and sorting
formatted_folders = format_and_sort_folders(st.session_state.folders)

if formatted_folders:
    folder_dropdown = st.selectbox(
        "Select Folder", 
        formatted_folders, 
        format_func=lambda x: x[1]  # Display the formatted name
    )
else:
    st.write("No folders found in the GCS bucket.")

selected_folder = folder_dropdown[0]

# Button to show inference images
if st.button("Show Inference Images") and st.session_state.folders:
    st.session_state.show_images = True

# Display images if available
if st.session_state.show_images:
    st.write(f"Displaying images from folder: {folder_dropdown[1]}")  # Use formatted name for display
    image_files = list_png_files_in_gcs_folder(GCS_BUCKET_NAME, selected_folder)  # Use original name for operations
    if image_files:
        for blob in image_files:
            img_bytes = blob.download_as_bytes()
            img = Image.open(BytesIO(img_bytes))
            st.image(img, caption=os.path.basename(blob.name))
    else:
        st.write("No images found in the selected folder.")

    # Button to download specific CSV files
    csv_files = list_specific_csv_files_in_gcs_folder(GCS_BUCKET_NAME, selected_folder)  # Use original name for operations
    if csv_files:
        for blob in csv_files:
            csv_bytes = blob.download_as_bytes()
            csv_name = os.path.basename(blob.name)
            if csv_name == 'results_x_pred_1.csv':
                if dataset_name != 'hw_patterns':
                    download_name = 'results_pores.csv'
                else:
                    download_name = 'results_cyclones.csv'
            elif csv_name == 'results_x_pred_0.csv':
                if dataset_name != 'hw_patterns':
                    download_name = 'results_throats.csv'
                else:
                    download_name = 'results_flows.csv'
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

# After displaying the inference images and CSV download buttons, add the following:
if st.session_state.show_images:
    st.write(f"Displaying images from folder: {folder_dropdown[1]}")  # Use formatted name for display
    image_files = list_png_files_in_gcs_folder(GCS_BUCKET_NAME, selected_folder)  # Use original name for operations
    if image_files:
        for blob in image_files:
            img_bytes = blob.download_as_bytes()
            img = Image.open(BytesIO(img_bytes))
            st.image(img, caption=os.path.basename(blob.name))
    else:
        st.write("No images found in the selected folder.")

    # Button to download specific CSV files
    csv_files = list_specific_csv_files_in_gcs_folder(GCS_BUCKET_NAME, selected_folder)  # Use original name for operations
    if csv_files:
        for blob in csv_files:
            csv_bytes = blob.download_as_bytes()
            csv_name = os.path.basename(blob.name)
            if csv_name == 'results_x_pred_1.csv':
                if dataset_name != 'hw_patterns':
                    download_name = 'results_pores.csv'
                else:
                    download_name = 'results_cyclones.csv'
            elif csv_name == 'results_x_pred_0.csv':
                if dataset_name != 'hw_patterns':
                    download_name = 'results_throats.csv'
                else:
                    download_name = 'results_flows.csv'
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

    # Add "Download All" button
    if st.button("Download All as Zip"):
        zip_bytes = create_zip_from_gcs(GCS_BUCKET_NAME, selected_folder)
        st.download_button(
            label="Download All",
            data=zip_bytes,
            file_name="inference_results.zip",
            mime="application/zip"
        )



# Help and Tips Section
st.sidebar.title("Help & Tips")

st.sidebar.subheader("Tips")
st.sidebar.write("""
- **Dataset Preparation:** Ensure your dataset is well-structured.
- **Inference:** Use the slider to adjust the detection threshold.
- **Uploading Files:** Use the 'Upload Files' section to add new data.
""")

st.sidebar.subheader("Tutorials")
st.sidebar.write("""
    - **Step 1:** Select the task you want to run (prepare, train, evaluate, inference).
    - **Step 2:** Choose the dataset name from the dropdown.
    - **Step 3:** Adjust the detection threshold using the slider.
    - **Step 4:** Optionally, use new data from the bucket by checking the respective box.
    - **Step 5:** Click 'Run Task' to execute the selected task.
    """)
