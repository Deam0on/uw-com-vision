import streamlit as st
import subprocess
import os

# Function to run shell commands
def run_command(command):
    """Runs a shell command and returns the output."""
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    return result.stdout, result.stderr

# Streamlit interface
st.title("Neural Network Control Panel")

# Task selection
st.header("Run Neural Network Script")
task = st.selectbox("Select Task", ["prepare", "train", "evaluate", "inference"])
dataset_name = st.text_input("Dataset Name")
visualize = st.checkbox("Visualize Results", value=False)
download = st.checkbox("Download Data", value=False)
upload = st.checkbox("Upload Results", value=True)

# Execute task
if st.button("Run Task"):
    visualize_flag = "--visualize" if visualize else ""
    download_flag = "--download" if download else ""
    upload_flag = "--upload" if upload else ""

    command = f"python3 main.py --task {task} --dataset_name {dataset_name} {visualize_flag} {download_flag} {upload_flag}"
    st.info(f"Running: {command}")
    stdout, stderr = run_command(command)
    st.text(stdout)
    if stderr:
        st.error(stderr)
    else:
        if st.button("Run Task"):
    visualize_flag = "--visualize" if visualize else ""
    download_flag = "--download" if download else ""
    upload_flag = "--upload" if upload else ""

    command = f"python3 main.py --task {task} --dataset_name {dataset_name} {visualize_flag} {download_flag} {upload_flag}"
    st.info(f"Running: {command}")
    stdout, stderr = run_command(command)
    st.text(stdout)
    
    if stderr:
        st.error(stderr)
    else:
        st.success(f"{task.capitalize()} task completed successfully!")

        # Example: Display images if available
        if task == 'inference':
            for img_file in os.listdir('/home/deamoon_uw_nn/'):
                if img_file.endswith('.png'):
                    st.image(os.path.join('/home/deamoon_uw_nn/', img_file), caption=img_file)
