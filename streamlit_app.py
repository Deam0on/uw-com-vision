import streamlit as st
import subprocess

def run_command(command):
    """Runs a shell command and returns the output."""
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    return result.stdout

# Streamlit interface
st.title("Neural Network Control Panel")

st.header("Run Neural Network Script")
task = st.selectbox("Select Task", ["prepare", "train", "evaluate", "inference"])
dataset_name = st.text_input("Dataset Name")
visualize = st.checkbox("Visualize Results", value=False)

if st.button("Run Task"):
    visualize_flag = "--visualize" if visualize else ""
    command = f"python3 main.py --task {task} --dataset_name {dataset_name} {visualize_flag}"
    output = run_command(command)
    st.success("Task completed successfully!")
    st.text_area("Output", output)

st.header("Upload Files to Google Cloud Storage")
uploaded_files = st.file_uploader("Choose files", accept_multiple_files=True)
bucket_name = st.text_input("GCS Bucket Name")

if st.button("Upload to GCS") and uploaded_files and bucket_name:
    for uploaded_file in uploaded_files:
        with open(uploaded_file.name, "wb") as f:
            f.write(uploaded_file.getbuffer())
        os.system(f"gsutil cp {uploaded_file.name} gs://{bucket_name}/{uploaded_file.name}")
        st.success(f"Uploaded {uploaded_file.name} to GCS.")
