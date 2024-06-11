import streamlit as st
import paramiko
from scp import SCPClient

# GCP VM SSH details
VM_HOST = "YOUR_GCP_VM_IP"
VM_USERNAME = "YOUR_GCP_VM_USERNAME"
SSH_KEY_PATH = "/path/to/your/private/key"

def execute_remote_command(command):
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(VM_HOST, username=VM_USERNAME, key_filename=SSH_KEY_PATH)

    stdin, stdout, stderr = ssh.exec_command(command)
    output = stdout.read().decode('utf-8')
    ssh.close()
    return output

def upload_file_to_vm(local_path, remote_path):
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(VM_HOST, username=VM_USERNAME, key_filename=SSH_KEY_PATH)

    with SCPClient(ssh.get_transport()) as scp:
        scp.put(local_path, remote_path)

    ssh.close()

# Streamlit interface
st.title("Neural Network Control Panel")

st.header("Run Neural Network Script")
task = st.selectbox("Select Task", ["prepare", "train", "evaluate", "inference"])
dataset_name = st.text_input("Dataset Name")
visualize = st.checkbox("Visualize Results", value=False)

if st.button("Run Task"):
    visualize_flag = "--visualize" if visualize else ""
    command = f"python main.py --task {task} --dataset_name {dataset_name} {visualize_flag}"
    output = execute_remote_command(command)
    st.success("Task completed successfully!")
    st.text_area("Output", output)

st.header("Upload Files to GCP VM")
uploaded_files = st.file_uploader("Choose files", accept_multiple_files=True)
remote_dir = st.text_input("Remote Directory on VM")

if st.button("Upload to VM") and uploaded_files and remote_dir:
    for uploaded_file in uploaded_files:
        local_path = uploaded_file.name
        with open(local_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        upload_file_to_vm(local_path, remote_dir)
        os.remove(local_path)
    st.success("Uploaded files to VM.")
