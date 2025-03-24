# Functions for file handling
import os

def save_uploaded_file(uploaded_file, temp_dir="temp"):
    """Save the uploaded file to a temporary directory."""
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)
    file_path = os.path.join(temp_dir, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path