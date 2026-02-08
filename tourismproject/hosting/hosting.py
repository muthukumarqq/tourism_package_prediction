from huggingface_hub import HfApi
import os

# Initialize Hugging Face API with token
api = HfApi(token=os.getenv("HF_TOKEN"))

# Upload Streamlit deployment files to Hugging Face Space
api.upload_folder(
    folder_path="tourismproject/deployment",      # local folder containing app.py & requirements.txt
    repo_id="muthukumarqq/Tourism-Package-Predict",  # Hugging Face Space repo
    repo_type="space",                             # space for Streamlit app
    path_in_repo="",                               # root of the Space
)

print("Streamlit app successfully uploaded to Hugging Face Spaces.")
