import os
from huggingface_hub import HfApi
from dotenv import load_dotenv

load_dotenv(override=True)

# Make sure your HF token is set as an environment variable or logged in via `huggingface-cli login`
# If you don't have it set, you can pass token="hf_your_token_here" to HfApi()
token = os.environ.get("HF_TOKEN")

if not token:
    token = input("Enter your Hugging Face token (starting with hf_): ").strip()

print(f"DEBUG: Using token starting with: {token[:6]}...")

api = HfApi(token=token)

print("Uploading to Hugging Face Spaces using large folder upload...")

# Upload the entire folder, but ignore large unneeded files and git history.
# The Hugging Face API automatically handles Git LFS for the .pt, .pth, and .onnx model weights!
api.upload_folder(
    folder_path=".",
    repo_id="Saransh482003/DSAI-Project-Navigation-Assistant",
    repo_type="space",
    allow_patterns=[
        "app/**",               # Include Streamlit app folder and all its contents
        "model_training/depth_estimation/model_weights/*.pth", # Explicit model inclusions
        "model_training/object_detection/best-weights/*.pt",   # Explicit model inclusions
        "mechanics/**",         # Any other core utilities
        "Depth-Anything-V2/**", # Your models dependencies
        "piper_voices/**",     # Explicitly include Piper voices
        "Dockerfile",           # The docker configuration
        "requirements-hf.txt",  # Specific HF space requirements
        "README.md"             # Used by Hugging Face to configure the space
    ]
)

print("Upload complete! Check your Space.")