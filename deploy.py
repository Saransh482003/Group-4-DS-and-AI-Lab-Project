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

# We generate the HF README explicitly so you don't have to keep the ugly YAML frontmatter in your GitHub repo!
hf_frontmatter = """---
title: DSAI Navigation Assistant
emoji: 👁️
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
---

"""
with open("README.md", "r", encoding="utf-8") as f:
    readme_content = f.read()

full_hf_readme = (hf_frontmatter + readme_content).encode("utf-8")

# Upload the entire folder, but ignore large unneeded files and git history.
# The Hugging Face API automatically handles Git LFS for the .pt, .pth, and .onnx model weights!
api.upload_folder(
    folder_path=".",
    repo_id="Saransh482003/DSAI-Project-Navigation-Assistant",
    repo_type="space",
    allow_patterns=[
        "app/**",               # Include Streamlit app folder and all its contents
        "assets/**",
        "model_training/depth_estimation/model_weights/*.pth", # Explicit model inclusions
        "model_training/object_detection/best-weights/*.pt",   # Explicit model inclusions
        "mechanics/**",         # Any other core utilities
        "Depth-Anything-V2/**", # Your models dependencies
        "piper/**",             # Explicitly include Piper execution
        "piper/piper_voices/**",# Explicitly include Piper voices
        "Dockerfile",           # The docker configuration
        "requirements-hf.txt"   # Specific HF space requirements
    ]
)

print("Uploading custom HF README.md...")
api.upload_file(
    path_or_fileobj=full_hf_readme,
    path_in_repo="README.md",
    repo_id="Saransh482003/DSAI-Project-Navigation-Assistant",
    repo_type="space"
)

print("Upload complete! Check your Space.")