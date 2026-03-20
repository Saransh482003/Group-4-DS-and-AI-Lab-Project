import os
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

# Define your model ID
MODEL_ID = "depth-anything/Depth-Anything-V2-Small-hf"

# Create a directory in your CWD
save_directory = "./depth_anything_weights"
os.makedirs(save_directory, exist_ok=True)

print(f"Downloading and saving weights to {save_directory}...")

# Load from Hugging Face cache (or download)
processor = AutoImageProcessor.from_pretrained(MODEL_ID)
model = AutoModelForDepthEstimation.from_pretrained(MODEL_ID)

# Save everything to your local folder
processor.save_pretrained(save_directory)
model.save_pretrained(save_directory)

print("Done! You can now load the model offline from the local folder.")