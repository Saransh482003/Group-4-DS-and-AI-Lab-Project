import os
import sys

# Ensure the app directory is added to the system path so internal module imports work
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "app")))

# Import the pre-configured Gradio demo instance
from huggingface_app import demo

if __name__ == "__main__":
    demo.launch()