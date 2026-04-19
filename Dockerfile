# Use a Python 3.10 slim base image for a smaller footprint
FROM python:3.10-slim

# Install necessary system dependencies for OpenCV and Piper TTS
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Create a non-root user (Hugging Face Spaces requirement)
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

# Set the working directory
WORKDIR /app

# Copy the requirements file and install dependencies
COPY --chown=user:user requirements-hf.txt /app/requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r /app/requirements.txt

# Copy all files to the working directory
COPY --chown=user:user . /app

# Expose the Streamlit default port
EXPOSE 8501

# Configure Streamlit defaults for HuggingFace Spaces
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS="0.0.0.0"
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS="false"

CMD ["streamlit", "run", "app/streamlit_app.py"]
