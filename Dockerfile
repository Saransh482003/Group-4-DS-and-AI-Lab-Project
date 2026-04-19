# Use a Python 3.10 slim base image for a smaller footprint
FROM python:3.10-slim

# Install necessary system dependencies for OpenCV, Piper TTS, and downloading tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    tar \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Download and extract Piper TTS globally so `sys.platform != "win32"` can use it
RUN wget -qO- https://github.com/rhasspy/piper/releases/download/2023.11.14-2/piper_linux_x86_64.tar.gz | tar -xz -C /opt && \
    ln -s /opt/piper/piper /usr/local/bin/piper

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

# Expose the Streamlit default port required by HF Spaces
EXPOSE 7860

# Configure Streamlit defaults for HuggingFace Spaces
ENV STREAMLIT_SERVER_PORT=7860
ENV STREAMLIT_SERVER_ADDRESS="0.0.0.0"
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS="false"

CMD ["streamlit", "run", "app/streamlit_app.py"]
