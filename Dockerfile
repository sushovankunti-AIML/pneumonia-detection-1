# 1. Specify a suitable base image
FROM python:3.9-slim-bookworm

# 2. Install system dependencies required for OpenCV and other packages
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory inside the container
WORKDIR /app

# Install wget to handle the download
RUN apt-get update && apt-get install -y wget && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .

# Download the model directly from your hosting link during the build
RUN wget -O ResNet50_model.keras "https://huggingface.co/Sushovankunti/pneumonia-detection-1/resolve/main/ResNet50_model.keras?download=true"

# Expose port and set entrypoint
EXPOSE 8501
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
