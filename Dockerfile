# 1. Specify a suitable base image
FROM python:3.9-slim-buster

# 2. Install system dependencies required for OpenCV and other packages
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 3. Copy the application script and the pre-trained model into the container
COPY app.py .
COPY ResNet50_model.keras .

# 4. Expose the default Streamlit port
EXPOSE 8501

# 5. Define the entry point to run the Streamlit application
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
