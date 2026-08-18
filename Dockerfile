# 1. Specify a suitable base image
FROM python:3.9-slim-buster

# 2. Install system dependencies required for OpenCV and other packages
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 3. Copy the app script and ALL split model pieces into the container
COPY app.py .
COPY ResNet50_model.keras.part-* .

# Recombine the parts into the single .keras model file, then delete the chunks
RUN cat ResNet50_model.keras.part-* > ResNet50_model.keras && rm ResNet50_model.keras.part-*

# 4. Expose the default Streamlit port
EXPOSE 8501

# 5. Define the entry point to run the Streamlit application
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
