FROM python:3.9-slim-bookworm

WORKDIR /app

RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# NUCLEAR OPTION: Force the exact TensorFlow version that understands Keras 3
RUN pip install --no-cache-dir tensorflow-cpu==2.17.0

COPY app.py .

EXPOSE 8501

# FIX FOR AXIOS 403: Disable CORS and XSRF protection
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.enableCORS=false", "--server.enableXsrfProtection=false"]
