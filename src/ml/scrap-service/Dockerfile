# Gunakan base image Python
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements.txt (pastikan file ini ada di scrap-service)
COPY src/ml/scrap-service/requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy seluruh direktori src
COPY . /app

# Download NLTK resources
RUN python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"

# Set environment untuk Python path
ENV PYTHONPATH=/app

# Expose port yang digunakan
EXPOSE 8000

# Jalankan aplikasi FastAPI
CMD ["uvicorn", "src.ml.scrap-service.app:app", "--host", "0.0.0.0", "--port", "8000"]