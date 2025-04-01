FROM python:3.13-slim

# Set environment variable untuk non-interactive install
ENV DEBIAN_FRONTEND=noninteractive 

WORKDIR /app

# Install dependensi sistem yang diperlukan sebelum menginstall Python packages
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    python3-dev \
    libblas-dev \
    liblapack-dev \
    libatlas-base-dev \
    && rm -rf /var/lib/apt/lists/*  # Membersihkan cache APT untuk mengurangi ukuran image

# Copy dan install dependensi Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy seluruh source code aplikasi
COPY . .

# Expose port
EXPOSE 8010

# Jalankan aplikasi
CMD [ "python", "app/app.py" ]
