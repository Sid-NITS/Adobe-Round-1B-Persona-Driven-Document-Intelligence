# Use lightweight Python base image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install OS dependencies including git
RUN apt-get update && apt-get install -y \
    wget \
    unzip \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install protoc
RUN wget -q https://github.com/protocolbuffers/protobuf/releases/download/v3.19.0/protoc-3.19.0-linux-x86_64.zip && \
    unzip -o protoc-3.19.0-linux-x86_64.zip -d /usr/local/ && \
    rm protoc-3.19.0-linux-x86_64.zip


# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Create necessary directories (if not already in source)
RUN mkdir -p "Collection 1/PDFs"

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# Expose port for possible future web UI
EXPOSE 8000

# Default entrypoint
CMD ["python", "solution.py"]
