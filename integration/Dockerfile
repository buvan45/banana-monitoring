# Use a lightweight Python base image
FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=7860 \
    HOME=/home/user

# Install system-level dependencies for image/video processing (OpenCV, FFmpeg, etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create a non-root user (Hugging Face expects user ID 1000)
RUN useradd -m -u 1000 user
USER user
ENV PATH=/home/user/.local/bin:$PATH

# Set working directory
WORKDIR $HOME/app

# Copy requirements and install Python dependencies
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Copy all application code
COPY --chown=user . .

# Ensure upload/snapshot directories exist with correct permissions
RUN mkdir -p snapshots temp_uploads && chmod 777 snapshots temp_uploads

# Expose default port
EXPOSE 7860

# Launch the Flask application
CMD ["python", "app.py"]
