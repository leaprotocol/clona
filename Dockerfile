# Use the official Python image as a base
FROM python:3.11

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgphoto2-dev \
    libcairo2-dev \
    libgirepository1.0-dev \
    build-essential \
    pkg-config \
    python3-dev \
    gir1.2-gtk-3.0 \
    libgl1-mesa-glx \
    libgl1-mesa-dri \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory in the container
WORKDIR /app

# Create datasets directory
RUN mkdir -p /app/datasets

# Create a volume for datasets
VOLUME /app/datasets

# Copy requirements file first for caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install setuptools && pip install -r requirements.txt

# Copy the rest of the application code
COPY . .

# Expose the port NiceGUI will run on
EXPOSE 8080

# Set environment variables for NiceGUI
ENV NICEGUI_HOST=0.0.0.0
ENV NICEGUI_PORT=8080

# Run the application
CMD ["python", "main.py"]