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

# Copy the current directory contents into the container at /app
COPY . .

# Install Python dependencies
RUN pip install setuptools && pip install -r requirements.txt

# Make port 8000 available to the world outside this container
EXPOSE 8000

# Run the application
CMD ["python", "main.py"]

# To run this container with USB device access and mount the datasets volume, use:
# docker run --device=/dev/bus/usb -p 8000:8000 -v ./datasets:/app/datasets clona-app

# Or with absolute path:
# docker run --device=/dev/bus/usb -p 8000:8000 -v $(pwd)/datasets:/app/datasets clona-app