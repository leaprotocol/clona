# Use the official Python image as a base
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -e .

# Make port 80 available to the world outside this container
EXPOSE 80

# Run the application
CMD ["python", "repo/clona/main.py"] 