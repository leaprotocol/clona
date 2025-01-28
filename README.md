# Lens Evaluation Application

## Overview

This project is part of a diploma thesis focused on developing an application for analyzing lens characteristics using at-home printable test charts. The application evaluates lens properties such as sharpness, bokeh, distortions, chromatic aberration, and vignetting.

## Features

- Image Acquisition in RAW format
- Image Processing and Analysis
- Evaluation of Lens Properties (Sharpness, Vignetting, PSF, Bokeh)
- Web Interface for User Interaction
- Session Management (Save and Load)

## Installation

### Docker

1. **Clone the repository:**
   ```bash
   git clone <repository_url>
   cd <repository_directory>
   ```

2. **Build and run the Docker container:**
   ```bash
   docker-compose up -d
   ```

3. **Access the application:**
   - Open your browser and go to `http://reangue.com/clona`.

### Non-Docker

1. **Clone the repository:**
   ```bash
   git clone <repository_url>
   cd <repository_directory>
   ```

2. **Create a virtual environment and activate it:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # on Linux/Mac
   venv\Scripts\activate  # on Windows
   ```

3. **Install the required dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Install system dependencies:**
   ```bash
   sudo apt-get update && sudo apt-get install -y \
       libgphoto2-dev libcairo2-dev libgirepository1.0-dev \
       build-essential pkg-config python3-dev gir1.2-gtk-3.0 \
       libgl1-mesa-glx libgl1-mesa-dri gphoto2
   ```

5. **Run the application:**
   ```bash
   python3 main.py
   ```

6. **Access the application:**
   - Open your browser and navigate to the local address provided by the NiceGUI server.

## Usage

1. Ensure your camera is connected via USB and is compatible with the GPhoto2 library.
2. Verify the camera is recognized by running:
   ```bash
   gphoto2 --auto-detect
   ```

## Testing

1. Navigate to the `repo/clona` directory.
2. Run the tests using `pytest`:
   ```bash
   python3 -m pytest tests/test_camera_manager.py -v
   ```

## File Structure

- `main.py`: Entry point of the application.
- `camera.py`: Functions for camera interaction.
- `ui.py`: Functions for user interface.
- `image_processing.py`: Functions for image processing and analysis.
- `config.py`: Configuration and global variables.
- `requirements.txt`: List of required Python packages.

## Thesis Description

### Introduction
In today's rapidly advancing digital age of photography, the quality of a camera lens can significantly influence the outcome of photographic or videographic work. This thesis aims to develop a Python application that captures images of calibration elements, assesses the quality of a lens by analyzing captured images, and results in a score (or a suite of scores) to describe the performance, suitable for comparison with other tested lenses.

### System Architecture and Methodology
The application is divided into four main components: image acquisition, segmentation and preprocessing, evaluating lens properties, and displaying the results. The backend, implemented in Python, interfaces with the camera, processes the images, and performs the analysis. The frontend, built using NiceGUI, provides a user-friendly interface for interacting with the application and viewing the results.

### Scenarios for Analysis
- **Scenario A**: White wall with different apertures to analyze vignetting.
- **Scenario B**: Point light with different apertures to analyze PSF, bokeh roundness, and evenness.
- **Scenario C**: Printed test chart (with Siemens chart, slanted edges, etc.) to analyze sharpness, MTF, geometric distortions, and chromatic aberrations.

### Analysis Algorithms
- **Sharpness**: Evaluated using keypoint response from the SIFT algorithm.
- **Vignetting**: Measured by comparing the luminance of the center and corners of the image.
- **PSF (Point Spread Function)**: Analyzed by summing pixel values around keypoints.
- **Bokeh**: Assessed by the standard deviation of pixel values around keypoints.

### Conclusion
This application provides a comprehensive tool for evaluating the quality of camera lenses. It is useful for both advanced users like lens technicians and beginners seeking to assess or compare lenses. Future work could explore motorization and cataloging of lenses for more extensive testing.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.

## Contact

For any questions or inquiries, please contact Lea Králová at lea.kralova00@gmail.com.
