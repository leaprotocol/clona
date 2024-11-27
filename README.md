# Lens Analysis Application

## Overview

This project is part of a diploma thesis focused on developing an application for analyzing lens characteristics using at-home printable test charts. The application is designed to evaluate various lens properties such as sharpness, bokeh, distortions, chromatic aberration, and vignetting.

The application consists of four main parts:
1. **Image Acquisition**: Interface with a camera to capture photographs at different settings relevant to lens properties.
2. **Segmentation and Preprocessing**: Use image processing algorithms to find boundaries and position of elements on captured images.
3. **Evaluating Lens Properties**: Analyze various lens properties based on the calibration elements.
4. **Displaying the Results**: Present the results to the user through a web interface.

## Features

- Image Acquisition in RAW format
- Image Processing and Analysis
- Evaluation of Lens Properties (Sharpness, Vignetting, PSF, Bokeh)
- Web Interface for User Interaction
- Session Management (Save and Load)

## Repository

The source code for this project is available on GitHub: [Clona Repository](https://github.com/N4M3L355/clona)

## Installation

1. Clone the repository:
   ```bash
   git clone <repository_url>
   cd <repository_directory>
   ```

2. Create a virtual environment and activate it:
   ```bash
   python -m venv venv
   source venv/bin/activate  # on Linux/Mac
   venv\Scripts\activate  # on Windows
   ```

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Ensure your camera is connected via USB and is compatible with the GPhoto2 library.
2. Run the main application:
   ```bash
   python main.py
   ```
3. Open your web browser and navigate to the local address provided by the NiceGUI server.

## File Structure

- `main.py`: Entry point of the application.
- `camera.py`: Functions for camera interaction.
- `ui.py`: Functions for user interface.
- `image_processing.py`: Functions for image processing and analysis.
- `config.py`: Configuration and global variables.
- `requirements.txt`: List of required Python packages.

## Thesis Description

### Introduction
In today's rapidly advancing digital age of photography, the quality of a camera lens can significantly influence the outcome of photographic or videographic work. However, evaluating a lens's quality involves intricate techniques that can be overwhelming and technically challenging. This thesis aims to develop a Python application that captures images of calibration elements, assesses the quality of a lens by analyzing captured images, and results in a score (or a suite of scores) to describe the performance, suitable for comparison with other tested lenses.

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

For any questions or inquiries, please contact Lea Králová at lea.kralova00@gmail.com .
