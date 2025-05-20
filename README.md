# Lens Evaluation Application

## Overview

This project is part of a diploma thesis focused on developing an application for analyzing lens characteristics using at-home printable test charts. The application evaluates lens properties such as sharpness, bokeh, geometric distortions, chromatic aberration, and vignetting.

## Features

- Image Acquisition in RAW format using gphoto2
- Image Processing and Analysis for five key lens properties:
    - Sharpness (combining MTF/edge-based analysis and SIFT keypoint response)
    - Vignetting (illumination fall-off analysis)
    - Bokeh (quality of out-of-focus areas: shape, color fringing, intensity)
    - Geometric Distortion (barrel, pincushion, and complex)
    - Chromatic Aberration (lateral and longitudinal)
- Web Interface for User Interaction (built with NiceGUI)
- Dataset and Scenario Management
- Session Management (Save and Load capabilities implied by dataset management)
- Visualization of analysis results (heatmaps, graphs, overlays)

## Installation

### Docker

1. **Clone the repository:**
   ```bash
   git clone https://github.com/N4M3L355/clona.git
   cd clona
   ```

2. **Build and run the Docker container:**
   ```bash
   # Ensure you are in the directory containing docker-compose.yml (likely 'clona/repo/clona' or 'clona')
   docker-compose up -d clona-app
   ```
   **Note**: The `docker-compose.yml` should define the `clona-app` service. If developing locally without Traefik, use `docker-compose up clona-app -d`. This typically makes the application accessible on `http://localhost:8080`.

3. **Access the application:**
   - Open your browser and go to `http://localhost:8080`.

### Non-Docker

1. **Clone the repository:**
   ```bash
   git clone https://github.com/N4M3L355/clona.git
   cd clona 
   ```
   (Navigate to the specific sub-directory containing `requirements.txt` and `main.py`, e.g., `repo/clona/`)

2. **Install system dependencies (Linux example):**
   ```bash
   sudo apt-get update && sudo apt-get install -y \
       libgphoto2-dev libcairo2-dev libgirepository1.0-dev \
       build-essential pkg-config python3-dev gir1.2-gtk-3.0 \
       libgl1-mesa-glx libgl1-mesa-dri gphoto2
   ```

3. **Create a virtual environment and activate it (Python 3.12 or higher recommended):**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # on Linux/Mac
   # venv\Scripts\activate  # on Windows
   ```

4. **Install the required dependencies:**
   ```bash
   # Ensure you are in the directory with requirements.txt
   pip install -r requirements.txt
   ```

5. **Run the application:**
   ```bash
   # Ensure you are in the directory with main.py (e.g., repo/clona/)
   python3 main.py 
   ```
   (Or `python main.py` if `python3` is not specifically required by your setup and `python` points to Python 3.12+)

6. **Access the application:**
   - Open your browser and navigate to the local address provided by the NiceGUI server (usually `http://localhost:8080`).

## Usage

1. Ensure your camera is connected via USB and is compatible with the GPhoto2 library.
2. Verify the camera is recognized by running `gphoto2 --auto-detect` in your terminal (if gphoto2 CLI is installed system-wide).
3. Use the application's UI to connect to the camera, manage datasets, define scenarios, and capture/import images for analysis.

## Testing

1. Navigate to the `repo/clona` directory (or wherever the tests are located).
2. Run the tests using `pytest`. For example:
   ```bash
   python3 -m pytest tests/test_camera_manager.py -v 
   # Or more generally:
   # pytest tests/
   ```
   (Ensure test dependencies are installed in your virtual environment).

## File Structure (Key Files within `repo/clona/`)

- `main.py` (or `app.py`): Entry point of the NiceGUI application.
- `analysis.py`: Core logic for lens property analysis algorithms.
- `camera_manager.py` (example name): Handles camera interactions via gphoto2.
- `dataset_manager.py` (example name): Manages datasets, scenarios, and photo metadata.
- `ui.py` (example name, or integrated into `main.py`): Defines NiceGUI user interface components and layouts.
- `requirements.txt`: List of required Python packages.
- `config.py` (if it exists): For configuration and global variables.

## Thesis Description

### Introduction
In today's rapidly advancing digital age of photography, the quality of a camera lens can significantly influence the outcome of photographic or videographic work. This thesis aims to develop a Python application that captures images of calibration elements, assesses the quality of a lens by analyzing captured images, and results in a score (or a suite of scores) to describe the performance, suitable for comparison with other tested lenses.

### System Architecture and Methodology
The application is divided into four main components: Camera Control, Dataset Management, Analysis Engine, and User Interface. The backend, implemented in Python, interfaces with the camera (via gphoto2), processes images, and performs analyses. The frontend, built using NiceGUI, provides a user-friendly interface for interacting with the application and viewing results.

### Scenarios for Analysis
- **Scenario A (Vignetting)**: An evenly lit surface (e.g., white wall) with different apertures.
- **Scenario B (Bokeh)**: Point light sources at varied distances against a dark background.
- **Scenario C (Comprehensive)**: Printed test chart (with Siemens star, slanted edges, grid patterns) for sharpness, MTF, geometric distortions, and chromatic aberrations.

### Analysis Algorithms
The following metrics are computed, with scores generally normalized to 0-100 (higher is better unless specified):

- **Sharpness (S)**: Combines traditional MTF/edge-based analysis (A) and SIFT keypoint response (K).
  - Formula: $S = w_a \cdot A + w_k \cdot K$ (e.g., $w_a=0.7, w_k=0.3$)
  - $A$ is derived from MTF values (e.g., weighted area under curve, capped at 95).
  - $K$ is derived from SIFT keypoint responses (center and corner).

- **Vignetting (V_score)**: Measures luminance fall-off towards image corners.
  - Formula: $V_{score} = 100 \cdot \text{average\_ratio}$ (capped at 100)
  - $\text{average\_ratio}$ is the mean of $V_i = I_{corner,i} / I_{center}$ for each corner, after IQR outlier removal for intensities.

- **Bokeh (B_score)**: Assesses the aesthetic quality of out-of-focus areas.
  - Formula: $B_{score} = (S_{shape} + F_{color} + I_{intensity}) / 3$
  - $S_{shape}$ (Shape Regularity): Combines circularity ($C = 4\pi \cdot Area / Perimeter^2$) and contour matching to an ideal circle ($M$). $S_{shape} = 50 \cdot C + 50 \cdot (1 - M)$.
  - $F_{color}$ (Color Fringing): $F_{color} = 100 \cdot (1 - \min(1, \text{avg\_color\_diff} / 255))$.
  - $I_{intensity}$ (Intensity Distribution): Average of uniformity ($U = 100 \cdot (1 - \min(1, \sigma_I / \mu_I))$) and falloff ($F_o = 100 \cdot (1 - \min(1, |\text{intensity\_gradient}| / 2.0))$).

- **Geometric Distortion (D_score)**: Measures deviation of straight lines.
  - Formula: $D_{score} = 100 \cdot (1 - \min(1, \text{total\_deviation}))$
  - $\text{total\_deviation}$ is the average of mean horizontal ($h_{dev}$), vertical ($v_{dev}$), and grid cell ($g_{dev}$) deviations from ideal geometry. Based on polynomial model: $r_{distorted} = r_{ideal} \cdot (1 + k_1 \cdot r_{ideal}^2 + k_2 \cdot r_{ideal}^4)$.

- **Chromatic Aberration (CA_overall)**: Quantifies color fringing.
  - Lateral CA Score ($CA_{lat}$): $CA_{lat} = 100 \cdot (1 - \min(1, W_{CA} / W_{threshold}))$ where $W_{CA}$ is max displacement between color channel centroids (e.g., $W_{threshold}=3.0$ pixels).
  - Longitudinal CA Score ($CA_{long}$): $CA_{long} = 100 \cdot (1 - \min(1, F_{max} / 255))$ where $F_{max}$ is max average color fringing.
  - Overall Score: $CA_{overall} = 0.6 \cdot CA_{lat} + 0.4 \cdot CA_{long}$.

### Conclusion
This application provides a comprehensive tool for evaluating the quality of camera lenses. It is useful for both advanced users like lens technicians and beginners seeking to assess or compare lenses. Future work could explore motorization and cataloging of lenses for more extensive testing.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.

## Contact

For any questions or inquiries, please contact Lea Králová at lea.kralova00@gmail.com.
