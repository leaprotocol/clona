import cv2
import numpy as np
import logging
from datetime import datetime

def analyze_vignetting(image_path):
    """Analyze vignetting in an image
    Returns a dictionary of measurements including center-to-corner ratios
    """
    # Read image and convert to grayscale
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Get image dimensions
    height, width = gray.shape

    # Define regions to measure
    center_size = min(width, height) // 8  # Size of center region

    # Center region coordinates
    cx = width // 2
    cy = height // 2
    center_region = gray[
                    cy - center_size // 2:cy + center_size // 2,
                    cx - center_size // 2:cx + center_size // 2
                    ]

    # Corner regions
    corner_size = center_size
    corners = {
        'top_left': gray[0:corner_size, 0:corner_size],
        'top_right': gray[0:corner_size, width - corner_size:width],
        'bottom_left': gray[height - corner_size:height, 0:corner_size],
        'bottom_right': gray[height - corner_size:height, width - corner_size:width]
    }

    # Calculate mean intensities
    center_intensity = np.mean(center_region)
    corner_intensities = {k: np.mean(v) for k, v in corners.items()}

    # Calculate ratios (1.0 means no vignetting)
    corner_ratios = {k: v / center_intensity for k, v in corner_intensities.items()}

    # Calculate average corner ratio
    avg_corner_ratio = np.mean(list(corner_ratios.values()))

    # Create result dictionary
    results = {
        'center_intensity': float(center_intensity),
        'corner_intensities': {k: float(v) for k, v in corner_intensities.items()},
        'corner_ratios': {k: float(v) for k, v in corner_ratios.items()},
        'average_corner_ratio': float(avg_corner_ratio),
        # Add overall vignetting score (0-100, where 100 is no vignetting)
        'vignetting_score': float(avg_corner_ratio * 100)
    }

    return results


def visualize_vignetting(image_path, output_path=None):
    """Create a visualization of the vignetting analysis"""
    # Read image
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape

    # Create heatmap
    normalized = gray.astype(float) / 255
    heatmap = cv2.applyColorMap(
        (normalized * 255).astype(np.uint8),
        cv2.COLORMAP_JET
    )

    # Add center cross
    cx, cy = width // 2, height // 2
    cross_size = 20
    cv2.line(heatmap, (cx - cross_size, cy), (cx + cross_size, cy), (255, 255, 255), 2)
    cv2.line(heatmap, (cx, cy - cross_size), (cx, cy + cross_size), (255, 255, 255), 2)

    if output_path:
        cv2.imwrite(output_path, heatmap)

    return heatmap


def analyze_scenario_photo(scenario, photo_info):
    """Analyze a single photo from a scenario"""
    if scenario['type'] != 'vignette':
        raise ValueError("This analysis is only for vignette scenarios")

    photo_path = photo_info['path']

    # Perform analysis
    try:
        results = analyze_vignetting(photo_path)

        # Generate visualization
        viz_path = photo_path.replace('.', '_analysis.')
        visualize_vignetting(photo_path, viz_path)

        # Add analysis results to photo metadata
        photo_info['analysis'] = {
            'vignetting_results': results,
            'visualization_path': viz_path,
            'analyzed_at': datetime.now().strftime("%Y%m%d_%H%M%S")
        }

        return True

    except Exception as e:
        logging.error(f"Error analyzing photo {photo_path}: {e}")
        return False