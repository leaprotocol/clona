# analysis.py

import cv2
import numpy as np
import logging
from datetime import datetime
import rawpy
import os


def analyze_vignetting(image_path):
    """Analyze vignetting in an image
    Returns a dictionary of measurements including center-to-corner ratios
    """
    logging.info(f"Starting analysis of image: {image_path}")

    try:
        # Check if file exists
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")

        # Handle RAW files
        if image_path.lower().endswith(('.cr2', '.nef', '.arw')):
            logging.info("Processing RAW file...")
            try:
                with rawpy.imread(image_path) as raw:
                    # Use sRGB color space and default settings
                    img = raw.postprocess(
                        use_camera_wb=True,
                        half_size=True,  # For faster processing
                        no_auto_bright=True,
                        output_bps=8
                    )
                    # Convert from RGB to BGR for OpenCV
                    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                    logging.info(f"RAW processing successful, image shape: {img.shape}")
            except Exception as e:
                logging.error(f"RAW processing failed: {str(e)}")
                raise
        else:
            logging.info("Processing regular image file...")
            img = cv2.imread(image_path)

        if img is None or img.size == 0:
            raise ValueError("Failed to load image data")

        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        logging.info(f"Converted to grayscale, shape: {gray.shape}")

        # Get image dimensions
        height, width = gray.shape

        # Define regions to measure
        center_size = min(width, height) // 8
        corner_size = center_size  # Add this line, it was missing

        # Center region coordinates
        cx = width // 2
        cy = height // 2

        logging.info(f"Analyzing regions - center at ({cx}, {cy}), size: {center_size}")

        # Get center region
        center_y1 = max(0, cy - center_size // 2)
        center_y2 = min(height, cy + center_size // 2)
        center_x1 = max(0, cx - center_size // 2)
        center_x2 = min(width, cx + center_size // 2)

        center_region = gray[center_y1:center_y2, center_x1:center_x2]

        # Define corner regions with bounds checking
        corners = {
            'top_left': gray[0:corner_size, 0:corner_size],
            'top_right': gray[0:corner_size, width - corner_size:width],
            'bottom_left': gray[height - corner_size:height, 0:corner_size],
            'bottom_right': gray[height - corner_size:height, width - corner_size:width]
        }

        # Calculate intensities
        center_intensity = np.mean(center_region)
        corner_intensities = {k: np.mean(v) for k, v in corners.items()}

        # Calculate ratios (1.0 means no vignetting)
        corner_ratios = {k: v / center_intensity for k, v in corner_intensities.items()}

        # Calculate average corner ratio and score
        avg_corner_ratio = np.mean(list(corner_ratios.values()))
        vignetting_score = avg_corner_ratio * 100

        logging.info(f"Analysis complete - score: {vignetting_score:.2f}")

        return {
            'center_intensity': float(center_intensity),
            'corner_intensities': {k: float(v) for k, v in corner_intensities.items()},
            'corner_ratios': {k: float(v) for k, v in corner_ratios.items()},
            'average_corner_ratio': float(avg_corner_ratio),
            'vignetting_score': float(vignetting_score)
        }

    except Exception as e:
        logging.error(f"Analysis failed: {str(e)}")
        raise


def create_vignetting_visualization(image_path, output_path):
    """Create a visualization of the vignetting analysis"""
    try:
        # Load and process image same way as analysis
        if image_path.lower().endswith(('.cr2', '.nef', '.arw')):
            with rawpy.imread(image_path) as raw:
                img = raw.postprocess(
                    use_camera_wb=True,
                    half_size=True,
                    no_auto_bright=True,
                    output_bps=8
                )
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        else:
            img = cv2.imread(image_path)

        if img is None or img.size == 0:
            raise ValueError("Failed to load image for visualization")

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        height, width = gray.shape

        # Create normalized intensity map
        normalized = gray.astype(float) / np.max(gray)

        # Create heatmap
        heatmap = cv2.applyColorMap(
            (normalized * 255).astype(np.uint8),
            cv2.COLORMAP_JET
        )

        # Add measurement regions overlay
        center_size = min(width, height) // 8
        cx = width // 2
        cy = height // 2

        # Draw center region
        cv2.rectangle(
            heatmap,
            (cx - center_size // 2, cy - center_size // 2),
            (cx + center_size // 2, cy + center_size // 2),
            (255, 255, 255),
            2
        )

        # Draw corner regions
        corner_size = center_size
        corners = [
            ((0, 0), (corner_size, corner_size)),
            ((width - corner_size, 0), (width, corner_size)),
            ((0, height - corner_size), (corner_size, height)),
            ((width - corner_size, height - corner_size), (width, height))
        ]

        for start, end in corners:
            cv2.rectangle(heatmap, start, end, (255, 255, 255), 2)

        # Save visualization
        cv2.imwrite(output_path, heatmap)
        logging.info(f"Visualization saved to: {output_path}")
        return output_path

    except Exception as e:
        logging.error(f"Visualization failed: {str(e)}")
        raise


def analyze_scenario_photo(scenario, photo_info):
    """Analyze a single photo from a scenario"""
    if scenario['type'] != 'vignette':
        raise ValueError("This analysis is only for vignette scenarios")

    photo_path = photo_info['path']

    try:
        # Create output paths
        base_path = photo_path.rsplit('.', 1)[0]
        viz_path = f"{base_path}_analysis.jpg"

        # Run analysis
        results = analyze_vignetting(photo_path)

        # Create visualization
        create_vignetting_visualization(photo_path, viz_path)

        # Store results in photo metadata
        photo_info['analysis'] = {
            'vignetting_results': results,
            'visualization_path': viz_path,
            'analyzed_at': datetime.now().strftime("%Y%m%d_%H%M%S")
        }

        return True

    except Exception as e:
        logging.error(f"Error analyzing photo {photo_path}: {e}")
        return False