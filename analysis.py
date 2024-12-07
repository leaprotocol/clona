import cv2
import numpy as np
import logging
import rawpy
from datetime import datetime
import os


def calculate_line_deviations(lines):
    """Calculate how much lines deviate from being straight"""
    if not lines:
        return np.array([])

    deviations = []
    for line in lines:
        x1, y1, x2, y2 = line
        length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        points = np.linspace(0, length, 20)
        ideal_y = np.linspace(y1, y2, 20)
        actual_y = points * (y2 - y1) / length + y1
        deviation = np.mean(np.abs(actual_y - ideal_y))
        deviations.append(deviation)

    return np.array(deviations)


def analyze_distortion(image_path):
    """Analyze lens distortion using a grid chart image"""
    logging.info(f"Starting distortion analysis of image: {image_path}")

    try:
        # Load and preprocess image
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

        if img is None:
            raise ValueError("Failed to load image")

        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Apply threshold
        _, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Detect lines
        lines = cv2.HoughLinesP(
            255 - binary,
            rho=1,
            theta=np.pi / 180,
            threshold=100,
            minLineLength=100,
            maxLineGap=10
        )

        if lines is None:
            raise ValueError("No lines detected in grid")

        # Separate horizontal and vertical lines
        h_lines = []
        v_lines = []

        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
            if angle < 45:  # Horizontal
                h_lines.append((x1, y1, x2, y2))
            elif angle > 45:  # Vertical
                v_lines.append((x1, y1, x2, y2))

        # Calculate deviations
        h_deviations = calculate_line_deviations(h_lines)
        v_deviations = calculate_line_deviations(v_lines)

        # Create visualization
        viz_img = img.copy()

        # Draw detected lines
        for x1, y1, x2, y2 in h_lines:
            cv2.line(viz_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        for x1, y1, x2, y2 in v_lines:
            cv2.line(viz_img, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Calculate metrics
        h_dev = np.mean(h_deviations) if len(h_deviations) > 0 else 0
        v_dev = np.mean(v_deviations) if len(v_deviations) > 0 else 0
        avg_deviation = (h_dev + v_dev) / 2 if (len(h_deviations) > 0 or len(v_deviations) > 0) else 0

        # Calculate distortion score (0-100, higher is better)
        distortion_score = 100 * (1 - min(avg_deviation / 50, 1))

        # Save visualization
        viz_path = os.path.join(
            os.path.dirname(image_path),
            f"distortion_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
        )
        cv2.imwrite(viz_path, viz_img)

        results = {
            'horizontal_deviations': h_deviations.tolist(),
            'vertical_deviations': v_deviations.tolist(),
            'average_deviation': float(avg_deviation),
            'distortion_score': float(distortion_score),
            'visualization_path': viz_path,
            'type': 'barrel' if avg_deviation > 0 else 'pincushion',
            'analysis_time': datetime.now().strftime("%Y%m%d_%H%M%S")
        }

        logging.info(f"Analysis complete. Score: {distortion_score:.2f}")
        return results

    except Exception as e:
        logging.error(f"Distortion analysis failed: {str(e)}")
        raise

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
        viz_path = os.path.join(
            os.path.dirname(image_path),
            f"vignette_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
        )
        create_vignetting_visualization(image_path, viz_path)

        return {
            'center_intensity': float(center_intensity),
            'corner_intensities': {k: float(v) for k, v in corner_intensities.items()},
            'corner_ratios': {k: float(v) for k, v in corner_ratios.items()},
            'average_corner_ratio': float(avg_corner_ratio),
            'vignetting_score': float(vignetting_score),
            'visualization_path': viz_path  # Make sure this is included
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

def convert_raw_to_jpeg(raw_path, jpeg_path):
    """Convert RAW image to JPEG for viewing"""
    try:
        with rawpy.imread(raw_path) as raw:
            rgb = raw.postprocess(
                use_camera_wb=True,
                no_auto_bright=True,
                output_bps=8
            )
            # Convert from RGB to BGR for OpenCV
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            cv2.imwrite(jpeg_path, bgr)
            return True
    except Exception as e:
        logging.error(f"Error converting RAW to JPEG: {e}")
        return False


def analyze_scenario_photo(scenario, photo_info):
    """Analyze a single photo from a scenario"""
    if scenario['type'] != 'vignette':
        raise ValueError("This analysis is only for vignette scenarios")

    photo_path = photo_info['path']

    try:
        # Create output paths
        base_path = photo_path.rsplit('.', 1)[0]
        viz_path = f"{base_path}_analysis.jpg"
        preview_path = f"{base_path}_preview.jpg"

        # Convert RAW to JPEG for viewing
        if photo_path.lower().endswith(('.cr2', '.nef', '.arw')):
            convert_raw_to_jpeg(photo_path, preview_path)

        # Run analysis
        results = analyze_vignetting(photo_path)

        # Create visualization
        create_vignetting_visualization(photo_path, viz_path)

        # Store results in photo metadata
        photo_info['analysis'] = {
            'vignetting_results': results,
            'visualization_path': viz_path,
            'preview_path': preview_path if photo_path.lower().endswith(('.cr2', '.nef', '.arw')) else photo_path,
            'analyzed_at': datetime.now().strftime("%Y%m%d_%H%M%S")
        }

        return True

    except Exception as e:
        logging.error(f"Error analyzing photo {photo_path}: {e}")
        return False


