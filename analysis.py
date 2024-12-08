import cv2
import numpy as np
import logging
import rawpy
from datetime import datetime
import os
import matplotlib.pyplot as plt



def analyze_sharpness(image_path):
    """Analyze image sharpness using edge detection and MTF calculations"""
    logging.info(f"Starting sharpness analysis of image: {image_path}")

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

        # Calculate edge response
        edges = cv2.Canny(gray, 50, 150)

        # Calculate local variance as a measure of detail retention
        kernel_size = 5
        local_var = cv2.Laplacian(gray, cv2.CV_64F).var()

        # Calculate MTF (Modulation Transfer Function)
        mtf_values = calculate_mtf(gray)

        # Calculate overall sharpness metrics
        edge_intensity = np.mean(edges)
        edge_density = np.count_nonzero(edges) / edges.size

        # Create visualization
        viz_path = os.path.join(
            os.path.dirname(image_path),
            f"sharpness_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
        )

        create_sharpness_visualization(img, edges, mtf_values, viz_path)

        # Calculate overall sharpness score (0-100)
        # Weight different factors
        edge_score = min(100, edge_intensity * 0.5)
        detail_score = min(100, local_var / 50)
        mtf_score = calculate_mtf_score(mtf_values)

        overall_score = (edge_score * 0.4 + detail_score * 0.3 + mtf_score * 0.3)

        results = {
            'sharpness_score': float(overall_score),
            'edge_intensity': float(edge_intensity),
            'edge_density': float(edge_density),
            'local_variance': float(local_var),
            'mtf_values': mtf_values.tolist(),
            'visualization_path': viz_path,
            'analysis_time': datetime.now().strftime("%Y%m%d_%H%M%S")
        }

        return results

    except Exception as e:
        logging.error(f"Sharpness analysis failed: {str(e)}")
        raise

def calculate_mtf(gray_image):
    """Calculate MTF (Modulation Transfer Function) values"""
    # Calculate frequency response using FFT
    f_transform = np.fft.fft2(gray_image)
    f_shift = np.fft.fftshift(f_transform)
    magnitude = np.abs(f_shift)

    # Calculate radial average
    center = (magnitude.shape[0] // 2, magnitude.shape[1] // 2)
    y, x = np.indices(magnitude.shape)
    r = np.sqrt((x - center[1]) ** 2 + (y - center[0]) ** 2)
    r = r.astype(int)

    # Calculate the mean for each radius
    mtf = np.zeros(min(center))
    for i in range(len(mtf)):
        mask = r == i
        if mask.any():
            mtf[i] = magnitude[mask].mean()

    # Normalize
    mtf = mtf / mtf[0] if mtf[0] != 0 else mtf

    return mtf

def calculate_mtf_score(mtf_values):
    """Convert MTF values to a 0-100 score"""
    # Calculate area under MTF curve
    area = np.trapz(mtf_values)
    # Normalize to 0-100 range
    score = min(100, area * 100 / len(mtf_values))
    return score

def create_sharpness_visualization(original, edges, mtf_values, output_path):
    """Create visualization of sharpness analysis"""
    # Create figure with subplots
    plt.figure(figsize=(15, 5))

    # Original image
    plt.subplot(131)
    plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    plt.title('Original Image')
    plt.axis('off')

    # Edge detection
    plt.subplot(132)
    plt.imshow(edges, cmap='gray')
    plt.title('Edge Detection')
    plt.axis('off')

    # MTF plot
    plt.subplot(133)
    plt.plot(mtf_values)
    plt.title('MTF Curve')
    plt.xlabel('Spatial Frequency')
    plt.ylabel('MTF')
    plt.grid(True)

    # Save visualization
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

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

def analyze_bokeh(image_path, click_x, click_y, metadata=None):
    """
    Analyze bokeh from a single clicked point

    Args:
        image_path: Path to image file
        click_x, click_y: Coordinates where user clicked

    Returns:
        Dictionary containing analysis results and visualizations
    """
    logging.info(f"Starting bokeh analysis at point ({click_x}, {click_y})")

    try:
        # Load and process image
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

        # Auto-detect bokeh region
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (15, 15), 0)

        # Use adaptive thresholding to isolate bright regions
        thresh = cv2.adaptiveThreshold(
            blurred, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            21, -2
        )

        # Find contours around bright regions
        contours, _ = cv2.findContours(
            thresh,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        # Find contour closest to click point
        min_dist = float('inf')
        bokeh_contour = None

        for contour in contours:
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                dist = np.sqrt((cx - click_x) ** 2 + (cy - click_y) ** 2)
                if dist < min_dist:
                    min_dist = dist
                    bokeh_contour = contour

        if bokeh_contour is None:
            raise ValueError("Could not detect bokeh region near click point")

        # Get enclosing circle
        (center_x, center_y), radius = cv2.minEnclosingCircle(bokeh_contour)
        center_x, center_y = int(center_x), int(center_y)
        radius = int(radius)

        # Extract ROI
        roi_size = int(radius * 2.5)
        roi_x1 = max(0, center_x - roi_size)
        roi_x2 = min(img.shape[1], center_x + roi_size)
        roi_y1 = max(0, center_y - roi_size)
        roi_y2 = min(img.shape[0], center_y + roi_size)

        roi = img[roi_y1:roi_y2, roi_x1:roi_x2]

        # 1. Analyze Shape Regularity
        regularity_score, shape_metrics = analyze_shape_regularity(
            bokeh_contour, radius
        )

        # 2. Analyze Color Fringing
        fringing_score, color_metrics = analyze_color_fringing(
            roi,
            (center_x - roi_x1, center_y - roi_y1),
            radius
        )

        # 3. Analyze Intensity Distribution
        intensity_score, intensity_metrics = analyze_intensity_distribution(
            roi,
            (center_x - roi_x1, center_y - roi_y1),
            radius
        )

        # Create visualization
        viz_path = os.path.join(
            os.path.dirname(image_path),
            f"bokeh_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
        )
        create_bokeh_visualization(
            roi,
            (center_x - roi_x1, center_y - roi_y1),
            radius,
            shape_metrics,
            color_metrics,
            intensity_metrics,
            viz_path,
            metadata
        )


        # Convert numpy values to Python native types in metrics
        shape_metrics = {
            'circularity': float(shape_metrics['circularity']),
            'contour_matching': float(shape_metrics['contour_matching']),
            # This was wrong before - it's 'contour_matching' not 'matching_score'
            'area': float(shape_metrics['area']),
            'perimeter': float(shape_metrics['perimeter'])
        }

        color_metrics = {
            'average_color_difference': float(color_metrics['average_color_difference']),
            'max_color_difference': float(color_metrics['max_color_difference']),
            'color_variation': float(color_metrics['color_variation'])
        }

        intensity_metrics = {
            'mean_intensity': float(intensity_metrics['mean_intensity']),
            'std_intensity': float(intensity_metrics['std_intensity']),
            'intensity_gradient': float(intensity_metrics['intensity_gradient']),
            'center_intensity': float(intensity_metrics['center_intensity'])
        }

        results = {
            'overall_score': float((regularity_score + fringing_score + intensity_score) / 3),
            'shape_regularity': {
                'score': float(regularity_score),
                'metrics': shape_metrics
            },
            'color_fringing': {
                'score': float(fringing_score),
                'metrics': color_metrics
            },
            'intensity_distribution': {
                'score': float(intensity_score),
                'metrics': intensity_metrics
            },
            'visualization_path': viz_path,
            'center_point': (float(center_x), float(center_y)),
            'radius': float(radius),
            'analysis_time': datetime.now().strftime("%Y%m%d_%H%M%S"),
            'metadata': metadata or {}  # Include the metadata in results
        }

        return results



    except Exception as e:
        logging.error(f"Bokeh analysis failed: {str(e)}")
        raise

def analyze_shape_regularity(contour, radius):
    """Analyze how circular/regular the bokeh shape is"""
    # Calculate circularity
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    circularity = 4 * np.pi * area / (perimeter * perimeter)

    # Calculate contour regularity
    ideal_circle = np.zeros((int(radius * 2.5), int(radius * 2.5)), dtype=np.uint8)
    cv2.circle(
        ideal_circle,
        (int(radius * 1.25), int(radius * 1.25)),
        radius,
        255,
        -1
    )

    # Compare with ideal circle
    matching_score = cv2.matchShapes(
        contour,
        cv2.findContours(ideal_circle, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0][0],
        cv2.CONTOURS_MATCH_I2,
        0.0
    )

    # Calculate score (0-100)
    regularity_score = (circularity * 50 + (1 - matching_score) * 50)

    metrics = {
        'circularity': circularity,
        'contour_matching': matching_score,
        'area': area,
        'perimeter': perimeter
    }

    return regularity_score, metrics

def analyze_color_fringing(roi, center, radius):
    """Analyze color fringing/chromatic aberration in bokeh"""
    center_x, center_y = center

    # Split into color channels
    b, g, r = cv2.split(roi)

    # Create circular mask
    mask = np.zeros(roi.shape[:2], dtype=np.uint8)
    cv2.circle(mask, (center_x, center_y), radius, 255, -1)

    # Calculate color channel differences along radius
    angles = np.linspace(0, 2 * np.pi, 360)
    color_diffs = []

    for angle in angles:
        points = []
        for i in range(radius):
            x = int(center_x + i * np.cos(angle))
            y = int(center_y + i * np.sin(angle))
            if 0 <= x < roi.shape[1] and 0 <= y < roi.shape[0]:
                points.append((r[y, x], g[y, x], b[y, x]))

        if points:
            points = np.array(points)
            # Calculate max difference between channels
            channel_diffs = np.max(points, axis=1) - np.min(points, axis=1)
            color_diffs.append(np.mean(channel_diffs))

    avg_color_diff = np.mean(color_diffs)
    max_color_diff = np.max(color_diffs)

    # Calculate score (0-100, lower color difference is better)
    fringing_score = 100 * (1 - min(avg_color_diff / 255, 1.0))

    metrics = {
        'average_color_difference': avg_color_diff,
        'max_color_difference': max_color_diff,
        'color_variation': np.std(color_diffs)
    }

    return fringing_score, metrics

def analyze_intensity_distribution(roi, center, radius):
    """Analyze the intensity distribution within bokeh"""
    center_x, center_y = center

    # Convert to grayscale
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    # Create circular mask
    mask = np.zeros_like(gray)
    cv2.circle(mask, (center_x, center_y), radius, 255, -1)

    # Get masked region
    masked = cv2.bitwise_and(gray, gray, mask=mask)

    # Calculate intensity metrics
    non_zero = masked[masked > 0]
    if len(non_zero) == 0:
        return 0, {}

    mean_intensity = np.mean(non_zero)
    std_intensity = np.std(non_zero)

    # Calculate radial intensity profile
    distances = []
    intensities = []

    y_indices, x_indices = np.nonzero(mask)
    for y, x in zip(y_indices, x_indices):
        dist = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
        distances.append(dist)
        intensities.append(gray[y, x])

    # Sort by distance
    sorted_indices = np.argsort(distances)
    distances = np.array(distances)[sorted_indices]
    intensities = np.array(intensities)[sorted_indices]

    # Calculate intensity falloff
    if len(distances) > 0:
        intensity_gradient = np.polyfit(distances, intensities, 1)[0]
    else:
        intensity_gradient = 0

    # Calculate score based on:
    # 1. Intensity uniformity (higher std = lower score)
    # 2. Gradual falloff (very negative gradient = lower score)
    uniformity_score = 100 * (1 - min(std_intensity / mean_intensity, 1.0))
    falloff_score = 100 * (1 - min(abs(intensity_gradient) / 2.0, 1.0))

    intensity_score = (uniformity_score + falloff_score) / 2

    metrics = {
        'mean_intensity': mean_intensity,
        'std_intensity': std_intensity,
        'intensity_gradient': intensity_gradient,
        'center_intensity': gray[center_y, center_x]
    }

    return intensity_score, metrics

def create_bokeh_visualization(roi, center, radius, shape_metrics, color_metrics,
                             intensity_metrics, output_path, metadata=None):
    """Create visualization of bokeh analysis"""
    viz = roi.copy()
    center_x, center_y = center

    # Draw detected circle
    cv2.circle(viz, (center_x, center_y), radius, (0, 255, 0), 2)

    # Add metric annotations
    font = cv2.FONT_HERSHEY_SIMPLEX
    y = 30

    # Add camera settings if available
    if metadata:
        settings = []
        if 'aperture' in metadata:
            settings.append(f"f/{metadata['aperture']}")
        if 'shutter_speed' in metadata:
            settings.append(f"1/{metadata['shutter_speed']}s")
        if 'iso' in metadata:
            settings.append(f"ISO {metadata['iso']}")

        settings_text = " • ".join(settings)
        cv2.putText(viz, settings_text, (10, y), font, 0.6, (255, 255, 255), 2)
        y += 25

    # Rest of the visualization code remains the same...

    cv2.putText(viz, f"Circularity: {shape_metrics['circularity']:.2f}",
                (10, y), font, 0.5, (255, 255, 255), 2)
    y += 20
    cv2.putText(viz, f"Color Fringing: {color_metrics['average_color_difference']:.2f}",
                (10, y), font, 0.5, (255, 255, 255), 2)
    y += 20
    cv2.putText(viz, f"Intensity Std: {intensity_metrics['std_intensity']:.2f}",
                (10, y), font, 0.5, (255, 255, 255), 2)

    cv2.imwrite(output_path, viz)

def analyze_chromatic_aberration(image_path):
    """Analyze chromatic aberration in an image"""
    logging.info(f"Starting chromatic aberration analysis of image: {image_path}")

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

        # Split into color channels
        b, g, r = cv2.split(img)

        # Calculate color differences between channels
        rg_diff = cv2.absdiff(r, g)
        rb_diff = cv2.absdiff(r, b)
        gb_diff = cv2.absdiff(g, b)

        # Calculate average color differences in high contrast areas
        edges = cv2.Canny(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 100, 200)
        kernel = np.ones((5, 5), np.uint8)
        edge_region = cv2.dilate(edges, kernel, iterations=1)

        # Measure color differences along edges
        rg_ca = cv2.mean(rg_diff, mask=edge_region)[0]
        rb_ca = cv2.mean(rb_diff, mask=edge_region)[0]
        gb_ca = cv2.mean(gb_diff, mask=edge_region)[0]

        # Create visualization
        viz_path = os.path.join(
            os.path.dirname(image_path),
            f"ca_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
        )

        # Create heatmap visualization
        ca_map = cv2.addWeighted(rg_diff, 0.33, rb_diff, 0.33, 0)
        ca_map = cv2.addWeighted(ca_map, 1.0, gb_diff, 0.33, 0)
        ca_heatmap = cv2.applyColorMap(ca_map, cv2.COLORMAP_JET)

        # Overlay on original image
        viz = cv2.addWeighted(img, 0.7, ca_heatmap, 0.3, 0)
        cv2.imwrite(viz_path, viz)

        # Calculate overall CA score (0-100, lower is better)
        avg_ca = (rg_ca + rb_ca + gb_ca) / 3.0
        ca_score = 100 * (1 - min(avg_ca / 50, 1.0))

        results = {
            'chromatic_aberration_score': float(ca_score),
            'channel_differences': {
                'red_green': float(rg_ca),
                'red_blue': float(rb_ca),
                'green_blue': float(gb_ca)
            },
            'visualization_path': viz_path,
            'analysis_time': datetime.now().strftime("%Y%m%d_%H%M%S")
        }

        logging.info(f"Analysis complete. Score: {ca_score:.2f}")
        return results

    except Exception as e:
        logging.error(f"Chromatic aberration analysis failed: {str(e)}")
        raise