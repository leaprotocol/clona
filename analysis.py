import cv2
import numpy as np
import logging
import rawpy
from datetime import datetime
import os
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d


def analyze_sharpness(image_path, patterns_dir):
    """
    Analyzes lens sharpness using both traditional MTF calculation and SIFT template matching.
    Combines both scores for a more comprehensive sharpness evaluation.

    Args:
        image_path: Path to captured image
        patterns_dir: Directory containing template patterns
    """
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

        # Method 1: Traditional MTF-based analysis
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Calculate MTF
        mtf_values = calculate_mtf_improved(gray)
        edge_intensity = cv2.mean(cv2.Canny(gray, 100, 200))[0]
        local_variance = cv2.Laplacian(gray, cv2.CV_64F).var()

        # Calculate traditional score
        traditional_score = calculate_calibrated_mtf_score(mtf_values)

        # Method 2: SIFT template matching analysis
        regional_sharpness = {
            'center': 0,
            'corners': [],
            'midpoints': []
        }

        # Analyze center pattern
        center_pattern = os.path.join(patterns_dir, "siemens_center.png")
        matches_mask, homography, center_points = match_template_sift(image_path, center_pattern)

        if center_points:
            center_roi = extract_roi(img, center_points[0], size=100)
            center_mtf = calculate_mtf_improved(cv2.cvtColor(center_roi, cv2.COLOR_BGR2GRAY))
            regional_sharpness['center'] = calculate_mtf_score(center_mtf)

        # Analyze corner patterns
        corner_pattern = os.path.join(patterns_dir, "siemens_corner.png")
        for corner in ['tl', 'tr', 'bl', 'br']:
            matches_mask, homography, corner_points = match_template_sift(image_path, corner_pattern)
            if corner_points:
                corner_roi = extract_roi(img, corner_points[0], size=100)
                corner_mtf = calculate_mtf_improved(cv2.cvtColor(corner_roi, cv2.COLOR_BGR2GRAY))
                regional_sharpness['corners'].append(calculate_mtf_score(corner_mtf))

        # Calculate template matching score
        template_score = regional_sharpness['center'] * 0.5 + \
                         sum(regional_sharpness['corners']) / max(len(regional_sharpness['corners']), 1) * 0.5

        # Combine scores with weights
        combined_score = traditional_score * 0.6 + template_score * 0.4

        # Create visualization
        viz_path = create_combined_sharpness_visualization(
            img,
            traditional_score,
            regional_sharpness,
            template_score,
            combined_score,
            mtf_values
        )

        return {
            'combined_sharpness_score': float(combined_score),
            'traditional_score': float(traditional_score),
            'template_score': float(template_score),
            'regional_analysis': {
                'center': float(regional_sharpness['center']),
                'corners': [float(x) for x in regional_sharpness['corners']]
            },
            'edge_intensity': float(edge_intensity),
            'local_variance': float(local_variance),
            'mtf_values': mtf_values.tolist(),
            'visualization_path': viz_path,
            'analysis_time': datetime.now().strftime("%Y%m%d_%H%M%S")
        }

    except Exception as e:
        logging.error(f"Sharpness analysis failed: {str(e)}")
        raise


def create_combined_sharpness_visualization(image, traditional_score, regional_sharpness,
                                            template_score, combined_score, mtf_values):
    """Create visualization showing results from both analysis methods"""
    plt.figure(figsize=(15, 5))

    # Original image with regional analysis
    plt.subplot(131)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    height, width = image.shape[:2]

    # Draw center region
    plt.plot(width // 2, height // 2, 'go', markersize=10)
    plt.text(width // 2 - 60, height // 2 - 60,
             f"Center: {regional_sharpness['center']:.1f}",
             color='green')

    # Draw corner regions
    corners = [(50, 50), (width - 50, 50), (50, height - 50), (width - 50, height - 50)]
    for i, corner in enumerate(corners):
        if i < len(regional_sharpness['corners']):
            plt.plot(corner[0], corner[1], 'go', markersize=10)
            plt.text(corner[0] - 20, corner[1] - 20,
                     f"{regional_sharpness['corners'][i]:.1f}",
                     color='green')

    plt.title('Regional Analysis')
    plt.axis('off')

    # MTF curve
    plt.subplot(132)
    plt.plot(mtf_values)
    plt.title('MTF Curve')
    plt.xlabel('Spatial Frequency')
    plt.ylabel('MTF')
    plt.grid(True)

    # Scores summary
    plt.subplot(133)
    scores = [traditional_score, template_score, combined_score]
    labels = ['Traditional', 'Template', 'Combined']
    plt.bar(labels, scores)
    plt.title('Sharpness Scores')
    plt.ylim(0, 100)

    # Save visualization
    viz_path = os.path.join(
        os.path.dirname(image.filename),
        f"combined_sharpness_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
    )
    plt.tight_layout()
    plt.savefig(viz_path)
    plt.close()

    return viz_path

def calculate_physical_mtf(image):
    """Calculate MTF with physical constraints applied"""
    try:
        # Calculate 2D FFT
        f_transform = np.fft.fft2(image.astype(float))
        f_shift = np.fft.fftshift(f_transform)
        magnitude = np.abs(f_shift)

        # Calculate radial average with physical constraints
        center = (magnitude.shape[0] // 2, magnitude.shape[1] // 2)
        y, x = np.indices(magnitude.shape)
        r = np.sqrt((x - center[1])**2 + (y - center[0])**2)
        r = r.astype(int)

        # Use fewer bins for stability
        n_bins = 100
        mtf = np.zeros(n_bins)

        for i in range(n_bins):
            mask = (r >= i * (r.max() / n_bins)) & (r < (i + 1) * (r.max() / n_bins))
            if mask.any():
                mtf[i] = magnitude[mask].mean()

        # Normalize
        mtf = mtf / mtf[0] if mtf[0] != 0 else mtf

        # Apply physical constraints
        nyquist_freq = n_bins // 2
        # MTF should decrease with frequency
        for i in range(1, len(mtf)):
            mtf[i] = min(mtf[i], mtf[i-1])
        # Apply diffraction limit
        mtf *= np.exp(-np.arange(len(mtf)) / nyquist_freq)

        return mtf

    except Exception as e:
        logging.error(f"Physical MTF calculation failed: {str(e)}")
        raise

def calculate_calibrated_mtf_score(mtf_values):
    """Calculate MTF score with physical calibration"""
    try:
        # Weight frequencies according to human vision
        weights = np.exp(-np.arange(len(mtf_values)) / 20)
        weighted_mtf = mtf_values * weights
        
        # Calculate area under weighted curve
        area = np.trapz(weighted_mtf)
        # Calculate theoretical maximum area
        max_area = np.trapz(weights)
        
        # Convert to 0-100 score with realistic scaling
        score = min(100, (area / max_area) * 100)
        
        # Apply physical limits
        score = min(95, score)  # Account for diffraction limit
        
        return score
    except Exception as e:
        logging.error(f"MTF score calculation failed: {str(e)}")
        return 0

def create_calibrated_visualization(image, edges, mtf_values, output_path):
    """Create visualization with calibrated results"""
    try:
        plt.figure(figsize=(15, 5))

        # Original image
        plt.subplot(131)
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.title('Original Image')
        plt.axis('off')

        # Edge detection
        plt.subplot(132)
        plt.imshow(edges, cmap='gray')
        plt.title('Pattern Detection')
        plt.axis('off')

        # MTF curve with physical limits
        plt.subplot(133)
        plt.plot(mtf_values)
        plt.title('MTF Curve')
        plt.xlabel('Spatial Frequency')
        plt.ylabel('MTF')
        plt.ylim(0, 1)  # Physical MTF cannot exceed 1
        plt.grid(True)

        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()

    except Exception as e:
        logging.error(f"Visualization creation failed: {str(e)}")
        raise

def calculate_mtf_improved(gray_image):
    """Calculate MTF with improved frequency analysis"""
    # Use edge detection to find strong edges
    edges = cv2.Canny(gray_image, 100, 200)
    
    # Calculate frequency response using FFT
    f_transform = np.fft.fft2(gray_image)
    f_shift = np.fft.fftshift(f_transform)
    magnitude = np.abs(f_shift)
    
    # Apply windowing to reduce frequency leakage
    rows, cols = gray_image.shape
    window = np.outer(np.hanning(rows), np.hanning(cols))
    magnitude = magnitude * window
    
    # Calculate radial average with improved sampling
    center = (magnitude.shape[0] // 2, magnitude.shape[1] // 2)
    y, x = np.indices(magnitude.shape)
    r = np.sqrt((x - center[1])**2 + (y - center[0])**2)
    r = r.astype(int)
    
    # Use more bins for better frequency resolution
    mtf = np.zeros(min(center))
    for i in range(len(mtf)):
        mask = r == i
        if mask.any():
            mtf[i] = magnitude[mask].mean()
    
    # Normalize and apply smoothing
    mtf = mtf / mtf[0] if mtf[0] != 0 else mtf
    mtf = np.convolve(mtf, np.ones(3)/3, mode='same')
    
    return mtf

def calculate_mtf_score(mtf_values):
    """Calculate MTF score with improved weighting of frequencies"""
    try:
        # Weight lower frequencies more heavily
        weights = np.exp(-np.arange(len(mtf_values)) / 20)  # Exponential decay
        weighted_mtf = mtf_values * weights
        
        # Calculate area under weighted curve
        area = np.trapz(weighted_mtf)
        max_possible_area = np.trapz(weights)  # Maximum possible area
        
        # Convert to 0-100 score with adjusted scaling
        score = min(100, (area / max_possible_area) * 150)  # Scale factor adjusted
        return score
    except Exception as e:
        logging.error(f"MTF score calculation failed: {str(e)}")
        return 0

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
    """
    Analyze lens distortion using both edge detection and SIFT template matching.
    Combines results for more robust distortion measurement.
    """
    logging.info(f"Starting comprehensive distortion analysis of image: {image_path}")

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

        # METHOD 1: Original edge-based analysis
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

        # Calculate edge-based deviations
        h_deviations = calculate_line_deviations(h_lines)
        v_deviations = calculate_line_deviations(v_lines)

        # METHOD 2: SIFT-based template matching
        # Create ideal grid template
        template = np.zeros((300, 300), dtype=np.uint8)
        spacing = 30
        for i in range(0, 300, spacing):
            cv2.line(template, (i, 0), (i, 299), 255, 2)
            cv2.line(template, (0, i), (299, i), 255, 2)

        # Initialize SIFT
        sift = cv2.SIFT_create()

        # Find keypoints and descriptors
        kp1, des1 = sift.detectAndCompute(template, None)
        kp2, des2 = sift.detectAndCompute(gray, None)

        if des1 is not None and des2 is not None:
            # Match features using FLANN
            FLANN_INDEX_KDTREE = 1
            index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
            search_params = dict(checks=50)
            flann = cv2.FlannBasedMatcher(index_params, search_params)
            matches = flann.knnMatch(des1, des2, k=2)

            # Get good matches
            good_matches = []
            for m, n in matches:
                if m.distance < 0.7 * n.distance:
                    good_matches.append(m)

            if len(good_matches) > 10:
                # Calculate SIFT-based metrics
                src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

                # Calculate displacements
                displacements = np.sqrt(np.sum((dst_pts - src_pts) ** 2, axis=2))
                sift_displacement = float(np.mean(displacements))

                # Calculate radial distortion
                center = np.mean(dst_pts, axis=0)[0]
                radial_distances = np.sqrt(np.sum((dst_pts - center) ** 2, axis=2))
                ideal_distances = np.sqrt(np.sum((src_pts - center) ** 2, axis=2))
                k = np.mean(radial_distances / ideal_distances) - 1
            else:
                sift_displacement = None
                k = None
        else:
            sift_displacement = None
            k = None

        # Create visualization
        viz_img = img.copy()

        # Draw detected lines from Method 1
        for x1, y1, x2, y2 in h_lines + v_lines:
            cv2.line(viz_img, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Draw SIFT matches if available
        if sift_displacement is not None:
            for pt1, pt2 in zip(src_pts, dst_pts):
                p1 = tuple(map(int, pt1[0]))
                p2 = tuple(map(int, pt2[0]))
                cv2.circle(viz_img, p2, 3, (0, 0, 255), -1)
                cv2.line(viz_img, p1, p2, (255, 0, 0), 1)

        # Save visualization
        viz_path = os.path.join(
            os.path.dirname(image_path),
            f"distortion_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
        )
        cv2.imwrite(viz_path, viz_img)

        # Calculate combined metrics
        h_dev = np.mean(h_deviations) if len(h_deviations) > 0 else 0
        v_dev = np.mean(v_deviations) if len(v_deviations) > 0 else 0
        edge_deviation = (h_dev + v_dev) / 2 if (len(h_deviations) > 0 or len(v_deviations) > 0) else 0

        # Calculate final combined score
        edge_score = 100 * (1 - min(edge_deviation / 50, 1))
        sift_score = 100 * (1 - min(sift_displacement / 50, 1)) if sift_displacement is not None else None

        # Combine scores with weights if both methods succeeded
        if sift_score is not None:
            final_score = 0.4 * edge_score + 0.6 * sift_score
        else:
            final_score = edge_score

        # Determine distortion type using both methods
        edge_type = 'barrel' if edge_deviation > 0 else 'pincushion'
        sift_type = 'barrel' if k is not None and k > 0 else 'pincushion'

        # Use SIFT type if available, otherwise fall back to edge type
        distortion_type = sift_type if k is not None else edge_type

        results = {
            'edge_based': {
                'horizontal_deviations': h_deviations.tolist() if len(h_deviations) > 0 else [],
                'vertical_deviations': v_deviations.tolist() if len(v_deviations) > 0 else [],
                'average_deviation': float(edge_deviation),
                'score': float(edge_score)
            },
            'sift_based': {
                'displacement': float(sift_displacement) if sift_displacement is not None else None,
                'radial_coefficient': float(k) if k is not None else None,
                'score': float(sift_score) if sift_score is not None else None
            },
            'combined_score': float(final_score),
            'distortion_type': distortion_type,
            'visualization_path': viz_path,
            'analysis_time': datetime.now().strftime("%Y%m%d_%H%M%S")
        }

        logging.info(f"Analysis complete. Combined score: {final_score:.2f}")
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

        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Get image dimensions
        height, width = gray.shape
        
        # Define regions
        center_size = min(width, height) // 3
        center_x = width // 2
        center_y = height // 2
        corner_size = min(width, height) // 4

        # Extract center region
        center_region = gray[
            center_y - center_size//2:center_y + center_size//2,
            center_x - center_size//2:center_x + center_size//2
        ]
        
        # Calculate center intensity with outlier removal
        center_values = center_region.flatten()
        q1, q3 = np.percentile(center_values, [25, 75])
        iqr = q3 - q1
        center_mask = (center_values >= q1 - 1.5*iqr) & (center_values <= q3 + 1.5*iqr)
        center_intensity = np.mean(center_values[center_mask])

        # Extract and analyze corner regions
        corners = {
            'top_left': gray[:corner_size, :corner_size],
            'top_right': gray[:corner_size, -corner_size:],
            'bottom_left': gray[-corner_size:, :corner_size],
            'bottom_right': gray[-corner_size:, -corner_size:]
        }

        # Calculate corner intensities with outlier removal
        corner_intensities = {}
        for corner_name, corner_region in corners.items():
            values = corner_region.flatten()
            q1, q3 = np.percentile(values, [25, 75])
            iqr = q3 - q1
            mask = (values >= q1 - 1.5*iqr) & (values <= q3 + 1.5*iqr)
            corner_intensities[corner_name] = np.mean(values[mask])

        # Calculate normalized ratios
        corner_ratios = {k: min(max(v / center_intensity, 0.0), 2.0) 
                        for k, v in corner_intensities.items()}

        # Calculate average corner ratio and score
        avg_corner_ratio = np.mean(list(corner_ratios.values()))
        vignetting_score = min(100, avg_corner_ratio * 100)

        logging.info(f"Analysis complete - score: {vignetting_score:.2f}")
        
        # Generate visualization
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
            'visualization_path': viz_path
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
    """
    Analyze both lateral and longitudinal chromatic aberration in an image.
    
    Returns:
        Dictionary containing:
        - chromatic_aberration_score: Overall score (0-100)
        - lateral_ca: Lateral chromatic aberration measurements
        - longitudinal_ca: Longitudinal chromatic aberration measurements
        - visualization_path: Path to generated visualization
        - channel_differences: Color channel difference metrics
    """
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
        
        # Edge detection on green channel (typically sharpest)
        edges = cv2.Canny(g, 100, 200)
        
        # Dilate edges to create analysis region
        kernel = np.ones((5, 5), np.uint8)
        edge_region = cv2.dilate(edges, kernel, iterations=2)

        # Analyze lateral CA (color misalignment at edges)
        lateral_ca = analyze_lateral_ca(r, g, b, edge_region)

        # Analyze longitudinal CA (color fringing across contrast boundaries)
        longitudinal_ca = analyze_longitudinal_ca(r, g, b, edge_region)

        # Calculate overall score
        lateral_score = calculate_lateral_ca_score(lateral_ca)
        longitudinal_score = calculate_longitudinal_ca_score(longitudinal_ca)
        
        # Weight lateral CA more heavily as it's typically more noticeable
        overall_score = (lateral_score * 0.6 + longitudinal_score * 0.4)

        # Create visualization
        viz_path = os.path.join(
            os.path.dirname(image_path),
            f"ca_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
        )
        create_ca_visualization(img, lateral_ca, longitudinal_ca, viz_path)

        return {
            'chromatic_aberration_score': float(overall_score),
            'lateral_ca': lateral_ca,
            'longitudinal_ca': longitudinal_ca,
            'visualization_path': viz_path,
            'analysis_time': datetime.now().strftime("%Y%m%d_%H%M%S")
        }

    except Exception as e:
        logging.error(f"Chromatic aberration analysis failed: {str(e)}")
        raise

def analyze_lateral_ca(r, g, b, edge_region):
    """Analyze lateral chromatic aberration by measuring RGB misalignment"""
    # Calculate center of mass for each channel along edges
    r_com = cv2.moments(cv2.bitwise_and(r, r, mask=edge_region))
    g_com = cv2.moments(cv2.bitwise_and(g, g, mask=edge_region))
    b_com = cv2.moments(cv2.bitwise_and(b, b, mask=edge_region))

    # Calculate centroid positions
    centers = {}
    for moments, channel in [(r_com, 'r'), (g_com, 'g'), (b_com, 'b')]:
        if moments['m00'] != 0:
            cx = moments['m10'] / moments['m00']
            cy = moments['m01'] / moments['m00']
            centers[channel] = (cx, cy)

    # Calculate displacement vectors between channels
    displacements = {
        'r_g': np.linalg.norm(np.array(centers['r']) - np.array(centers['g'])),
        'r_b': np.linalg.norm(np.array(centers['r']) - np.array(centers['b'])),
        'g_b': np.linalg.norm(np.array(centers['g']) - np.array(centers['b']))
    }

    return {
        'displacements': displacements,
        'centers': centers
    }

def analyze_longitudinal_ca(r, g, b, edge_region):
    """Analyze longitudinal chromatic aberration through color intensity ratios"""
    # Calculate color ratios along edges
    r_intensity = cv2.mean(r, mask=edge_region)[0]
    g_intensity = cv2.mean(g, mask=edge_region)[0]
    b_intensity = cv2.mean(b, mask=edge_region)[0]

    # Calculate intensity ratios
    ratios = {
        'r_g': r_intensity / g_intensity if g_intensity > 0 else 0,
        'r_b': r_intensity / b_intensity if b_intensity > 0 else 0,
        'g_b': g_intensity / b_intensity if b_intensity > 0 else 0
    }

    # Calculate color fringing metrics
    fringing = {
        'r_g': cv2.mean(cv2.absdiff(r, g), mask=edge_region)[0],
        'r_b': cv2.mean(cv2.absdiff(r, b), mask=edge_region)[0],
        'g_b': cv2.mean(cv2.absdiff(g, b), mask=edge_region)[0]
    }

    return {
        'intensity_ratios': ratios,
        'fringing_metrics': fringing
    }

def calculate_lateral_ca_score(lateral_ca):
    """Calculate score for lateral CA (0-100, higher is better)"""
    max_displacement = max(lateral_ca['displacements'].values())
    # Convert displacement to score (0-100)
    # Typical threshold for noticeable CA is around 1-2 pixels
    score = 100 * (1 - min(max_displacement / 3.0, 1.0))
    return score

def calculate_longitudinal_ca_score(longitudinal_ca):
    """Calculate score for longitudinal CA (0-100, higher is better)"""
    max_fringing = max(longitudinal_ca['fringing_metrics'].values())
    # Convert fringing to score (0-100)
    score = 100 * (1 - min(max_fringing / 255.0, 1.0))
    return score

def create_ca_visualization(img, lateral_ca, longitudinal_ca, output_path):
    """
    Create visualization of chromatic aberration analysis results.
    
    Args:
        img: Original image
        lateral_ca: Results from lateral CA analysis
        longitudinal_ca: Results from longitudinal CA analysis
        output_path: Path to save visualization
    """
    try:
        # Create a copy for visualization
        viz = img.copy()
        height, width = viz.shape[:2]
        
        # Split into channels and convert to float32 for calculations
        b, g, r = [channel.astype(np.float32) for channel in cv2.split(viz)]
        
        # Create heatmap of color differences
        heatmap = np.zeros((height, width), dtype=np.float32)
        for y in range(height):
            for x in range(width):
                r_val = float(r[y, x])
                g_val = float(g[y, x])
                b_val = float(b[y, x])
                # Calculate differences with proper type handling
                color_diff = max(
                    abs(r_val - g_val),
                    abs(r_val - b_val),
                    abs(g_val - b_val)
                )
                # Scale and clip the difference
                heatmap[y, x] = min(255.0, color_diff * 2.0)
        
        # Convert heatmap to uint8 for visualization
        heatmap = np.clip(heatmap, 0, 255).astype(np.uint8)
        
        # Rest of the visualization code remains the same...
        heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        
        # Create final visualization with both original and heatmap
        final_viz = np.zeros((height, width * 2, 3), dtype=np.uint8)
        final_viz[:, :width] = viz
        final_viz[:, width:] = heatmap_color
        
        # Add text annotations and metrics as before
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(final_viz, 'Original with CA Centers', 
                   (10, height - 20), font, 0.7, (255, 255, 255), 2)
        cv2.putText(final_viz, 'Color Difference Heatmap', 
                   (width + 10, height - 20), font, 0.7, (255, 255, 255), 2)
        
        # Save visualization
        cv2.imwrite(output_path, final_viz)
        return True
        
    except Exception as e:
        logging.error(f"Failed to create CA visualization: {str(e)}")
        return False

def import_raw_file(raw_path, processing_options=None):
    """Import and process a RAW image file with optional processing parameters
    
    Args:
        raw_path: Path to the RAW file
        processing_options: Dictionary of processing options for rawpy.postprocess
            Supported options include:
            - use_camera_wb (bool): Use camera white balance
            - no_auto_bright (bool): Disable auto brightness
            - output_bps (int): Bits per sample (8 or 16)
            - bright (float): Brightness multiplier
            - demosaic_algorithm (str): Demosaic algorithm to use
            
    Returns:
        numpy.ndarray: Processed image as a numpy array in RGB format
    """
    try:
        if not os.path.exists(raw_path):
            raise FileNotFoundError(f"RAW file not found: {raw_path}")
            
        # Default processing options
        default_options = {
            'use_camera_wb': True,
            'no_auto_bright': True,
            'output_bps': 8,
            'bright': None,
            'demosaic_algorithm': None
        }
        
        # Update defaults with provided options
        if processing_options:
            default_options.update(processing_options)
            
        with rawpy.imread(raw_path) as raw:
            # Process the RAW file with specified options
            rgb = raw.postprocess(
                use_camera_wb=default_options['use_camera_wb'],
                no_auto_bright=default_options['no_auto_bright'],
                output_bps=default_options['output_bps'],
                bright=default_options['bright'],
                demosaic_algorithm=default_options['demosaic_algorithm']
            )
            return rgb
            
    except Exception as e:
        logging.error(f"Error importing RAW file {raw_path}: {e}")
        raise


def match_template_sift(image_path, template_path, threshold=0.7):
    """
    Match template in image using SIFT features and homography.

    Args:
        image_path: Path to main image
        template_path: Path to template to find
        threshold: SIFT matching threshold (0-1)

    Returns:
        matches_mask: Mask of matched points
        homography: Transformation matrix between template and image
        match_points: Coordinates of matched points
    """
    # Load image and template, converting to grayscale for SIFT
    if image_path.lower().endswith(('.cr2', '.nef', '.arw')):
        with rawpy.imread(image_path) as raw:
            image = raw.postprocess(
                use_camera_wb=True,
                half_size=True,
                no_auto_bright=True,
                output_bps=8
            )
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)

    if image is None or template is None:
        raise ValueError("Failed to load image or template")

    # Initialize SIFT detector
    sift = cv2.SIFT_create()

    # Find keypoints and descriptors
    kp1, des1 = sift.detectAndCompute(template, None)
    kp2, des2 = sift.detectAndCompute(image, None)

    if des1 is None or des2 is None:
        return None, None, []

    # Match descriptors using FLANN
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)

    # Store good matches using Lowe's ratio test
    good_matches = []
    for m, n in matches:
        if m.distance < threshold * n.distance:
            good_matches.append(m)

    if len(good_matches) < 4:
        return None, None, []

    # Extract matched keypoints
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    # Find homography
    homography, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    matches_mask = mask.ravel().tolist()

    # Get matched point coordinates
    match_points = [(int(dst_pts[i][0][0]), int(dst_pts[i][0][1]))
                    for i, match in enumerate(matches_mask) if match]

    return matches_mask, homography, match_points


def create_template_visualization(image_path, template_path, matches_mask, match_points):
    """Create visualization of template matching results"""
    # Load images in color for visualization
    if image_path.lower().endswith(('.cr2', '.nef', '.arw')):
        with rawpy.imread(image_path) as raw:
            image = raw.postprocess()
    else:
        image = cv2.imread(image_path)

    # Draw matched points
    viz = image.copy()
    for point in match_points:
        cv2.circle(viz, point, 5, (0, 255, 0), -1)

    # Save visualization
    viz_path = os.path.join(
        os.path.dirname(image_path),
        f"template_match_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
    )
    cv2.imwrite(viz_path, viz)
    return viz_path




def extract_roi(image, center_point, size=100):
    """Extract region of interest around center point"""
    x, y = center_point
    half_size = size // 2
    return image[
           max(0, y - half_size):min(image.shape[0], y + half_size),
           max(0, x - half_size):min(image.shape[1], x + half_size)
           ]

