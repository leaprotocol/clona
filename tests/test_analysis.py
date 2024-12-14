import pytest
import numpy as np
import cv2
import os
from analysis import (
    analyze_sharpness,
    analyze_distortion,
    analyze_vignetting,
    analyze_bokeh,
    analyze_chromatic_aberration,
    calculate_physical_mtf,
    calculate_calibrated_mtf_score,
    create_calibrated_visualization,
    calculate_mtf_improved,
    calculate_mtf_score,
    create_sharpness_visualization,
    calculate_line_deviations,
    create_vignetting_visualization,
    convert_raw_to_jpeg,
    analyze_scenario_photo,
    analyze_shape_regularity,
    analyze_color_fringing,
    analyze_intensity_distribution,
    create_bokeh_visualization,
    analyze_lateral_ca,
    analyze_longitudinal_ca,
    calculate_lateral_ca_score,
    calculate_longitudinal_ca_score,
    create_ca_visualization,
    import_raw_file
)
from unittest.mock import MagicMock, patch

@pytest.fixture
def sample_image(temp_dir):
    """Create a test image with a simple pattern"""
    # Create a larger image for better analysis
    img = np.zeros((500, 500, 3), dtype=np.uint8)
    
    # Add grid pattern for distortion analysis
    for i in range(0, 500, 50):
        cv2.line(img, (i, 0), (i, 499), (255, 255, 255), 2)
        cv2.line(img, (0, i), (499, i), (255, 255, 255), 2)
    
    # Add some circles for bokeh analysis
    cv2.circle(img, (250, 250), 30, (200, 200, 200), -1)
    cv2.circle(img, (150, 150), 20, (180, 180, 180), -1)
    
    # Add diagonal lines for sharpness analysis
    cv2.line(img, (100, 100), (400, 400), (255, 255, 255), 2)
    cv2.line(img, (400, 100), (100, 400), (255, 255, 255), 2)
    
    # Add color variations for chromatic aberration analysis
    cv2.circle(img, (250, 250), 40, (255, 0, 0), 2)  # Blue
    cv2.circle(img, (250, 250), 42, (0, 255, 0), 2)  # Green
    cv2.circle(img, (250, 250), 44, (0, 0, 255), 2)  # Red
    
    # Add vignetting effect
    center = (250, 250)
    for y in range(500):
        for x in range(500):
            distance = np.sqrt((x - center[0])**2 + (y - center[1])**2)
            factor = 1 - min(1.0, distance / 350)
            img[y, x] = img[y, x] * factor
    
    # Save the image
    path = os.path.join(temp_dir, "test_image.jpg")
    cv2.imwrite(path, img)
    return path

def test_analyze_sharpness(sample_image):
    """Test sharpness analysis"""
    results = analyze_sharpness(sample_image)
    
    assert results is not None
    assert 'sharpness_score' in results
    assert 'edge_intensity' in results
    assert 'edge_density' in results
    assert 'local_variance' in results
    assert 'visualization_path' in results
    assert os.path.exists(results['visualization_path'])

def test_analyze_distortion(sample_image):
    """Test distortion analysis"""
    results = analyze_distortion(sample_image)
    
    assert results is not None
    assert 'distortion_score' in results
    assert 'visualization_path' in results
    assert os.path.exists(results['visualization_path'])

def test_analyze_vignetting(sample_image):
    """Test vignetting analysis"""
    results = analyze_vignetting(sample_image)
    
    assert results is not None
    assert 'vignetting_score' in results
    assert 'visualization_path' in results
    assert os.path.exists(results['visualization_path'])

def test_analyze_chromatic_aberration(sample_image):
    """Test chromatic aberration analysis"""
    results = analyze_chromatic_aberration(sample_image)
    
    assert results is not None
    assert 'chromatic_aberration_score' in results
    assert 'visualization_path' in results
    assert os.path.exists(results['visualization_path'])

def test_raw_processing(sample_image):
    """Test RAW file processing"""
    with patch('rawpy.imread') as mock_imread:
        # Mock RAW data
        mock_raw = MagicMock()
        mock_raw.raw_image = np.random.randint(0, 65535, (100, 100), dtype=np.uint16)
        mock_raw.raw_colors = np.zeros((100, 100), dtype=np.uint8)
        mock_raw.postprocess.return_value = np.zeros((100, 100, 3), dtype=np.uint8)
        mock_imread.return_value.__enter__.return_value = mock_raw
        
        # Process RAW file
        result = import_raw_file(sample_image)
        assert result is not None
        assert isinstance(result, np.ndarray)
        assert len(result.shape) == 3  # Should be RGB

def test_image_preprocessing(sample_image):
    """Test image preprocessing steps"""
    # Load test image
    img = cv2.imread(sample_image)
    
    # Test MTF calculation
    mtf = calculate_mtf_improved(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
    assert isinstance(mtf, np.ndarray)
    assert len(mtf) > 0
    
    # Test MTF score calculation
    score = calculate_mtf_score(mtf)
    assert isinstance(score, (int, float))
    assert 0 <= score <= 100

def test_feature_detection(sample_image):
    """Test feature detection algorithms"""
    img = cv2.imread(sample_image)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Test edge detection using Canny
    edges = cv2.Canny(gray, 100, 200)
    assert edges.shape == gray.shape
    assert edges.dtype == np.uint8
    
    # Test corner detection using Harris
    corners = cv2.goodFeaturesToTrack(gray, 25, 0.01, 10)
    assert corners is not None
    assert len(corners) > 0

def test_mtf_calculation(sample_image):
    """Test MTF calculation"""
    img = cv2.imread(sample_image)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Calculate MTF
    mtf = calculate_physical_mtf(gray)
    assert isinstance(mtf, np.ndarray)
    assert len(mtf) > 0
    assert all(0 <= x <= 1 for x in mtf)  # MTF values should be between 0 and 1

def test_result_calculation():
    """Test result calculation and formatting"""
    # Mock analysis results
    mock_results = {
        'mtf50': 0.5,
        'distortion_coefficient': 0.02,
        'vignetting_factor': 0.95,
        'ca_magnitude': 0.01
    }
    
    # Calculate MTF score
    mtf_score = calculate_calibrated_mtf_score(np.array([0.8, 0.6, 0.4, 0.2]))
    assert isinstance(mtf_score, (int, float))
    assert 0 <= mtf_score <= 100

def test_visualization_creation(sample_image, temp_dir):
    """Test visualization creation functions"""
    img = cv2.imread(sample_image)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    mtf = calculate_mtf_improved(gray)
    
    # Test sharpness visualization
    viz_path = os.path.join(temp_dir, "sharpness_viz.jpg")
    create_sharpness_visualization(img, edges, mtf, viz_path)
    assert os.path.exists(viz_path)
    
    # Test vignetting visualization
    viz_path = os.path.join(temp_dir, "vignette_viz.jpg")
    create_vignetting_visualization(sample_image, viz_path)
    assert os.path.exists(viz_path)

def test_bokeh_analysis(sample_image):
    """Test bokeh analysis functions"""
    # Test shape regularity analysis
    contour = np.array([[[0, 0]], [[0, 10]], [[10, 10]], [[10, 0]]], dtype=np.int32)
    score, metrics = analyze_shape_regularity(contour, 10)
    assert isinstance(score, (int, float))
    assert 'circularity' in metrics
    
    # Test color fringing analysis
    roi = cv2.imread(sample_image)
    score, metrics = analyze_color_fringing(roi, (50, 50), 20)
    assert isinstance(score, (int, float))
    assert 'average_color_difference' in metrics
    
    # Test intensity distribution analysis
    score, metrics = analyze_intensity_distribution(roi, (50, 50), 20)
    assert isinstance(score, (int, float))
    assert 'mean_intensity' in metrics

def test_chromatic_aberration_components(sample_image):
    """Test individual CA analysis components"""
    img = cv2.imread(sample_image)
    b, g, r = cv2.split(img)
    edges = cv2.Canny(g, 100, 200)
    
    # Test lateral CA analysis
    lateral_ca = analyze_lateral_ca(r, g, b, edges)
    assert 'displacements' in lateral_ca
    assert 'centers' in lateral_ca
    
    # Test longitudinal CA analysis
    longitudinal_ca = analyze_longitudinal_ca(r, g, b, edges)
    assert 'intensity_ratios' in longitudinal_ca
    assert 'fringing_metrics' in longitudinal_ca
    
    # Test CA score calculations
    lateral_score = calculate_lateral_ca_score(lateral_ca)
    assert isinstance(lateral_score, (int, float))
    assert 0 <= lateral_score <= 100
    
    longitudinal_score = calculate_longitudinal_ca_score(longitudinal_ca)
    assert isinstance(longitudinal_score, (int, float))
    assert 0 <= longitudinal_score <= 100

def test_raw_file_conversion(sample_image, temp_dir):
    """Test RAW to JPEG conversion"""
    jpeg_path = os.path.join(temp_dir, "test.jpg")
    result = convert_raw_to_jpeg(sample_image, jpeg_path)
    assert isinstance(result, bool)
    if result:
        assert os.path.exists(jpeg_path)

def test_scenario_analysis(sample_image):
    """Test scenario-based analysis"""
    scenario = {'type': 'vignette'}
    photo_info = {'path': sample_image}
    
    result = analyze_scenario_photo(scenario, photo_info)
    assert isinstance(result, bool)
    if result:
        assert 'analysis' in photo_info
        assert 'vignetting_results' in photo_info['analysis']