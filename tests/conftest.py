import pytest
import pytest_asyncio
import os
import shutil
import tempfile
from unittest.mock import MagicMock, patch
from camera_manager import CameraManager
from dataset_manager import DatasetManager

# Configure matplotlib to use non-interactive backend for testing
import matplotlib
matplotlib.use('Agg')

# Set default fixture loop scope for pytest-asyncio
pytest_asyncio.default_fixture_loop_scope = "function"

@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files"""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)

@pytest.fixture
def mock_camera():
    """Create a mock camera object"""
    camera = MagicMock()
    # Add common camera method mocks
    camera.get_config.return_value = MagicMock()
    camera.capture.return_value = MagicMock()
    camera.file_get.return_value = MagicMock()
    return camera

@pytest.fixture
def mock_camera_manager():
    """Create a mock camera manager with properly mocked methods"""
    manager = CameraManager()
    
    # Create mock camera
    mock_camera = MagicMock()
    
    # Setup basic camera mocks
    config = MagicMock()
    widget = MagicMock()
    widget.get_value.return_value = 0  # Camera ready
    config.get_child_by_name.return_value = widget
    mock_camera.get_config.return_value = config
    
    # Mock camera methods
    mock_camera.capture = MagicMock()
    mock_camera.file_get = MagicMock()
    mock_camera.exit = MagicMock()
    mock_camera.init = MagicMock()
    
    # Setup camera manager
    manager.camera = mock_camera
    manager.connected = True
    
    # Mock manager methods
    manager.initialize_camera = MagicMock(return_value=True)
    manager.wait_for_camera_ready = MagicMock(return_value=True)
    manager.capture_image = MagicMock()
    manager.get_camera_config = MagicMock(return_value={"iso": "400"})
    manager.set_camera_config = MagicMock(return_value=True)
    
    return manager

@pytest.fixture
def dataset_manager(temp_dir):
    """Create a dataset manager instance with temporary directory"""
    return DatasetManager(base_path=temp_dir)

@pytest.fixture
def sample_image(temp_dir):
    """Create a sample test image"""
    import numpy as np
    import cv2
    
    # Create a simple test image
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    cv2.circle(img, (50, 50), 20, (255, 255, 255), -1)
    
    # Save the image
    image_path = os.path.join(temp_dir, "test_image.jpg")
    cv2.imwrite(image_path, img)
    
    return image_path 