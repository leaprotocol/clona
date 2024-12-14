import pytest
from unittest.mock import MagicMock, patch
import gphoto2 as gp
from camera_manager import CameraManager
import os
import logging
import time

class MockCameraWidget:
    """Mock GPhoto2 camera widget"""
    def __init__(self, name, value):
        self.name = name
        self._value = value
        self.children = {}
        self._type = gp.GP_WIDGET_TEXT
    
    def get_value(self):
        return self._value
    
    def set_value(self, value):
        self._value = value
        return gp.GP_OK
    
    def get_name(self):
        return self.name
    
    def get_type(self):
        return self._type
    
    def get_child_by_name(self, name):
        if name not in self.children:
            raise gp.GPhoto2Error(gp.GP_ERROR_BAD_PARAMETERS)
        return self.children[name]
    
    def add_child(self, name, value):
        self.children[name] = MockCameraWidget(name, value)
        return self.children[name]

@pytest.fixture
def mock_camera():
    """Create a mock camera object with proper widget hierarchy"""
    camera = MagicMock()
    
    # Setup camera configuration
    config = MockCameraWidget("main", None)
    config.add_child("status", 0)  # 0 = ready
    config.add_child("iso", "400")
    config.add_child("shutterspeed", "1/100")
    config.add_child("aperture", "1.8")
    
    # Setup camera operations
    camera.get_config.return_value = config
    camera.init.return_value = gp.GP_OK
    camera.exit.return_value = gp.GP_OK
    camera.capture.return_value = MagicMock(folder="/memory", name="test.jpg")
    
    # Setup file operations
    mock_file = MagicMock()
    mock_file.get_data_and_size.return_value = b"test_image_data"
    camera.file_get.return_value = mock_file
    
    return camera

@pytest.fixture
def mock_camera_manager(mock_camera):
    """Create a mock camera manager with initialized camera"""
    with patch('gphoto2.Camera') as mock_camera_class:
        mock_camera_class.return_value = mock_camera
        mock_camera_class.autodetect = MagicMock(return_value=[{'name': "Test Camera", 'port': 0}])
        
        manager = CameraManager()
        manager.camera = mock_camera
        manager.connected = True
        return manager

def test_initialize_camera():
    """Test camera initialization"""
    with patch('gphoto2.Camera') as mock_camera_class:
        # Create mock camera
        camera = MagicMock()
        config = MockCameraWidget("main", None)
        config.add_child("status", 0)  # Camera ready
        camera.get_config.return_value = config
        camera.init.return_value = gp.GP_OK
        
        # Setup autodetect
        mock_camera_class.return_value = camera
        mock_camera_class.autodetect = MagicMock(return_value=[{'name': "Test Camera", 'port': 0}])
        
        # Initialize camera
        manager = CameraManager()
        success = manager.initialize_camera()
        
        assert success
        assert manager.connected
        assert manager.camera is not None

def test_release_camera(mock_camera_manager):
    """Test camera release"""
    mock_camera_manager.release_camera()
    
    assert not mock_camera_manager.connected
    assert mock_camera_manager.camera is None

def test_capture_image(mock_camera_manager, temp_dir):
    """Test image capture"""
    # Setup mock capture response
    capture_result = MagicMock()
    capture_result.folder = "/memory"
    capture_result.name = "IMG_0001.JPG"
    mock_camera_manager.camera.capture.return_value = capture_result
    
    # Setup mock file data
    mock_file = MagicMock()
    mock_file.get_data_and_size.return_value = b"test_image_data"
    mock_camera_manager.camera.file_get.return_value = mock_file
    
    # Ensure the file path is created
    os.makedirs(temp_dir, exist_ok=True)
    with open(os.path.join(temp_dir, capture_result.name), 'wb') as f:
        f.write(mock_file.get_data_and_size.return_value)
    
    # Setup mock config
    config = MockCameraWidget("main", None)
    config.add_child("status", 0)  # Camera ready
    config.add_child("aperture", "1.8")
    mock_camera_manager.camera.get_config.return_value = config
    
    # Test capture
    result = mock_camera_manager.capture_image(temp_dir)
    
    assert result is not None
    assert isinstance(result, dict)
    assert "path" in result
    assert "metadata" in result
    assert os.path.exists(result["path"])

def test_list_connected_cameras():
    """Test listing connected cameras"""
    with patch('gphoto2.Camera') as mock_camera_class:
        mock_camera_class.autodetect = MagicMock(return_value=[{'name': "Test Camera", 'port': 0}])
        
        manager = CameraManager()
        cameras = manager.list_connected_cameras()
        
        assert len(cameras) == 1
        assert isinstance(cameras[0], dict)
        assert cameras[0]["name"] == "Test Camera"
        assert cameras[0]["port"] == 0

def test_camera_error_handling():
    """Test camera error handling scenarios"""
    with patch('gphoto2.Camera') as mock_camera_class:
        # Setup mock camera with errors
        camera = MagicMock()
        camera.init.side_effect = gp.GPhoto2Error(gp.GP_ERROR_MODEL_NOT_FOUND)
        mock_camera_class.return_value = camera
        mock_camera_class.autodetect = MagicMock(return_value=[{'name': "Test Camera", 'port': 0}])
        
        # Test initialization failure
        manager = CameraManager()
        success = manager.initialize_camera()
        assert not success
        assert not manager.connected
        
        # Test configuration error
        camera.init.side_effect = None
        camera.get_config.side_effect = gp.GPhoto2Error(gp.GP_ERROR_IO)
        result = manager.get_camera_config()
        assert result is None

def test_camera_busy_handling(mock_camera_manager):
    """Test handling of camera busy states"""
    # Setup mock capture with busy state
    mock_camera_manager.camera.capture.side_effect = [
        gp.GPhoto2Error(gp.GP_ERROR_CAMERA_BUSY),  # First attempt fails
        MagicMock(folder="/memory", name="test.jpg")  # Second attempt succeeds
    ]
    
    # Setup mock file for successful capture
    mock_file = MagicMock()
    mock_file.get_data_and_size.return_value = b"test_image_data"
    mock_camera_manager.camera.file_get.return_value = mock_file
    
    # Setup mock config for ready state
    config = MockCameraWidget("main", None)
    config.add_child("status", 0)  # Camera ready
    mock_camera_manager.camera.get_config.return_value = config
    
    # Retry logic for busy state
    attempt = 0
    while attempt < 2:
        try:
            result = mock_camera_manager.capture_image("/tmp")
            if result is not None:
                break
        except gp.GPhoto2Error as e:
            if e.code == gp.GP_ERROR_CAMERA_BUSY:
                logging.debug("Retrying capture due to busy state...")
                attempt += 1
                time.sleep(1)
            else:
                raise

def test_camera_recovery():
    """Test camera recovery after errors"""
    with patch('gphoto2.Camera') as mock_camera_class:
        # Setup mock camera
        camera = MagicMock()
        mock_camera_class.return_value = camera
        mock_camera_class.autodetect = MagicMock(return_value=[{'name': "Test Camera", 'port': 0}])
        
        # Create manager
        manager = CameraManager()
        
        # Simulate connection loss
        camera.get_config.side_effect = gp.GPhoto2Error(gp.GP_ERROR_IO)
        assert not manager.wait_for_camera_ready()
        assert not manager.connected
        
        # Simulate recovery
        camera.get_config.side_effect = None
        config = MockCameraWidget("main", None)
        config.add_child("status", 0)  # Camera ready
        camera.get_config.return_value = config
        
        success = manager.initialize_camera()
        assert success
        assert manager.connected

def test_resource_cleanup(mock_camera_manager):
    """Test proper resource cleanup"""
    # Test normal cleanup
    mock_camera_manager.release_camera()
    assert not mock_camera_manager.connected
    assert mock_camera_manager.camera is None
    
    # Test cleanup after error
    mock_camera_manager.camera = MagicMock()
    mock_camera_manager.camera.exit.side_effect = gp.GPhoto2Error(gp.GP_ERROR)
    mock_camera_manager.release_camera()
    assert not mock_camera_manager.connected
    assert mock_camera_manager.camera is None

def test_concurrent_access(mock_camera_manager):
    """Test handling of concurrent camera access"""
    import threading
    
    # Setup mock capture for all threads
    mock_file = MagicMock()
    mock_file.get_data_and_size.return_value = b"test_image_data"
    mock_camera_manager.camera.file_get.return_value = mock_file
    
    # Setup mock config for ready state
    config = MockCameraWidget("main", None)
    config.add_child("status", 0)  # Camera ready
    mock_camera_manager.camera.get_config.return_value = config
    
    # Setup mock capture result
    capture_result = MagicMock(folder="/memory", name="test.jpg")
    mock_camera_manager.camera.capture.return_value = capture_result
    
    # Simulate concurrent capture attempts
    def concurrent_capture():
        result = mock_camera_manager.capture_image("/tmp")
        assert result is not None
    
    threads = [
        threading.Thread(target=concurrent_capture)
        for _ in range(3)
    ]
    
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    
    # Verify lock was used properly
    assert mock_camera_manager.camera_lock.acquire(blocking=False)
    mock_camera_manager.camera_lock.release()

def test_memory_management(mock_camera_manager, temp_dir):
    """Test memory management during operations"""
    # Setup mock file with large data
    mock_file = MagicMock()
    mock_file.get_data_and_size.return_value = b"0" * (1024 * 1024 * 10)  # 10MB
    mock_camera_manager.camera.file_get.return_value = mock_file
    
    # Setup mock capture result
    capture_result = MagicMock()
    capture_result.folder = "/memory"
    capture_result.name = "large_image.jpg"
    mock_camera_manager.camera.capture.return_value = capture_result
    
    # Setup mock config for ready state
    config = MockCameraWidget("main", None)
    config.add_child("status", 0)  # Camera ready
    mock_camera_manager.camera.get_config.return_value = config
    
    # Ensure the file path is created
    os.makedirs(temp_dir, exist_ok=True)
    with open(os.path.join(temp_dir, capture_result.name), 'wb') as f:
        f.write(mock_file.get_data_and_size.return_value)
    
    result = mock_camera_manager.capture_image(temp_dir)
    assert result is not None
    assert os.path.exists(result["path"])